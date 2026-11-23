"""Convert A2 runs on the stand into the static map: the wake force per point.

Copy the card files into the run first, `usd.py collect <card> --block a2_stand` puts them into
`<run>/sd/<role>/`. The victim's card holds the load cell and its rpm; it is mapped onto PC time
through its radio log, as in A3 on the stand (`a3_stand_convert.load_run`). The source's position
comes from the flight records, stamped in PC time at 100 Hz.

Each dwell of the design gives one row of `items.csv` over its used seconds, after the discarded
`a2.discard_s` until its end. The geometry is measured, never the nominal: the source's mean
position relative to the stand and its spread. The force is the load cell's change from the
**baseline of the item's part**, the mean over the used seconds of the part's first reference, where
the source is outside the jet: `fz_rel_N`, up positive. In a still run (S0) it is the wake force on
the unpowered airframe. In a spin run (S1) the first reference is also the spin-alone baseline, and
`dthrust_rpm_N` is the change of the rotor thrust from the measured rpm (A1's curve) against it, so
`dforce_wake_N = fz_rel_N - dthrust_rpm_N` is what the rotor speeds do not explain. A part that
ends on a reference gives its drift, `ref_drift_N`, the check of the tare.

An item is `ok`, `card_gap` (the card stalled inside its used seconds), `not_spun` (S1, the victim's
motors did not run through its part), `no_baseline` (its part's first reference is not ok) or
`not_flown`.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/a2_stand_convert.py \
        results/downwash/a2/stand_flight1_<still|spin>_<time> [--calibration FILE]
    pixi run -e sim2real python crazyflow_experiments/downwash/a2_stand_convert.py --combine \
        results/downwash/a2/stand_flight1_*

`--combine` writes the ok items of the converted runs to `static_map_S0.csv` (still) and
`static_map_S1.csv` (spin) in `results/downwash/a2/`.
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash import A2_RESULTS_DIR, DOWNWASH_PLOTS_DIR, plots
from crazyflow_experiments.downwash.a3_stand_convert import CARD_MAX_GAP, G, load_run

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("crazyflow_experiments.downwash.a2_stand_convert")  # Not __name__
logger.setLevel(logging.INFO)

STAND_PLOTS_DIR = DOWNWASH_PLOTS_DIR / "a2_stand"
MODES = {"still": "S0", "spin": "S1"}
MIN_SAMPLES = 50  # Card samples a dwell's used seconds need, at about 500 Hz
ITEM_KEYS = ("leg", "part", "kind", "height_sep", "dx")


def window(t: NDArray, t0: float, t1: float) -> NDArray:
    """Mask of the samples of `t` in [t0, t1]."""
    return (t >= t0) & (t <= t1)


def mean_over(t: NDArray, x: NDArray, t0: float, t1: float) -> tuple[NDArray, NDArray, int, float]:
    """Mean, standard deviation, sample count and longest gap of `x` (N, ...) over [t0, t1].

    NaNs are skipped, so a motor whose rpm had no reading leaves the others alone. The gap is the
    longest time between two samples of the window, the window's ends included.
    """
    mask = window(t, t0, t1)
    if mask.sum() < 2:
        nan = np.full(np.shape(x)[1:], np.nan)
        return nan, nan, int(mask.sum()), math.inf
    values = np.asarray(x, dtype=float)[mask]
    edges = np.concatenate([[t0], t[mask], [t1]])
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):  # A column without a reading is NaN
        mean = np.where(finite, values, 0.0).sum(axis=0) / count
        std = np.sqrt((np.where(finite, values - mean, 0.0) ** 2).sum(axis=0) / count)
    return mean, std, int(mask.sum()), float(np.max(np.diff(edges)))


def item_rows(
    run: Mapping, items: Sequence[Mapping], spun: Mapping[int, bool], mode: str, last_t: float
) -> list[dict]:
    """The rows of a run's items, see the module docstring.

    Args:
        run: The run on PC time, see `a3_stand_convert.load_run`.
        items: The run's design rows, `t_use_start` and `t_end` in timeline seconds.
        spun: Per part whether the victim's motors ran through it, a spin run only.
        mode: "still" or "spin".
        last_t: Timeline time of the last record, items after it were not flown.

    Returns:
        One row per item in flight order.
    """
    t, t_zero = run["t"], run["t_zero"]
    force = (run["weight_g"]) * 1e-3 * G
    stand = np.asarray(run["meta"]["stand_pos"], dtype=float)
    measured = {}
    for item in items:
        t0, t1 = t_zero + item["t_use_start"], t_zero + item["t_end"]
        fz, fz_std, n, gap = mean_over(t, force, t0, t1)
        thrust, _, _, _ = mean_over(t, run["thrust_rpm"], t0, t1)
        rpm, _, _, _ = mean_over(t, run["rpm"], t0, t1)
        valid = float(np.mean(np.isfinite(run["rpm"][window(t, t0, t1)]))) if n else math.nan
        mask = window(run["src_t"], t0, t1)
        rel = run["src_pos"][mask] - stand
        measured[int(item["leg"])] = {
            "fz": float(fz),
            "fz_std": float(fz_std),
            "n": n,
            "gap": gap,
            "thrust": float(thrust),
            "rpm": rpm,
            "rpm_valid": valid,
            "pos": rel.mean(axis=0) if len(rel) else np.full(3, np.nan),
            "pos_std": rel.std(axis=0) if len(rel) else np.full(3, np.nan),
            "vbat": float(mean_over(t, run["vbat"], t0, t1)[0]),
        }
    rows = []
    by_part: dict[int, list[Mapping]] = {}
    for item in items:
        by_part.setdefault(int(item["part"]), []).append(item)
    for part, part_items in by_part.items():
        first = measured[int(part_items[0]["leg"])]
        base_ok = (
            first["n"] >= MIN_SAMPLES and first["gap"] <= CARD_MAX_GAP and np.isfinite(first["fz"])
        )
        for item in part_items:
            m = measured[int(item["leg"])]
            status = "ok"
            if item["t_end"] > last_t:
                status = "not_flown"
            elif mode == "spin" and not spun.get(part, False):
                status = "not_spun"
            elif m["n"] < MIN_SAMPLES or m["gap"] > CARD_MAX_GAP:
                status = "card_gap"
            elif not base_ok:
                status = "no_baseline"
            nominal = np.array([item["dx"], item["dy"], item["height_sep"]], dtype=float)
            rows.append(
                {key: item[key] for key in ITEM_KEYS}
                | {
                    "mode": mode,
                    "status": status,
                    "dx_meas": m["pos"][0],
                    "dy_meas": m["pos"][1],
                    "dz_meas": m["pos"][2],
                    "dx_err": m["pos"][0] - nominal[0],
                    "dz_err": m["pos"][2] - nominal[2],
                    "pos_std_xy": float(np.linalg.norm(m["pos_std"][:2])),
                    "fz_N": m["fz"],
                    "fz_rel_N": m["fz"] - first["fz"],
                    "fz_std_N": m["fz_std"],
                    "thrust_rpm_N": m["thrust"],
                    "dthrust_rpm_N": m["thrust"] - first["thrust"],
                    "dforce_wake_N": (m["fz"] - first["fz"]) - (m["thrust"] - first["thrust"]),
                    "rpm_mean": float(np.nanmean(m["rpm"]))
                    if np.isfinite(m["rpm"]).any()
                    else math.nan,
                    "rpm_valid": m["rpm_valid"],
                    "vbat_V": m["vbat"],
                    "card_samples": m["n"],
                    "card_gap_s": m["gap"] if math.isfinite(m["gap"]) else math.nan,
                    "ref_drift_N": (m["fz"] - first["fz"])
                    if item["kind"] == "reference" and item is not part_items[0]
                    else math.nan,
                }
            )
    return rows


def quicklook(
    rows: Sequence[Mapping], run: Mapping, items: Sequence[Mapping], title: str, path: Path
) -> Path:
    """The load cell force over the run's dwells, the used seconds shaded, the references marked."""
    t_zero = run["t_zero"]
    fig, axes = plots.figure(2, row_height=2.4, sharex=False)
    t = run["t"] - t_zero
    force = run["weight_g"] * 1e-3 * G
    lo, hi = items[0]["t_dwell_start"] - 6.0, items[-1]["t_end"] + 3.0
    mask = (t >= lo) & (t <= hi)
    axes[0, 0].plot(
        t[mask], (force - np.nanmedian(force[mask]))[mask] * 1e3, color=plots.SERIES[0], lw=0.8
    )
    for item in items:
        color = plots.GRID if item["kind"] == "reference" else plots.SHADE
        axes[0, 0].axvspan(item["t_use_start"], item["t_end"], color=color, lw=0)
    axes[0, 0].set_ylabel("load cell [mN], from its median")
    axes[0, 0].set_title(f"{title}: used seconds shaded, references darker", loc="left")
    ok = [r for r in rows if r["status"] == "ok" and r["kind"] == "point"]
    for color, height in zip(plots.SERIES, sorted({r["height_sep"] for r in ok})):
        pick = [r for r in ok if r["height_sep"] == height]
        axes[1, 0].plot(
            [r["dx_meas"] for r in pick],
            [1e3 * r["fz_rel_N"] for r in pick],
            "o",
            color=color,
            label=f"Δz {height:.2f} m",
        )
    axes[1, 0].set_xlabel("measured Δx [m]")
    axes[1, 0].set_ylabel("ΔF_z from the part's baseline [mN]")
    plots.legend(axes[1, 0], loc="lower right", ncols=2)
    return plots.save(fig, path)


def convert(folder: Path, calibration: Path | None = None) -> list[dict]:
    """Convert a run: items.csv, the sync report and the quicklook; returns the rows."""
    folder = Path(folder)
    run = load_run(folder, calibration)
    meta = run["meta"]
    meta["folder"] = str(folder)
    mode = meta["mode"]
    items = sorted(meta["design_rows"], key=lambda r: int(r["index_in_flight"]))
    outcomes = meta["victim_schedule"]["legs"]
    spun = {int(part): bool(o["spun"]) for part, o in outcomes.items()}
    records = run["src_t"]
    last_t = float(records[-1] - run["t_zero"]) if len(records) else -math.inf
    rows = item_rows(run, items, spun, mode, last_t)
    for row in rows:
        row["run"] = folder.name
    write_rows(folder / "items.csv", rows)
    (folder / "sync.json").write_text(json.dumps(run["sync"], indent=2, default=float))
    quicklook(rows, run, items, folder.name, STAND_PLOTS_DIR / f"{folder.name}.png")
    counts = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
    logger.info(f"{folder.name}: {len(rows)} items, {counts}")
    for row in rows:
        if row["kind"] == "reference" and math.isfinite(row["ref_drift_N"]):
            logger.info(
                f"  part {int(row['part'])}: reference drift {row['ref_drift_N'] * 1e3:+.2f} mN"
                f" ({row['status']})"
            )
    return rows


def write_rows(path: Path, rows: Sequence[Mapping]):
    """Write rows with the union of their columns."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def combine(folders: Sequence[Path]) -> dict[str, list[dict]]:
    """Merge the ok items of converted runs into `static_map_S0.csv` and `static_map_S1.csv`."""
    maps: dict[str, list[dict]] = {"S0": [], "S1": []}
    for folder in folders:
        with open(Path(folder) / "items.csv", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok":
                    maps[MODES[row["mode"]]].append(row)
    for name, rows in maps.items():
        rows.sort(key=lambda r: (-float(r["height_sep"]), float(r["dx"]), r["run"]))
        if rows:
            write_rows(A2_RESULTS_DIR / f"static_map_{name}.csv", rows)
        logger.info(f"{len(rows)} ok items in static_map_{name}.csv")
    return maps


def main():
    """Command line: convert runs, or combine converted ones."""
    parser = argparse.ArgumentParser(description="Convert A2 runs on the stand.")
    parser.add_argument("folders", type=Path, nargs="+", help="Run folders")
    parser.add_argument("--calibration", type=Path, help="calibscale.py file replacing a and b")
    parser.add_argument("--combine", action="store_true", help="Merge converted runs")
    args = parser.parse_args()
    if args.combine:
        combine(args.folders)
        return
    for folder in args.folders:
        convert(folder, args.calibration)


if __name__ == "__main__":
    main()
