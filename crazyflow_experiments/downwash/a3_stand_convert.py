"""Convert A3 runs on the stand into per-leg force arrays around each crossing.

Copy the card files into the run first, `usd.py collect <card> --block a3_stand` puts them into
`<run>/sd/<role>/`. The victim's card holds the load cell and its rpm. It is mapped onto PC time
through its radio log of the load cell, as in the load cell test (`sync.sync_drone`); the source's
position comes from the flight records, stamped in PC time at 100 Hz.

Per leg, every window comes from the victim schedule the run recorded (`a3_stand.VictimSchedule`):

- the tare, motors off before the spin-up, zeroes the load cell for this leg, so the drift of the
  zero and a battery swap between runs drop out;
- the baseline, after the rpm settled and before the pass (the source still far out at the leg's
  start), gives the force and the rotor thrust the pass is compared with;
- the leg's arrays run from the baseline to the motor stop on a 500 Hz grid, with t = 0 where the
  source crosses over the victim, forces in N, up positive.

`force_z` is the load cell minus the leg's tare. `dforce_z` is its change from the baseline. In a
spin run `thrust_rpm` is the rotor thrust from the measured rpm through cf10's A1 curve, so the
battery's sag at the fixed command is accounted for, `dthrust_rpm` its change, and
`dforce_wake = dforce_z - dthrust_rpm` the part of the change the rotor speeds do not explain. The
rpm drift over each pass, against the baseline, is reported as well.

t = 0 is the *measured* crossing: the source's x from the flight records, which the onboard
controller follows 0.17 s late, passes the stand's x at a time that grows with speed past the
timeline's midpoint (`cross_nominal_s`, kept as a column, and `cross_lag_s` their difference). The
free-flight conversion (`a3_convert.derive`) uses the measured crossing as well, so the lags of the
two are comparable. The legs also report the measured `height_sep_meas`, `lateral_offset_meas` and
`cross_speed` over the settled window. A leg whose source never passes the stand's x is
`no_crossing`, its arrays aligned on the nominal crossing.

The load cell lags the rotor thrust. In a spin run the rpm-predicted thrust and the load cell rise
together at the spin-up on one card clock, so the shift that matches them (`a3_convert.lag`) is the
delay of the cell's filter and the stand's response, `cell_delay_s` per leg and its median in
`sync.json`. It is reported, not applied: the forces stay on the cell's own time until the tap test
(`stand_dynamics.toml`) says how to correct them. The windmilling of the unpowered props in a still
run shows in `rpm_pass_mean` and `rpm_pass_valid`, the mean rpm and the fraction of valid readings
over the pass.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/a3_stand_convert.py \
        results/downwash/a3/stand_flight<k>_<still|spin>_<time> [--calibration FILE]
    pixi run -e sim2real python crazyflow_experiments/downwash/a3_stand_convert.py --combine \
        results/downwash/a3/stand_flight*_*

Each run gets `legs/leg<NN>.npz`, `summary.csv` and `sync.json`, and a quicklook in
`plots/downwash/a3_stand/`. `--combine` merges converted runs, the parts of a flight split across
batteries and its reflown legs among them: per flight, mode and leg the latest run that flew it in
full counts, written to `results/downwash/a3/stand_summary.csv` with a plot of the peaks.
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import argparse
import csv
import json
import logging
import math
import pickle
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash import A3_RESULTS_DIR, DOWNWASH_PLOTS_DIR, plots, usd
from crazyflow_experiments.downwash.a3_convert import lag, lowpass, resample
from crazyflow_experiments.downwash.a3_stand import a1_curve
from crazyflow_experiments.downwash.sync import LOAD_CELL_SYNC, sync_drone
from crazyflow_experiments.downwash.thrust import rotor_thrust

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("crazyflow_experiments.downwash.a3_stand_convert")  # Not __name__
logger.setLevel(logging.INFO)

GRID_FREQ = 500.0  # Hz, as A3's legs
CARD_MAX_GAP = 0.01  # s, the card is NaN across longer gaps (card stalls)
MOCAP_MAX_GAP = 0.05  # s, the flight records are NaN across longer gaps
GAP_BEFORE_CROSS = -1.0  # s, card gaps from here to GAP_AFTER_CROSS mark a leg card_gap
GAP_AFTER_CROSS = 1.5  # s, the wake peaks within 0.6 s of the crossing (flight 1, 21 Nov)
PEAK_CUTOFF = 10.0  # Hz, low-pass of the force change before its peak is taken, as A3's labels
CELL_DELAY_BEFORE = 0.3  # s before the spin-on stamp where the delay fit starts
CELL_DELAY_AFTER = 1.0  # s after it where the fit ends; the stamp follows the arming and the write
CELL_DELAY_MAX = 0.3  # s, delays searched
MIN_EDGE_N = 0.2  # N the rpm-predicted thrust must rise over the fit window to count as an edge
G = 9.81  # m/s^2, grams of force to N
N_MOTORS = 4
STAND_PLOTS_DIR = DOWNWASH_PLOTS_DIR / "a3_stand"
COMBINED_PATH = A3_RESULTS_DIR / "stand_summary.csv"
LEG_KEYS = ("leg", "flight", "index_in_flight", "cell", "speed", "height_sep", "repeat")


def window_mean(t: NDArray, x: NDArray, t0: float, t1: float) -> NDArray:
    """Mean of `x` (N, ...) over the samples in [t0, t1), NaN-aware; NaN without samples."""
    inside = (t >= t0) & (t < t1)
    if not inside.any():
        return np.full(np.shape(x)[1:], np.nan)
    return np.nanmean(np.asarray(x, dtype=float)[inside], axis=0)


def peak(t_rel: NDArray, dforce: NDArray, t0: float, t1: float) -> tuple[float, float]:
    """Largest change of the force, signed, and its time, over [t0, t1] in leg time.

    The change is low-passed at `PEAK_CUTOFF` first, as A3 does with its labels, so a single noisy
    sample does not make the peak.
    """
    smooth = lowpass(dforce, GRID_FREQ, PEAK_CUTOFF)
    inside = (t_rel >= t0) & (t_rel <= t1) & np.isfinite(smooth)
    if not inside.any():
        return float("nan"), float("nan")
    k = np.flatnonzero(inside)[np.argmax(np.abs(smooth[inside]))]
    return float(smooth[k]), float(t_rel[k])


def rpm_drift(t: NDArray, rpm: NDArray, baseline: tuple[float, float], window: tuple) -> NDArray:
    """Relative change of each motor's mean rpm over `window` against the baseline, (4,)."""
    base = window_mean(t, rpm, *baseline)
    spinning = np.isfinite(base) & (base > 0)  # A still run's motors are off: no drift
    ratio = np.divide(
        window_mean(t, rpm, *window), base, out=np.full_like(base, np.nan), where=spinning
    )
    return ratio - 1.0


def measured_crossing(
    t: NDArray, x_rel: NDArray, speed: NDArray, t0: float, t1: float
) -> tuple[float, float]:
    """PC time and speed at which the source passes the stand's x inside [t0, t1].

    Where x changes sign more than once, e.g. a source that wobbles over the stand, the crossing at
    the highest speed counts, as in `a3_convert.derive`.

    Args:
        t: Record times in PC seconds (N,).
        x_rel: The source's x relative to the stand (N,).
        speed: The source's horizontal speed (N,).
        t0: Start of the search, PC seconds.
        t1: End of the search.

    Returns:
        The crossing time and the speed there, NaN for both when the source never crosses.
    """
    ok = np.isfinite(x_rel) & np.isfinite(speed) & (t >= t0) & (t <= t1)
    t, x_rel, speed = t[ok], x_rel[ok], speed[ok]
    if len(t) < 2:
        return math.nan, math.nan
    a, b = x_rel[:-1], x_rel[1:]
    crossings = np.flatnonzero(((a <= 0) & (b > 0)) | ((a >= 0) & (b < 0)))
    if len(crossings) == 0:
        return math.nan, math.nan
    k = int(crossings[np.argmax(speed[crossings])])
    cross = float(t[k] + (t[k + 1] - t[k]) * x_rel[k] / (x_rel[k] - x_rel[k + 1]))
    return cross, float(np.interp(cross, t, speed))


def spin_on_delay(t: NDArray, force: NDArray, thrust: NDArray, spin_on: float) -> float:
    """Delay of the load cell behind the rotor thrust at a spin-up, in s.

    The card logs both on one clock, and `lag` is the shift of the rpm-predicted `thrust` that
    best matches the cell's `force` over the window around the spin-on stamp. The stamp only places
    the window: it follows the arming and the write, so the edge itself lies a little before it.

    Args:
        t: Card time in PC seconds (N,).
        force: The load cell force in N (N,), any zero.
        thrust: The rotor thrust from the rpm in N (N,).
        spin_on: PC time stamped at the spin-on command.

    Returns:
        The delay, positive when the cell is late; NaN if the thrust does not rise in the window.
    """
    t_grid = np.arange(spin_on - CELL_DELAY_BEFORE, spin_on + CELL_DELAY_AFTER, 1 / GRID_FREQ)
    on_grid = resample(t, thrust, t_grid, CARD_MAX_GAP)
    if not np.isfinite(on_grid).any() or np.nanmax(on_grid) - np.nanmin(on_grid) < MIN_EDGE_N:
        return math.nan
    return lag(t_grid, resample(t, force, t_grid, CARD_MAX_GAP), t, thrust, CELL_DELAY_MAX)


def load_run(folder: Path, calibration: Path | None = None) -> dict:
    """Everything a run's legs are cut from, on PC time."""
    meta = json.loads((folder / "meta.json").read_text())
    with open(folder / "records.pkl", "rb") as f:
        records = pickle.load(f)["flight"]
    radio = dict(np.load(folder / "radio.npz"))
    own = {k.removeprefix("victim/"): v for k, v in radio.items() if k.startswith("victim/")}
    card = usd.decode(usd.sd_file(folder, "victim", meta))
    period = 1 / usd.role_config("lc", "victim").frequency
    clock, sync = sync_drone(own, "victim", card, period, LOAD_CELL_SYNC)
    cal = meta["load_cell"] if calibration is None else tomllib.loads(calibration.read_text())
    sync["calibration"] = {"a": cal["a"], "b": cal["b"], "from": str(calibration or "meta.json")}
    t = clock.pc_from_sd(card["t_us"])
    rpm = np.stack([card[f"motor.m{i}_rpm"] for i in range(1, N_MOTORS + 1)], axis=1).astype(float)
    rpm[rpm == usd.RPM_INVALID] = np.nan
    coef, fit = a1_curve()
    sync["thrust_curve"] = {"coef": coef.tolist(), "fit": fit}
    t_wall = np.array([r["t_wall"] for r in records])
    return {
        "meta": meta,
        "sync": sync,
        "t": t,
        "weight_g": cal["a"] * card["loadcell.rawWeight"].astype(float) + cal["b"],
        "rpm": rpm,
        "thrust_rpm": rotor_thrust(rpm, coef).sum(axis=1),
        "vbat": card["pm.vbatMV"].astype(float) / 1000,
        "t_zero": float(np.median([r["t_wall"] - r["t_perf"] for r in records])),
        "src_t": t_wall,
        "src_pos": np.array([r["source"]["pos"] for r in records]),
        "src_vel": np.array([r["source"]["vel"] for r in records]),
    }


def cut_leg(run: Mapping, row: Mapping, spin: Mapping, outcome: Mapping) -> tuple[dict, dict]:
    """The arrays and the summary row of one leg.

    Args:
        run: See `load_run`.
        row: The leg's design row.
        spin: Its schedule entry, timeline times, see `a3_stand.LegSpin`.
        outcome: Its outcome, wall times of the events, see `a3_stand.LegOutcome`.

    Returns:
        The arrays and the summary row.
    """
    meta = run["meta"]
    stand, cruise_half = meta["config"]["stand"], meta["config"]["a3"]["cruise_half"]
    events = outcome["events"]
    t_zero = run["t_zero"]
    cross_nominal = t_zero + spin["cross"]
    pass_start, pass_end = t_zero + spin["pass_start"], t_zero + spin["pass_end"]
    stand_pos = np.asarray(meta["stand_pos"], dtype=float)
    src_rel = run["src_pos"] - stand_pos
    src_speed = np.linalg.norm(run["src_vel"][:, :2], axis=1)
    cross, cross_speed = measured_crossing(
        run["src_t"], src_rel[:, 0], src_speed, pass_start, pass_end
    )
    crossed = math.isfinite(cross)
    if not crossed:
        cross = cross_nominal  # Aligned on the timeline, flagged no_crossing below
    half = cruise_half / row["speed"]
    settled = (run["src_t"] >= cross - half) & (run["src_t"] <= cross + half)
    baseline = (events["spin_on"] + stand["spin_settle_s"], pass_start)
    t, weight = run["t"], run["weight_g"]
    tare_g = float(window_mean(t, weight, events["tare_start"], events["tare_end"]))
    force = (weight - tare_g) * 1e-3 * G
    # Up to the stop of the motors, before their spin-down drops the force. Runs before 21 Nov
    # 23:30 stamped spin_off only after stop_spin returned, ~1 s after the stop went out, so the
    # schedule's own stop time bounds it as well.
    stop = min(events["spin_off"], t_zero + spin["spin_off"])
    t_grid = np.arange(baseline[0], stop, 1 / GRID_FREQ)
    t_rel = t_grid - cross
    on_grid = {
        "force_z": resample(t, force, t_grid, CARD_MAX_GAP),
        "thrust_rpm": resample(t, run["thrust_rpm"], t_grid, CARD_MAX_GAP),
        "rpm": resample(t, run["rpm"], t_grid, CARD_MAX_GAP),
        "src_pos_rel": resample(run["src_t"], src_rel, t_grid, MOCAP_MAX_GAP),
        "src_vel": resample(run["src_t"], run["src_vel"], t_grid, MOCAP_MAX_GAP),
    }
    base_force = float(window_mean(t, force, *baseline))
    base_thrust = float(window_mean(t, run["thrust_rpm"], *baseline))
    arrays = {"t": t_rel, **on_grid}
    arrays["dforce_z"] = on_grid["force_z"] - base_force
    arrays["dthrust_rpm"] = on_grid["thrust_rpm"] - base_thrust
    arrays["dforce_wake"] = arrays["dforce_z"] - arrays["dthrust_rpm"]
    window = (pass_start - cross, stop - cross)
    peak_force, t_peak = peak(t_rel, arrays["dforce_z"], *window)
    peak_wake, t_peak_wake = peak(t_rel, arrays["dforce_wake"], *window)
    drift = rpm_drift(t, run["rpm"], baseline, (pass_start, pass_end))
    # A card stall only spoils the leg in the baseline or while the wake acts around the crossing
    near = (t_grid < baseline[1]) | ((t_rel >= GAP_BEFORE_CROSS) & (t_rel <= GAP_AFTER_CROSS))
    in_pass = (t >= pass_start) & (t <= pass_end)
    rpm_pass = run["rpm"][in_pass]
    delay = (
        spin_on_delay(t, force, run["thrust_rpm"], events["spin_on"])
        if meta["mode"] == "spin" and outcome["spun"]
        else math.nan
    )
    summary = {key: row[key] for key in LEG_KEYS} | {
        "mode": meta["mode"],
        "run": Path(meta.get("folder", "")).name,
        "spun": outcome["spun"],
        "status": "ok",
        "cross_nominal_s": cross_nominal - t_zero,
        "cross_s": cross - t_zero,
        "cross_lag_s": cross - cross_nominal if crossed else math.nan,
        "cross_speed": cross_speed,
        "height_sep_meas": float(np.mean(src_rel[settled, 2])) if settled.any() else math.nan,
        "lateral_offset_meas": float(np.mean(src_rel[settled, 1])) if settled.any() else math.nan,
        "cell_delay_s": delay,
        "rpm_pass_mean": float(np.nanmean(rpm_pass)) if np.isfinite(rpm_pass).any() else math.nan,
        "rpm_pass_valid": float(np.isfinite(rpm_pass).mean()) if rpm_pass.size else math.nan,
        "tare_g": tare_g,
        "baseline_force_N": base_force,
        "baseline_thrust_rpm_N": base_thrust,
        "baseline_std_N": float(np.nanstd(on_grid["force_z"][t_grid < baseline[1]])),
        "peak_dforce_N": peak_force,
        "t_peak_s": t_peak,
        "peak_dforce_wake_N": peak_wake,
        "t_peak_wake_s": t_peak_wake,
        "rpm_drift_mean": float(np.nanmean(drift)) if np.isfinite(drift).any() else np.nan,
        "rpm_drift_max": float(np.nanmax(np.abs(drift))) if np.isfinite(drift).any() else np.nan,
        "vbat_baseline_V": float(window_mean(t, run["vbat"], *baseline)),
        "card_gaps": int(np.isnan(on_grid["force_z"]).sum()),
        "card_gaps_near": int(np.isnan(on_grid["force_z"][near]).sum()),
    }
    if meta["mode"] == "spin" and not outcome["spun"]:
        summary["status"] = "not_spun"  # Refly it: the motors did not run through the pass
    elif not crossed:
        summary["status"] = "no_crossing"  # The source never passed the stand's x
    elif summary["card_gaps_near"]:
        summary["status"] = "card_gap"
    return arrays, summary


def quicklook(rows: Sequence[Mapping], legs: Mapping[int, dict], title: str, path: Path) -> Path:
    """Change of the force over each leg, one panel per speed, colour by height separation."""
    speeds = sorted({float(r["speed"]) for r in rows})
    seps = sorted({float(r["height_sep"]) for r in rows})
    fig, axes = plots.figure(len(speeds), row_height=2.4)
    for ax, speed in zip(axes[:, 0], speeds):
        for row in rows:
            if float(row["speed"]) != speed or int(row["leg"]) not in legs:
                continue
            arrays = legs[int(row["leg"])]
            color = plots.SERIES[seps.index(float(row["height_sep"])) % len(plots.SERIES)]
            ax.plot(
                arrays["t"],
                arrays["dforce_z"] * 1e3,
                color=color,
                lw=plots.LINE_WIDTH,
                label=f"Δz {float(row['height_sep']):.2f} m, leg {int(row['leg'])}",
            )
        ax.axvline(0.0, color=plots.MUTED, lw=plots.LINE_WIDTH, ls=":")
        ax.set_ylabel(f"ΔF [mN], {speed:.1f} m/s")
        plots.legend(ax, fontsize=7, ncols=3)
    axes[-1, 0].set_xlabel("time from the crossing [s]")
    axes[0, 0].set_title(f"{title}: load cell change from the baseline, up +", loc="left")
    return plots.save(fig, path)


def convert(folder: Path, calibration: Path | None = None) -> list[dict]:
    """Cut every leg of a run, write its arrays, summary and sync report and the quicklook."""
    folder = Path(folder)
    run = load_run(folder, calibration)
    meta = run["meta"]
    meta["folder"] = str(folder)
    schedule = {int(s["leg"]): s for s in meta["victim_schedule"]["schedule"]}
    outcomes = {int(k): v for k, v in meta["victim_schedule"]["legs"].items()}
    out_dir = folder / "legs"
    out_dir.mkdir(exist_ok=True)
    rows, legs = [], {}
    for row in sorted(meta["design_rows"], key=lambda r: int(r["index_in_flight"])):
        leg = int(row["leg"])
        outcome = outcomes.get(leg, {"events": {}})
        if "spin_off" not in outcome["events"]:
            rows.append(
                {key: row[key] for key in LEG_KEYS} | {"mode": meta["mode"], "status": "not_flown"}
            )
            continue
        arrays, summary = cut_leg(run, row, schedule[leg], outcome)
        np.savez(out_dir / f"leg{leg:02d}.npz", **arrays)
        legs[leg] = arrays
        rows.append(summary)
        logger.info(
            f"Leg {leg:2d} (Δz {row['height_sep']:.2f} m, v {row['speed']:.1f} m/s): "
            f"{summary['status']}, peak ΔF {summary['peak_dforce_N'] * 1e3:+.1f} mN at "
            f"{summary['t_peak_s']:+.2f} s, rpm drift {summary['rpm_drift_mean'] * 100:+.2f} %"
        )
    write_rows(folder / "summary.csv", rows)
    run["sync"]["cell_delay"] = cell_delay_report(rows)
    (folder / "sync.json").write_text(json.dumps(run["sync"], indent=2, default=float))
    quicklook(meta["design_rows"], legs, folder.name, STAND_PLOTS_DIR / f"{folder.name}.png")
    return rows


def cell_delay_report(rows: Sequence[Mapping]) -> dict:
    """The load cell's delay behind the rotor thrust over the legs of a run, for sync.json."""
    per_leg = {
        int(r["leg"]): float(r["cell_delay_s"])
        for r in rows
        if "cell_delay_s" in r and math.isfinite(r["cell_delay_s"])
    }
    delays = np.array(list(per_leg.values()))
    return {
        "n": len(per_leg),
        "median_s": float(np.median(delays)) if len(delays) else math.nan,
        "std_s": float(np.std(delays)) if len(delays) else math.nan,
        "per_leg_s": per_leg,
        "note": "load cell behind the rpm-predicted thrust at the spin-up, reported, not applied",
    }


def write_rows(path: Path, rows: Sequence[Mapping]):
    """Write summary rows with the union of their columns."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def merge_runs(runs: Sequence[tuple[str, Sequence[Mapping]]]) -> list[dict]:
    """Every row converted with status ok, repeats of a leg from several runs included.

    Args:
        runs: (run folder name, its summary rows) pairs; the names end in their timestamp, so the
            latest sorts last.

    Returns:
        The kept rows, each with the run it came from, by flight, mode, leg and run.
    """
    kept = [
        dict(row) | {"run": name} for name, rows in runs for row in rows if row["status"] == "ok"
    ]
    return sorted(kept, key=lambda r: (int(r["flight"]), r["mode"], int(r["leg"]), r["run"][-19:]))


def combine(folders: Sequence[Path]) -> list[dict]:
    """Merge the summaries of converted runs, write the table and a plot of the peaks."""
    runs = []
    for folder in folders:
        with open(Path(folder) / "summary.csv", newline="") as f:
            runs.append((Path(folder).name, list(csv.DictReader(f))))
    rows = merge_runs(runs)
    write_rows(COMBINED_PATH, rows)
    speeds = sorted({float(r["speed"]) for r in rows})
    fig, axes = plots.figure(1, len(speeds), width=4.0 * max(len(speeds), 1), sharex=False)
    for ax, speed in zip(axes[0], speeds):
        for k, mode in enumerate(("still", "spin")):
            pick = [r for r in rows if r["mode"] == mode and float(r["speed"]) == speed]
            ax.plot(
                [float(r["height_sep"]) for r in pick],
                [float(r["peak_dforce_N"]) * 1e3 for r in pick],
                "o",
                color=plots.SERIES[k],
                label=mode,
            )
        ax.set_title(f"{speed:.1f} m/s", loc="left")
        ax.set_xlabel("height separation [m]")
        plots.legend(ax)
    axes[0, 0].set_ylabel("peak ΔF on the load cell [mN]")
    plots.save(fig, STAND_PLOTS_DIR / "stand_summary.png")
    logger.info(f"{len(rows)} legs in {COMBINED_PATH}")
    return rows


def main():
    """Command line: convert runs, or combine converted ones."""
    parser = argparse.ArgumentParser(description="Convert A3 runs on the stand.")
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
