"""Gain and delay of the stand against the rpm thrust at the victim's motor edges (export Step 1b).

No tap test was done. Every spun S1 leg has a spin-up and a spin-down: a thrust step that the rpm
and the load cell record on the victim's card clock. Around each edge (+-0.5 s) the load cell force
(scaled and signed as LEGS_FORMAT §6.2, no baseline) is fitted as `F(t) = K T(t - tau) + c`, with
`T` the victim's rotor thrust from its rpm on the A1 curve and `K`, `c` in closed form:

- (a) `tau` fixed at the existing `cell_delay_s`;
- (b) `tau` on a grid of 0 to 100 ms in 1 ms steps, the best fit kept.

Both fit the samples where a new load cell value arrived (`fz_new`): the cell converts at 80 Hz and
the card holds each value for about six records. (b) is also fitted on the held 500 Hz record
(`held`), the signal the chain of the dataset acts on: the hold delays it by about half a
conversion (6 ms) against the arrival of the new values, so `held` is what `cell_delay_s`
measures. Spin-ups are the primary set; spin-downs can carry the wake's tail and only cross-check.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/legs/stand_gain.py \
        --data ../crazyflow_experiments_data
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash.a3_stand_convert import load_run
from crazyflow_experiments.downwash.legs import (
    EVENING22_RUNS,
    MOUNT_SIGN,
    G,
    a1_curve_path,
    runs_dir,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

CELL_DELAY_S = 0.027  # s, `cell_delay_s` of `a3_stand_convert` (9 runs, std below 1 ms)
EDGE_HALF_S = 0.5  # s either side of the edge that is fitted
TAU_GRID = np.round(np.arange(0.0, 0.1005, 0.001), 6)  # s, delays of fit (b)
SEARCH_S = (-1.0, 1.5)  # s around the event stamp where the edge is searched
MIN_STEP_N = 0.2  # N the rpm thrust must change across an edge
MAX_CARD_GAP_S = 0.025  # s, an edge whose window holds a longer card stall is skipped
GAIN_TOL = 0.03  # Accept K within 1 +- this
TAU_TOL_S = 0.003  # Accept the delay of fit (b) within this of CELL_DELAY_S


def edge_time(t: NDArray, thrust: NDArray, t_event: float, rising: bool) -> float:
    """Time at which the rotor thrust crosses half of its step near an event stamp.

    Args:
        t: Times of the thrust samples (N,).
        thrust: Rotor thrust in N (N,), NaN where the rpm are invalid.
        t_event: The spin-on or spin-off stamp, same clock.
        rising: True for a spin-up.

    Returns:
        The interpolated crossing, NaN without a step of `MIN_STEP_N`.
    """
    inside = (t >= t_event + SEARCH_S[0]) & (t <= t_event + SEARCH_S[1]) & np.isfinite(thrust)
    tt, th = t[inside], thrust[inside]
    if len(tt) < 10:
        return math.nan
    low, high = np.percentile(th, [5, 95])
    if high - low < MIN_STEP_N:
        return math.nan
    above = th > (low + high) / 2
    crossing = (
        np.flatnonzero(above[1:] & ~above[:-1])
        if rising
        else np.flatnonzero(~above[1:] & above[:-1])
    )
    if len(crossing) == 0:
        return math.nan
    k = int(crossing[0])
    half = (low + high) / 2
    return float(tt[k] + (tt[k + 1] - tt[k]) * (half - th[k]) / (th[k + 1] - th[k]))


def fit_gain(
    t_force: NDArray, force: NDArray, t_thrust: NDArray, thrust: NDArray, tau: float
) -> tuple[float, float, float, int]:
    """Least squares `force = K thrust(t - tau) + c` over the given force samples.

    Returns:
        K, c, the rms residual in N and the number of samples used.
    """
    good = np.isfinite(thrust)
    x = np.interp(t_force - tau, t_thrust[good], thrust[good], left=np.nan, right=np.nan)
    ok = np.isfinite(x) & np.isfinite(force)
    if ok.sum() < 10:
        return math.nan, math.nan, math.nan, int(ok.sum())
    design = np.stack([x[ok], np.ones(ok.sum())], axis=1)
    (k_gain, c), *_ = np.linalg.lstsq(design, force[ok], rcond=None)
    residual = force[ok] - design @ np.array([k_gain, c])
    return float(k_gain), float(c), float(np.sqrt(np.mean(residual**2))), int(ok.sum())


def best_tau(
    t_force: NDArray, force: NDArray, t_thrust: NDArray, thrust: NDArray
) -> tuple[float, float, float]:
    """Fit (b): the delay of `TAU_GRID` with the smallest residual, its K and residual."""
    fits = [fit_gain(t_force, force, t_thrust, thrust, tau) for tau in TAU_GRID]
    rms = np.array([f[2] for f in fits])
    if not np.isfinite(rms).any():
        return math.nan, math.nan, math.nan
    k = int(np.nanargmin(rms))
    return float(TAU_GRID[k]), fits[k][0], fits[k][2]


def new_samples(raw: NDArray) -> NDArray:
    """Card records where a new load cell value arrived: the value changed from the record before."""
    out = np.zeros(len(raw), dtype=bool)
    out[1:] = raw[1:] != raw[:-1]
    return out


def edges_of_run(run: Mapping) -> list[dict]:
    """Fits of every motor edge of a spin run, see the module docstring."""
    meta = run["meta"]
    if meta["mode"] != "spin":
        return []
    t, t_rpm = run["t"], run["t_rpm"]
    force = MOUNT_SIGN * run["weight_g"] * 1e-3 * G
    is_new = new_samples(np.asarray(run["card"]["loadcell.rawWeight"]))
    gaps = np.diff(t)
    rows = []
    for leg, outcome in sorted(meta["victim_schedule"]["legs"].items(), key=lambda kv: int(kv[0])):
        events = outcome["events"]
        if not outcome.get("spun") or "spin_off" not in events:
            continue
        for kind, stamp in (("up", events["spin_on"]), ("down", events["spin_off"])):
            center = edge_time(t_rpm, run["thrust_rpm"], stamp, rising=kind == "up")
            row = {
                "run": Path(meta["folder"]).name,
                "leg": int(leg),
                "kind": kind,
                "t_edge": center,
            }
            if not math.isfinite(center):
                rows.append(row | {"skipped": "no_edge"})
                continue
            window = np.abs(t - center) <= EDGE_HALF_S
            stall = (gaps > MAX_CARD_GAP_S) & window[1:] & window[:-1]
            if stall.any():
                rows.append(row | {"skipped": "card_gap"})
                continue
            sel_new, sel_held = window & is_new, window
            k_a, c_a, rms_a, n_a = fit_gain(
                t[sel_new], force[sel_new], t_rpm, run["thrust_rpm"], CELL_DELAY_S
            )
            tau_b, k_b, rms_b = best_tau(t[sel_new], force[sel_new], t_rpm, run["thrust_rpm"])
            tau_h, k_h, rms_h = best_tau(t[sel_held], force[sel_held], t_rpm, run["thrust_rpm"])
            step = float(
                np.nanpercentile(run["thrust_rpm"][np.abs(t_rpm - center) <= EDGE_HALF_S], 95)
                - np.nanpercentile(run["thrust_rpm"][np.abs(t_rpm - center) <= EDGE_HALF_S], 5)
            )
            rows.append(
                row
                | {
                    "skipped": "",
                    "step_N": step,
                    "n_new": n_a,
                    "K_a": k_a,
                    "rms_a_N": rms_a,
                    "tau_b_s": tau_b,
                    "K_b": k_b,
                    "rms_b_N": rms_b,
                    "tau_held_s": tau_h,
                    "K_held": k_h,
                    "rms_held_N": rms_h,
                }
            )
    return rows


def summarize(rows: Sequence[Mapping]) -> dict:
    """Median and IQR of each fitted quantity per edge kind, and the verdict on the spin-ups."""
    out = {}
    for kind in ("up", "down"):
        fitted = [r for r in rows if r["kind"] == kind and not r["skipped"]]
        stats = {
            "n": len(fitted),
            "n_skipped": sum(r["kind"] == kind and bool(r["skipped"]) for r in rows),
        }
        for key in ("K_a", "tau_b_s", "K_b", "tau_held_s", "K_held", "rms_a_N"):
            values = np.array([r[key] for r in fitted], dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                q1, med, q3 = np.percentile(values, [25, 50, 75])
                stats[key] = {"median": float(med), "iqr": float(q3 - q1)}
        out[kind] = stats
    up = out["up"]
    gain = up.get("K_a", {}).get("median", math.nan)
    tau_new = up.get("tau_b_s", {}).get("median", math.nan)
    tau_held = up.get("tau_held_s", {}).get("median", math.nan)
    out["verdict"] = {
        "gain_ratio": gain,
        "gain_ok": bool(abs(gain - 1) <= GAIN_TOL),
        "tau_new_s": tau_new,
        "tau_held_s": tau_held,
        "tau_held_ok": bool(abs(tau_held - CELL_DELAY_S) <= TAU_TOL_S),
        "tau_new_ok": bool(abs(tau_new - CELL_DELAY_S) <= TAU_TOL_S),
        "cell_delay_s": CELL_DELAY_S,
    }
    return out


def stand_gain(folders: Sequence[Path], curve_path: Path) -> tuple[list[dict], dict]:
    """Edge fits of the spin runs among `folders` and their summary."""
    rows = []
    for folder in folders:
        meta = json.loads((Path(folder) / "meta.json").read_text())
        if meta["mode"] != "spin":
            continue
        run = load_run(Path(folder), curve_path=curve_path)
        run["meta"]["folder"] = str(folder)
        rows += edges_of_run(run)
    return rows, summarize(rows)


def main():
    """Command line: fit the edges of the selected spin runs and print the summary as JSON."""
    parser = argparse.ArgumentParser(description="Stand gain and delay at the motor edges.")
    parser.add_argument("--data", type=Path, required=True, help="crazyflow_experiments_data")
    parser.add_argument("--runs", default=EVENING22_RUNS, help="Glob of run folders")
    parser.add_argument("--out", type=Path, help="Write the edges and summary as JSON here")
    args = parser.parse_args()
    folders = sorted(runs_dir(args.data).glob(args.runs))
    rows, summary = stand_gain(folders, a1_curve_path(args.data))
    text = json.dumps({"edges": rows, "summary": summary}, indent=2, default=float)
    if args.out:
        args.out.write_text(text)
    print(json.dumps(summary, indent=2, default=float))  # noqa: T201


if __name__ == "__main__":
    main()
