"""Export the A3 stand legs as a LEGS_FORMAT v1 dataset (`docs/LEGS_FORMAT.md`).

Every flown leg of the selected runs becomes `legs/<version>/<leg_id>/{states.npz, meta.json}` in
the data repository, with `manifest.csv`, `dataset.json` and `chain_test.npz` beside them. Per run:

1. `a3_stand_convert.load_run` decodes the victim's card and maps it onto PC time through the
   radio (`sync.sync_drone`); `source_sd` decodes the source's card and maps it through the
   setpoint echo (`sync.fit_clock`).
2. The stand's gain and delay come from the victim's motor edges (`stand_gain`). The export stops
   when the gain ratio is off 1 by more than 3 %.
3. Per leg, t = 0 is the existing measured crossing (`a3_stand_convert.measured_crossing`, the
   source's mocap `pos` of the flight records against the stand's x). The grid holds t = 0 as a
   sample, from `window_pre[0] - 0.1` s (at most -2.5 s) to +4.0 s, in S1 to at most 0.2 s before
   the motors stop. `window_pre` is the converter's baseline window,
   [spin_on + spin_settle_s, pass_start], in S0 and S1 alike.
4. The labels follow LEGS_FORMAT §4 and §6: no filter, no shift, a constant baseline over the new
   load cell samples in `window_pre`, card stalls over 25 ms NaN, and in S1 the rotor thrust change
   through the stand's chain (a 27 ms delay) subtracted.
5. Legs flown more than once on a date: the last attempt that did not hard-fail is the leg, the
   earlier attempts are `superseded`, `rerun_of` links each attempt to the one before it.

Nothing is deleted: every flown leg is written, failures and duplicates flagged.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/legs/export.py \
        --data ../crazyflow_experiments_data --version legs_v2 [--runs GLOB] [--overwrite]
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import argparse
import csv
import hashlib
import json
import logging
import math
import pickle
import re
import shutil
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.downwash import usd
from crazyflow_experiments.downwash.a3_convert import (
    MOCAP_MAX_GAP,
    VEL_CUTOFF,
    fill_dropouts,
    lowpass,
    resample,
    resample_quat,
)
from crazyflow_experiments.downwash.a3_design import model_params
from crazyflow_experiments.downwash.a3_stand_convert import load_run, measured_crossing
from crazyflow_experiments.downwash.legs import (
    FORMAT,
    FORMAT_VERSION,
    GRID_FREQ,
    MOUNT_SIGN,
    G,
    a1_curve_path,
    dataset_dir,
    runs_dir,
    version_runs,
)
from crazyflow_experiments.downwash.legs.chain import chain, chain_test, stand_hash
from crazyflow_experiments.downwash.legs.source_sd import (
    card_signals,
    has_card,
    load_source_card,
    mocap_thrust,
)
from crazyflow_experiments.downwash.legs.stand_gain import CELL_DELAY_S, edges_of_run, summarize
from crazyflow_experiments.downwash.thrust import rotor_thrust

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("crazyflow_experiments.downwash.legs.export")  # Not __name__
logger.setLevel(logging.INFO)

BLOCK = "A3"
CONDITIONS = {"still": "S0", "spin": "S1"}
DT = 1 / GRID_FREQ
SPAN_START_MAX = -2.5  # s, the grid starts at the latest here
PRE_PAD = 0.1  # s before window_pre where the grid starts
SPAN_END = 4.0  # s
SPIN_DOWN_MARGIN = 0.2  # s, an S1 leg ends this long before the motors stop
FILTER_PAD = 1.0  # s of padding for the zero-phase filters of the source's velocity
SRC_MAX_GAP = 0.5  # s, the flight records skip poses when the PC loop stalls (50 to 420 ms)
LC_MAX_GAP = 0.025  # s, load cell: interpolate across at most two 80 Hz samples (LEGS_FORMAT §6.6)
RPM_MAX_GAP = 0.012  # s, card stall in the rpm bridged: five missing 500 Hz records
RPM_MAX_DROPOUT = 5  # Invalid rpm records interpolated
CROSS_HALF = 0.3  # s, half window of v_meas, dz_meas and y_meas
TAIL_S = 0.5  # s, the tail window at the end of the leg
PEAK_CUTOFF = 10.0  # Hz, the existing peak finder's low-pass
SPEED_TOL = 0.05
OFFSET_TOL = 0.03  # m
SYNC_RESID_TOL_MS = 5.0
NOISE_FLOOR_SIGMA = 5.0
OVERLOAD_FRACTION = 0.8  # Of the cell's rating
GAIN_TOL = 0.03
ROTOR_ORDER = ["FR", "RR", "RL", "FL"]
FLAGS = (
    "noise_floor",
    "card_gap",
    "sync_resid_high",
    "rpm_gap",
    "speed_err",
    "offset_err",
    "overload",
    "not_spun",
    "superseded",
    "aborted_run",
    "source_thrust_from_mocap",
    "victim_attitude_assumed",
    "windmill_unmeasured",
    "stand_dynamics_unknown",
    "nonlinear_processing",
)
MANIFEST_COLUMNS = (
    "leg_id",
    "run_id",
    "date",
    "design_version",
    "block",
    "condition",
    "flight",
    "leg_index",
    "repeat",
    "direction",
    "dz_nominal",
    "v_nominal",
    "y_nominal",
    "dz_meas",
    "v_meas",
    "y_meas",
    "noise_fz_N",
    "hard_fail",
    "flags",
    "rerun_of",
    "path",
)
UNITS = {
    "t": "s",
    "pos": "m",
    "quat": "1",
    "vel": "m/s",
    "ang_vel": "rad/s",
    "rotor_vel": "rpm",
    "motor_thrust": "N",
    "thrust_sum": "N",
    "fz_new": "bool",
    "fz_raw": "N",
    "fz_total": "N",
    "fz_wake": "N",
    "sim/freq": "Hz",
}
STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$")


def run_date(run_id: str) -> str:
    """Date of a run from the timestamp that ends its folder name, YYYY-MM-DD."""
    if (match := STAMP.search(run_id)) is None:
        raise ValueError(f"{run_id} does not end in a recording timestamp")
    return match.group(1)


def design_version(meta: Mapping) -> str:
    """Name of the design a run flew, from its heights and speeds, e.g. `a3_dz0.25-0.5_v0.5-...`."""
    a3 = meta["config"]["a3"]
    seps = "-".join(f"{float(x):g}" for x in a3["height_seps"])
    speeds = "-".join(f"{float(x):g}" for x in a3["speeds"])
    return f"a3_dz{seps}_v{speeds}"


def leg_id(condition: str, run_id: str, leg: int) -> str:
    """`<block>_<condition>_<run_id>_leg<NN>` of LEGS_FORMAT §1."""
    return f"{BLOCK}_{condition}_{run_id}_leg{leg:02d}"


def sha256(path: Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def leg_grid(start: float, end: float) -> NDArray:
    """Leg time on the 500 Hz grid that holds t = 0, covering [start, end]."""
    k0 = math.ceil(start * GRID_FREQ - 1e-9)
    k1 = math.floor(end * GRID_FREQ + 1e-9)
    return np.arange(k0, k1 + 1) * DT


def new_on_grid(t_card: NDArray, raw: NDArray, t_pc: NDArray) -> NDArray:
    """Grid samples at which a new load cell value has arrived since the sample before.

    A new value is a card record whose raw reading differs from the record before; it marks the
    first grid sample at or after its card time.
    """
    changed = np.flatnonzero(raw[1:] != raw[:-1]) + 1
    k = np.ceil((t_card[changed] - t_pc[0]) * GRID_FREQ - 1e-9).astype(int)
    out = np.zeros(len(t_pc), dtype=bool)
    out[k[(k >= 0) & (k < len(t_pc))]] = True
    return out


def gaps_in(t_src: NDArray, t0: float, t1: float, max_gap: float) -> list[list[float]]:
    """Intervals [a, b] of `t_src` spacing over `max_gap` that overlap [t0, t1]."""
    k = np.flatnonzero(np.diff(t_src) > max_gap)
    return [
        [float(t_src[i]), float(t_src[i + 1])] for i in k if t_src[i + 1] > t0 and t_src[i] < t1
    ]


def mean_on(mask: NDArray, x: NDArray) -> float:
    """NaN-aware mean of `x` over `mask`, NaN without samples."""
    values = x[mask & np.isfinite(x)]
    return float(values.mean()) if len(values) else math.nan


def load_session(folder: Path, data: Path) -> dict:
    """A run's victim data (`load_run`), its flight records and the source's card if present."""
    run = load_run(folder, curve_path=a1_curve_path(data))
    run["meta"]["folder"] = str(folder)
    run["folder"] = folder
    with open(folder / "records.pkl", "rb") as f:
        run["records"] = pickle.load(f)
    run["source_card"] = (
        load_source_card(folder, run["meta"], run["records"])
        if has_card(folder, run["meta"])
        else None
    )
    run["victim_quat"] = usd.quatdecompress(run["card"]["stateEstimateZ.quat"])
    return run


def curves(data: Path, meta: Mapping) -> dict:
    """The rotor thrust curve of each drone (LEGS_FORMAT §5.4), with source and sha256."""
    path = a1_curve_path(data)
    a1 = tomllib.loads(path.read_text())
    fit = a1["decision"]["fit"]
    scale = float(meta["config"]["drones"]["source"]["thrust_scale"])
    model = meta["config"]["drones"]["model"]
    source_id = meta["config"]["drones"]["source"]["id"]
    return {
        "victim": {
            "coeffs": [float(c) for c in a1[fit]["rpm2thrust"]],
            "source": f"A1 fit cf{meta['config']['drones']['victim']['id']} ({fit}, {path.name})",
            "sha256": sha256(path),
            "mass_kg": float(a1["mass_kg"]),
        },
        "source": {
            "coeffs": [scale * float(c) for c in a1["crazyflow"]["rpm2thrust"]],
            "source": f"crazyflow {model} x {scale:g} (cf{source_id}), curve as in {path.name}",
            "sha256": sha256(path),
            "mass_kg": float(meta["config"]["drones"]["source"]["mass_kg"]),
        },
    }


def stand_entry(gain: Mapping, n_edges: int) -> dict:
    """`rig.stand` of LEGS_FORMAT §5.5 with the gain of the motor edges."""
    return {
        "mount_sign": MOUNT_SIGN,
        "sample_rate_hz": 80,
        "dynamics": {
            "kind": "delay",
            "tau_eff_s": CELL_DELAY_S,
            "gain_ratio": round(float(gain["gain_ratio"]), 4),
            "source": (
                "cell_delay_s (a3_stand_convert, 9 runs, std < 1 ms); held-signal fit of the "
                f"spin-ups {gain['tau_held_s']:.3f} s; gain: median K of {n_edges} spin-up edges "
                "(export Step 1b)"
            ),
        },
        "processing": {"shift_s": 0.0, "filter": "none"},
    }


def build_leg(run: Mapping, row: Mapping, spin: Mapping, outcome: Mapping, ctx: Mapping) -> dict:
    """Arrays, meta and quality of one flown leg, before the duplicate rule.

    Args:
        run: See `load_session`.
        row: The leg's design row.
        spin: Its victim schedule entry (timeline times).
        outcome: Its outcome (PC times of the events).
        ctx: `stand`, `curves`, `params` and `created` of the dataset.

    Returns:
        `states`, `meta` and `manifest` (the row without the duplicate columns) of the leg.
    """
    meta = run["meta"]
    mode, condition = meta["mode"], CONDITIONS[meta["mode"]]
    rid = Path(meta["folder"]).name
    leg = int(row["leg"])
    events, t_zero = outcome["events"], run["t_zero"]
    stand_cfg = meta["config"]["stand"]
    stand_pos = np.asarray(meta["stand_pos"], dtype=float)
    direction = float(row["direction"])
    hard = []

    # Time zero: the existing measured crossing of the source's mocap pos over the stand's x
    pass_start, pass_end = t_zero + spin["pass_start"], t_zero + spin["pass_end"]
    src_t = run["src_t"]
    src_rel = run["src_pos"] - stand_pos
    src_speed = np.linalg.norm(run["src_vel"][:, :2], axis=1)
    cross, _ = measured_crossing(src_t, direction * src_rel[:, 0], src_speed, pass_start, pass_end)
    if not math.isfinite(cross):
        hard.append("no_crossing")
        cross = t_zero + spin["cross"]
    stop = min(events["spin_off"], t_zero + spin["spin_off"])
    window_pre = [events["spin_on"] + stand_cfg["spin_settle_s"] - cross, pass_start - cross]
    start = min(window_pre[0] - PRE_PAD, SPAN_START_MAX)
    spun = mode == "spin" and bool(outcome.get("spun", False))
    end = min(SPAN_END, stop - cross - SPIN_DOWN_MARGIN) if spun else SPAN_END
    t = leg_grid(start, end)
    t_pc = cross + t
    n = len(t)
    pre = (t >= window_pre[0]) & (t < window_pre[1])
    window_tail = [float(t[-1] - TAIL_S), float(t[-1])]
    tail = t >= window_tail[0]

    # Source: mocap pose of the flight records, velocity by central difference and 10 Hz zero-phase.
    # The records skip poses while the PC loop stalls; the pose is interpolated across such gaps up
    # to SRC_MAX_GAP and the gaps are listed in the quality
    pad = round(FILTER_PAD * GRID_FREQ)
    t_wide = (
        cross + np.arange(round(t[0] * GRID_FREQ) - pad, round(t[-1] * GRID_FREQ) + pad + 1) * DT
    )
    pos_wide = resample(src_t, run["src_pos"], t_wide, SRC_MAX_GAP)
    vel_src = lowpass(np.gradient(pos_wide, DT, axis=0), GRID_FREQ, VEL_CUTOFF)[pad:-pad]
    pos_src = pos_wide[pad:-pad]
    quat_src = resample_quat(
        src_t, np.array([r["source"]["quat"] for r in run["records"]["flight"]]), t_pc, SRC_MAX_GAP
    )
    mocap_gaps = [
        [a - cross, b - cross] for a, b in gaps_in(src_t, t_pc[0], t_pc[-1], MOCAP_MAX_GAP)
    ]
    if not np.isfinite(pos_src).all():
        hard.append("no_source_mocap")

    # Victim: the stand position, the onboard attitude estimate averaged over window_pre
    t_card = run["t"]
    in_pre_card = (t_card >= cross + window_pre[0]) & (t_card < cross + window_pre[1])
    q_vic = R.from_quat(run["victim_quat"][in_pre_card]).mean().as_quat()
    q_vic = q_vic if q_vic[3] >= 0 else -q_vic
    gyro = np.stack([run["card"][f"gyro.{a}"] for a in "xyz"], axis=1).astype(float)
    ang_vel_vic = np.deg2rad(resample(t_card, gyro, t_pc, RPM_MAX_GAP))

    # Rotors
    coef_vic = np.asarray(ctx["curves"]["victim"]["coeffs"])
    coef_src = np.asarray(ctx["curves"]["source"]["coeffs"])
    rotor_vel = np.full((n, 2, 4), np.nan)
    motor_thrust = np.full((n, 2, 4), np.nan)
    rpm_gaps = {"victim": [0] * 4, "source": [0] * 4}
    rpm_spikes = {"victim": [0] * 4, "source": [0] * 4}
    if mode == "spin":
        raw, spikes = run["rpm"], run["rpm_spike"]  # Despiked by `load_run`
        in_leg = (run["t_rpm"] >= t_pc[0]) & (run["t_rpm"] <= t_pc[-1])
        rpm_spikes["victim"] = [int(x) for x in spikes[in_leg].sum(axis=0)]
        filled = np.stack([fill_dropouts(raw[:, i], RPM_MAX_DROPOUT) for i in range(4)], axis=1)
        rotor_vel[:, 0] = resample(run["t_rpm"], filled, t_pc, RPM_MAX_GAP)
        was_nan = resample(run["t_rpm"], (~np.isfinite(raw)).astype(float), t_pc, RPM_MAX_GAP)
        rpm_gaps["victim"] = (
            ((np.nan_to_num(was_nan) > 0) & np.isfinite(rotor_vel[:, 0])).sum(axis=0).tolist()
        )
        motor_thrust[:, 0] = rotor_thrust(rotor_vel[:, 0], coef_vic)
    else:  # Unpowered: no windmill rpm logged, no thrust by definition
        rotor_vel[:, 0] = 0.0
        motor_thrust[:, 0] = 0.0
    source_from_mocap = run["source_card"] is None
    ang_vel_src = np.full((n, 3), np.nan)
    sync = {
        "victim_radio": {
            "latency_median_ms": 1e3 * float(run["sync"]["latency_median"]),
            "latency_p95_ms": 1e3 * float(run["sync"]["latency_p95"]),
            "offset_halves_spread_ms": float(np.ptp(run["sync"]["sd_offset_halves_ms"])),
            "match_fraction": float(run["sync"]["match_fraction"]),
        }
    }
    if source_from_mocap:
        thrust = mocap_thrust(
            src_t, run["src_pos"], t_wide, ctx["curves"]["source"]["mass_kg"], ctx["g"]
        )[pad:-pad]
        motor_thrust[:, 1] = thrust[:, None] / 4
    else:
        log, fit = run["source_card"]
        signals = card_signals(log, fit, t_pc, coef_src)
        rotor_vel[:, 1], motor_thrust[:, 1] = signals["rotor_vel"], signals["motor_thrust"]
        ang_vel_src = signals["ang_vel"]
        rpm_gaps["source"] = [int(x) for x in signals["rpm_filled"]]
        rpm_spikes["source"] = [int(x) for x in signals["rpm_spikes"]]
        sync["source_setpoint_echo"] = {
            "residual_std_ms": 1e3 * fit.residual_std,
            "n_matches": fit.n_matches,
            "drift_ppm": 1e6 * fit.drift,
        }
    thrust_sum = motor_thrust[:, 0].sum(axis=1)

    # Labels
    weight_g = run["weight_g"]
    force = MOUNT_SIGN * weight_g * 1e-3 * G
    fz = resample(t_card, force, t_pc, LC_MAX_GAP)
    fz_new = new_on_grid(t_card, np.asarray(run["card"]["loadcell.rawWeight"]), t_pc)
    fz_new &= np.isfinite(fz)
    fz_raw = fz - mean_on(fz_new & pre, fz)
    fz_total = fz_raw.copy()
    if mode == "spin":
        d_thrust = thrust_sum - mean_on(pre, thrust_sum)
        fz_wake = fz_total - chain(d_thrust, ctx["stand"])
        fz_wake -= mean_on(fz_new & pre, fz_wake)
    else:
        fz_wake = fz_total.copy()
    label = fz_wake if mode == "spin" else fz_total

    # Geometry over [-0.3, 0.3] s
    near = np.abs(t) <= CROSS_HALF + 1e-9
    rel = pos_src - stand_pos
    v_meas = float(np.mean(np.linalg.norm(vel_src[near, :2], axis=1)))
    dz_meas = float(np.mean(rel[near, 2]))
    y_meas = float(np.mean(direction * rel[near, 1]))  # Left of the path positive for +x passes

    # Quality
    noise = float(np.std(fz_total[fz_new & pre]))
    noise_wake = float(np.std(fz_wake[fz_new & pre & np.isfinite(fz_wake)]))
    smooth = lowpass(label, GRID_FREQ, PEAK_CUTOFF)
    v_ref = v_meas if v_meas > 0 else float(row["speed"])
    peak_window = (-0.5 / v_ref, 0.5 / v_ref + 1.0)
    in_peak = (t >= peak_window[0]) & (t <= peak_window[1]) & np.isfinite(smooth)
    k_peak = int(np.flatnonzero(in_peak)[np.argmin(smooth[in_peak])]) if in_peak.any() else -1
    peak_fz = float(smooth[k_peak]) if k_peak >= 0 else math.nan
    peak_time = float(t[k_peak]) if k_peak >= 0 else math.nan
    gaps = [[a - cross, b - cross] for a, b in gaps_in(t_card, t_pc[0], t_pc[-1], LC_MAX_GAP)]
    guard = (peak_window[0] - 0.2, peak_window[1])
    if any(b > guard[0] and a < guard[1] for a, b in gaps):
        hard.append("card_gap_in_wake")
    if mode == "spin" and not outcome.get("spun", False):
        hard.append("not_spun")
    flags = ["victim_attitude_assumed"]
    if gaps:
        flags.append("card_gap")
    sigma = noise_wake if mode == "spin" else noise
    if not abs(peak_fz) >= NOISE_FLOOR_SIGMA * sigma:
        flags.append("noise_floor")
    resid = [sync["victim_radio"]["offset_halves_spread_ms"]]
    if "source_setpoint_echo" in sync:
        resid.append(sync["source_setpoint_echo"]["residual_std_ms"])
    if max(resid) > SYNC_RESID_TOL_MS:
        flags.append("sync_resid_high")
    if (~np.isfinite(rotor_vel[:, 0])).any() or (
        not source_from_mocap and (~np.isfinite(rotor_vel[:, 1])).any()
    ):
        flags.append("rpm_gap")
    if abs(v_meas / float(row["speed"]) - 1) > SPEED_TOL:
        flags.append("speed_err")
    if abs(y_meas - float(row["lateral_offset"])) > OFFSET_TOL:
        flags.append("offset_err")
    capacity = float(meta["load_cell"].get("capacity_g", math.nan))
    in_leg = (t_card >= t_pc[0]) & (t_card <= t_pc[-1])
    if np.nanmax(np.abs(weight_g[in_leg])) > OVERLOAD_FRACTION * capacity:
        flags.append("overload")
    if mode == "spin" and not outcome.get("spun", False):
        flags.append("not_spun")
    if meta.get("abort_reason") or meta.get("error"):
        flags.append("aborted_run")
    if source_from_mocap:
        flags.append("source_thrust_from_mocap")
    if mode == "still":
        flags.append("windmill_unmeasured")

    lid = leg_id(condition, rid, leg)
    states = {
        "t": t,
        "pos": np.stack([np.broadcast_to(stand_pos, (n, 3)), pos_src], axis=1),
        "quat": np.stack([np.broadcast_to(q_vic, (n, 4)), quat_src], axis=1),
        "vel": np.stack([np.zeros((n, 3)), vel_src], axis=1),
        "ang_vel": np.stack([ang_vel_vic, ang_vel_src], axis=1),
        "rotor_vel": rotor_vel,
        "motor_thrust": motor_thrust,
        "thrust_sum": thrust_sum,
        "fz_new": fz_new,
        "fz_raw": fz_raw,
        "fz_total": fz_total,
        "fz_wake": fz_wake,
        "sim/freq": np.array(GRID_FREQ),
    }
    curves_ = ctx["curves"]
    date = run_date(rid)
    dv = design_version(meta)
    leg_meta = {
        "run": {
            "id": lid,
            "run_id": rid,
            "date": date,
            "plugin": "real",
            "created": ctx["created"],
            "producer_code_commit": ctx["commit"],
        },
        "design": {
            "design_version": dv,
            "block": BLOCK,
            "condition": condition,
            "dz": float(row["height_sep"]),
            "v": float(row["speed"]),
            "y": float(row["lateral_offset"]),
            "repeat": int(row["repeat"]),
            "direction": int(row["direction"]),
            "flight": int(row["flight"]),
            "leg_index": leg,
            "index_in_flight": int(row["index_in_flight"]),
            "cell": int(row["cell"]),
        },
        "derived": {
            "cross_time_abs": float(cross),
            "v_meas": v_meas,
            "dz_meas": dz_meas,
            "y_meas": y_meas,
            "t_pass_start": float(pass_start - cross),
            "t_spin_on": float(events["spin_on"] - cross) if mode == "spin" else None,
            "t_spin_down": float(stop - cross) if mode == "spin" else None,
            "window_pre": [float(x) for x in window_pre],
            "window_tail": window_tail,
            "victim_pose_source": "stand_mean",
            "victim_attitude_source": (
                "onboard estimate (victim card stateEstimateZ.quat), mean over window_pre; "
                "no mocap attitude of the stand was recorded"
            ),
            "source_pose_source": "records.pkl source.pos and source.quat (as the crossing)",
            "stand_pos": stand_pos.tolist(),
        },
        "sim_params": {
            "drone/rpm2thrust": [curves_["victim"]["coeffs"], curves_["source"]["coeffs"]],
            "drone/rpm2thrust_source": [curves_["victim"]["source"], curves_["source"]["source"]],
            "drone/mixing_matrix": np.asarray(ctx["params"]["mixing_matrix"]).tolist(),
            "drone/mass": [curves_["victim"]["mass_kg"], curves_["source"]["mass_kg"]],
        },
        "rig": {"stand": ctx["stand"]},
        "quality": {
            "noise_fz_N": noise,
            "noise_fz_wake_N": noise_wake,
            "n_fz_pre": int((fz_new & pre).sum()),
            "peak_fz_N": peak_fz,
            "peak_time_s": peak_time,
            "peak_window": [float(x) for x in peak_window],
            "tail_level_N": mean_on(fz_new & tail, label),
            "card_gaps": gaps,
            "mocap_gaps": mocap_gaps,
            "sync": sync,
            "rpm_gaps": rpm_gaps,
            "rpm_spikes": rpm_spikes,
            "hard_fail": bool(hard),
            "hard_fail_reason": ";".join(hard),
            "flags": flags,
            "load_cell_calibration": {
                "a": float(meta["load_cell"]["a"]),
                "b": float(meta["load_cell"]["b"]),
            },
        },
        "signals": {
            "quat_order": "xyzw",
            "rotor_order": ROTOR_ORDER,
            "fz_rate_hz": 80,
            "units": UNITS,
            "arrays": {k: list(np.shape(v)) for k, v in states.items()},
        },
    }
    manifest = {
        "leg_id": lid,
        "run_id": rid,
        "date": date,
        "design_version": dv,
        "block": BLOCK,
        "condition": condition,
        "flight": int(row["flight"]),
        "leg_index": leg,
        "repeat": int(row["repeat"]),
        "direction": int(row["direction"]),
        "dz_nominal": float(row["height_sep"]),
        "v_nominal": float(row["speed"]),
        "y_nominal": float(row["lateral_offset"]),
        "dz_meas": dz_meas,
        "v_meas": v_meas,
        "y_meas": y_meas,
        "noise_fz_N": noise,
        "path": lid,
    }
    return {"states": states, "meta": leg_meta, "manifest": manifest}


def resolve_duplicates(legs: Sequence[dict]) -> list[dict]:
    """Apply the duplicate rule to the legs, in place.

    Per (date, condition, flight, leg) the last attempt without a hard fail is the leg, earlier
    attempts are `superseded`, and `rerun_of` points to the attempt before.

    Returns:
        One entry per repeated leg for the notes: its attempts, the kept one and the reason.
    """
    groups: dict[tuple, list[dict]] = {}
    for item in legs:
        m = item["manifest"]
        groups.setdefault((m["date"], m["condition"], m["flight"], m["leg_index"]), []).append(item)
    report = []
    for key, attempts in sorted(groups.items()):
        attempts.sort(key=lambda it: it["manifest"]["run_id"][-19:])
        for previous, item in zip(attempts[:-1], attempts[1:]):
            item["manifest"]["rerun_of"] = previous["manifest"]["leg_id"]
        if len(attempts) == 1:
            continue
        usable = [i for i, it in enumerate(attempts) if not it["meta"]["quality"]["hard_fail"]]
        kept = usable[-1] if usable else len(attempts) - 1
        for i, item in enumerate(attempts):
            if i < kept:
                item["meta"]["quality"]["flags"].append("superseded")
        later_failed = [attempts[i]["manifest"]["leg_id"] for i in range(kept + 1, len(attempts))]
        report.append(
            {
                "key": {"date": key[0], "condition": key[1], "flight": key[2], "leg": key[3]},
                "attempts": [
                    {
                        "leg_id": it["manifest"]["leg_id"],
                        "hard_fail_reason": it["meta"]["quality"]["hard_fail_reason"],
                        "flags": list(it["meta"]["quality"]["flags"]),
                    }
                    for it in attempts
                ],
                "kept": attempts[kept]["manifest"]["leg_id"],
                "ambiguous": bool(later_failed) or not usable,
                "note": (
                    f"later attempts hard-failed: {later_failed}"
                    if later_failed
                    else ("no attempt without a hard fail" if not usable else "last attempt kept")
                ),
            }
        )
    for item in legs:
        q, m = item["meta"]["quality"], item["manifest"]
        m.setdefault("rerun_of", "")
        m["hard_fail"] = q["hard_fail"]
        m["flags"] = ";".join(q["flags"])
        q["usable"] = not q["hard_fail"] and "superseded" not in q["flags"]
    return report


def counts(legs: Sequence[dict]) -> list[dict]:
    """Legs per (date, design_version, block, condition, dz, v), total and usable."""
    table: dict[tuple, list[int]] = {}
    for item in legs:
        m = item["manifest"]
        key = (
            m["date"],
            m["design_version"],
            m["block"],
            m["condition"],
            m["dz_nominal"],
            m["v_nominal"],
        )
        entry = table.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += int(item["meta"]["quality"]["usable"])
    names = ("date", "design_version", "block", "condition", "dz_nominal", "v_nominal")
    return [
        dict(zip(names, key)) | {"total": v[0], "usable": v[1]} for key, v in sorted(table.items())
    ]


def code_state(repo: Path) -> tuple[str, bool]:
    """HEAD commit of the code repository and whether its tree is dirty."""
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return commit, bool(status.strip())


def json_default(value: object) -> object:
    """Encode numpy scalars and arrays for JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value)} is not JSON serializable")


def _finite(obj: object) -> object:
    """Replace NaN and inf by None, recursively, so the JSON stays strict."""
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_finite(v) for v in obj]
    if isinstance(obj, float | np.floating) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path, obj: object):
    """Strict JSON (NaN as null), indented."""
    path.write_text(json.dumps(_finite(obj), indent=2, default=json_default) + "\n")


def write_dataset(out: Path, legs: Sequence[dict], header: Mapping, stand: Mapping) -> dict:
    """Write the leg folders, `manifest.csv`, `chain_test.npz` and `dataset.json` into `out`.

    Args:
        out: The version folder; replaced if it exists.
        legs: Built legs after `resolve_duplicates`.
        header: `dataset.json` without `counts`.
        stand: The `rig.stand` of every leg, for `chain_test.npz`.

    Returns:
        The written `dataset.json` content.
    """
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for item in legs:
        folder = out / item["manifest"]["path"]
        folder.mkdir()
        np.savez_compressed(folder / "states.npz", **item["states"])
        write_json(folder / "meta.json", item["meta"])
    with open(out / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for item in sorted(legs, key=lambda it: it["manifest"]["leg_id"]):
            writer.writerow({k: item["manifest"][k] for k in MANIFEST_COLUMNS})
    np.savez(out / "chain_test.npz", **chain_test({stand_hash(stand): stand}))
    dataset = dict(header) | {"counts": counts(legs)}
    write_json(out / "dataset.json", dataset)
    return dataset


def export(
    data: Path,
    version: str,
    pattern: str | None = None,
    overwrite: bool = False,
    accept_gain: bool = False,
    out: Path | None = None,
) -> dict:
    """Build and write the dataset, see the module docstring.

    Args:
        data: The data repository.
        version: Dataset version, e.g. "legs_v1".
        pattern: Glob of the run folders under `results/downwash/a3/`; the version's runs in
            `VERSION_RUNS` if None.
        overwrite: Replace an existing, untagged version folder.
        accept_gain: Write even when the gain ratio is off 1 by more than `GAIN_TOL`.
        out: Folder to write instead of `legs/<version>/` of the data repository (drafts, tests).

    Returns:
        A summary for the notes: legs, duplicates, the stand gain fit.
    """
    out = dataset_dir(data, version) if out is None else Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; a tagged version is read-only, use a new version")
    tags = subprocess.run(
        ["git", "-C", str(data), "tag", "--list", version],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if tags:
        raise RuntimeError(f"{version} is tagged in {data}: it is read-only")
    pattern = version_runs(version, pattern)
    folders = sorted(runs_dir(data).glob(pattern))
    if not folders:
        raise FileNotFoundError(f"No runs match {pattern} in {runs_dir(data)}")
    runs = []
    for folder in folders:
        logger.info(f"Loading {folder.name}")
        runs.append(load_session(folder, data))
    edges = [row for run in runs for row in edges_of_run(run)]
    gain = summarize(edges)
    verdict = gain["verdict"]
    if not verdict["gain_ok"] and not accept_gain:
        raise RuntimeError(f"Gain ratio {verdict['gain_ratio']:.4f} outside 1 +- {GAIN_TOL}: ask")
    stand = stand_entry(verdict, gain["up"]["n"])
    commit, dirty = code_state(Path(__file__).parents[3])
    created = datetime.now().astimezone().isoformat(timespec="seconds")
    model = runs[0]["meta"]["config"]["drones"]["model"]
    ctx = {
        "stand": stand,
        "curves": curves(data, runs[0]["meta"]),
        "params": model_params(model),
        "g": G,
        "created": created,
        "commit": commit,
    }
    legs = []
    for run in runs:
        meta = run["meta"]
        schedule = {int(s["leg"]): s for s in meta["victim_schedule"]["schedule"]}
        for leg_key, outcome in meta["victim_schedule"]["legs"].items():
            if "spin_off" not in outcome["events"]:
                continue  # Not flown
            row = next(r for r in meta["design_rows"] if int(r["leg"]) == int(leg_key))
            legs.append(build_leg(run, row, schedule[int(leg_key)], outcome, ctx))
    duplicates = resolve_duplicates(legs)
    key = stand_hash(stand)
    header = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "version": version,
        "created": created,
        "producer": {
            "code_repo": "crazyflow_experiments",
            "code_commit": commit,
            "code_dirty": dirty,
        },
        "synthetic": False,
        "truth": None,
        "runs": [
            {
                "run_id": run["folder"].name,
                "date": run_date(run["folder"].name),
                "design_version": design_version(run["meta"]),
                "condition": CONDITIONS[run["meta"]["mode"]],
                "path": str(run["folder"].relative_to(data)),
                "aborted": run["meta"].get("abort_reason"),
            }
            for run in runs
        ],
        "calibrations": {
            "rpm2thrust": ctx["curves"],
            "stand": {
                "hash": key,
                "rig_stand": stand,
                "cell_delay_s_source": "a3_stand_convert.spin_on_delay, summary.csv cell_delay_s",
                "gain_fit": gain,
                "edges": edges,
            },
        },
    }
    dataset = write_dataset(out, legs, header, stand)
    logger.info(f"{len(legs)} legs -> {out}")
    return {
        "legs": legs,
        "duplicates": duplicates,
        "gain": gain,
        "edges": edges,
        "dataset": dataset,
    }


def main():
    """Command line, see the module docstring."""
    parser = argparse.ArgumentParser(description="Export the A3 stand legs in LEGS_FORMAT.")
    parser.add_argument("--data", type=Path, required=True, help="crazyflow_experiments_data")
    parser.add_argument("--version", required=True, help="Dataset version folder, e.g. legs_v2")
    parser.add_argument("--runs", help="Glob of the run folders, else the version's VERSION_RUNS")
    parser.add_argument("--overwrite", action="store_true", help="Replace an untagged version")
    parser.add_argument("--accept-gain", action="store_true", help="Ignore the gain ratio check")
    parser.add_argument("--report", type=Path, help="Write the duplicates and gain here as JSON")
    parser.add_argument("--out", type=Path, help="Write here instead of legs/<version>/")
    args = parser.parse_args()
    result = export(args.data, args.version, args.runs, args.overwrite, args.accept_gain, args.out)
    if args.report:
        write_json(
            args.report,
            {"duplicates": result["duplicates"], "gain": result["gain"], "edges": result["edges"]},
        )


if __name__ == "__main__":
    main()
