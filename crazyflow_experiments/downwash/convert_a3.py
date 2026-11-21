"""Convert one A3 flight into per-leg arrays in the thesis format.

The inputs are the flight folder of `deploy_a3.py` (`meta.json`, `records.pkl`), the SD logs of
both drones, copied into its `sd/victim/` and `sd/source/`, and the rosbag of `/tf` recorded during
the flight. numpy and scipy do the computing; crazyflow is only read for the drone parameters and
rosbags for the bag.

1. Decode both SD logs and decompress their quaternions.
2. Map each drone's clock onto PC time, `t_pc = a + (1 + b) t_onboard`, from the setpoint echo:
   `ctrltarget` on the card is the setpoint the PC sent, truncated to the mm by cflib, so every
   change of it is matched to the sent setpoint with the same value. The fit is checked against
   the IMU over the sync hops, where the vertical acceleration of the drone must line up with the
   motion capture's; if the two disagree, the conversion stops after writing `sync.json`.
3. Cut every leg from the start of its hover to the end of its pause on a 500 Hz grid in PC time.
   A gap in an SD log or the rpm, NaN on the grid, fails the leg only inside its label window, the
   settled window extended by the wake's arrival; elsewhere it is the soft flag
   `<sd|rpm>_gap_outside_label_<drone>`, as is a crossing speed off its nominal (`cross_speed_off`).
4. Write per leg `leg<NN>/states.npz` (arrays (T, 2, ...), index 0 the victim, 1 the source),
   `meta.json` and `quicklook.png`, and per flight `summary.csv`, `summary.png` and `checks.json`.

The force labels are body-frame forces on each drone that thrust, gravity and drag do not explain:
`force_imu` from the accelerometer, `force_fd` from the motion capture by finite differences. Each
has its mean over the leg's pre-pass hover subtracted (accelerometer bias, mass error), and the
subtracted bias is kept in `quality`. The thrust is the measured rpm on crazyflow's rpm2thrust of
the model, scaled by each drone's `thrust_scale` from the config. The bias does not replace it: a
wrong scale also distorts every change of thrust during the pass.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/convert_a3.py <flight folder> \
        --bag results/downwash/a3/bags/flight<k> [--victim-sd vicNN] [--source-sd srcNN]
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from crazyflow_experiments.downwash import ROLES, plots, usd
from crazyflow_experiments.downwash.design import SETTLED_EXT_MARGIN, gravity, model_params
from crazyflow_experiments.downwash.sync import ClockFit, fit_clock
from crazyflow_experiments.downwash.thrust import curve, rotor_thrust
from crazyflow_experiments.downwash.timeline import segment_leg_times, victim_hover_point

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(
    "crazyflow_experiments.downwash.convert_a3"
)  # Not __name__, the __main__
logger.setLevel(logging.INFO)

GRID_FREQ = 500.0  # Hz, the sim's rate
DRONE_NAMES = ("lower", "upper")  # The sim's names of index 0 and 1
QUAT_ORDER = "xyzw"
PHASE_NAMES = ("hover", "ramp_up", "cruise", "ramp_down", "pause")
PAD = 1.0  # s of signal kept around the legs for the zero-phase filters
MOCAP_MAX_GAP = 0.05  # s, the motion capture is NaN across longer gaps
ONBOARD_MAX_GAP = 0.01  # s, an SD log is NaN across longer gaps
RPM_MAX_GAP = 5  # samples, shorter rpm dropouts are interpolated, longer ones fail the leg
VEL_CUTOFF = 10.0  # Hz, zero-phase low-pass of the motion capture velocity and acceleration
SYNC_CUTOFF = 20.0  # Hz, low-pass of the vertical accelerations compared over the hops
LABEL_CUTOFF = 10.0  # Hz, low-pass of the force labels for the peak and the wake mask
HOP_PAD = 0.5  # s around each hop compared with the IMU
SYNC_GRID = 1000.0  # Hz, grid of the hop comparison
MAX_LAG = 0.05  # s, lags searched in the hop comparison
SYNC_RESIDUAL_TOL = 0.002  # s, std of the setpoint echo residuals
DRIFT_TOL = 0.02  # Clock drift; the cf21B clocks run ~1.3 % slow (`fit_clock`), more is a bad fit
SYNC_LAG_TOL = 0.005  # s, IMU and setpoint echo agree this well
SPEED_TOL = 0.05  # Relative crossing speed error that flags a leg (`cross_speed_off`)
MOTION_FRACTION = 0.05  # Of the nominal speed, the source moves above it
CRUISE_FRACTION = 0.95  # Of the nominal speed, the source cruises above it
WAKE_FRACTION = 0.01  # Of the peak, the victim is in the wake above it
WAKE_NOISE = 3.0  # Standard deviations of the pre-pass label the wake must exceed as well
Y_OFFSET_TOL = 0.03  # m, lateral offset flown off nominal that is flagged
LABEL_AGREEMENT = 0.2  # The two labels agree on the peak within this
THRUST_TOL = 0.03  # Hover thrust over m g off 1 by more is flagged
BIAS_STD_TOL = 0.003  # N, std of the label bias over the legs of a flight
LAG_TOL = 0.1  # s, peak lag off the predicted arrival by more is flagged


@dataclass
class OnboardLog:
    """One decoded SD log.

    Attributes:
        t: Onboard time in s, strictly increasing.
        values: Every logged variable as float, `motor.mX_rpm` NaN without a reading.
        quat: Decompressed `stateEstimateZ.quat` as (N, 4) in xyzw order.
        info: The decoder's file info.
    """

    t: NDArray
    values: dict[str, NDArray]
    quat: NDArray
    info: dict

    def stack(self, names: Sequence[str]) -> NDArray:
        """Variables as columns (N, len(names))."""
        return np.stack([self.values[name] for name in names], axis=1)


@dataclass
class Mocap:
    """Motion capture poses of one rigid body from the bag, stamped in PC time."""

    t: NDArray
    pos: NDArray
    quat: NDArray


def read_onboard(path: Path) -> OnboardLog:
    """Decode an SD log; rpm without a reading becomes NaN."""
    raw = usd.decode_usd(path)
    event = raw["fixedFrequency"]
    t = event["t_us"].astype(float) * 1e-6
    keep = np.concatenate([[True], np.diff(t) > 0])
    values = {}
    for name, value in event.items():
        if name in ("t_us", "stateEstimateZ.quat"):
            continue
        value = np.asarray(value, dtype=float)[keep]
        if name.endswith("_rpm"):
            value[value == usd.RPM_INVALID] = np.nan
        values[name] = value
    quat = usd.quatdecompress(event["stateEstimateZ.quat"])[keep]
    return OnboardLog(t[keep], values, quat, raw["_info"])


def read_mocap(bag: Path, frames: Mapping[str, str]) -> dict[str, Mocap]:
    """Poses of the drones from the `/tf` topic of a bag, stamped with their header stamps.

    Args:
        bag: The bag folder.
        frames: Child frame id per role, e.g. {"victim": "cf3"}.
    """
    typestore = get_typestore(Stores.ROS2_FOXY)  # tf2_msgs/TFMessage is the same in every distro
    role_of = {frame: role for role, frame in frames.items()}
    samples = {role: [] for role in frames}
    with Reader(bag) as reader:
        connections = [c for c in reader.connections if c.topic == "/tf"]
        for connection, _, raw in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(raw, connection.msgtype)
            for transform in msg.transforms:
                role = role_of.get(transform.child_frame_id)
                if role is None:
                    continue
                stamp = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
                p, q = transform.transform.translation, transform.transform.rotation
                samples[role].append((stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
    mocap = {}
    for role, rows in samples.items():
        if not rows:
            raise RuntimeError(f"No /tf poses of {frames[role]} in {bag}")
        data = np.array(rows)
        data = data[np.argsort(data[:, 0], kind="stable")]
        data = data[np.concatenate([[True], np.diff(data[:, 0]) > 0])]
        mocap[role] = Mocap(data[:, 0], data[:, 1:4], data[:, 4:8])
    return mocap


def lowpass(x: NDArray, fs: float, cutoff: float, order: int = 4) -> NDArray:
    """Zero-phase Butterworth low-pass along axis 0. NaNs are bridged for the filter and kept."""
    x = np.asarray(x, dtype=float)
    flat = x.reshape(len(x), -1)
    out = np.full_like(flat, np.nan)
    sos = butter(order, cutoff, fs=fs, output="sos")
    index = np.arange(len(x))
    for j in range(flat.shape[1]):
        good = np.isfinite(flat[:, j])
        if good.sum() < 32:
            continue
        filtered = sosfiltfilt(sos, np.interp(index, index[good], flat[good, j]))
        out[:, j] = np.where(good, filtered, np.nan)
    return out.reshape(x.shape)


def resample(t_src: NDArray, x_src: NDArray, t_grid: NDArray, max_gap: float) -> NDArray:
    """Linear interpolation onto `t_grid`, NaN outside the data, across gaps and next to NaNs."""
    x = np.asarray(x_src, dtype=float)
    flat = x.reshape(len(t_src), -1)
    out = np.full((len(t_grid), flat.shape[1]), np.nan)
    inside = (t_grid >= t_src[0]) & (t_grid <= t_src[-1])
    k = np.clip(np.searchsorted(t_src, t_grid[inside], side="right"), 1, len(t_src) - 1)
    t0, t1 = t_src[k - 1], t_src[k]
    w = ((t_grid[inside] - t0) / (t1 - t0))[:, None]
    values = flat[k - 1] * (1 - w) + flat[k] * w
    values[(t1 - t0) > max_gap] = np.nan
    out[inside] = values
    return out.reshape((len(t_grid), *x.shape[1:]))


def resample_quat(t_src: NDArray, q_src: NDArray, t_grid: NDArray, max_gap: float) -> NDArray:
    """Slerp of xyzw quaternions onto `t_grid`, NaN outside the data and across gaps."""
    out = np.full((len(t_grid), 4), np.nan)
    inside = (t_grid >= t_src[0]) & (t_grid <= t_src[-1])
    out[inside] = Slerp(t_src, R.from_quat(q_src))(t_grid[inside]).as_quat()
    k = np.clip(np.searchsorted(t_src, t_grid, side="right"), 1, len(t_src) - 1)
    out[(t_src[k] - t_src[k - 1]) > max_gap] = np.nan
    return out


def fill_dropouts(x: NDArray, max_run: int) -> NDArray:
    """Interpolate runs of at most `max_run` NaNs over the sample index; longer runs stay NaN."""
    x = np.array(x, dtype=float)
    bad = ~np.isfinite(x)
    if not bad.any() or bad.all():
        return x
    edges = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    index = np.arange(len(x))
    for start, end in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        if end - start <= max_run and start > 0 and end < len(x):
            x[start:end] = np.interp(index[start:end], index[~bad], x[~bad])
    return x


def lag(t: NDArray, reference: NDArray, t_other: NDArray, other: NDArray, max_lag: float) -> float:
    """Lag `d` in s with which `other(t - d)` best matches `reference(t)`.

    The mean-free RMS difference is swept over the lags at the grid step and the minimum refined
    with a parabola. A positive lag means the reference is late.
    """
    step = t[1] - t[0]
    lags = np.arange(-max_lag, max_lag + step / 2, step)
    cost = np.full(len(lags), np.nan)
    for i, d in enumerate(lags):
        shifted = np.interp(t - d, t_other, other, left=np.nan, right=np.nan)
        ok = np.isfinite(reference) & np.isfinite(shifted)
        if ok.sum() < 50:
            continue
        a, b = reference[ok] - reference[ok].mean(), shifted[ok] - shifted[ok].mean()
        cost[i] = np.mean((a - b) ** 2)
    if not np.isfinite(cost).any():
        return math.nan
    k = int(np.nanargmin(cost))
    best = float(lags[k])
    if 0 < k < len(lags) - 1 and np.isfinite(cost[k - 1 : k + 2]).all():
        y0, y1, y2 = cost[k - 1 : k + 2]
        if (curvature := y0 - 2 * y1 + y2) > 0:
            best += 0.5 * (y0 - y2) / curvature * step
    return best


def hop_lag(
    mocap: Mocap, log: OnboardLog, fit: ClockFit, window: tuple[float, float], g: float
) -> float:
    """Lag of the motion capture's vertical acceleration behind the mapped IMU over one hop."""
    t0, t1 = window
    t = np.arange(t0, t1, 1 / SYNC_GRID)
    z = lowpass(resample(mocap.t, mocap.pos[:, 2], t, MOCAP_MAX_GAP), SYNC_GRID, SYNC_CUTOFF)
    az_mocap = lowpass(np.gradient(np.gradient(z, t), t), SYNC_GRID, SYNC_CUTOFF)
    acc_world = R.from_quat(log.quat).apply(log.stack(["acc.x", "acc.y", "acc.z"]) * g)
    t_wide = np.arange(t0 - MAX_LAG - 0.1, t1 + MAX_LAG + 0.1, 1 / SYNC_GRID)
    az_imu = resample(fit.to_pc(log.t), acc_world[:, 2] - g, t_wide, ONBOARD_MAX_GAP)
    return lag(t, az_mocap, t_wide, lowpass(az_imu, SYNC_GRID, SYNC_CUTOFF), MAX_LAG)


def setpoint_echo(records: Sequence[Mapping], role: str, log: OnboardLog) -> tuple:
    """Sent and echoed setpoints of one drone in mm, on the axes its card logs."""
    axes = [i for i, axis in enumerate("xyz") if f"ctrltarget.{axis}" in log.values]
    if not axes:
        raise RuntimeError(f"The {role}'s log has no ctrltarget variable to synchronize with")
    recs = [r for r in records if r[role] is not None]
    pc_t = np.array([r["t_wall"] for r in recs])
    pc_mm = np.array([[r[role]["cmd_mm"][i] for i in axes] for r in recs])
    names = [f"ctrltarget.{'xyz'[i]}" for i in axes]
    onboard_mm = np.round(log.stack(names) * 1000).astype(int)
    return pc_t, pc_mm, log.t, onboard_mm


def drone_signals(
    t_grid: NDArray,
    mocap: Mocap | None,
    log: OnboardLog | None,
    fit: ClockFit | None,
    params: Mapping,
    g: float,
    thrust_scale: float,
) -> dict[str, NDArray]:
    """Every signal of one drone on the flight grid, NaN for a drone that did not fly.

    The rotor thrust is the measured rpm on the drone's calibrated curve, see `thrust.curve`.
    """
    n = len(t_grid)
    nan3, nan4 = np.full((n, 3), np.nan), np.full((n, 4), np.nan)
    signals = {
        "pos": nan3,
        "quat": nan4,
        "vel": nan3,
        "vel_dot": nan3,
        "pos_est": nan3,
        "vel_est": nan3,
        "quat_est": nan4,
        "ang_vel": nan3,
        "acc": nan3,
        "rotor_vel": nan4,
        "vbat": np.full(n, np.nan),
        "cmd_thrust": np.full(n, np.nan),
    }
    dt = 1 / GRID_FREQ
    if mocap is not None:
        pos = resample(mocap.t, mocap.pos, t_grid, MOCAP_MAX_GAP)
        vel = lowpass(np.gradient(pos, dt, axis=0), GRID_FREQ, VEL_CUTOFF)
        signals |= {
            "pos": pos,
            "quat": resample_quat(mocap.t, mocap.quat, t_grid, MOCAP_MAX_GAP),
            "vel": vel,
            "vel_dot": lowpass(np.gradient(vel, dt, axis=0), GRID_FREQ, VEL_CUTOFF),
        }
    if log is not None:
        t_pc = fit.to_pc(log.t)

        def onboard(names: Sequence[str]) -> NDArray:
            return resample(t_pc, log.stack(names), t_grid, ONBOARD_MAX_GAP)

        xyz = ("x", "y", "z")
        rpm = np.stack(
            [fill_dropouts(log.values[f"motor.m{i}_rpm"], RPM_MAX_GAP) for i in range(1, 5)], axis=1
        )
        vbat = (
            log.values["pm.vbatMV"] / 1000 if "pm.vbatMV" in log.values else log.values["pm.vbat"]
        )
        signals |= {
            "pos_est": onboard([f"stateEstimate.{a}" for a in xyz]),
            "vel_est": onboard([f"stateEstimate.v{a}" for a in xyz]),
            "quat_est": resample_quat(t_pc, log.quat, t_grid, ONBOARD_MAX_GAP),
            "ang_vel": np.deg2rad(onboard([f"gyro.{a}" for a in xyz])),
            "acc": onboard([f"acc.{a}" for a in xyz]) * g,
            "rotor_vel": resample(t_pc, rpm, t_grid, ONBOARD_MAX_GAP),
            "vbat": resample(t_pc, vbat, t_grid, ONBOARD_MAX_GAP),
        }
        if "ctrlMel.cmd_thrust" in log.values:
            signals["cmd_thrust"] = onboard(["ctrlMel.cmd_thrust"])[:, 0]
    coef = curve(params["rpm2thrust"], thrust_scale)
    signals["motor_thrust"] = rotor_thrust(signals["rotor_vel"], coef)
    return signals


def body_forces(signals: Mapping[str, NDArray], mass: float, params: Mapping, g: float) -> tuple:
    """Forces on one drone in its body frame that thrust, gravity and drag do not explain.

    `force_imu = m a_IMU - z sum(f) - C v_B` and
    `force_fd = R^T m (dv/dt - g) - z sum(f) - C v_B`, with the accelerometer's specific force
    `a_IMU`, the rotor thrusts `f` from the measured rpm, crazyflow's drag matrix `C` and the body
    velocity `v_B`, both from the motion capture.

    Returns:
        `force_imu` and `force_fd` as (N, 3).
    """
    n = len(signals["pos"])
    force_imu, force_fd = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    ok = np.isfinite(signals["quat"]).all(axis=1)
    if not ok.any():
        return force_imu, force_fd
    rot = R.from_quat(signals["quat"][ok])
    v_body = rot.inv().apply(np.nan_to_num(signals["vel"][ok]))
    v_body[~np.isfinite(signals["vel"][ok]).all(axis=1)] = np.nan
    drag = v_body @ np.asarray(params["drag_matrix"], dtype=float).T
    thrust = np.zeros((ok.sum(), 3))
    thrust[:, 2] = signals["motor_thrust"][ok].sum(axis=1)
    gravity_vec = np.array([0.0, 0.0, -g])
    force_imu[ok] = mass * signals["acc"][ok] - thrust - drag
    inertial = mass * (signals["vel_dot"][ok] - gravity_vec)
    inertial_body = rot.inv().apply(np.nan_to_num(inertial))
    inertial_body[~np.isfinite(inertial).all(axis=1)] = np.nan
    force_fd[ok] = inertial_body - thrust - drag
    return force_imu, force_fd


def _runs(mask: NDArray) -> list[tuple[int, int]]:
    """Inclusive index ranges of the True runs of a boolean array."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1))


def derive(
    t: NDArray, p_src: NDArray, v_src: NDArray, p_vic: NDArray, row: Mapping, a3: Mapping
) -> tuple[dict, NDArray]:
    """Pass quantities of one leg from the motion capture, and the phase of every sample.

    The phases follow the source's measured horizontal speed: it moves above 5 % and cruises above
    95 % of the nominal speed. The crossing is where the source passes the victim along x.

    Args:
        t: Seconds from the leg start (T,).
        p_src: Source position (T, 3).
        v_src: Source velocity (T, 3).
        p_vic: Victim position (T, 3), its hover point where it did not fly.
        row: The leg's design row.
        a3: The `[a3]` config.

    Returns:
        The `derived` entries and the phase index per sample (T,), -1 where undetermined.
    """
    nan = math.nan
    speed_nom = float(row["speed"])
    speed = np.linalg.norm(v_src[:, :2], axis=1)
    dx = p_src[:, 0] - p_vic[:, 0]
    phase = np.full(len(t), -1)
    derived = {"cross_time": nan, "cross_speed": nan, "cruise_start": nan, "cruise_end": nan}
    a, b = dx[:-1], dx[1:]  # Half-open, so a sample exactly at the victim counts once
    crossings = np.nonzero(((a <= 0) & (b > 0)) | ((a >= 0) & (b < 0)))[0]
    cruising = np.nan_to_num(speed) >= CRUISE_FRACTION * speed_nom
    moving = np.nan_to_num(speed) > MOTION_FRACTION * speed_nom
    if len(crossings) == 0 or not cruising.any():
        return derived, phase
    k = int(crossings[np.argmax(np.nan_to_num(speed[crossings]))])
    cross_time = float(t[k] + (t[k + 1] - t[k]) * dx[k] / (dx[k] - dx[k + 1]))
    cross_speed = float(np.interp(cross_time, t, speed))
    cruise = min(
        _runs(cruising),
        key=lambda run: 0 if run[0] <= k <= run[1] else min(abs(run[0] - k), abs(run[1] - k)),
    )
    motion = next(run for run in _runs(moving) if run[0] <= cruise[0] and run[1] >= cruise[1])
    phase[: motion[0]] = 0
    phase[motion[0] : cruise[0]] = 1
    phase[cruise[0] : cruise[1] + 1] = 2
    phase[cruise[1] + 1 : motion[1] + 1] = 3
    phase[motion[1] + 1 :] = 4
    in_cruise = slice(cruise[0], cruise[1] + 1)
    settled_half = a3["cruise_half"] / cross_speed
    settled_end = cross_time + settled_half
    heading = np.nanmean(v_src[in_cruise, :2], axis=0)
    phase_ends = [
        float(t[np.nonzero(phase == i)[0][-1]]) if (phase == i).any() else nan for i in range(5)
    ]
    derived = {
        "pass_distance": float(abs(p_src[motion[1], 0] - p_src[motion[0], 0])),
        "pass_duration": float(t[motion[1]] - t[motion[0]]),
        "height_sep": float(np.nanmean(p_src[in_cruise, 2] - p_vic[in_cruise, 2])),
        "lateral_offset": float(np.nanmean(p_src[in_cruise, 1] - p_vic[in_cruise, 1])),
        "closest_point": [float(np.interp(cross_time, t, p_src[:, i])) for i in range(3)],
        "upper_heading": float(math.atan2(heading[1], heading[0])),
        "direction": int(np.sign(heading[0])),
        "cruise_start": float(t[cruise[0]]),
        "cruise_end": float(t[cruise[1]]),
        "cross_time": cross_time,
        "cross_speed": cross_speed,
        "cruise_reach": float(
            min(
                abs(p_src[cruise[0], 0] - p_vic[cruise[0], 0]),
                abs(p_src[cruise[1], 0] - p_vic[cruise[1], 0]),
            )
        ),
        "settled_half": a3["cruise_half"],
        "settled_start": cross_time - settled_half,
        "settled_end": settled_end,
        "settled_end_ext": min(
            settled_end + float(row["arrival_s"]) + SETTLED_EXT_MARGIN, float(t[cruise[1]])
        ),
        "tail_time": float(t[-1] - t[motion[1]]),
        "phase_names": list(PHASE_NAMES),
        "phase_steps": [int((phase == i).sum()) for i in range(5)],
        "phase_ends": phase_ends,
    }
    return derived, phase


def wake(t: NDArray, fz: NDArray, settled: NDArray, search: NDArray, hover: NDArray) -> dict:
    """Peak of the victim's vertical force and the samples in its wake.

    The peak is the most negative low-passed `fz` inside `search`. The wake is the run of samples
    around it whose magnitude exceeds 1 % of the peak and three standard deviations of the
    pre-pass hover; on real labels the noise alone exceeds 1 % of the peak.

    Args:
        t: Seconds from the leg start (T,).
        fz: Low-passed vertical force on the victim, bias subtracted (T,).
        settled: The extended settled window (T,).
        search: Where the peak is searched (T,).
        hover: The pre-pass hover (T,).

    Returns:
        `peak_fz`, `peak_time`, the wake mask `in_wake` and its threshold.
    """
    candidates = np.where(search & np.isfinite(fz), fz, np.inf)
    if not np.isfinite(candidates).any():
        return {"peak_fz": math.nan, "peak_time": math.nan, "in_wake": np.zeros(len(t), bool)}
    k = int(np.argmin(candidates))
    peak = float(fz[k])
    threshold = max(WAKE_FRACTION * abs(peak), WAKE_NOISE * float(np.nanstd(fz[hover])))
    above = np.nan_to_num(-fz * np.sign(-peak)) > threshold
    in_wake = np.zeros(len(t), dtype=bool)
    for start, end in _runs(above):
        if start <= k <= end:
            in_wake[start : end + 1] = True
    return {"peak_fz": peak, "peak_time": float(t[k]), "in_wake": in_wake, "threshold": threshold}


@dataclass
class FlightData:
    """Everything of one flight on its 500 Hz grid in PC time.

    Attributes:
        folder: The flight folder.
        meta: The flight's meta.json.
        records: Records of the flight loop.
        roles: Drones that flew.
        t_zero: PC time of the timelines' zero.
        k_first: Grid index of the first sample, the grid being `t_zero + k / GRID_FREQ`.
        t_grid: PC time of every sample.
        signals: Signals per role, see `drone_signals`.
        forces: `force_imu` and `force_fd` per role, before the bias subtraction.
        invalid_rpm: Per role the PC times of the onboard samples and which rpm had no reading.
        sync: The clock fits and hop lags written to sync.json.
        params: Crazyflow parameters of the drone model.
        gravity: Magnitude of gravity in m/s^2.
    """

    folder: Path
    meta: dict
    records: list[dict]
    roles: tuple[str, ...]
    t_zero: float
    k_first: int
    t_grid: NDArray
    signals: dict[str, dict[str, NDArray]]
    forces: dict[str, tuple[NDArray, NDArray]]
    invalid_rpm: dict[str, tuple[NDArray, NDArray]]
    sync: dict
    params: dict
    gravity: float


def _both(data: FlightData, key: str, sl: slice) -> NDArray:
    """A signal of both drones as (T, 2, ...), the victim first."""
    return np.stack([data.signals[role][key][sl] for role in ROLES], axis=1)


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value if isinstance(value, (int, float, str, bool)) else repr(value)


def _finite(value: float) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def convert_leg(data: FlightData, row: Mapping) -> dict:
    """Write the folder of one flown leg and return its summary row."""
    cfg, a3 = data.meta["config"], data.meta["config"]["a3"]
    leg = int(row["leg"])
    times = segment_leg_times(data.meta["timeline"]["source"], leg)
    k0 = round(times["hover_start"] * GRID_FREQ) - data.k_first
    k1 = round(times["pause_end"] * GRID_FREQ) - data.k_first
    sl = slice(k0, k1 + 1)
    t = np.arange(k1 - k0 + 1) / GRID_FREQ
    t_wall = data.t_grid[sl]
    pos, vel = _both(data, "pos", sl), _both(data, "vel", sl)
    hover_point = victim_hover_point(cfg)
    p_vic = np.where(np.isfinite(pos[:, 0]), pos[:, 0], hover_point)
    derived, phase = derive(t, pos[:, 1], vel[:, 1], p_vic, row, a3)

    hover = t < a3["hover_time"]
    force_imu = np.stack([data.forces[role][0][sl] for role in ROLES], axis=1)
    force_fd = np.stack([data.forces[role][1][sl] for role in ROLES], axis=1)
    bias_imu, bias_fd = np.nanmean(force_imu[hover], axis=0), np.nanmean(force_fd[hover], axis=0)
    force_imu, force_fd = force_imu - bias_imu, force_fd - bias_fd
    fz = lowpass(force_imu[:, 0, 2], GRID_FREQ, LABEL_CUTOFF)
    fz_fd = lowpass(force_fd[:, 0, 2], GRID_FREQ, LABEL_CUTOFF)
    settled = (t >= derived.get("settled_start", math.nan)) & (
        t <= derived.get("settled_end_ext", math.nan)
    )
    search = phase >= 1 if (phase >= 1).any() else ~hover
    peak = wake(t, fz, settled, search, hover)
    peak_fd = wake(t, fz_fd, settled, search, hover)
    in_wake = peak.pop("in_wake")
    outside = int((in_wake & ~settled).sum())

    rotor_vel, motor_thrust = _both(data, "rotor_vel", sl), _both(data, "motor_thrust", sl)
    masses = data.meta["masses"]
    quality = {
        "baseline_pre": {
            "force_imu_bias": bias_imu,
            "force_fd_bias": bias_fd,
            "force_imu_std": np.nanstd(force_imu[hover], axis=0),
            "force_fd_std": np.nanstd(force_fd[hover], axis=0),
        },
        "peak_fz": peak["peak_fz"],
        "peak_time": peak["peak_time"],
        "lag": peak["peak_time"] - derived["cross_time"],
        "wake_threshold": peak.get("threshold", math.nan),
        "peak_fz_fd": peak_fd["peak_fz"],
        "peak_time_fd": peak_fd["peak_time"],
        "peak_fd_over_imu": peak_fd["peak_fz"] / peak["peak_fz"] if peak["peak_fz"] else math.nan,
        "cross_speed_err": (derived["cross_speed"] - row["speed"]) / row["speed"],
        "settled_samples": int(settled.sum()),
        "in_wake_samples": int(in_wake.sum()),
        "in_wake_outside_settled": outside,
        "rpm_dropouts": {},
        "thrust_ratio": {},
        "sync": {role: data.sync[role]["fit"] for role in data.roles},
    }
    hard, soft = [], [flag for flag in data.meta.get("leg_flags", {}).get(str(leg), [])]
    settled_wall = t_wall[settled] if settled.any() else np.array([])
    # The label window a gap must not touch: the settled window extended by the wake's arrival.
    # Unlike `settled` it is not cut at the end of the cruise, which a source cruising near
    # CRUISE_FRACTION ends early (F1 21:14 leg 1: 0.13 s after the crossing, its card stalled 0.55)
    guarded = (t >= derived.get("settled_start", math.nan)) & (
        t <= derived.get("settled_end", math.nan) + float(row["arrival_s"]) + SETTLED_EXT_MARGIN
    )
    for i, role in enumerate(ROLES):
        name = DRONE_NAMES[i]
        if role not in data.roles:
            continue
        t_pc, invalid = data.invalid_rpm[role]
        in_leg = (t_pc >= t_wall[0]) & (t_pc <= t_wall[-1])
        quality["rpm_dropouts"][name] = invalid[in_leg].sum(axis=0).tolist()
        if role == "victim" and len(settled_wall):
            in_settled = (t_pc >= settled_wall[0]) & (t_pc <= settled_wall[-1])
            quality["rpm_dropouts_in_settled"] = int(invalid[in_settled].sum())
            if quality["rpm_dropouts_in_settled"]:
                soft.append("rpm_dropout_in_settled")
        # The cards stall 100-600 ms a few times per flight (18 Nov). A gap only fails the leg
        # inside the label window; elsewhere its samples stay NaN
        for kind, signal in (("rpm", rotor_vel[:, i]), ("sd", _both(data, "acc", sl)[:, i])):
            gap = np.isnan(signal).any(axis=1)
            if (gap & guarded).any():
                hard.append(f"{kind}_gap_{name}")
            elif gap.any():
                soft.append(f"{kind}_gap_outside_label_{name}")
        if not np.isfinite(bias_imu[i]).all():
            hard.append(f"baseline_gap_{name}")  # No sample of the pre-pass hover to subtract
        if np.isnan(pos[:, i]).any():
            hard.append(f"mocap_gap_{name}")
        mass = masses[role]
        ratio = np.nanmean(motor_thrust[hover, i].sum(axis=1)) / (mass * data.gravity)
        quality["thrust_ratio"][name] = float(ratio)
        if abs(ratio - 1) > THRUST_TOL:
            soft.append(f"thrust_ratio_off_{name}")
    if "victim" in data.roles:
        quality["lower_min_z"] = float(np.nanmin(pos[:, 0, 2]))
        quality["lower_drop"] = float(hover_point[2] - np.nanmin(pos[:, 0, 2]))
        quality["lower_max_dxy"] = float(
            np.nanmax(np.linalg.norm(pos[:, 0, :2] - hover_point[:2], axis=1))
        )
    if not math.isfinite(derived["cross_time"]):
        hard.append("no_crossing")
    elif abs(quality["cross_speed_err"]) > SPEED_TOL:
        # A soft flag: at Δz 0.4 m the source speeds up over the victim (+9 to +12 % at 0.5 m/s in
        # both F1 runs of 18 Nov), and `cross_speed` keeps the speed it really had
        soft.append("cross_speed_off")
    if abs(derived.get("lateral_offset", 0.0) - row["lateral_offset"]) > Y_OFFSET_TOL:
        soft.append("y_off_nominal")
    if outside:
        soft.append("in_wake_outside_settled")
        logger.warning(f"Leg {leg}: {outside} wake samples outside the settled window")
    if abs(quality["peak_fd_over_imu"] - 1) > LABEL_AGREEMENT:
        soft.append("label_disagreement")
    quality["hard_fail"], quality["hard_fail_reasons"], quality["soft_flags"] = (
        bool(hard),
        hard,
        soft,
    )

    leg_dir = data.folder / f"leg{leg:02d}"
    leg_dir.mkdir(exist_ok=True)
    states = {
        "t": t,
        "phase": phase,
        "phase_ends": np.asarray(derived.get("phase_ends", [math.nan] * 5), dtype=float),
        "phase_names": np.array(PHASE_NAMES),
        "drone_names": np.array(DRONE_NAMES),
        "pos": pos,
        "vel": vel,
        "quat": _both(data, "quat", sl),
        "pos_est": _both(data, "pos_est", sl),
        "vel_est": _both(data, "vel_est", sl),
        "quat_est": _both(data, "quat_est", sl),
        "ang_vel": _both(data, "ang_vel", sl),
        "acc": _both(data, "acc", sl),
        "rotor_vel": rotor_vel,
        "motor_thrust": motor_thrust,
        "vbat": _both(data, "vbat", sl),
        "cmd_thrust": _both(data, "cmd_thrust", sl),
        "force_imu": force_imu,
        "force_fd": force_fd,
        **sim_params_arrays(data, cfg),
    }
    meta = leg_meta(data, row, times, derived, quality, states)
    meta["signals"]["arrays"] = {key: list(np.shape(value)) for key, value in states.items()}
    states["meta/json"] = np.array(json.dumps(meta, default=_json_default))
    np.savez(leg_dir / "states.npz", **states)
    (leg_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    quicklook(leg_dir / "quicklook.png", data, row, t, t_wall, states, derived, peak, fz, fz_fd)
    return {
        "leg": leg,
        "flight": int(row["flight"]),
        "index_in_flight": int(row["index_in_flight"]),
        "cell": int(row["cell"]),
        "speed": row["speed"],
        "height_sep": row["height_sep"],
        "direction": int(row["direction"]),
        "repeat": int(row["repeat"]),
        "status": "hard_fail" if hard else "ok",
        "cross_speed": derived["cross_speed"],
        "cross_speed_err": quality["cross_speed_err"],
        "height_sep_meas": derived.get("height_sep", math.nan),
        "lateral_offset_meas": derived.get("lateral_offset", math.nan),
        "cross_time": derived["cross_time"],
        "peak_fz": peak["peak_fz"],
        "peak_fz_fd": peak_fd["peak_fz"],
        "peak_time": peak["peak_time"],
        "lag": quality["lag"],
        "arrival_s": row["arrival_s"],
        "thrust_ratio_victim": quality["thrust_ratio"].get("lower", math.nan),
        "thrust_ratio_source": quality["thrust_ratio"].get("upper", math.nan),
        "bias_fx_victim": bias_imu[0, 0],
        "bias_fy_victim": bias_imu[0, 1],
        "bias_fz_victim": bias_imu[0, 2],
        "victim_drop": quality.get("lower_drop", math.nan),
        "settled_samples": quality["settled_samples"],
        "in_wake_outside_settled": outside,
        "hard_fail": ";".join(hard),
        "soft_flags": ";".join(soft),
    }


def sim_params_arrays(data: FlightData, cfg: Mapping) -> dict[str, NDArray]:
    """The `drone/*` and `sim/*` entries of states.npz, as the sim writes them."""
    params, masses = data.params, data.meta["masses"]
    scales = {role: cfg["drones"][role]["thrust_scale"] for role in ROLES}
    return {
        "drone/mass": np.array([masses["victim"]]),
        "drone/mass_upper": np.array([masses["source"]]),
        "drone/L": np.array([float(np.asarray(params["L"]))]),
        "drone/J": np.asarray(params["J"], dtype=float),
        "drone/rpm2thrust": curve(params["rpm2thrust"], scales["victim"]).reshape(1, -1),
        "drone/rpm2thrust_upper": curve(params["rpm2thrust"], scales["source"]).reshape(1, -1),
        "drone/rpm2torque": np.asarray(params["rpm2torque"], dtype=float).reshape(1, -1),
        "drone/mixing_matrix": np.asarray(params["mixing_matrix"], dtype=float),
        "drone/drag_matrix": np.asarray(params["drag_matrix"], dtype=float),
        "sim/freq": np.array(GRID_FREQ),
        "sim/control_freq": np.array(cfg["control"]["ctrl_freq"]),
    }


def leg_meta(
    data: FlightData,
    row: Mapping,
    times: Mapping,
    derived: Mapping,
    quality: Mapping,
    states: Mapping,
) -> dict:
    """meta.json of one leg in the layout of the real-run schema."""
    cfg, flight = data.meta["config"], data.meta
    a3, leg = cfg["a3"], int(row["leg"])
    direction = "p" if row["direction"] > 0 else "n"
    run_id = (
        f"{data.folder.name}_leg{leg:03d}_dz{row['height_sep']:.2f}_v{row['speed']:.2f}_{direction}"
    )
    table = {(s["leg"], s["phase"]): s for s in flight["timeline"]["source"]}
    vbat = states["vbat"][0]
    return {
        "run": {
            "id": run_id,
            "plugin": "real",
            "created": flight["created"],
            "tag": "A3",
            "sweep_id": data.folder.name,
            "argv": flight["argv"],
            "hostname": flight["hostname"],
            "hash": flight["git"]["hash"],
            "label": "real flight, hovering victim (A3, part B)",
            "device": "crazyflie",
            "wall_time": float(states["t"][-1]),
        },
        "scenario": {
            "lower_pos": victim_hover_point(cfg).tolist(),
            "upper_start": table[(leg, "pass")]["start"],
            "upper_end": table[(leg, "pass")]["end"],
            "speed": row["speed"],
            "ramp": a3["ramp"],
            "hover_time": a3["hover_time"],
            "pause_time": a3["pause_time"],
            "settle_time": a3["settle_time"],
            "cruise_half": a3["cruise_half"],
            "drone": cfg["drones"]["model"],
            "direction": int(row["direction"]),
            "victim_mode": "hover",
            "victim_power": "battery",
            "battery_v": {"lower": _finite(vbat[0]), "upper": _finite(vbat[1])},
            "timeline": dict(times),
        },
        "derived": {
            **derived,
            "control_freq": cfg["control"]["ctrl_freq"],
            "sim_freq": GRID_FREQ,
            "n_steps": len(states["t"]),
            "duration": float(states["t"][-1]),
        },
        "constants": {"real/gravity": data.gravity, "real/air_density": "[FILL]"},
        "sim_params": {
            "drone/mass": [flight["masses"]["victim"]],
            "drone/mass_upper": [flight["masses"]["source"]],
            "drone/rpm2thrust": states["drone/rpm2thrust"],
            "drone/rpm2thrust_upper": states["drone/rpm2thrust_upper"],
            "drone/rpm2thrust_source": (
                f"crazyflow {cfg['drones']['model']} times thrust_scale "
                f"{cfg['drones']['victim']['thrust_scale']} (lower), "
                f"{cfg['drones']['source']['thrust_scale']} (upper)"
            ),
            "drone/drag_matrix": states["drone/drag_matrix"],
            "drone/mixing_matrix": states["drone/mixing_matrix"],
        },
        "rig": {
            "firmware": {"version": "[FILL]", "git_hash": "[FILL]", "bidir_dshot": True},
            "mocap": {"system": "[FILL]", "frame": "mocap frame, z up", "topic": "/tf"},
            "radio": {
                role: f"radio://0/{cfg['drones'][role]['channel']}/2M" for role in data.roles
            },
            "sd": {
                role: flight["usd"][role] | {"sync": data.sync[role]["fit"]} for role in data.roles
            },
        },
        "sync": data.sync,
        "quality": quality,
        "design": dict(row),
        "sweep": {"key": f"real/A3/{leg}"},
        "files": {"folder": str(data.folder / f"leg{leg:02d}")},
        "code": {"git": flight["git"], "converter": "crazyflow_experiments.downwash.convert_a3"},
        "signals": {
            "quat_order": QUAT_ORDER,
            "units": {
                "pos": "m, motion capture",
                "vel": "m/s, motion capture by central difference, 10 Hz zero-phase low-pass",
                "ang_vel": "rad/s, from gyro deg/s, body frame",
                "acc": "m/s^2, from acc in g, body frame, label use only",
                "rotor_vel": "rpm, measured, dropouts up to 5 samples interpolated",
                "motor_thrust": "N per rotor, crazyflow rpm2thrust times thrust_scale of rotor_vel",
                "force_imu": "N, body frame, pre-pass hover mean subtracted",
                "force_fd": "N, body frame, pre-pass hover mean subtracted",
            },
        },
    }


def quicklook(
    path: Path,
    data: FlightData,
    row: Mapping,
    t: NDArray,
    t_wall: NDArray,
    states: Mapping[str, NDArray],
    derived: Mapping,
    peak: Mapping,
    fz: NDArray,
    fz_fd: NDArray,
):
    """Source x and speed against the setpoint, victim height, rotor speeds and force of a leg."""
    recs = data.records
    rec_t = np.array([r["t_wall"] for r in recs])
    cmd_pos = np.array([r["source"]["cmd_pos"] for r in recs])
    cmd_vel = np.array([r["source"]["cmd_vel"] for r in recs])
    cmd_x = np.interp(t_wall, rec_t, cmd_pos[:, 0])
    cmd_speed = np.interp(t_wall, rec_t, np.linalg.norm(cmd_vel[:, :2], axis=1))
    fig, axes = plots.figure(5, row_height=1.7)
    ax = axes[:, 0]
    for a in ax:
        if math.isfinite(derived.get("settled_start", math.nan)):
            a.axvspan(derived["settled_start"], derived["settled_end_ext"], color=plots.SHADE, lw=0)
        if math.isfinite(derived["cross_time"]):
            a.axvline(derived["cross_time"], color=plots.MUTED, lw=0.8)
    lw = plots.LINE_WIDTH
    ax[0].plot(t, states["pos"][:, 1, 0], color=plots.SOURCE, lw=lw, label="measured")
    ax[0].plot(t, cmd_x, color=plots.SOURCE, lw=lw, ls="--", label="setpoint")
    ax[0].set_ylabel("source x [m]")
    plots.legend(ax[0], loc="upper left", ncols=2)
    ax[1].plot(t, np.linalg.norm(states["vel"][:, 1, :2], axis=1), color=plots.SOURCE, lw=lw)
    ax[1].plot(t, cmd_speed, color=plots.SOURCE, lw=lw, ls="--")
    ax[1].set_ylabel("source speed [m/s]")
    ax[2].plot(t, states["pos"][:, 0, 2], color=plots.VICTIM, lw=lw)
    ax[2].axhline(data.meta["config"]["drones"]["victim"]["hover_z"], color=plots.MUTED, lw=0.8)
    ax[2].set_ylabel("victim z [m]")
    for i in range(4):
        ax[3].plot(
            t, states["rotor_vel"][:, 0, i], color=plots.SERIES[i], lw=0.8, label=f"m{i + 1}"
        )
    ax[3].set_ylabel("victim rotors [rpm]")
    plots.legend(ax[3], loc="upper left", ncols=4)
    ax[4].plot(t, fz * 1000, color=plots.SERIES[0], lw=lw, label="force_imu")
    ax[4].plot(t, fz_fd * 1000, color=plots.SERIES[2], lw=lw, label="force_fd")
    if math.isfinite(peak["peak_fz"]):
        ax[4].plot(peak["peak_time"], peak["peak_fz"] * 1000, "o", color=plots.SERIES[0], ms=4)
    ax[4].set_ylabel("victim Fz [mN]")
    ax[4].set_xlabel("time from the leg start [s]; shaded: settled window, line: crossing")
    plots.legend(ax[4], loc="lower left", ncols=2)
    ax[0].set_title(
        f"Leg {int(row['leg'])}: Δz {row['height_sep']:.1f} m, v {row['speed']:.1f} m/s,"
        f" direction {int(row['direction']):+d}; Fz low-passed at {LABEL_CUTOFF:.0f} Hz",
        loc="left",
    )
    plots.save(fig, path)


def summary_plot(path: Path, rows: Sequence[Mapping]) -> Path:
    """Peak Fz against height separation and lag against the predicted arrival, per speed."""
    fig, axes = plots.figure(1, 2, width=11.0, row_height=3.6, sharex=False)
    ax_peak, ax_lag = axes[0]
    flown = [r for r in rows if r["status"] in ("ok", "hard_fail")]
    speeds = sorted({r["speed"] for r in flown})
    for color, speed in zip(plots.SERIES, speeds):
        legs = sorted((r for r in flown if r["speed"] == speed), key=lambda r: r["height_sep"])
        dz = np.array([r["height_sep"] for r in legs])
        peak = np.array([r["peak_fz"] for r in legs]) * 1000
        ax_peak.plot(dz, peak, "o", color=color, ms=4, label=f"{speed:.1f} m/s")
        means = [(d, np.nanmean(peak[dz == d])) for d in np.unique(dz)]
        ax_peak.plot(*zip(*means), color=color, lw=plots.LINE_WIDTH)
        arrival = np.array([r["arrival_s"] for r in legs])
        ax_lag.plot(
            arrival, [r["lag"] for r in legs], "o", color=color, ms=4, label=f"{speed:.1f} m/s"
        )
    if flown:
        arrivals = np.array([r["arrival_s"] for r in flown])
        line = np.array([arrivals.min(), arrivals.max()])
        ax_lag.plot(line, line, color=plots.MUTED, lw=0.8)
        ax_lag.fill_between(line, line - LAG_TOL, line + LAG_TOL, color=plots.SHADE, lw=0)
    ax_peak.set_xlabel("height separation Δz [m]")
    ax_peak.set_ylabel("peak Fz on the victim [mN]")
    ax_peak.set_title("Peak force, force_imu", loc="left")
    ax_lag.set_xlabel("predicted arrival T(Δz) [s]")
    ax_lag.set_ylabel("peak time - crossing [s]")
    ax_lag.set_title(f"Lag, band ±{LAG_TOL} s around the prediction", loc="left")
    plots.legend(ax_peak, loc="lower right")
    plots.legend(ax_lag, loc="upper left")
    return plots.save(fig, path)


def flight_checks(rows: Sequence[Mapping], sync: Mapping) -> dict:
    """The checks of §7 on the converted flight."""
    flown = [r for r in rows if r["status"] in ("ok", "hard_fail")]
    checks = {"sync": {}, "thrust": {}, "bias": {}, "physics": {}, "labels": {}, "coverage": {}}
    for role, entry in sync.items():
        fit = entry["fit"]
        checks["sync"][role] = {
            "residual_std_ok": fit["residual_std"] < SYNC_RESIDUAL_TOL,
            "drift_ok": abs(fit["drift"]) < DRIFT_TOL,
            "hop_lags_ok": entry["hop_lags_ok"],
        }
    for role in ("victim", "source"):
        ratios = [
            r[f"thrust_ratio_{role}"] for r in flown if math.isfinite(r[f"thrust_ratio_{role}"])
        ]
        if ratios:
            checks["thrust"][role] = {
                "mean": float(np.mean(ratios)),
                "ok": bool(all(abs(x - 1) <= THRUST_TOL for x in ratios)),
            }
    for axis in ("fx", "fy", "fz"):
        biases = [
            r[f"bias_{axis}_victim"] for r in flown if math.isfinite(r[f"bias_{axis}_victim"])
        ]
        if len(biases) > 1:
            checks["bias"][axis] = {
                "std": float(np.std(biases)),
                "ok": float(np.std(biases)) < BIAS_STD_TOL,
            }
    for speed in sorted({r["speed"] for r in flown}):
        means = {
            dz: float(
                np.nanmean(
                    [r["peak_fz"] for r in flown if r["speed"] == speed and r["height_sep"] == dz]
                )
            )
            for dz in sorted({r["height_sep"] for r in flown if r["speed"] == speed})
        }
        magnitudes = [abs(means[dz]) for dz in sorted(means)]
        checks["physics"][f"v{speed:.1f}"] = {
            "peak_by_height": means,
            "decreasing_with_height": bool(all(a >= b for a, b in zip(magnitudes, magnitudes[1:]))),
        }
    lags = [r["lag"] - r["arrival_s"] for r in flown if math.isfinite(r["lag"])]
    checks["physics"]["lag_within_band"] = bool(lags) and all(abs(x) <= LAG_TOL for x in lags)
    ratios = [r["peak_fz_fd"] / r["peak_fz"] for r in flown if r["peak_fz"]]
    checks["labels"]["agree_within_20_percent"] = bool(ratios) and all(
        abs(x - 1) <= LABEL_AGREEMENT for x in ratios if math.isfinite(x)
    )
    checks["coverage"]["in_wake_inside_settled"] = all(
        r["in_wake_outside_settled"] == 0 for r in flown
    )
    checks["coverage"]["settled_samples"] = {
        f"v{speed:.1f}": [r["settled_samples"] for r in flown if r["speed"] == speed]
        for speed in sorted({r["speed"] for r in flown})
    }
    return checks


def find_sd_file(folder: Path, role: str, name: str | None, meta: Mapping) -> Path:
    """The SD log of a role: a path, a name in `sd/<role>/`, or the file the console reported."""
    if name is None:
        files = meta["usd"][role]["files"]
        if not files:
            raise RuntimeError(f"The console did not report the {role}'s SD file, name it")
        name = Path(files[-1]).name
    if (path := Path(name)).is_file():
        return path
    path = folder / "sd" / role / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: copy the {role}'s SD file there")
    return path


def load_flight(
    folder: Path, bag: Path, sd_names: Mapping[str, str | None], lag_tol: float
) -> FlightData:
    """Read and synchronize everything of one flight, see the module docstring."""
    meta = json.loads((folder / "meta.json").read_text())
    if "timeline" not in meta:
        raise RuntimeError(f"{folder}: no legs were flown, the flight ended at takeoff")
    with open(folder / "records.pkl", "rb") as f:
        records = pickle.load(f)
    cfg = meta["config"]
    model = cfg["drones"]["model"]
    params, g = model_params(model), gravity(model)
    roles = ("source",) if meta["only_source"] else ROLES
    logs = {
        role: read_onboard(find_sd_file(folder, role, sd_names.get(role), meta)) for role in roles
    }
    mocap = read_mocap(bag, {role: f"cf{cfg['drones'][role]['id']}" for role in roles})
    flight_records = records["flight"]
    t_zero = float(np.median([r["t_wall"] - r["t_perf"] for r in flight_records]))
    fits, sync = {}, {}
    for role in roles:
        log = logs[role]
        prior = meta["usd"][role]["t_wall_start"] - log.t[0]
        fit = fit_clock(*setpoint_echo(records["takeoff"] + flight_records, role, log), prior)
        hops = [s for s in meta["timeline"][role] if s["kind"] == "hop"]
        windows = [(t_zero + s["t_start"] - HOP_PAD, t_zero + s["t_end"] + HOP_PAD) for s in hops]
        lags = [hop_lag(mocap[role], log, fit, window, g) for window in windows]
        fits[role] = fit
        sync[role] = {
            "fit": asdict(fit) | {"drift_ppm": fit.drift * 1e6},
            "hop_lags_ms": [lag_s * 1000 for lag_s in lags],
            "hop_lags_ok": bool(lags) and all(math.isfinite(x) and abs(x) <= lag_tol for x in lags),
            "sd_file": log.info,
        }
        logger.info(
            f"{role}: {fit.n_matches} echoes, residual {fit.residual_std * 1000:.2f} ms, drift "
            f"{fit.drift * 1e6:.1f} ppm, hop lags {', '.join(f'{x * 1000:.1f}' for x in lags)} ms"
        )
    (folder / "sync.json").write_text(json.dumps(sync, indent=2, default=_json_default))
    failed = [role for role in roles if not sync[role]["hop_lags_ok"]]
    if failed:
        raise RuntimeError(
            f"The IMU over the hops disagrees with the setpoint echo by more than {lag_tol * 1000} ms"
            f" for {failed}, see {folder / 'sync.json'}. Look at both before picking one; a "
            "constant lag is the motion capture latency"
        )
    table = meta["timeline"]["source"]
    flown = [s for s in table if s["leg"] >= 0]
    k_first = math.floor((min(s["t_start"] for s in flown) - PAD) * GRID_FREQ)
    k_last = math.ceil((max(s["t_end"] for s in flown) + PAD) * GRID_FREQ)
    t_grid = t_zero + np.arange(k_first, k_last + 1) / GRID_FREQ
    signals, forces, invalid_rpm = {}, {}, {}
    for role in ROLES:
        flew = role in roles
        signals[role] = drone_signals(
            t_grid,
            mocap.get(role),
            logs.get(role) if flew else None,
            fits.get(role),
            params,
            g,
            cfg["drones"][role]["thrust_scale"],
        )
        mass = meta["masses"][role]
        forces[role] = body_forces(signals[role], mass, params, g)
        if flew:
            rpm = logs[role].stack([f"motor.m{i}_rpm" for i in range(1, 5)])
            invalid_rpm[role] = (fits[role].to_pc(logs[role].t), np.isnan(rpm))
    return FlightData(
        folder,
        meta,
        flight_records,
        roles,
        t_zero,
        k_first,
        t_grid,
        signals,
        forces,
        invalid_rpm,
        sync,
        params,
        g,
    )


def convert_flight(
    folder: Path, bag: Path, sd_names: Mapping[str, str | None], lag_tol: float
) -> list[dict]:
    """Convert every flown leg of a flight and write the flight summary.

    Returns:
        The summary rows.
    """
    data = load_flight(folder, bag, sd_names, lag_tol)
    last_t = data.records[-1]["t_perf"] if data.records else -math.inf
    skipped = set(data.meta.get("skipped_legs", []))
    rows = []
    for row in sorted(data.meta["design_rows"], key=lambda r: int(r["index_in_flight"])):
        leg = int(row["leg"])
        base = {
            key: row[key]
            for key in (
                "leg",
                "flight",
                "index_in_flight",
                "cell",
                "speed",
                "height_sep",
                "direction",
                "repeat",
            )
        }
        if leg in skipped:
            rows.append(base | {"status": "skipped"})
            continue
        times = segment_leg_times(data.meta["timeline"]["source"], leg)
        if times["pause_end"] > last_t:
            rows.append(base | {"status": "not_flown"})
            continue
        rows.append(convert_leg(data, row))
        logger.info(f"Leg {leg}: {rows[-1]['status']}, peak Fz {rows[-1]['peak_fz'] * 1000:.1f} mN")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(folder / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    full = [r for r in rows if r["status"] in ("ok", "hard_fail")]
    summary_plot(folder / "summary.png", full)
    checks = flight_checks(full, data.sync)
    (folder / "checks.json").write_text(json.dumps(checks, indent=2, default=_json_default))
    logger.info(f"Checks: {json.dumps(checks, default=_json_default)}")
    return rows


def main():
    """Command line: convert one flight folder."""
    parser = argparse.ArgumentParser(description="Convert one A3 flight into per-leg arrays.")
    parser.add_argument("folder", type=Path, help="Flight folder written by deploy_a3.py")
    parser.add_argument("--bag", type=Path, required=True, help="rosbag of /tf of the flight")
    parser.add_argument(
        "--victim-sd", default=None, help="Victim SD file, path or name in sd/victim"
    )
    parser.add_argument(
        "--source-sd", default=None, help="Source SD file, path or name in sd/source"
    )
    parser.add_argument(
        "--sync-lag-tol",
        type=float,
        default=SYNC_LAG_TOL,
        help="s the IMU over the hops may disagree with the setpoint echo",
    )
    args = parser.parse_args()
    sd_names = {"victim": args.victim_sd, "source": args.source_sd}
    convert_flight(args.folder, args.bag, sd_names, args.sync_lag_tol)


if __name__ == "__main__":
    main()
