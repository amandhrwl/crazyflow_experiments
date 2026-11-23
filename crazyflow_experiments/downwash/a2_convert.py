"""Convert one A2 free flight into the static map: the wake's force and torque per point.

The inputs are the flight folder of `a2_deploy.py --condition free` (`meta.json`, `records.pkl`),
the SD logs of both drones in its `sd/victim/` and `sd/source/` (`usd.py collect <card> --block a2`)
and the rosbag of `/tf` of the flight. The clocks are mapped onto PC time and the signals put on a
500 Hz grid by `a3_convert.load_flight`, which also writes `sync.json` and stops if the IMU and the
setpoint echo disagree.

Each dwell of the design gives one row of `items.csv`, over its used seconds, after the discarded
`a2.discard_s` until its end:

- the geometry measured by the motion capture, the source minus the victim, and the victim's drop
  and tilt;
- `force_imu` and `force_fd`, the body-frame forces of `a3_convert.body_forces`;
- the **steady labels**. With the source still the victim hovers: no acceleration on average, so the
  wake force and torque are what its own rotors balance. With the rotor thrusts `f_i` from the
  measured rpm, mass `m` and the onboard attitude `R` the wake force on the victim in its body frame
  is `R^T (0, 0, m g) - (0, 0, sum f_i)` and the torques are `-L (-f1 - f2 + f3 + f4)` about x and
  `-L (-f1 + f2 + f3 - f4)` about y, crazyflow's mixing matrix (the rows of `drone/mixing_matrix`).
  **[VERIFY]** that the real motors are numbered like crazyflow's; a hand tilt of the hovering
  drone shows the sign of each torque.

Every label is also given relative to the baseline of the item's part, the mean over the used
seconds of the part's first reference, where the victim is outside the jet: its own trim (tilt and
rotor imbalance) drops out, `*_rel`. `steady_minus_imu_*` is the steady label less the accelerometer's
`force_imu`, both relative: the check of the IMU label on every axis. A part that ends on a reference
gives the drift of the baseline, `ref_drift_*`.

An item is `ok`, `gap` (mocap, card or rpm samples missing in its used seconds), `no_baseline` (the
part's first reference is not ok) or `not_flown`. A flight with the source alone (`only_source`)
has the source's geometry only.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/a2_convert.py <flight folder> \
        --bag results/downwash/a2/bags/<name> [--victim-sd FILE] [--source-sd FILE]
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
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.downwash.a3_convert import SYNC_LAG_TOL, load_flight
from crazyflow_experiments.downwash.timeline import victim_hover_point

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from crazyflow_experiments.downwash.a3_convert import FlightData

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("crazyflow_experiments.downwash.a2_convert")  # Not __name__
logger.setLevel(logging.INFO)

MIN_FINITE = 0.95  # Fraction of the used seconds' grid samples that need a value
ITEM_KEYS = ("leg", "part", "kind", "height_sep", "dx")
VICTIM_WARN_DROP = 0.10  # m the victim sinks below its hover point that is flagged


def steady_labels(
    quat: NDArray, thrust: NDArray, mass: float, gravity: float, arm: float, mixing: NDArray
) -> tuple[NDArray, NDArray]:
    """The wake force and torque a hovering victim's rotors balance, per sample, in its body frame.

    Args:
        quat: The victim's attitude as xyzw, (N, 4).
        thrust: Thrust of each rotor from its rpm in N, (N, 4).
        mass: The victim's mass in kg.
        gravity: Magnitude of gravity in m/s^2.
        arm: Crazyflow's `L`, the lever arm of the mixing matrix in m.
        mixing: Crazyflow's mixing matrix (3, 4): the rows of x, y and z torque per rotor force.

    Returns:
        The force (N, 3) and the torque about x and y (N, 2): zero for a level, unloaded hover.
    """
    weight_body = R.from_quat(quat).inv().apply(np.array([0.0, 0.0, mass * gravity]))
    force = weight_body - np.stack(
        [np.zeros(len(thrust)), np.zeros(len(thrust)), thrust.sum(axis=1)], axis=1
    )
    torque = -(thrust * arm) @ np.asarray(mixing, dtype=float)[:2].T
    return force, torque


def window_mean(x: NDArray, mask: NDArray) -> tuple[NDArray, float]:
    """Mean of `x` (N, ...) over the samples of `mask`, skipping NaN, and the finite fraction."""
    values = np.asarray(x, dtype=float)[mask]
    if len(values) == 0:
        return np.full(np.shape(x)[1:], np.nan), 0.0
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(finite, values, 0.0).sum(axis=0) / count
    return mean, float(finite.all(axis=tuple(range(1, values.ndim))).mean())


def _vec(prefix: str, value: NDArray, axes: str = "xyz") -> dict[str, float]:
    return {
        f"{prefix}_f{a}_N" if len(axes) == 3 else f"{prefix}_t{a}_Nm": float(v)
        for a, v in zip(axes, value)
    }


def item_rows(data: FlightData, items: Sequence[Mapping], last_t: float) -> list[dict]:
    """The rows of a flight's items, see the module docstring.

    Args:
        data: The flight on its grid, see `a3_convert.load_flight`.
        items: The flight's design rows, times in timeline seconds.
        last_t: Timeline time of the last record, items that end after it were not flown.

    Returns:
        One row per item in flight order.
    """
    cfg = data.meta["config"]
    hover = victim_hover_point(cfg)
    has_victim = "victim" in data.roles
    mass = data.meta["masses"]["victim"]
    arm = float(np.asarray(data.params["L"]))
    mixing = np.asarray(data.params["mixing_matrix"], dtype=float)
    src, vic = data.signals["source"], data.signals["victim"]
    if has_victim:
        steady_f, steady_t = steady_labels(
            vic["quat_est"], vic["motor_thrust"], mass, data.gravity, arm, mixing
        )
        imu, fd = data.forces["victim"]
        thrust_sum = vic["motor_thrust"].sum(axis=1)
        euler = np.rad2deg(R.from_quat(vic["quat_est"]).as_euler("xyz"))
    measured = {}
    for item in items:
        t0 = data.t_zero + float(item["t_use_start"])
        t1 = data.t_zero + float(item["t_end"])
        mask = (data.t_grid >= t0) & (data.t_grid <= t1)
        p_src, ok_src = window_mean(src["pos"], mask)
        entry = {"p_src": p_src, "ok": ok_src}
        if has_victim:
            p_vic, ok_vic = window_mean(vic["pos"], mask)
            columns = {
                "steady_f": steady_f, "steady_t": steady_t, "imu": imu, "fd": fd,
                "thrust": thrust_sum, "euler": euler,
            }  # fmt: skip
            means, finite = {}, [ok_src, ok_vic]
            for name, array in columns.items():
                means[name], ok = window_mean(array, mask)
                finite.append(ok)
            entry |= {"p_vic": p_vic, "means": means, "ok": min(finite)}
        measured[int(item["leg"])] = entry
    rows = []
    parts: dict[int, list[Mapping]] = {}
    for item in items:
        parts.setdefault(int(item["part"]), []).append(item)
    for part_items in parts.values():
        first = measured[int(part_items[0]["leg"])]
        base_ok = first["ok"] >= MIN_FINITE
        for item in part_items:
            m = measured[int(item["leg"])]
            status = "ok"
            if float(item["t_end"]) > last_t:
                status = "not_flown"
            elif m["ok"] < MIN_FINITE:
                status = "gap"
            elif not base_ok:
                status = "no_baseline"
            row = {key: item[key] for key in ITEM_KEYS} | {"mode": "free", "status": status}
            nominal = np.array([float(item["dx"]), 0.0, float(item["height_sep"])])
            if has_victim:
                rel = m["p_src"] - m["p_vic"]
                row |= {
                    "dx_meas": rel[0], "dy_meas": rel[1], "dz_meas": rel[2],
                    "dx_err": rel[0] - nominal[0], "dz_err": rel[2] - nominal[2],
                    "victim_dz": m["p_vic"][2] - hover[2],
                    "victim_dxy": float(np.linalg.norm(m["p_vic"][:2] - hover[:2])),
                    "roll_deg": m["means"]["euler"][0], "pitch_deg": m["means"]["euler"][1],
                    "thrust_sum_N": float(m["means"]["thrust"]),
                    "weight_ratio": float(m["means"]["thrust"]) / (mass * data.gravity),
                }  # fmt: skip
                for name, key, axes in (
                    ("steady", "steady_f", "xyz"),
                    ("steady", "steady_t", "xy"),
                    ("imu", "imu", "xyz"),
                    ("fd", "fd", "xyz"),
                ):
                    value, base = m["means"][key], first["means"][key]
                    row |= _vec(name, value, axes) | _vec(f"{name}_rel", value - base, axes)
                diff = (m["means"]["steady_f"] - first["means"]["steady_f"]) - (
                    m["means"]["imu"] - first["means"]["imu"]
                )
                row |= _vec("steady_minus_imu", diff)
                if item["kind"] == "reference" and item is not part_items[0]:
                    row |= _vec(
                        "ref_drift_steady", m["means"]["steady_f"] - first["means"]["steady_f"]
                    )
                    row |= _vec("ref_drift_imu", m["means"]["imu"] - first["means"]["imu"])
                row["victim_sinks"] = bool(-row["victim_dz"] > VICTIM_WARN_DROP)
            else:
                row |= {
                    "dx_meas": m["p_src"][0] - hover[0],
                    "dy_meas": m["p_src"][1] - hover[1],
                    "dz_meas": m["p_src"][2] - hover[2],
                    "dx_err": m["p_src"][0] - hover[0] - nominal[0],
                    "dz_err": m["p_src"][2] - hover[2] - nominal[2],
                }
            rows.append(row)
    return rows


def convert_flight(
    folder: Path, bag: Path, sd_names: Mapping[str, str | None], lag_tol: float
) -> list[dict]:
    """Convert a free flight: items.csv and checks.json; returns the rows."""
    folder = Path(folder)
    data = load_flight(folder, bag, sd_names, lag_tol)
    items = sorted(data.meta["design_rows"], key=lambda r: int(r["index_in_flight"]))
    last_t = data.records[-1]["t_perf"] if data.records else -math.inf
    rows = item_rows(data, items, last_t)
    for row in rows:
        row["run"] = folder.name
    write_rows(folder / "items.csv", rows)
    checks = flight_checks(rows, data.sync)
    (folder / "checks.json").write_text(json.dumps(checks, indent=2, default=float))
    counts = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
    logger.info(
        f"{folder.name}: {len(rows)} items, {counts}; checks {json.dumps(checks, default=float)}"
    )
    return rows


def flight_checks(rows: Sequence[Mapping], sync: Mapping) -> dict:
    """What the first look at a flight needs: sync, the references' drift, the victim's lowest point."""
    checks: dict = {
        "sync": {
            role: {
                "residual_std_s": entry["fit"]["residual_std"],
                "hop_lags_ok": entry["hop_lags_ok"],
            }
            for role, entry in sync.items()
        }
    }
    ok = [r for r in rows if r["status"] == "ok"]
    for kind in ("steady", "imu"):
        drifts = [
            [r[f"ref_drift_{kind}_f{a}_N"] for a in "xyz"]
            for r in ok
            if f"ref_drift_{kind}_fx_N" in r
        ]
        drifts = [d for d in drifts if all(math.isfinite(v) for v in d)]
        if drifts:
            checks[f"ref_drift_{kind}_max_N"] = np.abs(np.array(drifts)).max(axis=0).tolist()
    drops = [r["victim_dz"] for r in rows if "victim_dz" in r and math.isfinite(r["victim_dz"])]
    if drops:
        checks["victim_min_dz_m"] = float(min(drops))
    return checks


def write_rows(path: Path, rows: Sequence[Mapping]):
    """Write rows with the union of their columns."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Command line: convert one flight folder."""
    parser = argparse.ArgumentParser(description="Convert one A2 free flight into the static map.")
    parser.add_argument("folder", type=Path, help="Flight folder written by a2_deploy.py")
    parser.add_argument("--bag", type=Path, required=True, help="rosbag of /tf of the flight")
    parser.add_argument(
        "--victim-sd", default=None, help="Victim SD file, path or name in sd/victim"
    )
    parser.add_argument(
        "--source-sd", default=None, help="Source SD file, path or name in sd/source"
    )
    parser.add_argument(
        "--sync-lag-tol", type=float, default=SYNC_LAG_TOL, help="s, IMU against echo"
    )
    args = parser.parse_args()
    sd_names = {"victim": args.victim_sd, "source": args.source_sd}
    convert_flight(args.folder, args.bag, sd_names, args.sync_lag_tol)


if __name__ == "__main__":
    main()
