"""The checks of LEGS_FORMAT §8 on an exported dataset; exits non-zero on any failure.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/legs/validate.py \
        ../crazyflow_experiments_data/legs/legs_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash.legs.chain import chain
from crazyflow_experiments.downwash.thrust import rotor_thrust

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

GRID_STEP = 0.002
STEP_TOL = 1e-6
CHAIN_TOL = 1e-9
THRUST_TOL = 1e-9
BASELINE_TOL = 1e-12
QUAT_NORM_TOL = 1e-6
QUAT_W_MIN = 0.99
SHAPES = {  # LEGS_FORMAT §4, T first
    "t": (),
    "pos": (2, 3),
    "quat": (2, 4),
    "vel": (2, 3),
    "ang_vel": (2, 3),
    "rotor_vel": (2, 4),
    "motor_thrust": (2, 4),
    "thrust_sum": (),
    "fz_new": (),
    "fz_raw": (),
    "fz_total": (),
    "fz_wake": (),
}
NEVER_NAN = ("t", "pos", "quat", "vel")
COUNT_KEYS = ("date", "design_version", "block", "condition", "dz_nominal", "v_nominal")


def inside(t: NDArray, intervals: list, pad: float) -> NDArray:
    """Samples of `t` inside any of the intervals, widened by `pad` on both sides."""
    out = np.zeros(len(t), dtype=bool)
    for a, b in intervals:
        out |= (t >= a - pad) & (t <= b + pad)
    return out


def check_leg(folder: Path, meta: Mapping, states: Mapping) -> list[str]:
    """Checks 1 to 7 of LEGS_FORMAT §8 on one leg; the failures as messages."""
    errors = []
    t = states["t"]
    n = len(t)
    for name, tail in SHAPES.items():
        if name not in states:
            errors.append(f"missing array {name}")
        elif states[name].shape != (n, *tail):
            errors.append(f"{name} has shape {states[name].shape}, expected {(n, *tail)}")
    if "sim/freq" not in states or float(states["sim/freq"]) != 500:
        errors.append("sim/freq is not 500")
    if errors:
        return errors
    if np.abs(np.diff(t) - GRID_STEP).max() > STEP_TOL:
        errors.append("t is not uniform at 0.002 s")
    if states["fz_new"].dtype != bool:
        errors.append("fz_new is not boolean")
    q = meta["quality"]
    flags = set(q["flags"])
    # 2. NaN only where the contract allows it
    for name in NEVER_NAN:
        if not np.isfinite(states[name]).all():
            errors.append(f"NaN in {name}")
    gaps = inside(t, q["card_gaps"], GRID_STEP)
    for name in ("fz_raw", "fz_total"):
        if (~np.isfinite(states[name]) & ~gaps).any():
            errors.append(f"NaN in {name} outside the card gaps")
    tau = meta["rig"]["stand"]["dynamics"].get("tau_eff_s", 0.0)
    thrust_nan = ~np.isfinite(states["thrust_sum"])
    reach = int(math.ceil(tau / GRID_STEP)) + 1
    after_thrust_nan = np.zeros(n, dtype=bool)
    for k in np.flatnonzero(thrust_nan):
        after_thrust_nan[k : k + reach + 1] = True
    if (~np.isfinite(states["fz_wake"]) & ~gaps & ~after_thrust_nan).any():
        errors.append("NaN in fz_wake outside the card and rpm gaps")
    if (
        ~np.isfinite(states["motor_thrust"][:, 0]) != ~np.isfinite(states["rotor_vel"][:, 0])
    ).any():
        errors.append("victim motor_thrust and rotor_vel disagree on NaN")
    # 3. t = 0 at the crossing
    if "no_crossing" not in q["hard_fail_reason"]:
        d = np.array([meta["design"]["direction"], 0.0, 0.0])
        s = (states["pos"][:, 1] - states["pos"][:, 0]) @ d
        changes = np.flatnonzero(np.sign(s[1:]) != np.sign(s[:-1]))
        if len(changes) == 0:
            errors.append("the source never crosses the stand")
        else:
            k = changes[np.argmin(np.abs(t[changes]))]
            t_cross = t[k] - s[k] * (t[k + 1] - t[k]) / (s[k + 1] - s[k])
            if abs(t_cross) > GRID_STEP:
                errors.append(f"t = 0 is {t_cross * 1e3:.2f} ms off the crossing")
    # 4. Static victim pose
    pos_v, quat_v = states["pos"][:, 0], states["quat"][:, 0]
    if np.ptp(pos_v, axis=0).max() > 0 or np.ptp(quat_v, axis=0).max() > 0:
        errors.append("the victim's pose is not constant")
    if abs(np.linalg.norm(quat_v[0]) - 1) > QUAT_NORM_TOL or quat_v[0, 3] <= QUAT_W_MIN:
        errors.append(f"victim quaternion {quat_v[0]} not unit or w <= {QUAT_W_MIN}")
    # 5. Thrust from each drone's own curve
    curves = meta["sim_params"]["drone/rpm2thrust"]
    for i, name in enumerate(("victim", "source")):
        if name == "source" and "source_thrust_from_mocap" in flags:
            continue
        rpm, thrust = states["rotor_vel"][:, i], states["motor_thrust"][:, i]
        ok = np.isfinite(rpm)
        if ok.any() and np.abs(rotor_thrust(rpm[ok], curves[i]) - thrust[ok]).max() > THRUST_TOL:
            errors.append(f"{name} motor_thrust is not its rpm2thrust(rotor_vel)")
    if meta["design"]["condition"] == "S0" and np.abs(states["motor_thrust"][:, 0]).max() != 0:
        errors.append("victim motor_thrust is not 0 in S0")
    # 6. Baseline by construction, noise over the new samples
    label = states["fz_wake"] if meta["design"]["condition"] == "S1" else states["fz_total"]
    w0, w1 = meta["derived"]["window_pre"]
    pre = states["fz_new"] & (t >= w0) & (t < w1)
    finite = pre & np.isfinite(label)
    if finite.any() and abs(label[finite].mean()) > BASELINE_TOL:
        errors.append(f"label mean over window_pre is {label[finite].mean():.3e}, not 0")
    if int(pre.sum()) != q["n_fz_pre"]:
        errors.append("n_fz_pre is not the number of fz_new samples in window_pre")
    noise = float(np.std(states["fz_total"][pre]))
    if q["noise_fz_N"] is None or abs(noise - q["noise_fz_N"]) > 1e-12:
        errors.append("noise_fz_N is not the std over the fz_new samples in window_pre")
    # 7. S0 peaks are negative
    if (
        meta["design"]["condition"] == "S0"
        and "noise_floor" not in flags
        and not q["hard_fail"]
        and not (q["peak_fz_N"] is not None and q["peak_fz_N"] < 0)
    ):
        errors.append(f"S0 peak_fz_N {q['peak_fz_N']} is not negative")
    return errors


def check_chain_test(folder: Path) -> list[str]:
    """Check 8 on the producer's side: chain_test.npz reproduces with this chain."""
    errors = []
    data = np.load(folder / "chain_test.npz")
    keys = {k.split("/")[0] for k in data.files if "/" in k}
    for key in keys:
        stand = json.loads(str(data[f"{key}/stand"]))
        for kind in ("impulse", "rect"):
            diff = np.abs(chain(data[kind], stand) - data[f"{key}/{kind}"])
            if np.nanmax(diff) > CHAIN_TOL:
                errors.append(f"chain_test {key}/{kind} differs by {np.nanmax(diff):.2e}")
    return errors


def validate(folder: Path) -> dict[str, list[str]]:
    """All checks; failures per leg_id, `dataset` for the dataset-wide ones."""
    folder = Path(folder)
    failures: dict[str, list[str]] = {}
    for name in ("dataset.json", "manifest.csv", "chain_test.npz"):
        if not (folder / name).is_file():
            failures.setdefault("dataset", []).append(f"missing {name}")
    if failures:
        return failures
    dataset = json.loads((folder / "dataset.json").read_text())
    with open(folder / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["leg_id"] for r in rows]
    dataset_errors = []
    if len(set(ids)) != len(ids):
        dataset_errors.append("leg_ids are not unique")
    folders = {p.name for p in folder.iterdir() if p.is_dir()}
    if folders != {r["path"] for r in rows}:
        dataset_errors.append(
            f"manifest and folders differ: {sorted(folders ^ {r['path'] for r in rows})}"
        )
    expected = Counter(tuple(str(r[k]) for k in COUNT_KEYS) for r in rows)
    listed = Counter({tuple(str(c[k]) for k in COUNT_KEYS): c["total"] for c in dataset["counts"]})
    if expected != listed:
        dataset_errors.append("counts do not match the manifest")
    dataset_errors += check_chain_test(folder)
    if dataset_errors:
        failures["dataset"] = dataset_errors
    for row in rows:
        leg_dir = folder / row["path"]
        if not (leg_dir / "meta.json").is_file() or not (leg_dir / "states.npz").is_file():
            failures[row["leg_id"]] = ["missing meta.json or states.npz"]
            continue
        meta = json.loads((leg_dir / "meta.json").read_text())
        states = dict(np.load(leg_dir / "states.npz"))
        if errors := check_leg(leg_dir, meta, states):
            failures[row["leg_id"]] = errors
    return failures


def main():
    """Command line: validate a dataset folder, print the failures, exit 1 if there are any."""
    parser = argparse.ArgumentParser(description="Validate a LEGS_FORMAT dataset (§8).")
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    failures = validate(args.folder)
    for key, errors in failures.items():
        for error in errors:
            print(f"{key}: {error}")  # noqa: T201
    print(f"{'FAIL' if failures else 'OK'}: {args.folder}")  # noqa: T201
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
