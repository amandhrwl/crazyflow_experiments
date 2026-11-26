"""A fake stand run for the legs export: the arrays `load_session` returns, built in closed form.

The source passes the stand at 1 m/s, `cross_offset` after the schedule's nominal crossing. The
victim's load cell reads its rotor thrust 27 ms late plus a Gaussian wake pulse, sampled at the
card's 500 Hz (every record a new value) or held at 80 Hz, with a little white noise.
"""

from __future__ import annotations

import numpy as np

from crazyflow_experiments.downwash.legs import G
from crazyflow_experiments.downwash.thrust import rotor_thrust

T_ZERO = 1000.0  # s, PC time of the run's timeline zero
CARD_DT = 0.002  # s
STAND_POS = np.array([0.5, 0.02, 1.52])
SPEED = 1.0  # m/s
DZ = 0.3  # m, source above the stand
Y_OFF = 0.01  # m
CROSS_OFFSET = 0.013  # s, measured crossing after the nominal one
CELL_DELAY = 0.027  # s
WAKE_PEAK = -0.1  # N
WAKE_AT = 0.1  # s after the crossing
WAKE_WIDTH = 0.05  # s
HOVER_RPM = 17000.0
BUMP_RPM = 400.0  # rpm the victim rises by while the wake acts
A1_COEF = [0.0, -6.064710107326551e-07, 4.376601568313169e-10]
SOURCE_COEF = [0.0, -2.88e-07, 4.05e-10]
CAL_A = 1e-4  # g per raw count
NOISE_N = 2e-5  # N, white noise of the cell, so that every conversion reads a new value
SCHEDULE = {
    "leg": 8,
    "tare_start": 1.0,
    "arm_at": 3.0,
    "spin_on": 4.5,
    "baseline_start": 6.0,
    "pass_start": 8.0,
    "cross": 9.9,
    "pass_end": 11.8,
    "spin_off": 13.3,
    "leg_end": 16.3,
}
ROW = {
    "leg": 8,
    "flight": 2,
    "index_in_flight": 0,
    "cell": 9,
    "speed": SPEED,
    "height_sep": 0.25,
    "lateral_offset": 0.0,
    "direction": 1,
    "repeat": 0,
}
STAND = {
    "mount_sign": 1,
    "sample_rate_hz": 80,
    "dynamics": {"kind": "delay", "tau_eff_s": CELL_DELAY, "gain_ratio": 1.0, "source": "fake"},
    "processing": {"shift_s": 0.0, "filter": "none"},
}


def cross_time() -> float:
    """PC time of the fake run's true crossing."""
    return T_ZERO + SCHEDULE["cross"] + CROSS_OFFSET


def wake(t: np.ndarray) -> np.ndarray:
    """The wake force on the victim in N at PC times `t`."""
    return WAKE_PEAK * np.exp(-(((t - cross_time() - WAKE_AT) / WAKE_WIDTH) ** 2))


def rpm(t: np.ndarray, spin: bool) -> np.ndarray:
    """The victim's rotor rpm (N, 4): a 0.1 s ramp at spin-on, a bump during the wake."""
    if not spin:
        return np.zeros((len(t), 4))
    on, off = T_ZERO + SCHEDULE["spin_on"], T_ZERO + SCHEDULE["spin_off"]
    level = HOVER_RPM * np.clip((t - on) / 0.1, 0, 1) * (t < off)
    bump = BUMP_RPM * np.exp(-(((t - cross_time() - WAKE_AT) / 0.08) ** 2)) * (t < off)
    return np.repeat((level + bump)[:, None], 4, axis=1)


def fake_run(
    mode: str = "spin",
    card_gap: tuple[float, float] | None = None,
    held: bool = False,
    run_id: str = "stand_flight2_spin_2026-11-22_20-24-13",
    spun: bool = True,
) -> tuple[dict, dict, dict, dict]:
    """A fake run, its design row, schedule entry and outcome, see the module docstring.

    Args:
        mode: "spin" or "still".
        card_gap: Card records removed over this interval, in s after the true crossing.
        held: Hold the load cell at 80 Hz instead of a new value on every record.
        run_id: Folder name of the run.
        spun: Whether the victim's motors ran (S1).
    """
    spin = mode == "spin" and spun
    t = T_ZERO + np.arange(0.0, 20.0, CARD_DT)
    if card_gap is not None:
        keep = ~((t > cross_time() + card_gap[0]) & (t < cross_time() + card_gap[1]))
        t = t[keep]
    rotor = rpm(t, spin)
    thrust = rotor_thrust(rotor, A1_COEF).sum(axis=1)
    thrust_late = np.interp(t - CELL_DELAY, t, thrust)
    force = thrust_late + wake(t) + np.random.default_rng(0).normal(0.0, NOISE_N, len(t))
    if held:  # The cell converts at 80 Hz, the card holds the last value
        t_conv = np.floor((t - T_ZERO) * 80) / 80 + T_ZERO
        force = np.interp(t_conv, t, force)
    weight_g = force / (1e-3 * G)
    t_rec = T_ZERO + np.arange(0.0, 20.0, 0.01)
    x = STAND_POS[0] + SPEED * (t_rec - cross_time())
    src_pos = np.stack(
        [x, np.full_like(x, STAND_POS[1] + Y_OFF), np.full_like(x, STAND_POS[2] + DZ)], axis=1
    )
    src_vel = np.tile([SPEED, 0.0, 0.0], (len(t_rec), 1))
    records = {
        "takeoff": [],
        "flight": [{"t_wall": tt, "source": {"quat": [0.0, 0.0, 0.0, 1.0]}} for tt in t_rec],
    }
    events = {
        "tare_start": T_ZERO + SCHEDULE["tare_start"],
        "tare_end": T_ZERO + SCHEDULE["tare_start"] + 2.0,
        "spin_on": T_ZERO + SCHEDULE["spin_on"],
        "spin_off": T_ZERO + SCHEDULE["spin_off"],
    }
    meta = {
        "mode": mode,
        "folder": f"/data/{run_id}",
        "config": {
            "stand": {"spin_settle_s": 1.5},
            "a3": {"height_seps": [0.25, 0.5], "speeds": [0.5, 1.0, 1.5, 2.0]},
        },
        "stand_pos": STAND_POS.tolist(),
        "load_cell": {"a": CAL_A, "b": 0.0, "capacity_g": 100},
        "abort_reason": None,
    }
    run = {
        "meta": meta,
        "sync": {
            "latency_median": 0.0013,
            "latency_p95": 0.003,
            "sd_offset_halves_ms": [-65.73, -65.75],
            "match_fraction": 0.95,
        },
        "t": t,
        "weight_g": weight_g,
        "t_rpm": t,
        "rpm": rotor,
        "thrust_rpm": thrust,
        "t_zero": T_ZERO,
        "src_t": t_rec,
        "src_pos": src_pos,
        "src_vel": src_vel,
        "card": {
            "loadcell.rawWeight": np.round(weight_g / CAL_A).astype(np.int64),
            "gyro.x": np.zeros(len(t)),
            "gyro.y": np.zeros(len(t)),
            "gyro.z": np.zeros(len(t)),
        },
        "records": records,
        "source_card": None,
        "victim_quat": np.tile([0.001, -0.002, 0.0, 1.0], (len(t), 1)),
    }
    outcome = {"spun": spun if mode == "spin" else False, "events": events}
    return run, dict(ROW), dict(SCHEDULE), outcome


def fake_ctx(stand: dict | None = None) -> dict:
    """The dataset context `build_leg` takes, with the fake curves."""
    return {
        "stand": STAND if stand is None else stand,
        "curves": {
            "victim": {"coeffs": A1_COEF, "source": "fake A1", "sha256": "", "mass_kg": 0.05425},
            "source": {"coeffs": SOURCE_COEF, "source": "fake", "sha256": "", "mass_kg": 0.04525},
        },
        "params": {
            "mixing_matrix": [
                [-1.0, -1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0, 1.0],
            ]
        },
        "g": G,
        "created": "2026-11-26T00:00:00+01:00",
        "commit": "0" * 40,
    }


def header(stand: dict | None = None) -> dict:
    """`dataset.json` without counts for a fake dataset."""
    return {
        "format": "LEGS_FORMAT",
        "format_version": 1,
        "version": "legs_test",
        "created": "2026-11-26T00:00:00+01:00",
        "producer": {
            "code_repo": "crazyflow_experiments",
            "code_commit": "0" * 40,
            "code_dirty": True,
        },
        "synthetic": True,
        "truth": None,
        "runs": [],
        "calibrations": {"stand": {"rig_stand": STAND if stand is None else stand}},
    }
