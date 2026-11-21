"""An A1 block faked from known clocks and thrust curve, to check that convert recovers them.

The victim's card logs the load cell and the motors at 500 Hz, its radio streams the load cell at
100 Hz, as in case S. The rotors follow each command with a first-order lag, the thrust is
crazyflow's curve times `SCALE`, the load cell converts at 80 Hz behind a short low-pass and its
zero drifts slowly over the block. The clocks come from the load cell test's fakes.
"""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash.a1_stand import load_config
from tests.load_cell import fakes

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

CF_COEF = np.array([0.0, -3.133427287299859e-7, 4.407354891648379e-10])
SCALE = 0.91  # The true thrust over crazyflow's
MOTOR_SPREAD = np.array([1.0, 1.01, 0.99, 1.005])  # Rpm of each motor relative to the mean
ROTOR_TAU = 0.05  # s, spin-up lag of the rotors
CELL_TAU = 0.02  # s, low-pass of the load cell
CELL_NOISE = 0.012  # g
TARE_DRIFT = 0.05  # g over the block
MASS = 0.050  # kg
G = 9.81
DISCARD = 1.0  # s
# Kind, duration and command of each step; tares have no command
STEPS = [
    ("tare", 4.0, None),
    ("idle", 1.0, 7000),
    ("pulse", 0.3, 26000),
    ("idle", 1.0, 7000),
    ("ramp", 3.0, 12000),
    ("ramp", 3.0, 16000),
    ("ramp", 3.0, 20000),
    ("ramp", 3.0, 24000),
    ("level", 3.0, 20000),
    ("level", 3.0, 12000),
    ("center", 3.0, 16000),
    ("level", 3.0, 24000),
    ("tare", 4.0, None),
    ("idle", 1.0, 7000),
    ("pulse", 0.3, 26000),
    ("idle", 1.0, 7000),
    ("tare", 4.0, None),
]


def rpm_of(command: NDArray) -> NDArray:
    """Steady mean rotor speed at a command, 0 with the motors off."""
    command = np.asarray(command, dtype=float)
    return np.where(command > 0, 4800 + 0.35 * command, 0.0)


def lag(t: NDArray, target: NDArray, tau: float) -> NDArray:
    """A first-order lag of `target`, sampled at the times `t`."""
    out = np.empty_like(target)
    out[0] = target[0]
    alpha = 1 - np.exp(-np.diff(t, prepend=t[0]) / tau)
    for i in range(1, len(t)):
        out[i] = out[i - 1] + alpha[i] * (target[i] - out[i - 1])
    return out


def events(t0: float) -> list[dict]:
    """The steps on PC time from `t0`, as a1_sweep writes them."""
    rows, t = [], t0
    for i, (kind, duration, command) in enumerate(STEPS):
        commands = (0,) * 4 if command is None else (command,) * 4
        rows.append(
            {"step": i, "kind": kind, "level_id": -1, "motor": 0, "repeat": 0}
            | {f"m{k + 1}": c for k, c in enumerate(commands)}
            | {"duration": duration, "t_start": t, "t_end": t + duration}
            | {"predicted_N": 0.0, "measured_N": np.nan, "skipped": ""}
        )
        t += duration
    return rows


def command_at(rows: list[dict], t_pc: NDArray) -> NDArray:
    """The command of motor 1 at PC times."""
    starts = np.array([r["t_start"] for r in rows])
    commands = np.array([r["m1"] for r in rows])
    i = np.clip(np.searchsorted(starts, t_pc, side="right") - 1, 0, len(rows) - 1)
    inside = (t_pc >= starts[0]) & (t_pc < rows[-1]["t_end"])
    return np.where(inside, commands[i], 0)


def write_block(folder: Path, clock: fakes.Clock = fakes.VICTIM_CLOCK, seed: int = 0) -> list[dict]:
    """Write events.csv, meta.json, radio.npz and the card file of a faked block all.

    Returns:
        The events.
    """
    rng = np.random.default_rng(seed)
    t0 = fakes.PC_BOOT + 30.0
    rows = events(t0 + 1.0)

    # The truth on a 1 kHz grid of PC time: rotor speeds, thrust, load cell weight
    t = t0 + np.arange(0.0, fakes.DURATION, 1e-3)
    rpm = lag(t, rpm_of(command_at(rows, t)), ROTOR_TAU)
    thrust = SCALE * sum(
        np.where(rpm > 0, CF_COEF[1] * (s * rpm) + CF_COEF[2] * (s * rpm) ** 2, 0.0)
        for s in MOTOR_SPREAD
    )
    drift = TARE_DRIFT * (t - t[0]) / fakes.DURATION
    weight = lag(t, thrust / G * 1e3, CELL_TAU) + drift

    # The load cell converts at 80 Hz on its own oscillator
    tick0 = clock.tick(t0)
    changes = tick0 - 1.0 + np.cumsum(rng.uniform(12.4e-3, 12.6e-3, int(62.0 / 12.5e-3)))
    w_at = np.interp(clock.pc(changes), t, weight) + rng.normal(0, CELL_NOISE, len(changes))
    raw = np.round((w_at - fakes.B) / fakes.A).astype(np.int32)
    sd_ticks, v_sd, stamps, v_radio, arrival = fakes.drone_logs(clock, rng, changes, raw, t0)
    t_sd_pc = clock.pc(sd_ticks)
    rpm_sd = np.interp(t_sd_pc, t, rpm)
    motor_cmd = command_at(rows, t_sd_pc).astype(np.uint16)
    columns = [v_sd]
    variables = [("loadcell.rawWeight", "i")]
    for k, s in enumerate(MOTOR_SPREAD, start=1):
        noisy = s * rpm_sd * (1 + rng.normal(0, 0.003, len(rpm_sd)))
        columns.append(np.round(noisy).astype(np.uint16))
        variables.append((f"motor.m{k}_rpm", "H"))
    for k in range(1, 5):
        columns.append(motor_cmd)
        variables.append((f"motor.m{k}", "H"))
    columns.append(np.full(len(rpm_sd), 4000, dtype=np.uint16))
    variables.append(("pm.vbatMV", "H"))
    (folder / "sd" / "victim").mkdir(parents=True)
    (folder / "sd" / "victim" / "a100").write_bytes(
        fakes.usd_file(variables, clock.sd_us(sd_ticks), columns)
    )
    np.savez(
        folder / "radio.npz",
        **{
            "victim/loadcell/t_pc": arrival,
            "victim/loadcell/t_tick_ms": stamps,
            "victim/loadcell/loadcell.rawWeight": v_radio,
            "victim/loadcell/loadcell.weight": fakes.A * v_radio + fakes.B,
        },
    )
    config = load_config()
    config["load_cell"] |= {"a": fakes.A, "b": fakes.B}
    config["sweep"] |= {"discard_s": DISCARD}
    config["victim"] |= {"mass_kg": MASS}
    meta = {
        "config": config,
        "case": "S",
        "block": "all",
        "mass_kg": MASS,
        "crazyflow_rpm2thrust": CF_COEF.tolist(),
        "sd": {"victim": {"files": ["a100"]}},
    }
    (folder / "meta.json").write_text(json.dumps(meta))
    with open(folder / "events.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
