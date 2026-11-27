"""The source's card on the stand runs, and its thrust from the motion capture (export Step 4).

The stand conversion never decodes the source's card. Its clock is mapped onto PC time the way
`a3_convert` maps it in free flight: the card logs the setpoint the PC sent (`ctrltarget.x`,
`ctrltarget.z`), so `sync.fit_clock` on the setpoint echo gives offset and drift, with the
residual of the matched echoes as the check. The hop check of `a3_convert` needs a bag, which the
stand runs do not have.

Without a card, the total thrust comes from the motion capture instead,
`T = m |a + g z|` with `a` the twice differentiated position, low-passed at 10 Hz zero phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash import usd
from crazyflow_experiments.downwash.a3_convert import (
    MOCAP_MAX_GAP,
    ONBOARD_MAX_GAP,
    RPM_MAX_GAP,
    VEL_CUTOFF,
    fill_dropouts,
    lowpass,
    read_onboard,
    resample,
    setpoint_echo,
)
from crazyflow_experiments.downwash.a3_stand_convert import despike_rpm
from crazyflow_experiments.downwash.sync import fit_clock
from crazyflow_experiments.downwash.thrust import rotor_thrust

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from crazyflow_experiments.downwash.a3_convert import OnboardLog
    from crazyflow_experiments.downwash.sync import ClockFit

N_MOTORS = 4
LOG_HEADER = 0xBC  # First byte of a uSD log file, see `usd._parse`


def has_card(folder: Path, meta: Mapping) -> bool:
    """Whether the run holds a readable source card file: present and with the log header.

    An empty file (flight 2 still, 22 Nov 20:09) counts as missing; the export then takes the
    thrust from the motion capture.
    """
    directory = folder / "sd" / "source"
    names = usd.cards(meta).get("source", {}).get("files", [])
    if names:
        path = directory / names[-1]
    else:
        found = sorted(p for p in directory.iterdir() if p.is_file()) if directory.is_dir() else []
        path = found[-1] if found else directory / "missing"
    if not path.is_file():
        return False
    with open(path, "rb") as f:
        head = f.read(5)
    return len(head) == 5 and head[0] == LOG_HEADER


def load_source_card(folder: Path, meta: Mapping, records: Mapping) -> tuple[OnboardLog, ClockFit]:
    """Decode the source's card and map its clock onto PC time through the setpoint echo.

    Args:
        folder: The run folder.
        meta: Its `meta.json`.
        records: Its `records.pkl` (`takeoff` and `flight`).

    Returns:
        The decoded log and its clock fit.
    """
    log = read_onboard(usd.sd_file(folder, "source", meta))
    prior = usd.cards(meta)["source"]["t_wall_start"] - log.t[0]
    fit = fit_clock(*setpoint_echo(records["takeoff"] + records["flight"], "source", log), prior)
    return log, fit


def card_signals(
    log: OnboardLog, fit: ClockFit, t_grid: NDArray, coef: NDArray
) -> dict[str, NDArray]:
    """The source's rotor rpm, rotor thrust and body rates on a PC-time grid.

    Telemetry glitches (`despike_rpm`) count as dropouts. rpm dropouts of up to `RPM_MAX_GAP`
    card records are interpolated (`rpm_filled` counts them per
    motor on the grid); longer ones and card stalls over `ONBOARD_MAX_GAP` stay NaN.
    """
    t_pc = fit.to_pc(log.t)
    raw, spikes = despike_rpm(log.stack([f"motor.m{i}_rpm" for i in range(1, N_MOTORS + 1)]))
    in_grid = (t_pc >= t_grid[0]) & (t_pc <= t_grid[-1])
    filled = np.stack([fill_dropouts(raw[:, i], RPM_MAX_GAP) for i in range(N_MOTORS)], axis=1)
    rotor_vel = resample(t_pc, filled, t_grid, ONBOARD_MAX_GAP)
    was_nan = resample(t_pc, (~np.isfinite(raw)).astype(float), t_grid, ONBOARD_MAX_GAP)
    rpm_filled = (np.nan_to_num(was_nan) > 0) & np.isfinite(rotor_vel)
    gyro = log.stack(["gyro.x", "gyro.y", "gyro.z"])
    return {
        "rotor_vel": rotor_vel,
        "motor_thrust": rotor_thrust(rotor_vel, coef),
        "ang_vel": np.deg2rad(resample(t_pc, gyro, t_grid, ONBOARD_MAX_GAP)),
        "rpm_filled": rpm_filled.sum(axis=0),
        "rpm_spikes": spikes[in_grid].sum(axis=0),
    }


def mocap_thrust(t: NDArray, pos: NDArray, t_grid: NDArray, mass: float, g: float) -> NDArray:
    """Total thrust in N from the motion capture, `m |a + g z|`, on a uniform grid.

    Args:
        t: Times of the positions (N,).
        pos: Positions (N, 3).
        t_grid: Uniform grid to compute on; pad it by about a second for the filters' edges.
        mass: The drone's mass in kg.
        g: Gravity in m/s^2.
    """
    fs = 1 / (t_grid[1] - t_grid[0])
    p = resample(t, pos, t_grid, MOCAP_MAX_GAP)
    vel = lowpass(np.gradient(p, 1 / fs, axis=0), fs, VEL_CUTOFF)
    acc = lowpass(np.gradient(vel, 1 / fs, axis=0), fs, VEL_CUTOFF)
    acc[:, 2] += g
    return mass * np.linalg.norm(acc, axis=1)
