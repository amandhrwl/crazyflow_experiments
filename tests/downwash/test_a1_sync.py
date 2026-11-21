"""The A1 clock syncs of cases R1 and R2, built on `sync.py`, on faked clocks.

R1: the card holds the rpm but not the load cell, so the rpm, which the DShot telemetry updates on
every loop pass, syncs the card to the radio; a constant offset and the 85/84 card timer are both
recovered. R2: the load cell comes from a second device with its own crystal; each device maps
onto PC time through its radio arrivals, and a sync pulse seen by both lands where it happened.
"""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.downwash.a1_convert import R1_SYNC, edge_time  # noqa: E402
from crazyflow_experiments.downwash.sync import (  # noqa: E402
    fit_radio_clock,
    sync_drone,
    unwrap_ticks,
)
from tests.downwash import sync_fakes as fakes  # noqa: E402

SD_PERIOD = 0.002  # s, 500 Hz
TOLERANCE = 0.5e-3  # s, card records against their true PC time, after the latency floor
PULSE_AT = 30.0  # s after the start of the recording
ROTOR_TAU, CELL_TAU, CELL_DELAY = 0.05, 0.02, 0.008  # s


def rpm_logs(clock: fakes.Clock, rng: np.random.Generator, t0: float) -> tuple[dict, dict]:
    """The victim's card and radio logs of motor 1's rpm, new on every loop pass."""
    changes = fakes.loop_passes(clock, t0)
    pulse = clock.pc(changes) - t0 >= PULSE_AT
    true = np.where(pulse, 15000.0, 7300.0) - np.where(
        pulse, 7700 * np.exp(-(clock.pc(changes) - t0 - PULSE_AT) / ROTOR_TAU), 0.0
    )
    rpm = np.round(true + rng.normal(0, 40, len(changes))).astype(np.uint16)
    sd_ticks, v_sd, stamps, v_radio, arrival = fakes.drone_logs(clock, rng, changes, rpm, t0)
    sd = {"t_us": clock.sd_us(sd_ticks).astype(np.int64), "motor.m1_rpm": v_sd}
    radio = {"motors/t_pc": arrival, "motors/t_tick_ms": stamps, "motors/motor.m1_rpm": v_radio}
    return sd, radio | {"_true_sd_pc": clock.pc(sd_ticks)}


@pytest.mark.parametrize(
    "clock",
    [fakes.Clock(rate=1.0, sd_offset=-65.77e-3), fakes.Clock(1.0016, 4.1e-3, sd_rate=85 / 84)],
    ids=["constant_offset", "drift_and_85_84"],
)
def test_r1_card_syncs_on_the_rpm(clock: fakes.Clock):
    """The card lands on its true PC time, late by the latency floor, at either card timer rate.

    The first clock is cf10's of the load cell test of 21 Nov: tick and card timer 65.77 ms apart.
    """
    rng = np.random.default_rng(1)
    sd, radio = rpm_logs(clock, rng, fakes.PC_BOOT + 30.0)
    true_pc = radio.pop("_true_sd_pc")
    mapped, stats = sync_drone(radio, "victim", sd, SD_PERIOD, sync=R1_SYNC)
    assert stats["sd_rate"] == clock.sd_rate
    assert abs(stats["sd_offset_halves_ms"][1] - stats["sd_offset_halves_ms"][0]) < 1.0
    error = mapped.pc_from_sd(sd["t_us"]) - true_pc
    assert np.abs(error - fakes.LATENCY_FLOOR).max() < TOLERANCE


def test_r2_pulse_lands_on_both_devices_where_it_happened():
    """Victim card (rpm) and stand device (load cell) on separate crystals agree on the pulse.

    The force edge follows the rpm edge by the load cell's delay and filter; after the sync the
    measured lag matches the true one within 2 ms although the two clocks drift 0.3 % apart.
    """
    rng = np.random.default_rng(2)
    t0 = fakes.PC_BOOT + 30.0
    victim = fakes.Clock(rate=1.0016, sd_offset=-65.77e-3)
    stand = fakes.Clock(rate=0.9987, sd_offset=0.0)
    sd, radio = rpm_logs(victim, rng, t0)
    radio.pop("_true_sd_pc")
    mapped, _ = sync_drone(radio, "victim", sd, SD_PERIOD, sync=R1_SYNC)
    t_rpm = mapped.pc_from_sd(sd["t_us"]) - t0

    # The stand device streams the force at 100 Hz, stamped with its own tick
    stamps = np.arange(
        np.ceil(stand.tick(t0) * 1000) + 100, np.floor(stand.tick(t0 + 59) * 1000), 10
    )
    sample_pc = stand.pc(stamps / 1000)
    arrival = sample_pc + fakes.LATENCY_FLOOR + rng.exponential(2e-3, len(stamps))
    after = np.clip(sample_pc - t0 - PULSE_AT - CELL_DELAY, 0, None)
    force = np.where(after > 0, 1 - np.exp(-after / CELL_TAU), 0.0) + rng.normal(
        0, 1e-3, len(after)
    )
    clock, _ = fit_radio_clock(unwrap_ticks(stamps), arrival)
    t_force = clock.pc_from_tick(unwrap_ticks(stamps)) - t0

    rpm_edge = edge_time(t_rpm, sd["motor.m1_rpm"].astype(float), PULSE_AT, 0.3)
    force_edge = edge_time(t_force, force, PULSE_AT, 0.3)
    true_rpm_edge = PULSE_AT + ROTOR_TAU * np.log(7700 / (7700 - (15000 - 7300) / 2))
    true_force_edge = PULSE_AT + CELL_DELAY + CELL_TAU * np.log(2)
    assert rpm_edge - true_rpm_edge == pytest.approx(fakes.LATENCY_FLOOR, abs=2e-3)
    assert (force_edge - rpm_edge) == pytest.approx(true_force_edge - true_rpm_edge, abs=2e-3)
