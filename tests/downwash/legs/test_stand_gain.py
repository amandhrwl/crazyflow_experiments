"""Gain and delay of the stand at the motor edges, on a cell that converts at 80 Hz."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import math  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.downwash.legs.stand_gain import (  # noqa: E402
    best_tau,
    edge_time,
    edges_of_run,
    fit_gain,
    new_samples,
    summarize,
)
from tests.downwash.legs import fakes  # noqa: E402

K_TRUE, TAU_TRUE, OFFSET = 0.98, 0.020, 0.3


def cell(held: bool) -> tuple:
    """Card records of a 0.55 N spin-up at 1.05 s, seen by the cell K x thrust, tau late."""
    t = np.arange(0.0, 3.0, 0.002)
    thrust = 0.55 * np.clip((t - 1.0) / 0.1, 0, 1)
    truth = K_TRUE * np.interp(t - TAU_TRUE, t, thrust) + OFFSET
    if held:  # The cell converts at 80 Hz with a little noise, the card holds its last value
        truth = truth + np.random.default_rng(3).normal(0.0, 1e-4, len(t))
        truth = np.interp(np.floor(t * 80) / 80, t, truth)
    return t, thrust, truth


def test_edge_time_is_the_half_crossing():
    t, thrust, _ = cell(held=False)
    assert edge_time(t, thrust, 1.0, rising=True) == pytest.approx(1.05, abs=1e-3)
    assert math.isnan(edge_time(t, 0.01 * thrust, 1.0, rising=True))  # No step


def test_fit_gain_exact_delay():
    t, thrust, force = cell(held=False)
    k_gain, c, rms, _ = fit_gain(t, force, t, thrust, TAU_TRUE)
    assert k_gain == pytest.approx(K_TRUE, abs=1e-6)
    assert c == pytest.approx(OFFSET, abs=1e-6)
    assert rms < 1e-6


def test_new_samples_and_held_signal_differ_by_the_hold():
    t, thrust, force = cell(held=True)
    is_new = new_samples(force)
    assert 70 < is_new.sum() / 3.0 < 90  # About 80 new values per second
    tau_new, k_new, _ = best_tau(t[is_new], force[is_new], t, thrust)
    tau_held, k_held, _ = best_tau(t, force, t, thrust)
    assert tau_new == pytest.approx(TAU_TRUE, abs=0.0025)
    assert k_new == pytest.approx(K_TRUE, abs=0.01)
    assert 0.003 <= tau_held - tau_new <= 0.009  # Half a conversion of 12.5 ms
    assert k_held == pytest.approx(K_TRUE, abs=0.01)


def test_edges_of_a_fake_run():
    run, _, _, outcome = fakes.fake_run("spin", held=True)
    run["meta"]["victim_schedule"] = {"legs": {"8": outcome}}
    rows = edges_of_run(run)
    assert [r["kind"] for r in rows] == ["up", "down"]
    summary = summarize(rows)
    assert summary["verdict"]["gain_ok"]
    assert summary["up"]["K_a"]["median"] == pytest.approx(1.0, abs=0.02)
    # The fake cell is 27 ms late before the 80 Hz hold, so the held signal reads about 6 ms more
    assert summary["up"]["tau_b_s"]["median"] == pytest.approx(fakes.CELL_DELAY, abs=0.003)
    assert summary["up"]["tau_held_s"]["median"] > summary["up"]["tau_b_s"]["median"]


def test_still_runs_have_no_edges():
    run, _, _, outcome = fakes.fake_run("still")
    run["meta"]["victim_schedule"] = {"legs": {"8": outcome}}
    assert edges_of_run(run) == []
