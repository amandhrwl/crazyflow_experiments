"""Card stalls in the exported legs: NaN across more than 25 ms, hard fail only near the wake."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import numpy as np  # noqa: E402

from crazyflow_experiments.downwash.legs.export import build_leg, gaps_in  # noqa: E402
from tests.downwash.legs import fakes  # noqa: E402


def leg_with_gap(gap: tuple[float, float], mode: str = "spin") -> dict:
    run, row, spin, outcome = fakes.fake_run(mode, card_gap=gap)
    return build_leg(run, row, spin, outcome, fakes.fake_ctx())


def test_gaps_in():
    t = np.array([0.0, 0.002, 0.004, 0.1, 0.102, 0.2])
    assert gaps_in(t, 0.0, 0.2, 0.025) == [[0.004, 0.1], [0.102, 0.2]]
    assert gaps_in(t, 0.15, 0.3, 0.025) == [[0.102, 0.2]]


def test_gap_in_the_wake_hard_fails():
    leg = leg_with_gap((0.05, 0.15))
    q, st = leg["meta"]["quality"], leg["states"]
    assert q["hard_fail"] and "card_gap_in_wake" in q["hard_fail_reason"]
    nan = ~np.isfinite(st["fz_raw"])
    assert nan.any()
    t = st["t"]
    ((a, b),) = q["card_gaps"]
    assert np.all((t[nan] > a) & (t[nan] < b))


def test_gap_away_from_the_wake_is_a_flag():
    leg = leg_with_gap((-2.0, -1.9), mode="still")
    q, st = leg["meta"]["quality"], leg["states"]
    assert not q["hard_fail"]
    assert "card_gap" in q["flags"]
    nan = ~np.isfinite(st["fz_total"])
    assert 40 <= nan.sum() <= 52  # The 100 ms stall, nothing filled
    assert np.isfinite(st["fz_total"][st["t"] > -1.5]).all()


def test_short_stall_is_bridged():
    leg = leg_with_gap((-2.0, -1.985), mode="still")  # 15 ms: under two 80 Hz samples
    q, st = leg["meta"]["quality"], leg["states"]
    assert q["card_gaps"] == []
    assert np.isfinite(st["fz_total"]).all()


def test_rpm_stall_keeps_fz_wake_nan_local():
    leg = leg_with_gap((1.0, 1.1))
    st = leg["states"]
    wake_nan = np.flatnonzero(~np.isfinite(st["fz_wake"]))
    t = st["t"]
    assert t[wake_nan].min() > 0.99 and t[wake_nan].max() < 1.14  # Gap plus the 27 ms chain


def test_rpm_glitch_becomes_a_filled_dropout():
    from crazyflow_experiments.downwash.legs.source_sd import despike_rpm

    rpm = np.full((100, 4), 17000.0) + np.arange(100)[:, None]
    rpm[40, 2] = 62000.0  # A telemetry glitch
    rpm[60:, 0] = 17500.0  # A real step stays
    cleaned, spikes = despike_rpm(rpm)
    assert spikes.sum(axis=0).tolist() == [0, 0, 1, 0] and spikes[40, 2]
    assert np.isnan(cleaned[40, 2]) and np.isfinite(np.delete(cleaned, 40, axis=0)).all()
    assert np.isfinite(despike_rpm(np.zeros((20, 4)))[0]).all()  # Motors off: nothing removed
