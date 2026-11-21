"""Steady windows, tare and checks of the A1 conversion, on a faked block and small cases."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import json  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.downwash import a1_convert  # noqa: E402
from crazyflow_experiments.downwash.a1_convert import (  # noqa: E402
    check_rises,
    directions,
    edge_time,
    gap_mask,
    read_steady,
    tare_line,
    window_stats,
)
from tests.downwash import a1_fakes  # noqa: E402
from tests.load_cell import fakes  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(params=[fakes.VICTIM_CLOCK, fakes.SOURCE_CLOCK], ids=["fixed_timer", "85_84"])
def block(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A faked block, converted, with the timeline kept out of the repository."""
    monkeypatch.setattr(a1_convert, "A1_PLOTS_DIR", tmp_path / "plots")
    folder = tmp_path / "all_fake"
    folder.mkdir()
    a1_fakes.write_block(folder, clock=request.param)
    a1_convert.convert(folder)
    return folder


def test_windows_recover_the_thrust_curve_and_crazyflows_prediction(block: Path):
    """Every level window holds the true thrust, the faked scale times crazyflow's of its rpm."""
    rows = read_steady(block / "steady.csv")
    assert [r["kind"] for r in rows] == [
        k for k, _, _ in a1_fakes.STEPS if k in a1_convert.WINDOW_KINDS
    ]
    for row in rows:
        assert row["ok"], row["flags"]
        assert row["F_mean"] == pytest.approx(a1_fakes.SCALE * row["F_cf"], rel=0.01)
        expected_rpm = a1_fakes.rpm_of(row["m1"])
        assert row["sum_rpm"] == pytest.approx(expected_rpm * a1_fakes.MOTOR_SPREAD.sum(), rel=5e-3)


def test_tare_line_removes_the_drift_and_the_sync_finds_the_card_rate(block: Path):
    """The zero drift over the block is measured, and the card timer's rate is the faked one."""
    report = json.loads((block / "sync.json").read_text())
    assert report["tare"]["drift_g"] == pytest.approx(a1_fakes.TARE_DRIFT * 40 / 60, abs=0.02)
    assert report["tare"]["noise_std_g"] < 3 * a1_fakes.CELL_NOISE
    assert report["clock"]["match_fraction"] > 0.8
    assert report["correlation"] > 0.999


def test_sync_pulses_put_the_force_behind_the_rpm_by_the_load_cell_lag(block: Path):
    """Both pulses give the same force-minus-rpm lag, a few tens of ms at 80 SPS."""
    pulses = json.loads((block / "sync.json").read_text())["pulses"]
    assert len(pulses) == 2
    lags = [p["force_minus_rpm_ms"] for p in pulses]
    assert all(0 < lag < 40 for lag in lags)
    assert abs(lags[1] - lags[0]) < 10


def test_directions_follow_the_command_order(block: Path):
    """Windows reached from below are up, from above down, per block."""
    rows = read_steady(block / "steady.csv")
    assert [r["direction"] for r in rows] == [
        "first", "up", "up", "up", "down", "down", "up", "up"
    ]  # fmt: skip


def test_gap_mask_drops_the_samples_beside_a_card_stall():
    """A 300 ms stall loses the records next to it, the others stay."""
    t = np.r_[np.arange(0, 1, 0.002), np.arange(1.3, 2, 0.002)]
    valid, gaps = gap_mask(t)
    k = 499
    assert not valid[k] and not valid[k + 1] and valid[k - 1] and valid[k + 2]
    assert gaps[:, 1] == pytest.approx([0.302])


def test_tare_line_interpolates_between_tare_windows():
    """A linear drift of the zero is followed between the windows and held beyond them."""
    t = np.linspace(0, 10, 1001)
    weight = 0.1 * t
    tare, windows = tare_line(t, weight, np.ones_like(t, bool), [(0.0, 1.0), (9.0, 10.0)])
    assert [w["mean_g"] for w in windows] == pytest.approx([0.05, 0.95], abs=1e-3)
    assert tare[500] == pytest.approx(0.5, abs=1e-3)
    with pytest.raises(ValueError, match="no load cell samples"):
        tare_line(t, weight, np.ones_like(t, bool), [(20.0, 21.0)])


def signals(force: np.ndarray, rpm: np.ndarray, t: np.ndarray) -> dict:
    """Signals for window_stats on one time base."""
    ones = np.ones(len(t), dtype=bool)
    return {
        "force": (t, force, ones),
        "rpm": (t, rpm, ones),
        "vbat": (t, np.full(len(t), 4.0), ones),
    }


def test_window_flags_trend_unsteady_rpm_invalid_rpm_and_gaps():
    """Each fault of a window raises its own flag, a clean window none."""
    t = np.arange(0, 2, 0.002)
    rpm = np.full((len(t), 4), 15000.0)
    clean = window_stats(
        signals(np.full(len(t), 0.3), rpm, t), 0, 2, a1_fakes.CF_COEF, {"force": 500}
    )
    assert clean["ok"] and clean["flags"] == ""
    trending = window_stats(signals(0.3 + 0.01 * t, rpm, t), 0, 2, a1_fakes.CF_COEF, {"force": 500})
    assert "force_trending" in trending["flags"]
    shaky = rpm * (1 + 0.05 * np.sin(40 * t))[:, None]
    unsteady = window_stats(
        signals(np.full(len(t), 0.3), shaky, t), 0, 2, a1_fakes.CF_COEF, {"force": 500}
    )
    assert "rpm_unsteady" in unsteady["flags"]
    invalid = rpm.copy()
    invalid[10, 2] = np.nan
    flagged = window_stats(
        signals(np.full(len(t), 0.3), invalid, t), 0, 2, a1_fakes.CF_COEF, {"force": 500}
    )
    assert "rpm_invalid" in flagged["flags"] and flagged["rpm3"] == pytest.approx(15000)
    with_gap = signals(np.full(len(t), 0.3), rpm, t) | {"gaps": (np.array([1.0]), np.array([0.3]))}
    gapped = window_stats(with_gap, 0, 2, a1_fakes.CF_COEF, {"force": 500})
    assert "sd_gap" in gapped["flags"] and gapped["longest_gap_ms"] == pytest.approx(300)


def test_reversed_force_fails_the_rise_check():
    """A force that falls as the rpm rise is a sign error, the conversion stops."""
    rows = [{"F_mean": -0.1 * k, "F_cf": 0.1 * k} for k in range(1, 6)]
    with pytest.raises(RuntimeError, match="force_sign"):
        check_rises(rows)
    assert check_rises([{"F_mean": 0.1 * k, "F_cf": 0.1 * k} for k in range(1, 6)]) > 0.99


def test_edge_time_finds_the_half_level_of_a_step():
    """The half crossing of a step 20 ms after the command lands at 20 ms."""
    t = np.arange(-0.5, 0.5, 0.002)
    x = np.where(t >= 0.02, 1.0, 0.0)
    assert edge_time(t, x, 0.0, 0.3) == pytest.approx(0.019, abs=0.002)


def test_directions_compare_each_motor_with_its_own_windows():
    """In the single-motor block a motor's first window is first, not down from the last one."""
    rows = [
        {"motor": 1, "m1": 20000, "m2": 7000, "m3": 7000, "m4": 7000},
        {"motor": 1, "m1": 30000, "m2": 7000, "m3": 7000, "m4": 7000},
        {"motor": 2, "m1": 7000, "m2": 10000, "m3": 7000, "m4": 7000},
    ]
    assert directions(rows) == ["first", "up", "first"]
