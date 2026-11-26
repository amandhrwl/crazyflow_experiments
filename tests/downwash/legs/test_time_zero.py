"""Time zero of the exported legs: the mocap crossing, a sample of the grid."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.downwash.legs.export import build_leg, leg_grid, new_on_grid  # noqa: E402
from crazyflow_experiments.downwash.legs.validate import check_leg  # noqa: E402
from tests.downwash.legs import fakes  # noqa: E402


def test_leg_grid_holds_zero():
    t = leg_grid(-4.371, 3.1)
    assert 0.0 in t
    assert t[0] >= -4.371 and t[0] - 0.002 < -4.371
    np.testing.assert_allclose(np.diff(t), 0.002, atol=1e-12)


def test_new_values_mark_the_first_grid_sample_after_them():
    t_card = np.array([0.0, 0.0021, 0.0043, 0.0061, 0.0083])
    raw = np.array([5, 5, 7, 7, 9])
    t_pc = np.arange(0.0, 0.012, 0.002)
    assert np.flatnonzero(new_on_grid(t_card, raw, t_pc)).tolist() == [3, 5]


@pytest.mark.parametrize("mode", ["spin", "still"])
def test_crossing_is_time_zero(mode: str):
    run, row, spin, outcome = fakes.fake_run(mode)
    leg = build_leg(run, row, spin, outcome, fakes.fake_ctx())
    derived, t = leg["meta"]["derived"], leg["states"]["t"]
    assert derived["cross_time_abs"] == pytest.approx(fakes.cross_time(), abs=1e-9)
    assert np.any(t == 0.0)
    assert t[0] <= -2.5
    pos = leg["states"]["pos"]
    k = int(np.flatnonzero(t == 0.0)[0])
    assert pos[k, 1, 0] - pos[k, 0, 0] == pytest.approx(0.0, abs=1e-6)
    on = fakes.T_ZERO + fakes.SCHEDULE["spin_on"]
    assert derived["window_pre"][0] == pytest.approx(on + 1.5 - fakes.cross_time())
    assert derived["window_pre"][1] == pytest.approx(
        fakes.T_ZERO + fakes.SCHEDULE["pass_start"] - fakes.cross_time()
    )
    assert "time zero" not in " ".join(check_leg(None, leg["meta"], leg["states"]))


def test_validator_catches_a_shifted_time_zero():
    run, row, spin, outcome = fakes.fake_run("still")
    leg = build_leg(run, row, spin, outcome, fakes.fake_ctx())
    states = dict(leg["states"])
    pos = states["pos"].copy()
    pos[:, 1, 0] += 0.01  # 10 ms at 1 m/s
    states["pos"] = pos
    assert any("off the crossing" in e for e in check_leg(None, leg["meta"], states))


def test_measured_geometry():
    run, row, spin, outcome = fakes.fake_run("spin")
    derived = build_leg(run, row, spin, outcome, fakes.fake_ctx())["meta"]["derived"]
    assert derived["v_meas"] == pytest.approx(fakes.SPEED, rel=0.01)
    assert derived["dz_meas"] == pytest.approx(fakes.DZ, abs=1e-6)
    assert derived["y_meas"] == pytest.approx(fakes.Y_OFF, abs=1e-6)
