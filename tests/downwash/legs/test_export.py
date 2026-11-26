"""The labels and flags of one exported leg, and the duplicate rule."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.downwash.legs.export import (  # noqa: E402
    build_leg,
    leg_id,
    resolve_duplicates,
)
from tests.downwash.legs import fakes  # noqa: E402


def leg(mode: str = "spin", **kwargs: object) -> dict:
    run, row, spin, outcome = fakes.fake_run(mode, **kwargs)
    return build_leg(run, row, spin, outcome, fakes.fake_ctx())


def pre_mask(item: dict) -> np.ndarray:
    st, w = item["states"], item["meta"]["derived"]["window_pre"]
    return st["fz_new"] & (st["t"] >= w[0]) & (st["t"] < w[1])


def test_leg_id_is_unique_per_run():
    assert leg_id("S1", "stand_flight2_spin_2026-11-22_20-24-13", 8) == (
        "A3_S1_stand_flight2_spin_2026-11-22_20-24-13_leg08"
    )


def test_s1_rotor_correction_goes_through_the_chain():
    item = leg("spin")
    st = item["states"]
    t_pc = fakes.cross_time() + st["t"]
    truth = fakes.wake(t_pc)
    err_new = np.abs(st["fz_wake"] - truth)
    pre = pre_mask(item)
    old = st["fz_total"] - (st["thrust_sum"] - st["thrust_sum"][pre].mean())
    err_old = np.abs(old - truth)
    window = np.abs(st["t"]) < 1.0
    assert err_new[window].max() < 5e-4  # N
    assert err_old[window].max() > 2e-3  # The raw subtraction leaves the 27 ms artifact
    assert item["meta"]["quality"]["peak_fz_N"] == pytest.approx(fakes.WAKE_PEAK, rel=0.15)
    assert item["meta"]["quality"]["peak_time_s"] == pytest.approx(fakes.WAKE_AT, abs=0.01)


@pytest.mark.parametrize("mode", ["spin", "still"])
def test_baseline_is_zero_over_the_new_samples(mode: str):
    item = leg(mode)
    st, pre = item["states"], pre_mask(item)
    label = st["fz_wake"] if mode == "spin" else st["fz_total"]
    assert abs(label[pre].mean()) < 1e-12
    np.testing.assert_array_equal(st["fz_total"], st["fz_raw"])
    q = item["meta"]["quality"]
    assert q["n_fz_pre"] == pre.sum()
    assert q["noise_fz_N"] == pytest.approx(np.std(st["fz_total"][pre]))


def test_s0_victim_is_unpowered():
    item = leg("still")
    st, flags = item["states"], item["meta"]["quality"]["flags"]
    assert np.all(st["motor_thrust"][:, 0] == 0) and np.all(st["rotor_vel"][:, 0] == 0)
    np.testing.assert_array_equal(st["fz_wake"], st["fz_total"])
    assert {"windmill_unmeasured", "victim_attitude_assumed", "source_thrust_from_mocap"} <= set(
        flags
    )
    assert item["meta"]["design"]["condition"] == "S0"


def test_static_victim_pose_and_source_thrust_from_mocap():
    st = leg("still")["states"]
    q = st["quat"][:, 0]
    assert np.ptp(q, axis=0).max() == 0 and np.ptp(st["pos"][:, 0], axis=0).max() == 0
    assert np.linalg.norm(q[0]) == pytest.approx(1.0, abs=1e-9) and q[0, 3] > 0.99
    level = 0.04525 * 9.81 / 4  # Level flight at constant speed
    np.testing.assert_allclose(st["motor_thrust"][500:-500, 1], level, rtol=1e-3)
    assert np.isnan(st["rotor_vel"][:, 1]).all()


def test_spin_leg_ends_before_the_motors_stop():
    item = leg("spin")
    stop = item["meta"]["derived"]["t_spin_down"]
    assert item["states"]["t"][-1] <= min(4.0, stop - 0.2) + 1e-9


def test_not_spun_hard_fails_and_keeps_its_span():
    item = leg("spin", spun=False)
    q = item["meta"]["quality"]
    assert q["hard_fail"] and "not_spun" in q["hard_fail_reason"] and "not_spun" in q["flags"]
    assert item["states"]["t"][-1] == pytest.approx(4.0)


def test_duplicates_last_usable_attempt_is_kept():
    first = leg("spin", run_id="stand_flight2_spin_2026-11-22_20-09-20")
    second = leg("spin", run_id="stand_flight2_spin_2026-11-22_20-17-53")
    report = resolve_duplicates([second, first])
    assert "superseded" in first["meta"]["quality"]["flags"]
    assert "superseded" not in second["meta"]["quality"]["flags"]
    assert second["manifest"]["rerun_of"] == first["manifest"]["leg_id"]
    assert first["manifest"]["rerun_of"] == ""
    assert report[0]["kept"] == second["manifest"]["leg_id"] and not report[0]["ambiguous"]
    assert not first["meta"]["quality"]["usable"] and second["meta"]["quality"]["usable"]


def test_duplicates_later_failed_attempt_keeps_the_earlier():
    first = leg("spin", run_id="stand_flight2_spin_2026-11-22_20-09-20")
    second = leg("spin", run_id="stand_flight2_spin_2026-11-22_20-17-53", card_gap=(0.05, 0.15))
    report = resolve_duplicates([first, second])
    assert "superseded" not in first["meta"]["quality"]["flags"]
    assert report[0]["kept"] == first["manifest"]["leg_id"] and report[0]["ambiguous"]


def test_flight_numbers_reused_across_dates_are_not_duplicates():
    a = leg("spin", run_id="stand_flight2_spin_2026-11-21_23-43-15")
    b = leg("spin", run_id="stand_flight2_spin_2026-11-22_20-24-13")
    assert resolve_duplicates([a, b]) == []
    assert "superseded" not in a["meta"]["quality"]["flags"]
