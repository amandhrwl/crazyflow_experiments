"""A3 on the stand: geometry, hover command, spin schedule, the victim's thread and the legs."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import time  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.downwash import a3_deploy, a3_stand  # noqa: E402
from crazyflow_experiments.downwash.a3_design import (  # noqa: E402
    build_design,
    check_design,
    flights,
    load_config,
)
from crazyflow_experiments.downwash.a3_stand_convert import (  # noqa: E402
    cell_delay_report,
    cut_leg,
    merge_runs,
    peak,
    rpm_drift,
)
from crazyflow_experiments.downwash.pair import SafetyLimits, safety_violation  # noqa: E402
from crazyflow_experiments.downwash.timeline import (  # noqa: E402
    build_flight,
    takeoff_timelines,
)

CFG = load_config(require_masses=False)
STAND = np.array([0.0, 0.12, 1.53])  # The victim on the stand, 21 Nov
SOURCE = np.array([-0.3, -1.0, 0.04])
ROWS = build_design(CFG)[0]
NEAR_FLIGHT = flights(ROWS)[2]  # The stand's heights, 0.5 and 0.25 m, in the shakedown order


def stand_plan(legs: list[int] | None = None) -> tuple[dict, list[dict], list[a3_stand.LegSpin]]:
    """The stand's config, rows and spin schedule of flight 1."""
    cfg, rows = a3_stand.stand_flight(CFG, a3_stand.select_legs(NEAR_FLIGHT, legs), STAND)
    _, source = build_flight(rows, cfg)
    return cfg, rows, a3_stand.spin_schedule(source, rows, CFG["stand"])


def test_legs_are_centred_on_the_stand_at_their_height_above_it():
    cfg, rows = a3_stand.stand_flight(CFG, NEAR_FLIGHT, STAND)
    originals = {int(r["leg"]): r for r in NEAR_FLIGHT}
    # Only the separations of [stand] height_seps fly (at most 0.5 m, user, 21 Nov)
    assert sorted({float(r["height_sep"]) for r in rows}) == sorted(CFG["stand"]["height_seps"])
    assert len(rows) == len(NEAR_FLIGHT) == 8  # Every leg of the near flight is a stand leg
    assert len(a3_stand.stand_rows(flights(ROWS)[1], CFG)) == 0  # The far flight has none
    for row in rows:
        original = originals[int(row["leg"])]
        assert row["x_start"] == pytest.approx(original["x_start"] + STAND[0])
        assert row["z_src"] == pytest.approx(STAND[2] + original["height_sep"])
    assert cfg["drones"]["victim"]["hover_xy"] == pytest.approx(STAND[:2].tolist())
    assert cfg["drones"]["victim"]["hover_z"] == pytest.approx(STAND[2])
    assert cfg["safety"]["arena_max"][2] == CFG["stand"]["arena_max_z"]
    assert CFG["drones"]["victim"]["hover_z"] == 0.8  # The A3 config itself is untouched
    check_design(rows, cfg)
    with pytest.raises(ValueError, match="arena_max_z"):
        a3_stand.stand_flight(CFG, NEAR_FLIGHT, STAND + [0.0, 0.0, 0.9])  # 0.5 m pass at 2.93 m


def test_the_takeoff_climbs_no_higher_than_the_stands_highest_pass():
    """22 Nov: it climbed to the stand + the free flights' 1.0 m, 2.52 m, and the source was lost."""
    cfg, rows = a3_stand.stand_flight(CFG, NEAR_FLIGHT, STAND)
    _, source = build_flight(rows, cfg)
    plan = takeoff_timelines({"source": SOURCE}, {"source": source.state(0.0)[0]}, cfg)["source"]
    top = STAND[2] + max(CFG["stand"]["height_seps"])
    z = plan.state(np.linspace(0.0, plan.duration, 4001))[0][:, 2]
    assert z.max() == pytest.approx(top)
    assert max(CFG["a3"]["height_seps"]) > max(
        CFG["stand"]["height_seps"]
    )  # The config is untouched


def test_a_flight_without_stand_legs_is_refused_with_a_reason():
    far = flights(ROWS)[1]  # 1.0 and 0.75 m, no separation of the stand
    with pytest.raises(ValueError, match="near flights"):
        a3_deploy.plan_stand(CFG, far, STAND, SOURCE, None)
    a3_deploy.plan_stand(CFG, NEAR_FLIGHT, STAND, SOURCE, None)


def test_an_off_centre_stand_leaves_the_passes_no_room():
    """The 2 m/s passes span ±2.8 m, the arena ±3.5 m after its margin."""
    cfg, rows = a3_stand.stand_flight(CFG, NEAR_FLIGHT, STAND + [0.75, 0.0, 0.0])
    with pytest.raises(ValueError, match="leaves the arena"):
        check_design(rows, cfg)


def test_the_source_takes_off_clear_of_the_stand():
    a3_stand.check_source_start(SOURCE, STAND, CFG["stand"]["clearance"])
    with pytest.raises(ValueError, match="clearance"):
        a3_stand.check_source_start(STAND + [0.2, 0.2, -1.5], STAND, CFG["stand"]["clearance"])


def test_legs_parse_and_select():
    assert a3_stand.parse_legs("3,5-7") == [3, 5, 6, 7]
    with pytest.raises(ValueError):
        a3_stand.parse_legs("3,3")
    legs = [int(r["leg"]) for r in NEAR_FLIGHT]
    assert [r["leg"] for r in a3_stand.select_legs(NEAR_FLIGHT, legs[2:4])] == legs[2:4]
    with pytest.raises(ValueError, match="not in this flight"):
        a3_stand.select_legs(NEAR_FLIGHT, [999])


def steady(command: int, force: float, ok: bool = True, uniform: bool = True) -> dict:
    """One row of A1's steady.csv."""
    other = command if uniform else command + 1000
    return {"ok": ok, "m1": command, "m2": command, "m3": other, "m4": command, "F_mean": force}


def test_hover_command_interpolates_the_ok_windows_of_a1():
    rows = [
        steady(40000, 0.40),
        steady(48000, 0.50),
        steady(52000, 0.60, ok=False),  # Flagged, left out
        steady(50000, 0.55, uniform=False),  # One motor apart, left out
        steady(56000, 0.62),
    ]
    command, how = a3_stand.hover_command(rows, 0.56)
    assert command == round(48000 + 8000 * (0.06 / 0.12))
    assert how["below"]["cmd"] == 48000 and how["above"]["cmd"] == 56000
    with pytest.raises(ValueError, match="outside"):
        a3_stand.hover_command(rows, 0.7)


def test_the_spin_windows_fit_every_leg_and_follow_each_other():
    _, _, schedule = stand_plan()
    stand = CFG["stand"]
    previous_off = -np.inf
    for leg in schedule:
        assert leg.tare_start >= previous_off + stand["rest_gap_s"]
        assert leg.tare_start < leg.arm_at < leg.spin_on < leg.baseline_start < leg.pass_start
        assert leg.pass_start - leg.baseline_start == pytest.approx(stand["baseline_s"])
        assert leg.spin_off == pytest.approx(leg.pass_end + stand["spin_after_s"])
        assert leg.spin_off <= leg.leg_end
        previous_off = leg.spin_off
    budget = a3_stand.battery_budget(schedule, stand)
    assert budget["legs_per_battery"] == int(
        stand["battery_spin_s"] // max(leg.spin_time for leg in schedule)
    )
    assert budget["batteries"] * budget["legs_per_battery"] >= len(schedule)


def test_a_schedule_that_overlaps_the_next_leg_is_refused():
    cfg, rows = a3_stand.stand_flight(CFG, NEAR_FLIGHT, STAND)
    _, source = build_flight(rows, cfg)
    with pytest.raises(ValueError, match="rest_gap_s|tare"):
        a3_stand.spin_schedule(source, rows, CFG["stand"] | {"rest_gap_s": 60.0})


def test_a_subset_of_legs_plans_on_its_own():
    legs = [int(r["leg"]) for r in NEAR_FLIGHT][4:6]
    cfg, rows, schedule = stand_plan(legs)
    assert [leg.leg for leg in schedule] == legs
    check_design(rows, cfg)


def test_the_source_lands_clear_of_the_stand():
    cfg, _ = a3_stand.stand_flight(CFG, NEAR_FLIGHT, STAND)
    assert a3_stand.landing_waypoints(STAND + [1.5, 0.0, 0.4], STAND, cfg) == []
    climb, lane = a3_stand.landing_waypoints(STAND + [0.1, 0.0, 0.3], STAND, cfg)
    assert climb[2] == pytest.approx(STAND[2] + max(CFG["stand"]["height_seps"]))
    assert lane[1] == pytest.approx(STAND[1] + CFG["a3"]["return_lane"])
    assert np.linalg.norm(lane[:2] - STAND[:2]) > cfg["stand"]["clearance"]


def test_the_separation_rule_holds_over_the_stand():
    limits = SafetyLimits.from_config(a3_stand.stand_flight(CFG, NEAR_FLIGHT, STAND)[0])
    stand = {"pos": STAND, "is_outdated": False}
    low = STAND + [0.1, 0.0, 0.1]  # Below min_vertical_sep, 0.15 m
    reason = safety_violation(limits, {"pos": low, "is_outdated": False}, low, stand, STAND)
    assert reason is not None and reason.startswith("separation")
    high = STAND + [0.1, 0.0, 0.4]
    assert safety_violation(limits, {"pos": high, "is_outdated": False}, high, stand, STAND) is None


class FakeRadioLog:
    """The victim's battery as the status packets would report it."""

    def __init__(self, vbat: float):
        """Report `vbat` until it changes."""
        self.vbat = vbat

    def recent(self, variable: str, seconds: float) -> np.ndarray:
        """A few readings of the present voltage."""
        return np.full(2, self.vbat)


class FakeVictim:
    """A victim whose battery sags while it spins."""

    def __init__(self, vbat: float, sag: float = 0.0):
        """Start at `vbat` at rest; spinning takes `sag` off it, stopping gives half of it back."""
        self.radio_log = FakeRadioLog(vbat)
        self.sag = sag
        self.calls: list[str] = []
        self._spinning = False

    @property
    def spinning(self) -> bool:
        """Whether the motors run."""
        return self._spinning

    def spin(self, pwm: int):
        """Arm and run the motors."""
        self.calls.append(f"spin {pwm}")
        self._spinning = True
        self.radio_log.vbat -= self.sag  # Under load

    def stop_spin(self):
        """Stop the motors."""
        self.calls.append("stop")
        self._spinning = False
        self.radio_log.vbat += self.sag * 0.5  # Recovers in part at rest


SPEED = 50.0  # The schedule's clock runs this much faster than real time


def run_schedule(
    victim: FakeVictim, spin: bool, stand: dict | None = None
) -> a3_stand.VictimSchedule:
    """Play the spin schedule of three legs of flight 2 against a fast clock."""
    _, _, schedule = stand_plan([int(r["leg"]) for r in a3_stand.stand_rows(NEAR_FLIGHT, CFG)][:3])
    start = time.perf_counter()
    plan = a3_stand.VictimSchedule(
        victim,
        schedule,
        (stand or CFG["stand"]) | {"hover_cmd": 49649},
        spin,
        clock=lambda: (time.perf_counter() - start) * SPEED,
    )
    plan.start(0.0)
    deadline = time.perf_counter() + schedule[-1].leg_end / SPEED + 5.0
    while plan._thread.is_alive() and time.perf_counter() < deadline:
        time.sleep(0.01)
    plan.stop()
    return plan


def test_each_leg_spins_around_its_pass_and_stops():
    victim = FakeVictim(3.9)
    plan = run_schedule(victim, spin=True)
    assert victim.calls == ["spin 49649", "stop"] * 3
    assert all(outcome.spun for outcome in plan.outcomes.values())
    assert plan.refly() == [] and plan.error is None
    for outcome in plan.outcomes.values():
        events = outcome.events
        assert events["tare_start"] < events["tare_end"] <= events["spin_on"] < events["spin_off"]


def test_a_low_battery_at_rest_ends_the_spinning_and_lists_the_legs_to_refly():
    victim = FakeVictim(3.75, sag=0.2)  # 3.55 V under load, 3.65 then 3.55 V at rest
    plan = run_schedule(victim, spin=True)
    spun = [leg for leg, outcome in plan.outcomes.items() if outcome.spun]
    assert len(spun) == 2 and plan.battery_low
    assert plan.refly() == [leg for leg in plan.outcomes if leg not in spun]
    assert victim.calls.count("spin 49649") == 2


def test_the_load_floor_stops_only_the_victims_motors():
    victim = FakeVictim(3.7, sag=0.4)  # 3.3 V under load, below vbat_load_min
    plan = run_schedule(victim, spin=True, stand=CFG["stand"] | {"vbat_rest_min": 3.0})
    first = plan.outcomes[min(plan.outcomes)]
    assert not first.spun and "under load" in first.reason
    assert victim.calls[:2] == ["spin 49649", "stop"]


def test_a_still_run_stamps_the_same_windows_with_the_motors_off():
    victim = FakeVictim(3.9)
    plan = run_schedule(victim, spin=False)
    assert victim.calls == [] and plan.refly() == []
    assert all("spin_off" in outcome.events for outcome in plan.outcomes.values())


def test_peak_and_rpm_drift():
    t = np.arange(-2.0, 2.0, 1 / 500)
    dforce = -0.01 * np.exp(-(((t - 0.3) / 0.15) ** 2))
    value, t_peak = peak(t, dforce, -1.0, 2.0)
    assert value == pytest.approx(-0.01, rel=0.05) and t_peak == pytest.approx(0.3, abs=0.01)
    rpm = np.tile([18000.0, 18000.0, 18000.0, 18000.0], (len(t), 1))
    rpm[t > 0] *= 0.98
    drift = rpm_drift(t, rpm, (-2.0, -1.0), (0.5, 1.5))
    assert drift == pytest.approx([-0.02] * 4)
    assert np.isnan(rpm_drift(t, np.zeros_like(rpm), (-2.0, -1.0), (0.5, 1.5))).all()


T_ZERO, CROSS = 1000.0, 50.0  # Run clock zero and the timeline's crossing of the synthetic leg
SOURCE_LAG = 0.35  # s the synthetic source crosses the stand after its timeline says
CELL_DELAY = 0.04  # s the synthetic load cell follows the rotor thrust
SPIN = {"cross": CROSS, "pass_start": CROSS - 2.0, "pass_end": CROSS + 2.0, "spin_off": CROSS + 3.5}
EVENTS = {"tare_start": 1040.0, "tare_end": 1042.0, "spin_on": 1043.5, "spin_off": 1053.5}


def synthetic_leg(mode: str = "spin", crosses: bool = True) -> tuple[dict, dict, dict]:
    """A leg's run, design row and outcome: thrust 0.53 N from rpm, the wake takes 10 mN of it.

    The source passes the stand at 1 m/s `SOURCE_LAG` after the timeline's crossing, 0.02 m to its
    side and 0.8 m above it, and the load cell follows the thrust `CELL_DELAY` late. The dip lies
    0.2 s after the measured crossing. A still leg has no thrust, only 300 rpm of windmilling with
    every fifth reading invalid.
    """
    stand = CFG["stand"]
    t = np.arange(1039.0, 1055.0, 1 / 500)
    t_cross = T_ZERO + CROSS + SOURCE_LAG
    spinning = (mode == "spin") & (t >= EVENTS["spin_on"]) & (t < EVENTS["spin_off"])
    dip = -0.010 * np.exp(-(((t - (t_cross + 0.2)) / 0.2) ** 2))
    thrust = np.where(spinning, 0.53, 0.0)
    force_n = np.interp(t - CELL_DELAY, t, thrust) + dip * spinning
    rpm = np.tile(np.where(spinning, 18000.0, 300.0)[:, None], (1, 4))
    if mode == "still":
        rpm[::5] = np.nan
    pos = np.tile(STAND + [0.0, 0.02, 0.8], (len(t), 1))
    vel = np.zeros((len(t), 3))
    if crosses:
        pos[:, 0] = STAND[0] + 1.0 * (t - t_cross)
        vel[:, 0] = 1.0
    run = {
        "meta": {
            "config": {"stand": stand, "a3": CFG["a3"]},
            "mode": mode,
            "stand_pos": STAND.tolist(),
        },
        "t": t,
        "weight_g": 120.0 + force_n / 9.81 * 1e3,  # The tare's zero, 120 g, drops out
        "rpm": rpm,
        "thrust_rpm": thrust,
        "vbat": np.full(len(t), 3.8),
        "t_zero": T_ZERO,
        "src_t": t,
        "src_pos": pos,
        "src_vel": vel,
    }
    row = {k: 0 for k in ("leg", "flight", "index_in_flight", "cell", "repeat")}
    row |= {"speed": 1.0, "height_sep": 0.8}
    return run, row, {"events": EVENTS, "spun": mode == "spin"}


def test_a_leg_is_zeroed_on_its_tare_and_compared_with_its_baseline():
    run, row, outcome = synthetic_leg()
    arrays, summary = cut_leg(run, row, SPIN, outcome)
    assert summary["status"] == "ok" and summary["tare_g"] == pytest.approx(120.0)
    assert summary["baseline_force_N"] == pytest.approx(0.53, abs=1e-4)
    assert summary["peak_dforce_N"] == pytest.approx(-0.010, rel=0.1)
    assert summary["t_peak_s"] == pytest.approx(0.2, abs=0.02)  # From the measured crossing
    assert summary["peak_dforce_wake_N"] == pytest.approx(-0.010, rel=0.1)  # rpm held
    assert np.nanmax(np.abs(arrays["dthrust_rpm"])) == pytest.approx(0.0, abs=1e-9)
    assert arrays["t"][0] == pytest.approx(
        EVENTS["spin_on"] + CFG["stand"]["spin_settle_s"] - T_ZERO - CROSS - SOURCE_LAG
    )
    assert np.allclose(arrays["src_pos_rel"][:, 1:], [0.02, 0.8], atol=1e-6)


def test_the_crossing_is_the_measured_one_not_the_timelines():
    run, row, outcome = synthetic_leg()
    _, summary = cut_leg(run, row, SPIN, outcome)
    assert summary["cross_nominal_s"] == pytest.approx(CROSS)
    assert summary["cross_s"] == pytest.approx(CROSS + SOURCE_LAG, abs=0.01)
    assert summary["cross_lag_s"] == pytest.approx(SOURCE_LAG, abs=0.01)
    assert summary["cross_speed"] == pytest.approx(1.0)
    assert summary["height_sep_meas"] == pytest.approx(0.8)
    assert summary["lateral_offset_meas"] == pytest.approx(0.02)


def test_a_source_that_never_passes_the_stand_flags_the_leg():
    run, row, outcome = synthetic_leg(crosses=False)
    arrays, summary = cut_leg(run, row, SPIN, outcome)
    assert summary["status"] == "no_crossing" and np.isnan(summary["cross_lag_s"])
    # Still aligned on the timeline's crossing
    assert arrays["t"][0] == pytest.approx(
        EVENTS["spin_on"] + CFG["stand"]["spin_settle_s"] - T_ZERO - CROSS
    )


def test_the_load_cell_delay_is_read_off_the_spin_up():
    run, row, outcome = synthetic_leg()
    _, summary = cut_leg(run, row, SPIN, outcome)
    assert summary["cell_delay_s"] == pytest.approx(CELL_DELAY, abs=0.005)
    report = cell_delay_report([summary, summary | {"leg": 1, "cell_delay_s": 0.06}])
    assert report["n"] == 2 and report["median_s"] == pytest.approx(0.05, abs=0.005)


def test_a_still_leg_reports_the_windmilling_and_no_delay():
    run, row, outcome = synthetic_leg("still")
    _, summary = cut_leg(run, row, SPIN, outcome)
    assert summary["status"] == "ok" and np.isnan(summary["cell_delay_s"])
    assert summary["rpm_pass_mean"] == pytest.approx(300.0)
    assert summary["rpm_pass_valid"] == pytest.approx(0.8, abs=0.01)
    assert cell_delay_report([summary])["n"] == 0


def test_runs_merge_keeping_every_ok_repeat():
    """Both repeats of an ok leg stay, in run order; a leg that did not spin is left out."""
    early = [{"flight": "1", "mode": "spin", "leg": "3", "status": "ok", "peak_dforce_N": "-0.01"}]
    part = [{"flight": "1", "mode": "spin", "leg": "4", "status": "not_spun"}]
    refly = [{"flight": "1", "mode": "spin", "leg": "3", "status": "ok", "peak_dforce_N": "-0.02"}]
    rows = merge_runs(
        [
            ("stand_flight1_spin_2026-11-22_11-00-00", refly),
            ("stand_flight1_spin_2026-11-22_10-00-00", early + part),
        ]
    )
    assert [row["peak_dforce_N"] for row in rows] == ["-0.01", "-0.02"]
    assert rows[1]["run"].endswith("11-00-00")
