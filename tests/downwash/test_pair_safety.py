"""Safety rules, flight loop, takeoff and landing of the two-drone downwash experiment.

The loop runs against fake drones that track their setpoints perfectly, on a fake clock, so a whole
flight takes a fraction of a second and no radio or ROS is involved.
"""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import copy  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import signal  # noqa: E402
import threading  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.downwash import deploy_a3, pair, usd  # noqa: E402
from crazyflow_experiments.downwash.design import (  # noqa: E402
    build_design,
    drag_per_speed,
    load_config,
)
from crazyflow_experiments.downwash.pair import SafetyLimits, safety_violation  # noqa: E402
from crazyflow_experiments.downwash.timeline import (  # noqa: E402
    CEILING_MARGIN,
    build_flight,
    takeoff_timelines,
    victim_hover_point,
)
from tests.downwash.fakes import (  # noqa: E402
    FakeClock,
    FakeConsole,
    FakeDrone,
    fake_read_once,
    patch_pair,
)

VICTIM = np.array([0.0, 0.0, 1.0])
SOURCE = np.array([-1.15, 0.0, 2.2])


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(require_masses=False)


@pytest.fixture(scope="module")
def limits(cfg: dict) -> SafetyLimits:
    return SafetyLimits.from_config(cfg)


def obs(pos: np.ndarray, outdated: bool = False) -> dict:
    return {"pos": np.asarray(pos, dtype=float), "is_outdated": outdated}


def violation(
    limits: SafetyLimits, src: np.ndarray, vic: np.ndarray | None, **kwargs: object
) -> str | None:
    """Check with the commands where the drones are, unless given."""
    cmd_src = kwargs.get("cmd_src", src)
    cmd_vic = kwargs.get("cmd_vic", vic)
    obs_vic = None if vic is None else obs(vic, kwargs.get("vic_outdated", False))
    return safety_violation(
        limits, obs(src, kwargs.get("src_outdated", False)), cmd_src, obs_vic, cmd_vic
    )


def test_nominal_flight_passes(limits: SafetyLimits):
    assert violation(limits, SOURCE, VICTIM) is None
    over = VICTIM + np.array([0.0, 0.0, 0.4])  # The closest pass, straight above the victim
    assert violation(limits, over, VICTIM) is None


@pytest.mark.parametrize("drone", ["src", "vic"])
def test_outdated_mocap_aborts(limits: SafetyLimits, drone: str):
    reason = violation(limits, SOURCE, VICTIM, **{f"{drone}_outdated": True})
    assert reason is not None and reason.startswith("mocap")


@pytest.mark.parametrize(
    ("pos", "inside"),
    [
        ([2.45, 0.0, 1.4], True),
        ([2.55, 0.0, 1.4], False),
        ([0.0, -2.05, 1.4], False),
        ([-1.0, 0.0, 2.29], True),
        ([-1.0, 0.0, 2.31], False),
        ([-1.0, 0.0, -0.01], False),
    ],
)
def test_arena(limits: SafetyLimits, pos: list[float], inside: bool):
    reason = violation(limits, np.array(pos), VICTIM)
    assert (reason is None) == inside
    if not inside:
        assert reason.startswith("arena")


def test_separation(limits: SafetyLimits):
    low = VICTIM + np.array([0.1, 0.0, 0.2])  # 0.2 m above, 0.1 m to the side
    assert violation(limits, low, VICTIM).startswith("separation")
    assert violation(limits, VICTIM + np.array([0.0, 0.0, 0.26]), VICTIM) is None
    assert (
        violation(limits, VICTIM + np.array([0.45, 0.0, 0.1]), VICTIM) is None
    )  # Outside xy_guard
    assert violation(limits, VICTIM + np.array([0.1, 0.0, 0.2]), None) is None  # Victim not flying


@pytest.mark.parametrize("drone", ["src", "vic"])
def test_tracking(limits: SafetyLimits, drone: str):
    offset = {"cmd_src": SOURCE, "cmd_vic": VICTIM}
    off = np.array([0.0, 0.2, 0.0])
    pos = {"src": SOURCE, "vic": VICTIM}
    pos[drone] = pos[drone] + off
    assert violation(limits, pos["src"], pos["vic"], **offset) is None
    pos[drone] = pos[drone] + off  # 0.4 m off its setpoint
    assert violation(limits, pos["src"], pos["vic"], **offset).startswith("tracking")


def test_tracking_tolerates_lag_but_not_leaving_the_path(limits: SafetyLimits):
    t = np.linspace(-pair.TRACKING_WINDOW, 0.0, 31)
    path = SOURCE + np.outer(1.4 * t, [1.0, 0.0, 0.0])  # The last 0.3 s of a pass at 1.4 m/s
    late = path[10]  # 0.2 s late: 0.28 m behind the current setpoint, but on the path
    assert np.linalg.norm(late - path[-1]) > limits.max_tracking_error
    assert safety_violation(limits, obs(late), path) is None
    off = late + np.array([0.0, 0.26, 0.0])
    assert safety_violation(limits, obs(off), path).startswith("tracking: the source")


def test_victim_drop(limits: SafetyLimits):
    assert violation(limits, SOURCE, VICTIM - np.array([0.0, 0.0, 0.2])) is None
    assert violation(limits, SOURCE, VICTIM - np.array([0.0, 0.0, 0.3])).startswith("victim drop")


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    patch_pair(monkeypatch, pair, clock)
    return clock


def battery(monkeypatch: pytest.MonkeyPatch, vbat: float | None):
    """Answer the battery reads with `vbat`, or time out when it is None."""

    def read_once(drone: FakeDrone, names: list[str], timeout: float = 2.0) -> dict:
        if vbat is None and "pm.vbat" in names:
            raise TimeoutError("no sample")
        return {name: vbat if name == "pm.vbat" else drone.log_value(name) for name in names}

    monkeypatch.setattr(usd, "read_once", read_once)
    monkeypatch.setattr(threading, "excepthook", lambda args: None)  # The failed read is expected


def drones_at_start(cfg: dict, rows: list[dict]) -> tuple[FakeDrone, FakeDrone]:
    _, source = build_flight(rows, cfg)
    return FakeDrone(3, victim_hover_point(cfg)), FakeDrone(1, source.state(0.0)[0])


def single_leg(cfg: dict, speed: float, height_sep: float) -> list[dict]:
    return build_design(cfg, single=(speed, height_sep))[0]


def test_flight_completes_with_perfect_tracking(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
):
    battery(monkeypatch, 4.0)
    rows = single_leg(cfg, 1.0, 0.8)
    victim, source = drones_at_start(cfg, rows)
    flight = pair.run_flight(victim, source, rows, cfg)
    assert flight.abort_reason is None
    assert flight.skipped == []
    duration = flight.source_timeline.duration
    assert len(flight.records) == pytest.approx(duration * cfg["control"]["ctrl_freq"], abs=2)
    stats, flags = flight.leg_stats[0], flight.leg_flags[0]
    assert stats["cruise_speed"] == pytest.approx(1.0, rel=0.01)
    # A drone is observed before its next setpoint, so it lags one tick of travel at cruise
    assert stats["tracking_error"] == pytest.approx(1.0 / cfg["control"]["ctrl_freq"], abs=1e-6)
    assert stats["victim_min_dz"] == pytest.approx(0.0, abs=1e-9)
    assert stats["victim_hover_std"] == pytest.approx(0.0, abs=1e-9)
    assert flags == []
    record = flight.records[len(flight.records) // 2]
    assert set(record) == {"t_perf", "t_wall", "victim", "source"}
    assert {"phase", "leg", "cmd_mm", "cmd_pos", "pos_raw", "is_outdated"} <= set(record["source"])
    sent = [call[1][0] for call in source.calls if call[0] == "send_state"]
    assert max(p[0] for p in sent) == pytest.approx(rows[0]["x_end"], abs=1e-6)
    updates = victim.names().count("update_estimators")
    assert updates == len(flight.records)


def test_setpoints_carry_the_drag_feedforward(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
):
    """In F0 (18 Nov) the source sank to 0.95 m/s after each 1.4 m/s ramp without it."""
    battery(monkeypatch, 4.0)
    rows = single_leg(cfg, 1.0, 0.8)
    flight = pair.run_flight(*drones_at_start(cfg, rows), rows, cfg)
    drag = drag_per_speed(cfg["drones"]["model"])
    assert drag == pytest.approx(0.5, abs=0.05)
    cruise = [r for r in flight.records if r["source"]["phase"] == "cruise"]
    assert cruise
    for record in cruise:  # The timeline's own acceleration is zero in cruise
        vel = record["source"]["cmd_vel"]
        assert vel[0] == pytest.approx(1.0)
        assert record["source"]["cmd_acc"] == pytest.approx([drag * vel[0], 0.0, 0.0])
        assert record["victim"]["cmd_acc"] == pytest.approx([0.0, 0.0, 0.0])  # Hovering
    pos, vel, acc = pair.with_drag((np.zeros(3), np.array([0.3, -0.2, 0.5]), np.ones(3)), drag)
    assert acc == pytest.approx([1.0 + 0.3 * drag, 1.0 - 0.2 * drag, 1.0])  # Not vertical


def test_close_leg_skipped_on_a_low_battery(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
):
    low = cfg["safety"]["vbat_min_close_leg"] - 0.05
    battery(monkeypatch, low)
    rows = single_leg(cfg, 1.0, 0.4)
    victim, source = drones_at_start(cfg, rows)
    flight = pair.run_flight(victim, source, rows, cfg)
    assert flight.abort_reason is None
    assert flight.skipped == [0]
    assert flight.vbat == {0: low}
    assert flight.leg_flags[0] == ["skipped"]
    sent = np.array([call[1][0] for call in source.calls if call[0] == "send_state"])
    assert sent[:, 0].max() < -1.0  # The source never crossed over the victim
    nominal, _ = build_flight(rows, cfg)
    assert flight.source_timeline.duration == pytest.approx(nominal.duration)


def test_close_leg_skipped_without_a_reading(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
):
    battery(monkeypatch, None)
    rows = single_leg(cfg, 1.0, 0.4)
    victim, source = drones_at_start(cfg, rows)
    flight = pair.run_flight(victim, source, rows, cfg)
    assert flight.skipped == [0]
    assert flight.vbat == {0: None}


def test_close_leg_flown_on_a_full_battery(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
):
    battery(monkeypatch, 3.9)
    rows = single_leg(cfg, 1.0, 0.4)
    flight = pair.run_flight(*drones_at_start(cfg, rows), rows, cfg)
    assert flight.skipped == []
    assert flight.vbat == {0: 3.9}


def test_outdated_mocap_ends_the_flight(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
):
    battery(monkeypatch, 4.0)
    rows = single_leg(cfg, 1.0, 0.8)
    victim, source = drones_at_start(cfg, rows)
    original = victim.update_estimators

    def go_stale():
        original()
        victim.outdated = victim.names().count("update_estimators") >= 500

    victim.update_estimators = go_stale
    flight = pair.run_flight(victim, source, rows, cfg)
    assert flight.abort_reason.startswith("mocap: the victim")
    assert len(flight.records) == 500


def test_tracking_loss_ends_the_flight(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
):
    battery(monkeypatch, 4.0)
    rows = single_leg(cfg, 1.0, 0.8)
    victim, source = drones_at_start(cfg, rows)
    source.frozen = True  # Stays at the start while the pass begins
    flight = pair.run_flight(victim, source, rows, cfg)
    assert flight.abort_reason.startswith("tracking: the source")
    t_pass = flight.source_timeline.leg_times(0)["pass_start"]
    assert t_pass < flight.records[-1]["t_perf"] < t_pass + 1.0 + pair.TRACKING_WINDOW


def test_a_late_source_flies_the_fast_pass(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
):
    """F0: the source followed its setpoints 0.17 s late, 0.24 m behind at 1.4 m/s."""
    battery(monkeypatch, 4.0)
    rows = single_leg(cfg, 1.4, 0.8)
    victim, source = drones_at_start(cfg, rows)
    sent, send = deque(maxlen=18), source.send_state

    def late(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, yaw: float):
        sent.append(np.array(pos, dtype=float))
        send(sent[0], vel, acc, yaw)  # The setpoint of 17 ticks ago

    source.send_state = late
    flight = pair.run_flight(victim, source, rows, cfg)
    assert flight.abort_reason is None
    stats, flags = flight.leg_stats[0], flight.leg_flags[0]
    assert stats["tracking_error"] > 0.2
    assert stats["path_error"] < 0.02
    assert "tracking_error_high" not in flags


def test_operator_ends_the_flight(cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch):
    battery(monkeypatch, 4.0)
    calls = iter(range(10_000))
    monkeypatch.setattr(pair, "rclpy", SimpleNamespace(ok=lambda: next(calls) < 300))
    rows = single_leg(cfg, 1.0, 0.8)
    flight = pair.run_flight(*drones_at_start(cfg, rows), rows, cfg)
    assert flight.abort_reason.startswith("operator")
    assert len(flight.records) == 300


def ctrl_c_after(drone: FakeDrone, n_setpoints: int):
    """Press Ctrl+C, a real SIGINT, right after the drone's `n_setpoints`-th setpoint."""
    send, calls = drone.send_state, iter(range(1, 1_000_000))

    def send_then_ctrl_c(*args: object):
        send(*args)
        if next(calls) == n_setpoints:
            signal.raise_signal(signal.SIGINT)

    drone.send_state = send_then_ctrl_c


def test_ctrl_c_ends_the_flight_and_keeps_its_records(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
):
    """Ctrl+C raised inside `run_flight` and lost every leg of A3 F1 (19 Nov)."""
    battery(monkeypatch, 4.0)
    rows = single_leg(cfg, 1.0, 0.8)
    victim, source = drones_at_start(cfg, rows)
    ctrl_c_after(source, 300)
    before = signal.getsignal(signal.SIGINT)
    with pair.ctrl_c_ends_flight():
        flight = pair.run_flight(victim, source, rows, cfg)
        with pytest.raises(KeyboardInterrupt):  # A second Ctrl+C raises as usual
            signal.raise_signal(signal.SIGINT)
    assert flight.abort_reason == "operator: Ctrl+C"
    assert len(flight.records) == 300
    assert signal.getsignal(signal.SIGINT) is before
    assert not pair.operator_stopped()  # The next flight starts afresh


def test_source_flies_alone(cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch):
    battery(monkeypatch, None)  # Never read: without a victim there is no battery check
    rows = single_leg(cfg, 1.4, 0.4)
    _, source = drones_at_start(cfg, rows)
    flight = pair.run_flight(None, source, rows, cfg)
    assert flight.abort_reason is None
    assert flight.skipped == []
    assert all(record["victim"] is None for record in flight.records)
    assert "victim_min_dz" not in flight.leg_stats[0]


def test_leg_summary_flags_moving_air(cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch):
    battery(monkeypatch, 4.0)
    rows = single_leg(cfg, 0.5, 1.2)
    victim, source = drones_at_start(cfg, rows)
    rng = np.random.default_rng(0)
    original = victim.obs

    def drifting_obs() -> dict:
        o = original()
        o["pos"] = o["pos"] + rng.normal(0.0, 0.02, 3)  # 2 cm of jitter, still air allows 1 cm
        return o

    victim.obs = drifting_obs
    flight = pair.run_flight(victim, source, rows, cfg)
    assert flight.abort_reason is None
    assert "air_not_still" in flight.leg_flags[0]


PLACEMENTS = [  # Where the victim and the source stand before takeoff, (x, y)
    ((0.0, 0.0), (-2.0, 0.0)),  # Below their targets' sides
    ((-1.15, 0.0), (0.0, 0.0)),  # Swapped: each stands below the other's target
    ((1.5, -1.0), (-1.0, 1.2)),
    ((0.5, 0.5), (-0.5, 0.5)),  # The minimum spacing
]


def separation_broken(cfg: dict, p_src: np.ndarray, p_vic: np.ndarray) -> np.ndarray:
    safety = cfg["safety"]
    dxy = np.linalg.norm(p_src[:, :2] - p_vic[:, :2], axis=1)
    return (dxy < safety["xy_guard"]) & (p_src[:, 2] - p_vic[:, 2] < safety["min_vertical_sep"])


@pytest.mark.parametrize(("victim_xy", "source_xy"), PLACEMENTS)
@pytest.mark.parametrize("source_z", [1.4, 2.2])  # Start of the closest and the highest first leg
def test_takeoff_from_anywhere_keeps_the_pair_apart(
    cfg: dict, victim_xy: tuple, source_xy: tuple, source_z: float
):
    grounds = {"victim": np.array([*victim_xy, 0.02]), "source": np.array([*source_xy, 0.02])}
    targets = {"victim": VICTIM, "source": np.array([-1.15, 0.0, source_z])}
    plans = takeoff_timelines(grounds, targets, cfg)
    assert plans["victim"].duration == pytest.approx(plans["source"].duration)
    for role, plan in plans.items():
        np.testing.assert_allclose(plan.state(0.0)[0], grounds[role], atol=1e-12)
        np.testing.assert_allclose(plan.state(plan.duration)[0], targets[role], atol=1e-12)
    t = np.linspace(0.0, plans["source"].duration, 5001)
    p_vic, p_src = plans["victim"].state(t)[0], plans["source"].state(t)[0]
    assert not separation_broken(cfg, p_src, p_vic).any()
    ceiling = cfg["safety"]["arena_max"][2] - CEILING_MARGIN
    assert max(p_vic[:, 2].max(), p_src[:, 2].max()) <= ceiling + 1e-9


def test_placement_needs_the_arena_margin_and_spacing(cfg: dict):
    def drones(victim_xy: tuple, source_xy: tuple) -> dict:
        return {
            "victim": FakeDrone(3, np.array([*victim_xy, 0.02])),
            "source": FakeDrone(4, np.array([*source_xy, 0.02])),
        }

    for victim_xy, source_xy in PLACEMENTS:
        deploy_a3.check_placement(drones(victim_xy, source_xy), cfg)
    with pytest.raises(RuntimeError, match="apart"):
        deploy_a3.check_placement(drones((0.3, -0.4), (0.3, 0.4)), cfg)
    with pytest.raises(RuntimeError, match="arena margin"):
        deploy_a3.check_placement(drones((0.0, 0.0), (-2.4, 0.0)), cfg)


@pytest.mark.parametrize(("victim_xy", "source_xy"), PLACEMENTS[:2])
def test_takeoff_streams_both_to_their_targets(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, victim_xy: tuple, source_xy: tuple
):
    done = threading.Event()
    done.set()
    monkeypatch.setattr(pair, "calibrate_async", lambda drone_id, delay: done)
    victim = FakeDrone(3, np.array([*victim_xy, 0.02]))
    source = FakeDrone(1, np.array([*source_xy, 0.02]))
    targets = {"victim": VICTIM, "source": SOURCE}
    takeoff = pair.takeoff_pair(victim, source, targets, cfg)
    assert takeoff.error is None
    np.testing.assert_allclose(victim.pos, VICTIM, atol=1e-5)
    np.testing.assert_allclose(source.pos, SOURCE, atol=1e-5)
    for drone in (victim, source):  # Re-armed before the first setpoint
        names = drone.names()
        assert names.index("arm") < names.index("send_state")
    assert takeoff.records[-1]["source"]["phase"] == "calibrate"
    p_src = np.array([r["source"]["cmd_pos"] for r in takeoff.records])
    p_vic = np.array([r["victim"]["cmd_pos"] for r in takeoff.records])
    assert not separation_broken(cfg, p_src, p_vic).any()


def test_takeoff_gives_up_on_a_stuck_calibration(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(pair, "calibrate_async", lambda drone_id, delay: threading.Event())
    victim = FakeDrone(3, np.array([0.0, 0.0, 0.02]))
    source = FakeDrone(1, np.array([-2.0, 0.0, 0.02]))
    takeoff = pair.takeoff_pair(victim, source, {"victim": VICTIM, "source": SOURCE}, cfg)
    assert takeoff.error == "calibration did not finish"


def test_takeoff_raises_on_outdated_mocap(cfg: dict, clock: FakeClock):
    victim = FakeDrone(3, np.array([0.0, 0.0, 0.02]))
    source = FakeDrone(1, np.array([-2.0, 0.0, 0.02]))
    source.outdated = True
    with pytest.raises(RuntimeError, match="outdated"):
        pair.takeoff_pair(victim, source, {"victim": VICTIM, "source": SOURCE}, cfg)


def test_landing_far_apart_lands_both_in_place(cfg: dict, clock: FakeClock):
    victim, source = FakeDrone(3, VICTIM), FakeDrone(1, SOURCE)
    pair.land_pair(victim, source, cfg)
    for drone in (victim, source):
        names = drone.names()
        assert names[:4] == ["send_stop_setpoint", "send_notify_setpoint_stop", "set_value", "arm"]
        assert "go_to" not in names
        assert names.count("land") == 1
        assert names[-1] == "stop"
        assert "update_estimators" in names


def test_landing_close_lets_the_upper_drone_climb_first(cfg: dict, clock: FakeClock):
    victim = FakeDrone(3, VICTIM)
    source = FakeDrone(1, VICTIM + np.array([0.1, 0.0, 0.4]))
    pair.land_pair(victim, source, cfg)
    names = source.names()
    assert names.index("go_to") < names.index("land")
    x, y, z, _, duration = next(call[1] for call in source.calls if call[0] == "go_to")
    assert (x, y, z, duration) == pytest.approx((0.1, 0.0, 1.7, pair.CLIMB_TIME))
    assert victim.names().count("land") == 1 and "go_to" not in victim.names()


def test_landing_climb_stays_below_the_ceiling(cfg: dict, clock: FakeClock):
    victim = FakeDrone(3, VICTIM)
    source = FakeDrone(1, VICTIM + np.array([0.0, 0.0, 1.2]))  # The highest pass, at 2.2 m
    pair.land_pair(victim, source, cfg)
    z = next(call[1][2] for call in source.calls if call[0] == "go_to")
    assert z <= cfg["safety"]["arena_max"][2] - 0.1 + 1e-9


def test_landing_kills_a_tipped_drone(cfg: dict, clock: FakeClock):
    victim, source = FakeDrone(3, VICTIM), FakeDrone(1, SOURCE, tilt=120.0)
    pair.land_pair(victim, source, cfg)
    assert source.names().count("packet") == 1
    assert "packet" not in victim.names()
    assert victim.names()[-1] == "stop"


@pytest.fixture
def deploy_env(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> SimpleNamespace:
    """`deploy_a3.fly` against fake drones, SD cards and calibration, writing into tmp_path."""
    victim = FakeDrone(cfg["drones"]["victim"]["id"], np.array([0.0, 0.0, 0.02]))
    source = FakeDrone(cfg["drones"]["source"]["id"], np.array([-2.0, 0.0, 0.02]))
    sd_calls: list[tuple[str, str]] = []
    done = threading.Event()
    done.set()
    monkeypatch.setattr(pair, "connect_pair", lambda cfg, only_source: (victim, source))
    monkeypatch.setattr(pair, "calibrate_async", lambda drone_id, delay: done)
    names = {id(drone.cf): drone.name for drone in (victim, source)}
    monkeypatch.setattr(usd, "ConsoleCapture", lambda cf: FakeConsole(f"{names[id(cf)]}_00"))
    monkeypatch.setattr(usd, "check_variables", lambda cf, name, variables: None)
    monkeypatch.setattr(usd, "read_once", fake_read_once)
    for action in ("start", "stop"):

        def log(cf: object, name: str, action: str = action) -> float:
            sd_calls.append((action, name))
            return clock.time()

        monkeypatch.setattr(usd, f"{action}_logging", log)
    monkeypatch.setattr(deploy_a3, "A3_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(deploy_a3, "git_info", lambda: {"hash": "0" * 40, "dirty": False})
    flight_cfg = copy.deepcopy(cfg)
    for role in ("victim", "source"):
        flight_cfg["drones"][role]["mass_kg"] = 0.045
    rows = single_leg(flight_cfg, 1.0, 0.8)
    return SimpleNamespace(
        victim=victim, source=source, sd_calls=sd_calls, cfg=flight_cfg, rows=rows, folder=tmp_path
    )


def test_connect_pair_leaves_both_disarmed(
    cfg: dict, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
):
    """Neither idles armed while the other connects: the firmware locks it 2 s on."""
    monkeypatch.setattr(
        pair,
        "connect",
        lambda config, control: FakeDrone(config["drone"]["id"], np.zeros(3), armed=True),
    )
    victim, source = pair.connect_pair(cfg)
    _, alone = pair.connect_pair(cfg, only_source=True)
    drag = drag_per_speed(cfg["drones"]["model"])
    for drone in (victim, source, alone):
        assert not drone.armed
        # The zero setpoint brings the drone out of WarningLevelOut, which ignores the disarm
        assert drone.calls[:2] == [("send_setpoint", (0, 0, 0, 0)), ("arm", (False,))]
        # Without it the onboard velocity estimate ran 0.27 m/s high at 1.4 m/s in F0 (18 Nov)
        assert drone.calls[2:] == [
            ("set_value", ("kalman.dragB_x", drag)),
            ("set_value", ("kalman.dragB_y", drag)),
        ]


def test_disarm_fails_when_the_drone_cannot_be_armed_again(clock: FakeClock):
    """cf3 was locked when the takeoff armed it (A3 F2, 17 Nov), and the request was dropped."""
    drone = FakeDrone(3, np.zeros(3), armed=True)
    drone.info_disarmed = 0x0240
    t_start = clock.now
    with pytest.raises(RuntimeError, match=r"cf3 cannot be armed \(supervisor.info 0x0240\)"):
        pair.disarm(drone)
    assert pair.DISARM_TIMEOUT <= clock.now - t_start <= pair.DISARM_TIMEOUT + 2 * pair.DISARM_POLL


def test_takeoff_refuses_a_drone_that_cannot_be_armed(cfg: dict, clock: FakeClock):
    """The firmware drops an arming request outside PreFlChecksPassed and reports nothing."""
    victim = FakeDrone(3, np.array([0.0, 0.0, 0.02]))
    source = FakeDrone(4, np.array([-2.0, 0.0, 0.02]))
    victim.info_disarmed = 0x0240
    takeoff = pair.takeoff_pair(victim, source, {"victim": VICTIM, "source": SOURCE}, cfg)
    assert takeoff.error == (
        "arming: the victim cannot be armed (supervisor.info 0x0240): it is locked, reboot it"
    )
    assert not takeoff.records
    for drone in (victim, source):
        assert "arm" not in drone.names() and "send_state" not in drone.names()


@pytest.mark.parametrize(
    ("info", "reason"),
    [(0x0201, "arming rpm check"), (0x0241, "locked, reboot"), (0x0221, "tumbled")],
)
def test_takeoff_ends_low_when_a_drone_did_not_arm(
    cfg: dict, clock: FakeClock, info: int, reason: str
):
    """cf3's M3 failed the arming rpm check, so cf4 took off alone (A3 F1, 17 Nov)."""
    victim = FakeDrone(3, np.array([0.0, 0.0, 0.02]))
    source = FakeDrone(4, np.array([-2.0, 0.0, 0.02]))
    victim.info_armed = info
    takeoff = pair.takeoff_pair(victim, source, {"victim": VICTIM, "source": SOURCE}, cfg)
    assert takeoff.error.startswith(f"arming: the victim cannot fly (supervisor.info 0x{info:04x})")
    assert reason in takeoff.error and "source" not in takeoff.error
    assert takeoff.records[-1]["t_perf"] < pair.ARMING_CHECK_AT + 0.5
    assert max(r["source"]["cmd_pos"][2] for r in takeoff.records) < 0.3  # Still low


def flight_meta(folder: Path) -> dict:
    (path,) = folder.glob("flight1_*/meta.json")
    return json.loads(path.read_text())


def test_deploy_flies_lands_and_records(
    deploy_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    battery(monkeypatch, 4.0)
    env = deploy_env
    deploy_a3.fly(env.rows, env.cfg, 1, False, Path("design.csv"))
    meta = flight_meta(env.folder)
    assert meta["abort_reason"] is None and meta["error"] is None
    vic, src = (f"cf{env.cfg['drones'][role]['id']}" for role in ("victim", "source"))
    assert meta["usd"]["victim"]["files"] == [f"{vic}_00"]
    assert {"timeline", "leg_stats", "leg_flags", "t_wall_zero"} <= set(meta)
    assert env.sd_calls == [("start", vic), ("start", src), ("stop", vic), ("stop", src)]
    for drone in (env.victim, env.source):
        names = drone.names()
        assert names.count("land") == 1
        assert names[-1] == "close"
    with open(next(env.folder.glob("flight1_*/records.pkl")), "rb") as f:
        records = pickle.load(f)
    assert records["takeoff"] and records["flight"]


def test_deploy_lands_both_when_a_drone_did_not_arm(deploy_env: SimpleNamespace):
    env = deploy_env
    env.victim.info_armed = 0x0201  # Disarmed by the arming rpm check
    deploy_a3.fly(env.rows, env.cfg, 1, False, Path("design.csv"))
    meta = flight_meta(env.folder)
    assert meta["abort_reason"].startswith("arming: the victim cannot fly")
    assert [call[0] for call in env.sd_calls].count("stop") == 2  # The SD files are closed
    for drone in (env.victim, env.source):
        names = drone.names()
        assert names.count("land") == 1 and names.index("land") < names.index("close")


def test_deploy_lands_and_keeps_the_legs_on_ctrl_c(
    deploy_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    """Ctrl+C in the legs lands both and writes the records and timeline the conversion needs."""
    battery(monkeypatch, 4.0)
    env = deploy_env
    run = pair.run_flight

    def run_with_ctrl_c(victim: FakeDrone, source: FakeDrone, rows: list, cfg: dict) -> pair.Flight:
        ctrl_c_after(source, 300)
        return run(victim, source, rows, cfg)

    monkeypatch.setattr(pair, "run_flight", run_with_ctrl_c)
    deploy_a3.fly(env.rows, env.cfg, 1, False, Path("design.csv"))
    meta = flight_meta(env.folder)
    assert meta["abort_reason"] == "operator: Ctrl+C" and meta["error"] is None
    assert "timeline" in meta
    with open(next(env.folder.glob("flight1_*/records.pkl")), "rb") as f:
        assert len(pickle.load(f)["flight"]) == 300
    assert [call[0] for call in env.sd_calls].count("stop") == 2
    for drone in (env.victim, env.source):
        names = drone.names()
        assert names.count("land") == 1 and names.index("land") < names.index("close")


def test_deploy_abort_path_lands_stops_logging_and_records(
    deploy_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    """B5: the motion capture drops out during the takeoff."""
    env = deploy_env
    original, calls = env.source.obs, iter(range(100_000))

    def covered_markers() -> dict:
        obs = original()
        obs["is_outdated"] = next(calls) > 300
        return obs

    env.source.obs = covered_markers
    with pytest.raises(RuntimeError, match="outdated"):
        deploy_a3.fly(env.rows, env.cfg, 1, False, Path("design.csv"))
    meta = flight_meta(env.folder)
    assert "outdated" in meta["abort_reason"] and "outdated" in meta["error"]
    assert [call[0] for call in env.sd_calls].count("stop") == 2
    for drone in (env.victim, env.source):
        names = drone.names()
        assert names.count("land") == 1  # Landed, not dropped
        assert names.index("land") < names.index("close")
        assert names[-1] == "close"
