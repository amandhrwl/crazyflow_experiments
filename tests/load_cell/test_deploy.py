"""Flight plan and safety checks of the load cell test, without a drone."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import threading  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import toml  # noqa: E402

from crazyflow_experiments.load_cell import CONFIG_PATH, deploy  # noqa: E402
from crazyflow_experiments.load_cell.deploy import (  # noqa: E402
    plan_flight,
    settle,
    spin_pwm,
    violation,
)
from crazyflow_experiments.load_cell.radio import ARMED, CAN_FLY, Victim  # noqa: E402

CONFIG = toml.load(CONFIG_PATH)
VICTIM = np.array([0.2, 0.1, 0.5])


def test_holds_sit_beside_and_over_the_victim():
    """The source holds aside, then over the victim at height_sep, then aside again."""
    flight = CONFIG["flight"]
    start = VICTIM + np.array([*flight["aside"], -0.5])
    plan = plan_flight(CONFIG, VICTIM, start)
    np.testing.assert_allclose(plan.over, VICTIM + [0.0, 0.0, flight["height_sep"]])
    np.testing.assert_allclose(plan.aside, plan.over + [*flight["aside"], 0.0])
    phases = {name: (t0, t1) for name, t0, t1 in plan.phases}
    assert phases["over"][1] - phases["over"][0] == pytest.approx(flight["over_time"])
    assert phases["move_in"][1] - phases["move_in"][0] == pytest.approx(flight["move_time"])
    for name, at in (("aside_1", plan.aside), ("over", plan.over), ("aside_2", plan.aside)):
        middle = sum(phases[name]) / 2
        np.testing.assert_allclose(plan.trajectory.pos(middle), at, atol=1e-6)
    assert plan.trajectory.duration == pytest.approx(phases["aside_2"][1])


def test_a_takeoff_path_across_the_victim_is_refused():
    """A source standing on the far side would climb over the victim to its hold, so no flight."""
    start = VICTIM - np.array([*CONFIG["flight"]["aside"], 0.5])
    with pytest.raises(ValueError, match="takeoff path"):
        plan_flight(CONFIG, VICTIM, start)


def test_violations_stop_the_source():
    """Straying from the setpoint, or sinking toward the victim while over it, stops the flight."""
    safety = CONFIG["safety"]
    limit = safety["max_tracking_error"]
    over = VICTIM + [0.0, 0.0, CONFIG["flight"]["height_sep"]]
    assert violation(over, over, VICTIM, safety, limit) is None
    assert "setpoint" in violation(over + [0.3, 0.0, 0.0], over, VICTIM, safety, limit)
    low = VICTIM + [0.0, 0.0, 0.3]
    assert "above the victim" in violation(low, low, VICTIM, safety, limit)
    beside = VICTIM + [1.0, 0.0, 0.0]
    assert violation(beside, beside, VICTIM, safety, limit) is None


class FakeSource:
    """A source that closes `step` of its distance to the setpoint on every tick."""

    def __init__(self, pos: np.ndarray, step: float):
        """Start at `pos`."""
        self.pos, self.step = np.array(pos, dtype=float), step

    def obs(self) -> dict:
        """The observation of `Drone.obs`, exact and never outdated."""
        pos = self.pos.copy()
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        return {"pos": pos, "vel": np.zeros(3), "quat": quat, "pos_raw": pos, "is_outdated": False}

    def send_action(self, action: np.ndarray):
        """Fly toward the setpoint."""
        self.pos += self.step * (action[:3] - self.pos)

    def update_estimators(self):
        """Nothing to update."""


@pytest.fixture
def fast_settle(monkeypatch: pytest.MonkeyPatch) -> dict:
    """The config with a fast control loop and a short settle, and ROS running."""
    monkeypatch.setattr(deploy.rclpy, "ok", lambda: True)
    flight = CONFIG["flight"] | {"ctrl_freq": 1000, "settle_time": 0.05, "settle_timeout": 2.0}
    return CONFIG | {"flight": flight}


def test_settle_holds_aside_until_the_source_has_arrived(fast_settle: dict):
    """The source left 0.26 m off its hold by the takeoff is not stopped, the holds wait for it."""
    start = VICTIM + np.array([*CONFIG["flight"]["aside"], -0.5])
    plan = plan_flight(fast_settle, VICTIM, start)
    source = FakeSource(plan.aside + [0.0, 0.2, -0.16], step=0.02)
    records, reason = settle(source, plan, fast_settle, threading.Event())
    assert reason is None
    errors = [np.linalg.norm(r["pos"] - plan.aside) for r in records]
    assert errors[0] > CONFIG["safety"]["max_tracking_error"]
    assert errors[-1] < CONFIG["flight"]["settle_tolerance"]


def test_settle_lands_a_source_that_leaves_its_clearance(fast_settle: dict):
    """A source that drifts off its hold by more than the clearance is landed before the holds."""
    start = VICTIM + np.array([*CONFIG["flight"]["aside"], -0.5])
    plan = plan_flight(fast_settle, VICTIM, start)
    source = FakeSource(plan.aside + [0.0, 0.6, 0.0], step=0.0)
    _, reason = settle(source, plan, fast_settle, threading.Event())
    assert "setpoint" in reason


def test_spin_command_is_the_thrust_fraction_of_the_victims_weight():
    """Half the weight over four motors through the linear thrust map, and no lift-off fraction."""
    params = deploy.drone_params(CONFIG["victim"]["model"])
    config = CONFIG | {"spin": CONFIG["spin"] | {"thrust_fraction": 0.5}}
    thrust = spin_pwm(config) / params["pwm_max"] * params["thrust_max"] * 4
    assert thrust == pytest.approx(0.5 * params["mass"] * deploy.G, rel=1e-4)
    with pytest.raises(ValueError, match="lift"):
        spin_pwm(CONFIG | {"spin": CONFIG["spin"] | {"thrust_fraction": 1.0}})


class FakeLink:
    """The parts of a cflib link the spinning victim uses, with a supervisor that arms at once."""

    def __init__(self, arms: bool):
        """Arm on request if `arms`, as with every motor idling in its rpm range."""
        self.params: dict[str, float] = {}
        self.setpoints = 0
        self.info = 0x0001
        self.arms = arms
        self.param = self
        self.commander = self
        self.supervisor = self

    def set_value(self, name: str, value: float):
        """Write a parameter."""
        self.params[name] = value

    def get_value(self, name: str) -> float:
        """Read a parameter."""
        return self.params[name]

    def send_setpoint(self, roll: float, pitch: float, yawrate: float, thrust: int):
        """Count the setpoints that feed the watchdog."""
        self.setpoints += 1

    def send_arming_request(self, arm: bool):
        """Arm or disarm, the supervisor reports it in supervisor.info."""
        self.info = (ARMED | CAN_FLY) if arm and self.arms else 0x0001

    def latest(self, variable: str) -> int:
        """The radio log's last supervisor.info."""
        return self.info


def fake_victim(link: FakeLink) -> Victim:
    """A Victim on the fake link, without connecting."""
    victim = Victim.__new__(Victim)
    victim.name, victim.cf, victim.radio_log = "cf4", link, link
    victim._feeding, victim._stop_feeding = None, threading.Event()
    return victim


def test_spinning_victim_arms_feeds_the_watchdog_and_disarms(monkeypatch: pytest.MonkeyPatch):
    """The motors get the command once armed, setpoints flow until the stop, which disarms."""
    monkeypatch.setattr("crazyflow_experiments.load_cell.radio.STATUS_PERIOD_MS", 50)
    link = FakeLink(arms=True)
    victim = fake_victim(link)
    victim.spin(17000)
    assert [link.params[f"motorPowerSet.m{i}"] for i in range(1, 5)] == [17000] * 4
    assert link.params["motorPowerSet.enable"] == 1 and victim.spinning
    victim.stop_spin()
    assert link.info & ARMED == 0 and link.params["motorPowerSet.enable"] == 0
    assert link.setpoints > 1 and not victim.spinning


def test_a_victim_that_does_not_arm_never_gets_a_motor_command(monkeypatch: pytest.MonkeyPatch):
    """A failed arming, e.g. a motor out of its idle rpm range, stops before the motors are set."""
    monkeypatch.setattr("crazyflow_experiments.load_cell.radio.ARM_TIMEOUT", 0.3)
    monkeypatch.setattr("crazyflow_experiments.load_cell.radio.STATUS_PERIOD_MS", 50)
    link = FakeLink(arms=False)
    victim = fake_victim(link)
    with pytest.raises(RuntimeError, match="did not arm"):
        victim.spin(17000)
    assert "motorPowerSet.m1" not in link.params and not victim.spinning
