"""Flight plan and safety checks of the load cell test, without a drone."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import threading  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import toml  # noqa: E402

from crazyflow_experiments.load_cell import CONFIG_PATH, deploy  # noqa: E402
from crazyflow_experiments.load_cell.deploy import plan_flight, settle, violation  # noqa: E402

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
