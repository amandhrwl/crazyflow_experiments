"""Plan, limits and runner of the A1 sweep, without a drone."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import threading  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.downwash import a1_sweep  # noqa: E402
from crazyflow_experiments.downwash.a1_stand import (  # noqa: E402
    ThrustPredictor,
    check_limits,
    load_config,
    plan_all,
    plan_single,
    thrust_cap,
    uniform,
)
from crazyflow_experiments.downwash.a1_sweep import Stand, run_block  # noqa: E402

CONFIG = load_config()
MASS = 0.050
G = CONFIG["g"]


def test_block_all_tares_at_start_middle_and_end_and_repeats_every_level():
    """Tare first, between the repeats and last; every level in the ramp and once per repeat."""
    steps = plan_all(CONFIG)
    kinds = [s.kind for s in steps]
    assert kinds[0] == "tare" and kinds[-1] == "tare" and kinds.count("tare") == 4
    assert kinds.count("pulse") == 2
    levels = CONFIG["sweep"]["cmd_levels"]
    ramp = [s.commands[0] for s in steps if s.kind == "ramp"]
    assert ramp == levels
    counts = Counter(s.commands[0] for s in steps if s.kind == "level")
    assert all(counts[c] == CONFIG["sweep"]["repeats"] for c in levels)
    # A center level after every center_every levels of a repeat
    run = 0
    for step in steps:
        if step.kind == "level":
            run += 1
        elif step.kind in ("center", "tare"):
            assert run <= CONFIG["sweep"]["center_every"]
            run = 0
    assert plan_all(CONFIG) == steps  # Seeded


def test_single_block_idles_the_other_motors():
    """Each motor ramps alone from above idle, the others at idle_cmd, a tare before each."""
    steps = plan_single(CONFIG, MASS)
    idle = CONFIG["sweep"]["idle_cmd"]
    for motor in range(1, 5):
        own = [s for s in steps if s.motor == motor and s.kind == "level"]
        assert len(own) == CONFIG["sweep"]["single_motor_levels"]
        commands = [s.commands[motor - 1] for s in own]
        assert commands == sorted(commands) and commands[0] > idle
        assert all(c == idle for s in own for m, c in enumerate(s.commands, 1) if m != motor)


def test_resting_victim_caps_below_its_weight_and_clamped_at_twice_hover():
    """Resting: 0.8 m g; clamped (the stand): 2 m g; a filled-in load cell rating caps both."""
    assert CONFIG["victim"]["clamped"]
    resting = CONFIG | {"victim": CONFIG["victim"] | {"clamped": False}}
    cap, reason = thrust_cap(resting, MASS)
    assert cap == pytest.approx(0.8 * MASS * G) and "resting" in reason
    assert thrust_cap(CONFIG, MASS)[0] == pytest.approx(2 * MASS * G)
    rated = CONFIG | {"load_cell": CONFIG["load_cell"] | {"capacity_g": 50}}
    assert thrust_cap(rated, MASS) == pytest.approx((0.8 * 0.05 * G, "load cell capacity"))
    with pytest.raises(ValueError, match="capacity_g"):
        check_limits(CONFIG | {"load_cell": CONFIG["load_cell"] | {"capacity_g": 0}})
    check_limits(rated)


def test_predictor_uses_measurements_inside_and_never_drops_below_the_linear_map_beyond():
    """Linear map until measured; interpolation inside the measured commands; margin beyond."""
    predictor = ThrustPredictor(CONFIG["victim"]["model"])
    assert predictor.total(uniform(32000)) == pytest.approx(4 * predictor.linear(32000))
    for command in (12000, 16000, 20000):
        predictor.add(command, 4 * 1e-10 * command**2)  # Quadratic, below the linear map
    assert predictor.total(uniform(16000)) == pytest.approx(4e-10 * 16000**2)
    assert predictor.total(uniform(14000)) < 4 * predictor.linear(14000)
    beyond = predictor.per_motor(np.array([40000]))[0]
    assert beyond >= predictor.linear(40000)


class FakeStand:
    """The parts of `Stand` the runner uses; thrust = 4e-10 cmd^2 per motor."""

    def __init__(self, cap: float, fail_at: int | None = None):
        """A stand whose thrust follows its commands at once."""
        self.config = CONFIG | {
            "sweep": CONFIG["sweep"]
            | {"hold_s": 0.02, "discard_s": 0.005, "tare_s": 0.02, "idle_s": 0.01}
        }
        self.cap, self.fail_at, self.tare = cap, fail_at, None
        self.commands, self.history = None, []

    @property
    def on(self) -> bool:
        """Whether the motors run."""
        return self.commands is not None

    def set(self, commands: tuple[int, int, int, int]):
        """Run the motors at their commands."""
        self.commands = commands
        self.history.append(commands)

    def stop(self):
        """Stop the motors."""
        self.commands = None
        self.history.append(None)

    def weight(self, seconds: float) -> float:
        """Load cell reading in g, zero with the motors off."""
        return 0.0 if self.commands is None else self.thrust() / G * 1e3

    def thrust(self, seconds: float = 0.2) -> float:
        """Thrust of the commands in N."""
        return 0.0 if self.commands is None else sum(4e-10 * c**2 for c in self.commands)

    def violation(self) -> str | None:
        """A violation once `fail_at` commands were sent."""
        if self.fail_at is not None and len(self.history) >= self.fail_at:
            return "fake violation"
        return None

    def screen(self, *args: object) -> str:
        """No live line."""
        return ""


def short_plan(stand: FakeStand) -> list:
    """Block all with the fake stand's short durations."""
    return plan_all(stand.config)


def test_runner_skips_levels_over_the_cap_ends_the_ramp_and_lowers_the_center(
    monkeypatch: pytest.MonkeyPatch,
):
    """Only levels predicted under the cap run, the ramp stops at the first one over it."""
    monkeypatch.setattr(a1_sweep, "LOOP_PERIOD", 0.001)
    cap = 0.8 * MASS * G  # 0.392 N: 4e-10 * 4 * c^2 allows up to about 15600 ... 31300
    stand = FakeStand(cap)
    predictor = ThrustPredictor(CONFIG["victim"]["model"])
    rows, abort = run_block(stand, short_plan(stand), predictor, np.zeros(3), threading.Event())
    assert abort is None
    ran = [r for r in rows if not r["skipped"] and r["kind"] in ("ramp", "level", "center")]
    assert ran and all(r["measured_N"] <= cap for r in ran)
    ramp = [r for r in rows if r["kind"] == "ramp"]
    first_skip = next(i for i, r in enumerate(ramp) if r["skipped"])
    assert all(r["skipped"] for r in ramp[first_skip:])
    centers = [r for r in rows if r["kind"] == "center" and not r["skipped"]]
    assert centers and all(4e-10 * 4 * r["m1"] ** 2 <= cap for r in centers)
    # Motors off for every tare, and stopped at the end
    assert stand.history[-1] is None


def test_runner_stops_on_a_violation(monkeypatch: pytest.MonkeyPatch):
    """A violation ends the block at once and names itself."""
    monkeypatch.setattr(a1_sweep, "LOOP_PERIOD", 0.001)
    stand = FakeStand(1.0, fail_at=3)
    predictor = ThrustPredictor(CONFIG["victim"]["model"])
    rows, abort = run_block(stand, short_plan(stand), predictor, np.zeros(3), threading.Event())
    assert abort == "fake violation" and len(rows) < len(short_plan(stand))


class FakeLog:
    """Radio values for `Stand.violation`."""

    def __init__(self):
        """Healthy values: armed, 4 V, 12000 rpm, no load."""
        self.values = {"pm.vbat": 4.0, "supervisor.info": 0x000B, "loadcell.weight": 0.0}
        self.values |= {f"motor.m{i}_rpm": 12000 for i in range(1, 5)}

    def latest(self, name: str) -> float:
        """The value of a variable."""
        return self.values[name]

    def recent(self, name: str, seconds: float) -> np.ndarray:
        """The value of a variable, as the only recent one."""
        return np.array([self.values[name]])


class FakeVictim:
    """A victim with a fake radio log and a link whose loss can be reported."""

    def __init__(self):
        """A victim whose link is up."""
        self.radio_log = FakeLog()
        self.cf = self
        self.connection_lost = self
        self.callbacks = []

    def add_callback(self, callback: object):
        """Register the link-lost callback."""
        self.callbacks.append(callback)


def test_stand_violations(monkeypatch: pytest.MonkeyPatch):
    """Thrust over the cap, invalid rpm for too long, a disarm, a low battery and a lost link."""
    victim = FakeVictim()
    stand = Stand(victim, CONFIG, cap=0.3)
    stand.tare, stand.commands = 0.0, uniform(20000)
    log = victim.radio_log.values
    assert stand.violation() is None
    log["loadcell.weight"] = 0.35 / G * 1e3
    assert "over the cap" in stand.violation()
    log["loadcell.weight"] = 0.0
    log["motor.m2_rpm"] = 65535
    assert stand.violation() is None  # Not yet for long
    monkeypatch.setattr(a1_sweep.time, "monotonic", lambda: 1e9)
    assert "motor 2 rpm invalid" in stand.violation()
    log["motor.m2_rpm"] = 12000
    log["supervisor.info"] = 0x0001
    assert "disarmed" in stand.violation()
    log["supervisor.info"] = 0x000B
    log["pm.vbat"] = 3.3
    assert "battery" in stand.violation()
    log["pm.vbat"] = 4.0
    victim.callbacks[0]("uri", "gone")
    assert stand.violation() == "radio link lost"
