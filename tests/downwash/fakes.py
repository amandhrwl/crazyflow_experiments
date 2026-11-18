"""Fake clock and fake drones for running the downwash flight loop without hardware."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from collections.abc import Callable

WALL_OFFSET = 1.7e9  # s, wall time of the fake clock's zero


class FakeClock:
    """`perf_counter`, `time` and `sleep` of the `time` module, on a clock only sleeping moves."""

    def __init__(self):
        """Start the clock at 100 s."""
        self.now = 100.0

    def perf_counter(self) -> float:
        """Seconds of the fake clock."""
        return self.now

    def time(self) -> float:
        """Wall time of the fake clock."""
        return WALL_OFFSET + self.now

    def sleep(self, duration: float):
        """Advance the clock."""
        self.now += duration


class FakeDrone:
    """A drone that is always where its last setpoint put it; its cflib calls are recorded."""

    def __init__(self, drone_id: int, pos: np.ndarray, tilt: float = 0.0, armed: bool = False):
        """Place the drone.

        Args:
            drone_id: Drone ID.
            pos: Position as (3,).
            tilt: Roll in degrees.
            armed: Start armed, as `Drone._reset` leaves a drone that just connected, instead of
                disarmed, as `pair.connect_pair` leaves it.
        """
        self.id, self.name = drone_id, f"cf{drone_id}"
        self.pos = np.asarray(pos, dtype=float)
        self.quat = R.from_euler("x", tilt, degrees=True).as_quat()
        self.outdated = False
        self.frozen = False  # Ignore setpoints, as a drone that stopped tracking
        self.calls: list[tuple] = []
        self.armed = armed
        # supervisor.info once armed (armed, can fly) and disarmed (can be armed), as the firmware
        # reports them; set them to make the arming fail
        self.info_armed, self.info_disarmed = 0x000A, 0x0001
        # The other firmware log variables as `usd.read_once` reads them
        self.log_values = {"pm.vbat": 4.0, **{f"motor.m{i}_rpm": 2000 for i in range(1, 5)}}
        record = self._record
        self.cf = SimpleNamespace(
            commander=SimpleNamespace(
                send_setpoint=record("send_setpoint"),
                send_stop_setpoint=record("send_stop_setpoint"),
                send_notify_setpoint_stop=record("send_notify_setpoint_stop"),
            ),
            param=SimpleNamespace(set_value=record("set_value")),
            supervisor=SimpleNamespace(send_arming_request=self._arm),
            high_level_commander=SimpleNamespace(
                land=record("land"), go_to=record("go_to"), stop=record("stop")
            ),
            send_packet=record("packet"),
        )

    def _record(self, name: str) -> Callable:
        return lambda *args: self.calls.append((name, args))

    def _arm(self, do_arm: bool):
        """Record the arming request and follow it."""
        self.calls.append(("arm", (do_arm,)))
        self.armed = do_arm

    def log_value(self, name: str) -> float:
        """A firmware log variable as `usd.read_once` reads it."""
        if name == "supervisor.info":
            return self.info_armed if self.armed else self.info_disarmed
        return self.log_values[name]

    def names(self) -> list[str]:
        """Names of the recorded calls in order."""
        return [call[0] for call in self.calls]

    def obs(self) -> dict:
        """Observation in the layout of `Drone.obs`."""
        zeros = np.zeros(3)
        return {
            "pos": self.pos.copy(),
            "quat": self.quat.copy(),
            "vel": zeros,
            "ang_vel": zeros,
            "pos_raw": self.pos.copy(),
            "quat_raw": self.quat.copy(),
            "is_outdated": self.outdated,
            "rpy": zeros,
        }

    def send_state(self, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, yaw: float):
        """Fly to the setpoint at once, unless frozen."""
        self.calls.append(("send_state", (np.array(pos),)))
        if not self.frozen:
            self.pos = np.array(pos, dtype=float)

    def update_estimators(self):
        """Record the pose update."""
        self.calls.append(("update_estimators", ()))

    def close(self):
        """Record the emergency stop and disconnect of `Drone.close`."""
        self.calls.append(("close", ()))


class FakeConsole:
    """Console capture that reports one SD file."""

    def __init__(self, file: str):
        """Report `file` as the file the card opened."""
        self.lines = [f"uSD: Logging to: {file}"]

    def files(self) -> list[str]:
        """The reported file."""
        return [self.lines[0].split()[-1]]

    def skipped_variables(self) -> list[str]:
        """No skipped variables."""
        return []


def fake_read_once(drone: FakeDrone, names: list[str], timeout: float = 2.0) -> dict:
    """`usd.read_once` answered from the fake drone's `log_values`."""
    return {name: drone.log_value(name) for name in names}


def patch_pair(monkeypatch: object, pair: object, clock: FakeClock):
    """Run the pair module on the fake clock with ROS always up and log reads faked."""
    monkeypatch.setattr(pair.usd, "read_once", fake_read_once)
    monkeypatch.setattr(pair, "time", clock)
    monkeypatch.setattr(pair, "sleep_until_next_cycle", lambda tick, freq: clock.sleep(1 / freq))
    monkeypatch.setattr(pair, "rclpy", SimpleNamespace(ok=lambda: True))
