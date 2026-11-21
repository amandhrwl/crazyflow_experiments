"""Two Crazyflies flown from one control loop: the victim hovers, the source flies the legs.

After takeoff a Crazyflie needs a continuous setpoint stream, otherwise the commander watchdog
levels it and cuts the motors after about 2 s. Both drones are therefore served by one loop at all
times, from the joint takeoff through the legs to the joint landing; the repo's single-drone
takeoff, landing and return home block and cannot be used for the pair. Only the public `Drone`
methods and the cflib `Crazyflie` in `drone.cf` are used, so `deploy/` stays unchanged.

The source may be flown alone (`victim=None`), with the victim on the ground and unconnected.
"""

from __future__ import annotations

import logging
import math
import signal
import struct
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import rclpy
from cflib.crazyflie import Localization
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from crazyflow.control import Control
from scipy.spatial.transform import Rotation as R

from crazyflow_experiments.control.loop import sleep_until_next_cycle
from crazyflow_experiments.deploy.experiment import connect
from crazyflow_experiments.downwash import usd
from crazyflow_experiments.downwash.design import drag_per_speed
from crazyflow_experiments.downwash.timeline import (
    CEILING_MARGIN,
    build_flight,
    takeoff_timelines,
    victim_hover_point,
)
from crazyflow_experiments.utils.ros import calibrate_async

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from types import FrameType

    from numpy.typing import NDArray

    from crazyflow_experiments.deploy.crazyflie import Drone
    from crazyflow_experiments.downwash.timeline import Timeline

logger = logging.getLogger(__name__)

SPIN_UP_TIME = 1.5  # s, as in Drone._reset: the motors spin up after arming
# The firmware disarms a drone whose motors do not all idle in its arming rpm range within 2 s,
# and nothing reports it: the drone then stays on the ground while the other one takes off
# s into the takeoff, 2.1 s after the arming request, when both drones' state is read: the firmware
# has decided by then (2 s spin-up timeout), and setpoints keep streaming while the read runs
ARMING_CHECK_AT = 0.6
ARMING_CHECK_TIMEOUT = 3.0  # s for that read to answer before the takeoff is ended
DISARM_TIMEOUT = 1.0  # s for a disarmed drone to reach the one state that accepts an arming request
DISARM_POLL = 0.1  # s between the reads of its state
SUPERVISOR_CAN_ARM = 0x0001  # Bits of supervisor.info, see the firmware's supervisor.c
SUPERVISOR_ARMED = 0x0002
SUPERVISOR_CAN_FLY = 0x0008
SUPERVISOR_TUMBLED = 0x0020
SUPERVISOR_LOCKED = 0x0040
CALIBRATION_DELAY = 0.5  # s of clean hover before the orientation calibration
CALIBRATION_TIMEOUT = 40.0  # s, the service call itself times out after 30 s
CLOSE_HEIGHT_SEP = 0.4  # m, legs at or below it need the victim's battery check
VBAT_LEAD = 6.0  # s before a close leg starts, when the victim's battery is read
VBAT_DECIDE = 0.2  # s before a close leg starts, when the leg is flown or skipped
CLIMB = 0.3  # m the source climbs before landing when it is above the victim
CLIMB_TIME = 1.0  # s
LAND_HEIGHT = 0.05  # m
LAND_DURATION = 3.0  # s
LAND_WAIT = 3.5  # s of pose updates after the land command
MAX_TILT = 90.0  # deg, a landing drone tilted further is killed, as in Drone.return_to_start
AIR_STILL_STD = 0.01  # m, victim position spread in the pre-pass hover that counts as still air
F0_SPEED_TOL = 0.05  # Relative cruise speed error of the source that needs a look (§0)
F0_TRACKING_TOL = 0.10  # m, path error of the source that needs a look (§0)
# s of commanded path a drone is compared with: the onboard controller follows full-state setpoints
# 0.17 s late (F0), which alone puts the source 0.24 m behind its setpoint at 1.4 m/s
TRACKING_WINDOW = 0.3
OBS_KEYS = ("pos", "quat", "vel", "ang_vel", "pos_raw", "quat_raw")

_operator_stop = threading.Event()


@contextmanager
def ctrl_c_ends_flight() -> Iterator[None]:
    """Turn the first Ctrl+C into a stop that `takeoff_pair` and `run_flight` check every tick.

    Left alone, Ctrl+C raises KeyboardInterrupt wherever the loop happens to be, before rclpy's own
    handler ends `rclpy.ok()`, and the records of every leg flown are lost with `run_flight` (A3 F1,
    19 Nov). Inside this context the first Ctrl+C only sets the stop, so the loop returns its records
    and the caller lands both; it also puts Python's handler back, so a second Ctrl+C raises as usual.
    """
    _operator_stop.clear()

    def on_sigint(signum: int, frame: FrameType | None):
        _operator_stop.set()
        signal.signal(signal.SIGINT, signal.default_int_handler)
        logger.warning("Ctrl+C: ending the flight, both land. Ctrl+C again raises")

    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous or signal.default_int_handler)
        _operator_stop.clear()


def operator_stopped() -> bool:
    """Whether the operator pressed Ctrl+C, see `ctrl_c_ends_flight`, or ROS shut down."""
    return _operator_stop.is_set() or not rclpy.ok()


@dataclass(frozen=True)
class SafetyLimits:
    """Limits checked every tick of the flight, from the `[safety]` config.

    Attributes:
        arena_min: Lower corner of the flight volume as (3,).
        arena_max: Upper corner of the flight volume as (3,).
        min_vertical_sep: Height the source keeps above the victim while horizontally close.
        xy_guard: Horizontal distance below which the vertical separation is enforced.
        max_tracking_error: Distance of the measured position from the commanded path of the last
            `TRACKING_WINDOW`, either drone.
        victim_hover_z: Height the victim hovers at.
        victim_max_drop: Drop of the victim below its hover height that aborts.
    """

    arena_min: NDArray
    arena_max: NDArray
    min_vertical_sep: float
    xy_guard: float
    max_tracking_error: float
    victim_hover_z: float
    victim_max_drop: float

    @classmethod
    def from_config(cls, cfg: Mapping) -> SafetyLimits:
        """Limits of the experiment config."""
        safety = cfg["safety"]
        return cls(
            arena_min=np.asarray(safety["arena_min"], dtype=float),
            arena_max=np.asarray(safety["arena_max"], dtype=float),
            min_vertical_sep=safety["min_vertical_sep"],
            xy_guard=safety["xy_guard"],
            max_tracking_error=safety["max_tracking_error"],
            victim_hover_z=cfg["drones"]["victim"]["hover_z"],
            victim_max_drop=safety["victim_max_drop"],
        )


def path_error(pos: NDArray, cmds: NDArray) -> float:
    """Distance from the position (3,) to the nearest of the commanded positions (N, 3) or (3,)."""
    cmds = np.atleast_2d(np.asarray(cmds, dtype=float))
    return float(np.min(np.linalg.norm(cmds - np.asarray(pos, dtype=float), axis=1)))


def path_errors(pos: NDArray, cmd: NDArray, window: int) -> NDArray:
    """`path_error` of every tick (T,) against the commands of the `window` ticks up to it.

    Args:
        pos: Measured positions as (T, 3).
        cmd: Commanded positions as (T, 3), one per tick.
        window: Number of earlier ticks whose commands count, besides the tick's own.
    """
    errors = np.linalg.norm(pos - cmd, axis=1)
    for lag in range(1, min(window, len(pos) - 1) + 1):
        errors[lag:] = np.minimum(errors[lag:], np.linalg.norm(pos[lag:] - cmd[:-lag], axis=1))
    return errors


def with_drag(
    cmd: tuple[NDArray, NDArray, NDArray], drag: float
) -> tuple[NDArray, NDArray, NDArray]:
    """The setpoint with the acceleration that balances the drag at its velocity added.

    Without it the onboard controller tilts against the drag only once the drone has fallen behind
    its setpoint: in F0 (18 Nov) the source sank from 1.4 to 0.95 m/s right after each ramp and
    held speed again 0.8 s later, 27 to 30 cm behind. Horizontal only, the part F0 measured.

    Args:
        cmd: Position, velocity and acceleration as (3,).
        drag: Deceleration per m/s of speed in 1/s, see `design.drag_per_speed`.
    """
    pos, vel, acc = cmd
    acc = np.array(acc, dtype=float)
    acc[:2] += drag * np.asarray(vel, dtype=float)[:2]
    return pos, vel, acc


def safety_violation(
    limits: SafetyLimits,
    obs_source: Mapping,
    cmd_source: NDArray,
    obs_victim: Mapping | None = None,
    cmd_victim: NDArray | None = None,
) -> str | None:
    """The first broken safety rule, or None.

    The tracking rule measures how far a drone is from its commanded path, the nearest of its
    recent setpoints, so a drone that follows late but on its path passes.

    Args:
        limits: The limits.
        obs_source: Observation of the source with `pos` and `is_outdated`.
        cmd_source: Commanded positions of the source over the last `TRACKING_WINDOW` as (N, 3),
            or only the current one as (3,).
        obs_victim: Observation of the victim, None when it does not fly.
        cmd_victim: Commanded positions of the victim as for the source, None when it does not
            fly.

    Returns:
        The rule and what broke it, or None when every rule holds.
    """
    drones = {"source": (obs_source, cmd_source)}
    if obs_victim is not None:
        drones["victim"] = (obs_victim, cmd_victim)
    for name, (obs, _) in drones.items():
        if obs["is_outdated"]:
            return f"mocap: the {name}'s measurement is outdated"
    for name, (obs, _) in drones.items():
        pos = np.asarray(obs["pos"])
        if np.any(pos < limits.arena_min) or np.any(pos > limits.arena_max):
            return f"arena: the {name} is at {np.round(pos, 3).tolist()}"
    if obs_victim is not None:
        p_src, p_vic = np.asarray(obs_source["pos"]), np.asarray(obs_victim["pos"])
        dxy = float(np.linalg.norm(p_src[:2] - p_vic[:2]))
        dz = float(p_src[2] - p_vic[2])
        if dxy < limits.xy_guard and dz < limits.min_vertical_sep:
            return f"separation: the source is {dz:.3f} m above the victim at {dxy:.3f} m"
    for name, (obs, cmd) in drones.items():
        error = path_error(obs["pos"], cmd)
        if error > limits.max_tracking_error:
            return f"tracking: the {name} is {error:.3f} m off its commanded path"
    if obs_victim is not None:
        z = float(np.asarray(obs_victim["pos"])[2])
        if z < limits.victim_hover_z - limits.victim_max_drop:
            return f"victim drop: the victim is at z = {z:.3f} m"
    return None


def drone_config(cfg: Mapping, role: str) -> dict:
    """Sub-config of one drone in the form `deploy.experiment.connect` reads."""
    drones = cfg["drones"]
    drone = drones[role]
    return {
        "drone": {
            "model": drones["model"],
            "id": drone["id"],
            "channel": drone["channel"],
            "pos_landing": [0.0, 0.0, 0.0],  # Unused: land_pair lands both where they are
            "pos_update_freq": drones["pos_update_freq"],
        }
    }


def connect_pair(cfg: Mapping, only_source: bool = False) -> tuple[Drone | None, Drone]:
    """Connect the victim, then the source, both on radio 0.

    Each connection power-cycles the drone, so its card's config.txt is read afresh, resets its
    estimator from the motion capture, arms it and spins the motors up. Each is disarmed again right
    away: the firmware locks an armed drone 2 s after its last setpoint (see `disarm`), and the
    victim would wait about 6 s for the source alone. `takeoff_pair` arms both. Each estimator is
    given the drag of the model, see `set_estimator_drag`.

    Args:
        cfg: The experiment config.
        only_source: Leave the victim unconnected.

    Returns:
        The victim, None with `only_source`, and the source.
    """
    if only_source:
        return None, _connect_disarmed(cfg, "source")
    victim = _connect_disarmed(cfg, "victim")
    connected = False
    try:
        source = _connect_disarmed(cfg, "source")
        connected = True
    finally:
        if not connected:  # Do not leave the victim connected when the source fails to connect
            victim.close()
    return victim, source


def _connect_disarmed(cfg: Mapping, role: str) -> Drone:
    drone = connect(drone_config(cfg, role), Control.state)
    disarm(drone)
    set_estimator_drag(drone, drag_per_speed(cfg["drones"]["model"]))
    return drone


def set_estimator_drag(drone: Drone, drag: float):
    """Give the onboard Kalman filter the horizontal drag of the drone, in 1/s.

    The firmware's default is no drag (`DRAG_B_X` 0 in platform_defaults.h), so in flight the
    filter takes the forward tilt that holds a cruise against drag for an acceleration. In F0
    (18 Nov) its velocity ran 0.27 m/s above the true 1.4 m/s between the 20 Hz pose updates, and
    the onboard Mellinger braked on it (`kd_xy` is 0.2 N per m/s, 4.4 m/s^2 on this drone): the
    source sank to 1.0 m/s after each ramp and cruised 20 to 30 cm behind its setpoint.
    """
    for axis in ("x", "y"):
        drone.cf.param.set_value(f"kalman.dragB_{axis}", drag)


def disarm(drone: Drone):
    """Disarm the drone `Drone._reset` armed and wait until it accepts an arming request again.

    `_reset` ends on one setpoint above idle thrust, so the firmware's supervisor counts the drone
    as flying, and its commander watchdog moves it to WarningLevelOut 0.5 s after that setpoint.
    WarningLevelOut ignores a disarm request and locks the drone 2 s after the setpoint, and an
    arming request is accepted only in PreFlChecksPassed, so `rearm` would be dropped silently. A
    zero setpoint first returns the drone to Flying, where the disarm request takes it through
    Reset to PreFlChecksPassed within a few supervisor ticks (25 Hz).

    Raises:
        RuntimeError: The drone cannot be armed again `DISARM_TIMEOUT` after the request.
    """
    drone.cf.commander.send_setpoint(0, 0, 0, 0)  # Back from WarningLevelOut, motors to idle
    drone.cf.supervisor.send_arming_request(False)
    t_end = time.perf_counter() + DISARM_TIMEOUT
    while not (info := _supervisor_info(drone)) & SUPERVISOR_CAN_ARM:
        if time.perf_counter() > t_end:
            raise RuntimeError(f"Disarming after the connection: {_cannot_arm(drone.name, info)}")
        time.sleep(DISARM_POLL)


def _supervisor_info(drone: Drone) -> int:
    """The drone's `supervisor.info`, read once."""
    return int(usd.read_once(drone, ["supervisor.info"])["supervisor.info"])


def _cannot_arm(name: str, info: int) -> str:
    """Why a drone with this `supervisor.info` would drop an arming request."""
    if info & SUPERVISOR_LOCKED:
        reason = "it is locked, reboot it"
    elif info & SUPERVISOR_ARMED:
        reason = "it is still armed"
    else:
        reason = "its preflight checks do not pass"
    return f"{name} cannot be armed (supervisor.info 0x{info:04x}): {reason}"


def rearm(drone: Drone):
    """Repeat the arming of `Drone._reset` for a drone that `connect_pair` disarmed."""
    drone.cf.commander.send_setpoint(0, 0, 0, 0)  # Release the commander's thrust lock
    drone.cf.supervisor.send_arming_request(True)  # Arm the brushless drone
    drone.cf.commander.send_setpoint(0.0, 0.0, 0.0, 7000)  # Minimum thrust spins the motors up


def _arming_failure(name: str, sample: Mapping) -> str:
    """Why a drone that cannot fly cannot, from its `supervisor.info`."""
    info = int(sample["supervisor.info"])
    if info & SUPERVISOR_LOCKED:
        reason = "it is locked, reboot it"
    elif info & SUPERVISOR_TUMBLED:
        reason = "it thinks it is tumbled, check that it stands level"
    else:
        reason = (
            "it disarmed, most likely a motor failed the arming rpm check; its console line"
            " names the motor, see the end of this run's output"
        )
    return f"the {name} cannot fly (supervisor.info 0x{info:04x}): {reason}"


class _ArmingCheck:
    """One background read of `supervisor.info` from every drone."""

    def __init__(self, drones: Mapping[str, Drone]):
        self.samples: dict[str, dict] = {}
        self.done = threading.Event()
        threading.Thread(target=self._read, args=(dict(drones),), daemon=True).start()

    def _read(self, drones: dict[str, Drone]):
        self.samples = {
            name: usd.read_once(drone, ["supervisor.info"]) for name, drone in drones.items()
        }
        self.done.set()  # Never set on a failure


def emergency_stop(drone: Drone):
    """Stop the motors at once, with the packet `Drone._emergency_stop` sends."""
    pk = CRTPPacket()
    pk.port = CRTPPort.LOCALIZATION
    pk.channel = Localization.GENERIC_CH
    pk.data = struct.pack("<B", Localization.EMERGENCY_STOP)
    drone.cf.send_packet(pk)


def _flying(victim: Drone | None, source: Drone) -> dict[str, Drone]:
    return {"victim": victim, "source": source} if victim is not None else {"source": source}


def _snapshot(obs: Mapping) -> dict:
    """Copy of the recorded observation fields."""
    snapshot = {key: np.array(obs[key], dtype=float) for key in OBS_KEYS}
    snapshot["is_outdated"] = bool(obs["is_outdated"])
    return snapshot


def _drone_record(
    obs: Mapping, cmd: tuple[NDArray, NDArray, NDArray], phase: str, leg: int
) -> dict:
    """Record of one drone at one tick; `cmd` is exactly what was passed to `send_state`."""
    return {
        "phase": phase,
        "leg": leg,
        # The position in mm as cflib packs it, int() of the value times 1000 in its own dtype, so
        # the conversion can match it to the onboard ctrltarget echo exactly
        "cmd_mm": [int(value * 1000) for value in cmd[0]],
        "cmd_pos": np.array(cmd[0], dtype=float),
        "cmd_vel": np.array(cmd[1], dtype=float),
        "cmd_acc": np.array(cmd[2], dtype=float),
        **_snapshot(obs),
    }


@dataclass
class Takeoff:
    """Result of the joint takeoff.

    Attributes:
        records: One record per tick, as in `run_flight`, phase "takeoff" or "calibrate".
        error: Why the flight must not continue, or None.
    """

    records: list[dict]
    error: str | None = None


def takeoff_pair(
    victim: Drone | None, source: Drone, targets: Mapping[str, NDArray], cfg: Mapping
) -> Takeoff:
    """Fly both drones up together, hover and calibrate both, streaming setpoints throughout.

    Each drone flies from wherever it stands to its target along `takeoff_timelines`, turning to
    yaw 0 during the climb, then hovers. After `hover_time` both orientations are calibrated in the
    background while the hover keeps streaming, until both calibrations have returned. Both drones
    are re-armed first (`connect_pair` disarmed them), once both read as able to be armed, since the
    firmware drops an arming request silently otherwise. `ARMING_CHECK_AT` into the climb their
    state is read again: a drone the firmware disarmed, because a motor failed its arming rpm check,
    ends the takeoff while both are still low, the source about 0.2 m up, instead of leaving it
    there.

    Args:
        victim: The victim, None when only the source flies.
        source: The source.
        targets: Target position as (3,) per role: the victim's hover point and the source's start
            point of the first leg.
        cfg: The experiment config.

    Returns:
        The records and an error, if the takeoff ended without both drones calibrated.
    """
    drones = _flying(victim, source)
    ctrl_freq, climb_time = cfg["control"]["ctrl_freq"], cfg["takeoff"]["duration"]
    drag = drag_per_speed(cfg["drones"]["model"])
    refused = [
        _cannot_arm(f"the {name}", info)
        for name, drone in drones.items()
        if not (info := _supervisor_info(drone)) & SUPERVISOR_CAN_ARM
    ]
    if refused:
        logger.error(f"Arming refused: {'; '.join(refused)}")
        return Takeoff([], f"arming: {'; '.join(refused)}")
    for drone in drones.values():
        rearm(drone)
    time.sleep(SPIN_UP_TIME)
    standing = {name: drone.obs() for name, drone in drones.items()}
    plans = takeoff_timelines({name: obs["pos"] for name, obs in standing.items()}, targets, cfg)
    yaw_start = {name: float(obs["rpy"][2]) for name, obs in standing.items()}
    n_ticks = round(plans["source"].duration * ctrl_freq) + 1
    records, calibrations, tick = [], None, 0
    arming: _ArmingCheck | None = None
    armed = False
    t0 = t_calibration = time.perf_counter()
    while not operator_stopped():
        t = time.perf_counter()
        obs = {name: drone.obs() for name, drone in drones.items()}
        for name, o in obs.items():
            if o["is_outdated"]:
                raise RuntimeError(f"Takeoff: the {name}'s measurement is outdated")
        t_wall = time.time()
        record = {"t_perf": t - t0, "t_wall": t_wall, "victim": None}
        t_plan = tick / ctrl_freq
        for name, drone in drones.items():
            cmd = with_drag(plans[name].state(t_plan), drag)
            yaw = yaw_start[name] * (1.0 - min(t_plan / climb_time, 1.0))  # To 0 in the climb
            drone.send_state(*cmd, yaw)
            phase = "takeoff" if calibrations is None else "calibrate"
            record[name] = _drone_record(obs[name], cmd, phase, -1)
        for drone in drones.values():
            drone.update_estimators()
        records.append(record)
        if not armed:  # Both drones armed, see ARMING_CHECK_AT
            if arming is None and t_plan >= ARMING_CHECK_AT:
                arming = _ArmingCheck(drones)
            if arming is not None and arming.done.is_set():
                failing = {
                    name: sample
                    for name, sample in arming.samples.items()
                    if not int(sample["supervisor.info"]) & SUPERVISOR_CAN_FLY
                }
                if failing:
                    failures = [_arming_failure(name, sample) for name, sample in failing.items()]
                    logger.error(f"Arming failed: {'; '.join(failures)}")
                    return Takeoff(records, f"arming: {'; '.join(failures)}")
                logger.info(f"Armed: {', '.join(drones)} can fly")
                armed = True
            elif arming is not None and t_plan > ARMING_CHECK_AT + ARMING_CHECK_TIMEOUT:
                return Takeoff(records, "arming: the drones' state could not be read")
        if tick == n_ticks - 1:
            if calibrations is None:
                logger.info("Takeoff finished, calibrating both drones")
                calibrations = {
                    name: calibrate_async(drone.id, delay=CALIBRATION_DELAY)
                    for name, drone in drones.items()
                }
                t_calibration = t
            elif all(event.is_set() for event in calibrations.values()):
                logger.info("Calibration complete")
                return Takeoff(records)
            elif t - t_calibration > CALIBRATION_TIMEOUT:
                return Takeoff(records, "calibration did not finish")
        tick = min(tick + 1, n_ticks - 1)
        sleep_until_next_cycle(t, ctrl_freq)
    return Takeoff(records, "operator: Ctrl+C during takeoff")


class _VbatReading:
    """One background read of the victim's battery voltage."""

    def __init__(self, drone: Drone):
        self.value: float | None = None
        self.done = threading.Event()
        threading.Thread(target=self._read, args=(drone,), daemon=True).start()

    def _read(self, drone: Drone):
        self.value = float(usd.read_once(drone, ["pm.vbat"])["pm.vbat"])  # Never set on a failure
        self.done.set()


@dataclass
class Flight:
    """Result of `run_flight`.

    Attributes:
        records: One record per tick: `t_perf` since the loop started, `t_wall`, and per drone
            its phase and leg, the commanded position, velocity and acceleration, and the
            observation (`pos, quat, vel, ang_vel, pos_raw, quat_raw, is_outdated`). The victim's
            entry is None when only the source flies.
        abort_reason: Why the loop ended early, or None when the flight was completed.
        skipped: Legs replaced by a hold because of the victim's battery.
        vbat: Battery voltage read before every close leg, None when the read failed.
        leg_stats: Per-leg numbers printed after each leg, see `leg_summary`.
        leg_flags: Flags of the legs, e.g. `air_not_still`.
        victim_timeline: The victim's timeline as flown.
        source_timeline: The source's timeline as flown.
    """

    records: list[dict] = field(default_factory=list)
    abort_reason: str | None = None
    skipped: list[int] = field(default_factory=list)
    vbat: dict[int, float | None] = field(default_factory=dict)
    leg_stats: dict[int, dict] = field(default_factory=dict)
    leg_flags: dict[int, list[str]] = field(default_factory=dict)
    victim_timeline: Timeline | None = None
    source_timeline: Timeline | None = None


def leg_summary(
    records: Sequence[Mapping], row: Mapping, times: Mapping, cfg: Mapping
) -> tuple[dict, list[str]]:
    """Numbers of one flown leg from its records, for the line printed after it.

    Args:
        records: Records of the flight; those of the leg are picked by the source's leg label.
        row: The leg's design row.
        times: The leg's boundaries in the source timeline, see `Timeline.leg_times`.
        cfg: The experiment config.

    Returns:
        The numbers and the leg's flags.
    """
    leg = int(row["leg"])
    recs = [r for r in records if r["source"]["leg"] == leg]
    if not recs:
        return {}, ["not_flown"]
    t = np.array([r["t_perf"] for r in recs])
    src_pos = np.array([r["source"]["pos"] for r in recs])
    src_raw = np.array([r["source"]["pos_raw"] for r in recs])
    src_cmd = np.array([r["source"]["cmd_pos"] for r in recs])
    window = round(TRACKING_WINDOW * cfg["control"]["ctrl_freq"])
    stats = {
        "tracking_error": float(np.max(np.linalg.norm(src_pos - src_cmd, axis=1))),
        "path_error": float(np.max(path_errors(src_pos, src_cmd, window))),
    }
    flags = []
    half_window = cfg["a3"]["cruise_half"] / float(row["speed"])
    t0, t1 = times["cross"] - half_window, times["cross"] + half_window
    if np.isfinite(times["cross"]) and t[0] <= t0 and t[-1] >= t1:
        i0, i1 = int(np.searchsorted(t, t0)), int(np.searchsorted(t, t1))
        distance = float(np.linalg.norm(src_raw[i1, :2] - src_raw[i0, :2]))
        stats["cruise_speed"] = distance / (t[i1] - t[i0])
        speed_error = abs(stats["cruise_speed"] - float(row["speed"])) / float(row["speed"])
        if speed_error > F0_SPEED_TOL:
            flags.append("cruise_speed_off")
    if stats["path_error"] > F0_TRACKING_TOL:
        flags.append("tracking_error_high")
    if recs[0]["victim"] is not None:
        p_vic = victim_hover_point(cfg)
        vic_pos = np.array([r["victim"]["pos"] for r in recs])
        stats["victim_min_dz"] = float(np.min(vic_pos[:, 2] - p_vic[2]))
        stats["victim_max_dxy"] = float(np.max(np.linalg.norm(vic_pos[:, :2] - p_vic[:2], axis=1)))
        hover = np.array([r["source"]["phase"] == "hover" for r in recs])
        if hover.sum() > 1:
            spread = vic_pos[hover] - vic_pos[hover].mean(axis=0)
            stats["victim_hover_std"] = float(np.sqrt(np.mean(np.sum(spread**2, axis=1))))
            if stats["victim_hover_std"] > AIR_STILL_STD:
                flags.append("air_not_still")
    return stats, flags


def _log_leg(row: Mapping, stats: Mapping, flags: Sequence[str], skipped: bool):
    head = f"Leg {int(row['leg']):2d} (Δz {row['height_sep']:.1f} m, v {row['speed']:.1f} m/s)"
    if skipped:
        logger.warning(f"{head}: skipped, the victim's battery was low or unread")
        return
    nan = float("nan")
    parts = [
        f"cruise {stats.get('cruise_speed', nan):.3f} m/s",
        f"source off path max {stats.get('path_error', nan) * 100:.1f} cm"
        f" (behind setpoint {stats.get('tracking_error', nan) * 100:.1f} cm)",
    ]
    if "victim_min_dz" in stats:
        parts += [
            f"victim min dz {stats['victim_min_dz'] * 100:+.1f} cm",
            f"max dxy {stats['victim_max_dxy'] * 100:.1f} cm",
            f"pre-pass std {stats.get('victim_hover_std', nan) * 100:.2f} cm",
        ]
    logger.info(f"{head}: {', '.join(parts)}" + (f"  [{', '.join(flags)}]" if flags else ""))


def run_flight(
    victim: Drone | None, source: Drone, rows: Sequence[Mapping], cfg: Mapping
) -> Flight:
    """Fly the legs of one flight from one loop at the control rate.

    Every tick: observe both drones, check the safety rules, send both setpoints, update both
    estimators, record, sleep. Any broken rule ends the loop with its reason, and so does Ctrl+C
    inside `ctrl_c_ends_flight`; the caller lands the pair afterwards.

    Before every leg at or below `CLOSE_HEIGHT_SEP` the victim's battery is read in the background;
    below `vbat_min_close_leg`, or without a reading, the leg is replaced by a hold of the same
    duration (see `build_flight`) and marked skipped.

    Args:
        victim: The victim, None when only the source flies.
        source: The source.
        rows: The design rows of the flight.
        cfg: The experiment config.

    Returns:
        The records, the abort reason and the per-leg results.
    """
    limits = SafetyLimits.from_config(cfg)
    drag = drag_per_speed(cfg["drones"]["model"])
    ctrl_freq = cfg["control"]["ctrl_freq"]
    rows = sorted(rows, key=lambda row: int(row["index_in_flight"]))
    flight = Flight()
    tl_vic, tl_src = build_flight(rows, cfg)
    leg_times = {int(r["leg"]): tl_src.leg_times(int(r["leg"])) for r in rows}
    close_legs = [
        int(r["leg"])
        for r in rows
        if victim is not None and r["height_sep"] <= CLOSE_HEIGHT_SEP + 1e-9
    ]
    readings: dict[int, _VbatReading] = {}
    window = round(TRACKING_WINDOW * ctrl_freq)  # Setpoints sent before this tick's, per drone
    recent = {"source": deque(maxlen=window), "victim": deque(maxlen=window)}
    first_record: dict[int, int] = {}  # Index of each leg's first record
    pending = list(rows)  # Legs whose summary is not printed yet
    zero_yaw = 0.0
    t0 = time.perf_counter()
    while True:
        if operator_stopped():
            flight.abort_reason = "operator: Ctrl+C"
            break
        tick = time.perf_counter()
        t = tick - t0
        if t > tl_src.duration:
            break
        for leg in [leg for leg in close_legs if leg not in flight.vbat]:  # Battery checks
            leg_start = leg_times[leg]["start"]
            if leg not in readings and t >= leg_start - VBAT_LEAD:
                readings[leg] = _VbatReading(victim)
            if leg in readings and t >= leg_start - VBAT_DECIDE:
                vbat = readings[leg].value
                flight.vbat[leg] = vbat
                if vbat is None or vbat < cfg["safety"]["vbat_min_close_leg"]:
                    flight.skipped.append(leg)
                    tl_vic, tl_src = build_flight(rows, cfg, skipped=flight.skipped)
                    leg_times = {int(r["leg"]): tl_src.leg_times(int(r["leg"])) for r in rows}
                    logger.warning(f"Leg {leg} skipped: victim battery {vbat} V")
        obs_src = source.obs()
        obs_vic = victim.obs() if victim is not None else None
        cmd_src = with_drag(tl_src.state(t), drag)
        cmd_vic = with_drag(tl_vic.state(t), drag) if victim is not None else None
        path_src = np.array([*recent["source"], cmd_src[0]])
        path_vic = np.array([*recent["victim"], cmd_vic[0]]) if victim is not None else None
        reason = safety_violation(limits, obs_src, path_src, obs_vic, path_vic)
        if reason is not None:
            logger.error(f"Abort: {reason}")
            flight.abort_reason = reason
            break
        t_wall = time.time()
        source.send_state(*cmd_src, zero_yaw)
        recent["source"].append(cmd_src[0])
        if victim is not None:
            victim.send_state(*cmd_vic, zero_yaw)
            recent["victim"].append(cmd_vic[0])
        source.update_estimators()
        if victim is not None:
            victim.update_estimators()
        phase, leg = tl_src.label(t)
        record = {
            "t_perf": t,
            "t_wall": t_wall,
            "source": _drone_record(obs_src, cmd_src, phase, leg),
            "victim": None,
        }
        if victim is not None:
            record["victim"] = _drone_record(obs_vic, cmd_vic, *tl_vic.label(t))
        flight.records.append(record)
        first_record.setdefault(leg, len(flight.records) - 1)
        while pending and t >= leg_times[int(pending[0]["leg"])]["end"]:
            row = pending.pop(0)
            leg = int(row["leg"])
            if leg in flight.skipped:
                stats, flags = {}, ["skipped"]
            else:
                recs = flight.records[first_record.get(leg, len(flight.records)) :]
                stats, flags = leg_summary(recs, row, leg_times[leg], cfg)
            flight.leg_stats[leg], flight.leg_flags[leg] = stats, flags
            _log_leg(row, stats, flags, leg in flight.skipped)
        sleep_until_next_cycle(tick, ctrl_freq)
    flight.victim_timeline, flight.source_timeline = tl_vic, tl_src
    return flight


def _killed_by_tilt(name: str, drone: Drone) -> bool:
    """Emergency-stop a landing drone tilted beyond `MAX_TILT`, as `Drone.return_to_start` does."""
    body_z = R.from_quat(drone.obs()["quat"]).apply([0.0, 0.0, 1.0])
    if body_z[2] >= math.cos(math.radians(MAX_TILT)):
        return False
    emergency_stop(drone)
    logger.warning(f"Landing the {name} failed: tilt above {MAX_TILT:.0f} deg, killed it")
    return True


def _wait_landing(drones: dict[str, Drone], duration: float, rate: float):
    """Keep the poses coming for `duration` and kill any drone that tips over."""
    t_end = time.perf_counter() + duration
    while time.perf_counter() < t_end and drones:
        for name, drone in list(drones.items()):
            if _killed_by_tilt(name, drone):
                del drones[name]
                continue
            drone.update_estimators()
        time.sleep(1 / rate)


def land_pair(victim: Drone | None, source: Drone, cfg: Mapping):
    """Land both drones in place with the high-level commander.

    Both switch to the high-level commander as `Drone.land` does. If the two are horizontally
    within `xy_guard`, the upper one, normally the source, first climbs up to `CLIMB` straight up,
    capped below the arena ceiling, while the lower one starts landing right away. Throughout, the
    poses keep coming at `pos_update_freq` and a drone tilted beyond `MAX_TILT` is killed.

    Args:
        victim: The victim, None when only the source flies.
        source: The source.
        cfg: The experiment config.
    """
    drones = _flying(victim, source)
    rate = cfg["drones"]["pos_update_freq"]
    for drone in drones.values():
        drone.cf.commander.send_stop_setpoint()
        drone.cf.commander.send_notify_setpoint_stop()  # Low-level setpoints end here
        drone.cf.param.set_value("commander.enHighLevel", 1)
        drone.cf.supervisor.send_arming_request(True)
    climber = None
    if victim is not None:
        p_src, p_vic = np.asarray(source.obs()["pos"]), np.asarray(victim.obs()["pos"])
        if np.linalg.norm(p_src[:2] - p_vic[:2]) < cfg["safety"]["xy_guard"]:
            climber = "source" if p_src[2] >= p_vic[2] else "victim"
    for name, drone in drones.items():
        if name != climber:
            drone.cf.high_level_commander.land(LAND_HEIGHT, LAND_DURATION)
    landing = dict(drones)  # A killed drone drops out
    if climber is not None:
        p = np.asarray(drones[climber].obs()["pos"])
        z = min(p[2] + CLIMB, cfg["safety"]["arena_max"][2] - CEILING_MARGIN)
        logger.warning(f"The drones are close, the {climber} climbs to {z:.2f} m before landing")
        drones[climber].cf.high_level_commander.go_to(p[0], p[1], z, 0.0, CLIMB_TIME)
        _wait_landing(landing, CLIMB_TIME, rate)
        if climber in landing:
            drones[climber].cf.high_level_commander.land(LAND_HEIGHT, LAND_DURATION)
    _wait_landing(landing, LAND_WAIT, rate)
    for drone in drones.values():
        drone.cf.high_level_commander.stop()
