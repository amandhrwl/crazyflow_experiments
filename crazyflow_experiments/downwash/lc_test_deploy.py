"""Load cell setup test: the source hovers beside, over and beside the victim on the load cell.

The victim stays on the load cell with its motors off and streams the load cell over the radio. The
script records still air first, then the source takes off from where it stands, holds beside the
victim, moves over it, holds there, moves back, holds and lands where it took off, and still air is
recorded again. Both cards log throughout. See the load cell setup test in README.md for the
terminals to start first.

With `--spin` the victim runs its motors on the load cell at a fixed command (`[spin]` in the
config) from after the still air until the source has landed, first alone, so the load cell
measures its thrust, then under the source's wake, as a flying victim would meet it.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/lc_test_deploy.py \
        [--dry-run] [--spin]
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default
# The estimator node pins its ROS discovery to LOCALHOST, and a SUBNET participant does not
# reliably discover it. rclpy and the ROS processes of the drone inherit the range from here
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

import argparse
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import rclpy
import toml
from crazyflow.control import Control
from drone_estimators.ros_nodes.ros2_connector import ROSConnector

from crazyflow_experiments.control.config import build_trajectory, drone_params
from crazyflow_experiments.control.loop import sleep_until_next_cycle
from crazyflow_experiments.control.state_controller import StateController
from crazyflow_experiments.control.trajectories import CompositeTrajectory
from crazyflow_experiments.deploy.experiment import connect
from crazyflow_experiments.deploy.takeoff import takeoff
from crazyflow_experiments.downwash import LC_CONFIG_PATH, LC_RESULTS_DIR, usd
from crazyflow_experiments.downwash.radio import RadioLog, Victim
from crazyflow_experiments.utils.paths import new_recording_path

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from crazyflow_experiments.control.trajectories import Trajectory
    from crazyflow_experiments.deploy.crazyflie import Drone

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("crazyflow_experiments.downwash.lc_test_deploy")  # Not __name__
logging.getLogger("crazyflow_experiments").setLevel(logging.INFO)

# Streamed by the source at 100 Hz: its accelerometer is also on its card and syncs the two (see
# sync.py), supervisor.info shows whether it armed (a motor out of the arming rpm range disarms it)
SOURCE_RADIO_VARIABLES = ["acc.z", "stateEstimate.z", "pm.vbat", "supervisor.info"]
STILL_SPREAD = 0.005  # m, largest motion capture spread of a drone standing still
G = 9.81
# Largest thrust of the spinning victim, as a fraction of its weight: below it the victim stays
# pressed onto the load cell even where it only rests on it
MAX_SPIN_FRACTION = 0.8


@dataclass(frozen=True)
class Plan:
    """Where the source flies, from the motion capture before anything is connected.

    Attributes:
        victim: Position of the victim's marker frame.
        source_start: Position the source takes off from and lands at.
        aside: Hold beside the victim.
        over: Hold over the victim.
        trajectory: Holds and moves from the end of the takeoff on.
        phases: Name, start and end of every hold and move in trajectory time.
    """

    victim: NDArray[np.floating]
    source_start: NDArray[np.floating]
    aside: NDArray[np.floating]
    over: NDArray[np.floating]
    trajectory: CompositeTrajectory
    phases: list[tuple[str, float, float]]


def segment_distance(point: NDArray, a: NDArray, b: NDArray) -> float:
    """Horizontal distance of a point from the segment from a to b."""
    point, a, b = point[:2], a[:2], b[:2]
    ab = b - a
    s = np.clip(np.dot(point - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(a + s * ab - point))


def plan_flight(config: dict, victim: NDArray, source_start: NDArray) -> Plan:
    """Place the holds around the victim and check the paths against the safety limits.

    Args:
        config: The experiment config.
        victim: Position of the victim's marker frame.
        source_start: Position of the source on the ground.
    """
    flight, safety = config["flight"], config["safety"]
    if flight["height_sep"] <= safety["min_height_sep"]:
        raise ValueError("height_sep must exceed min_height_sep, the hold over would trip it")
    if np.linalg.norm(flight["aside"]) < safety["clearance"]:
        raise ValueError(f"The hold aside {flight['aside']} is closer than the clearance")
    over = victim + np.array([0.0, 0.0, flight["height_sep"]])
    aside = over + np.array([*flight["aside"], 0.0])
    # The takeoff ramps straight to the hold aside, the return flies straight back at its height
    if (d := segment_distance(victim, source_start, aside)) < safety["clearance"]:
        raise ValueError(
            f"The takeoff path passes {d:.2f} m from the victim, less than the clearance. Place "
            f"the source nearer to the hold aside at {aside[:2].round(2)}."
        )

    def hold(pos: NDArray, duration: float) -> Trajectory:
        """A zero-size ellipse stays at its origin."""
        return build_trajectory("ellipse", 1, duration, [0.0, 0.0, 0.0], pos.tolist())

    times = (flight["aside_time"], flight["over_time"], flight["aside_time"])
    holds = (hold(aside, times[0]), hold(over, times[1]), hold(aside, times[2]))
    trajectory = CompositeTrajectory(holds, transition_time=flight["move_time"])
    trajectory.warm_up(0.0)  # Compile now, not between the arming and the takeoff
    s = trajectory.starts
    phases = [
        ("aside_1", s[0], s[0] + times[0]),
        ("move_in", s[0] + times[0], s[1]),
        ("over", s[1], s[1] + times[1]),
        ("move_out", s[1] + times[1], s[2]),
        ("aside_2", s[2], s[2] + times[2]),
    ]
    return Plan(victim, source_start, aside, over, trajectory, phases)


def spin_pwm(config: dict) -> int:
    """Motor command at which the victim's thrust is `thrust_fraction` of its weight.

    With the linear thrust map of `deploy.crazyflie.Drone`, thrust per motor / thrust_max * pwm_max.
    The load cell measures the thrust it really makes.
    """
    fraction = config["spin"]["thrust_fraction"]
    if not 0 < fraction <= MAX_SPIN_FRACTION:
        raise ValueError(
            f"thrust_fraction {fraction} must lie in (0, {MAX_SPIN_FRACTION}], higher could lift "
            "the victim off the load cell"
        )
    params = drone_params(config["victim"]["model"])
    thrust = fraction * params["mass"] * G / 4
    return round(thrust / params["thrust_max"] * params["pwm_max"])


def still_position(mocap: ROSConnector, name: str, duration: float = 1.0) -> NDArray:
    """Mean motion capture position of a drone over `duration`, which must stand still."""
    samples = []
    for _ in range(20):
        samples.append(np.array(mocap.pos[name]))
        time.sleep(duration / 20)
    samples = np.array(samples)
    if (spread := np.ptp(samples, axis=0).max()) > STILL_SPREAD:
        raise RuntimeError(f"{name} moved {spread * 100:.1f} cm in the motion capture")
    return samples.mean(axis=0)


def violation(
    pos: NDArray, setpoint: NDArray, victim: NDArray, safety: dict, max_error: float
) -> str | None:
    """Why the source has to stop, or None while it flies within the limits.

    Args:
        pos: Position of the source.
        setpoint: Its setpoint.
        victim: Position of the victim's marker frame.
        safety: The `[safety]` section of the config.
        max_error: Largest distance of the source from its setpoint in m.
    """
    if (error := np.linalg.norm(pos - setpoint)) > max_error:
        return f"source {error:.2f} m from its setpoint"
    above = pos[2] - victim[2]
    near = np.linalg.norm(pos[:2] - victim[:2]) < safety["xy_guard"]
    if near and above < safety["min_height_sep"]:
        return f"source only {above:.2f} m above the victim"
    return None


def record(obs: dict[str, NDArray], setpoint: NDArray) -> dict[str, NDArray | float]:
    """One tick of the flight record."""
    keys = ("pos", "vel", "quat", "pos_raw")
    return {"t_pc": time.time(), **{k: obs[k] for k in keys}, "setpoint": setpoint}


def settle(
    drone: Drone, plan: Plan, config: dict, stop: threading.Event
) -> tuple[list[dict], str | None]:
    """Hold beside the victim until the source has settled there after the takeoff.

    The takeoff ends with the calibration of the source's orientation, before the source has
    worked off the offset that the uncalibrated orientation leaves: in the first test it hovered
    0.25 m off its hold, its attitude estimate 2° from level, and the tracking limit tripped on the
    first tick. The holds start once the source stays within `settle_tolerance` of the hold aside
    for `settle_time`, or after `settle_timeout`, and the tracking limit applies from there on.
    Meanwhile it only has to keep within the clearance of its hold, which keeps it off the victim.

    Returns:
        The record of every tick and why the source has to land, None if the holds may start.
    """
    flight, safety = config["flight"], config["safety"]
    hold = StateController({"t": 0.0}, plan.trajectory)({"t": 0.0})  # The start of the holds
    t_start, settled_since, records = time.perf_counter(), None, []
    while rclpy.ok():
        t = time.perf_counter()
        obs = drone.obs()
        if obs["is_outdated"]:
            return records, "motion capture outdated"
        if stop.is_set():
            return records, "Ctrl+C"
        if reason := violation(obs["pos"], hold[:3], plan.victim, safety, safety["clearance"]):
            return records, reason
        drone.send_action(hold)
        records.append(record(obs, hold[:3]))
        error = float(np.linalg.norm(obs["pos"] - hold[:3]))
        if error > flight["settle_tolerance"]:
            settled_since = None
        elif settled_since is None:
            settled_since = t
        if settled_since is not None and t - settled_since >= flight["settle_time"]:
            logger.info(f"Settled {error * 100:.0f} cm from the hold aside in {t - t_start:.1f} s")
            return records, None
        if t - t_start >= flight["settle_timeout"]:
            logger.warning(
                f"Not settled in {flight['settle_timeout']:.0f} s, {error * 100:.0f} cm from the "
                "hold aside. Starting the holds, the tracking limit stops them if it is too far"
            )
            return records, None
        drone.update_estimators()
        sleep_until_next_cycle(t, flight["ctrl_freq"])
    return records, "ROS shut down"


def fly(
    drone: Drone, plan: Plan, config: dict, stop: threading.Event
) -> tuple[list[dict], float, str | None]:
    """Fly the holds and moves through the state interface, checking the limits every tick.

    Returns:
        The record of every tick, the PC time of trajectory zero and why the flight stopped early,
        None if it did not.
    """
    t_start, t_zero = time.perf_counter(), time.time()
    controller = StateController({"t": t_start}, plan.trajectory)
    records = []
    while rclpy.ok():
        t = time.perf_counter()
        obs = drone.obs()
        if obs["is_outdated"]:
            return records, t_zero, "motion capture outdated"
        if stop.is_set():
            return records, t_zero, "Ctrl+C"
        action = controller({"t": t})
        safety = config["safety"]
        if reason := violation(
            obs["pos"], action[:3], plan.victim, safety, safety["max_tracking_error"]
        ):
            return records, t_zero, reason
        drone.send_action(action)
        records.append(record(obs, action[:3]))
        if controller.done({"t": t}):
            return records, t_zero, None
        drone.update_estimators()
        sleep_until_next_cycle(t, config["flight"]["ctrl_freq"])
    return records, t_zero, "ROS shut down"


def phase_summary(radio: dict[str, NDArray], phases: list[tuple[str, float, float]]) -> list[str]:
    """Mean load cell reading of every phase over the radio, relative to the still air before."""
    t, weight = radio["loadcell/t_pc"], radio["loadcell/loadcell.weight"]
    _, t0, t1 = phases[0]
    rest = weight[(t >= t0) & (t < t1)]
    lines = [f"still air before: {rest.mean():+.3f} g, std {rest.std():.3f} g (the zero)"]
    for name, t0, t1 in phases[1:]:
        if (inside := (t >= t0) & (t < t1)).any():
            lines.append(f"{name}: {weight[inside].mean() - rest.mean():+.3f} g")
    return lines


def save(folder: Path, meta: dict, victim: RadioLog, source: RadioLog | None, records: list[dict]):
    """Write the metadata, both radio logs and the source's flight records."""
    with open(folder / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=lambda x: np.asarray(x).tolist())
    radio = {f"victim/{k}": v for k, v in victim.arrays().items()}
    if source is not None:
        radio |= {f"source/{k}": v for k, v in source.arrays().items()}
    np.savez(folder / "radio.npz", **radio)
    if records:
        np.savez(
            folder / "flight.npz", **{k: np.array([r[k] for r in records]) for k in records[0]}
        )
    logger.info(f"Recording in {folder}")


def main(dry_run: bool = False, spin: bool = False):
    """Plan the flight from the motion capture, then fly and record it.

    Args:
        dry_run: Only plan and check the flight, connect to no drone.
        spin: Run the victim's motors on the load cell while the source flies.
    """
    config = toml.load(LC_CONFIG_PATH)
    flight = config["flight"]
    if not rclpy.ok():
        rclpy.init()
    victim_name, source_name = f"cf{config['victim']['id']}", f"cf{config['source']['id']}"
    mocap = ROSConnector(tf_names=[victim_name, source_name], timeout=10.0)
    plan = plan_flight(
        config, still_position(mocap, victim_name), still_position(mocap, source_name)
    )
    mocap.close()
    logger.info(
        f"Victim at {plan.victim.round(3)}, source at {plan.source_start.round(3)}. Holds aside at "
        f"{plan.aside.round(3)} and over at {plan.over.round(3)}, {plan.trajectory.duration:.0f} s"
    )
    if spin:
        logger.info(f"The victim spins at {spin_pwm(config)}")
    if dry_run:
        return

    victim_config = config["victim"]
    victim = Victim(
        victim_config["id"], victim_config["channel"], victim_config["radio"], config["load_cell"]
    )
    folder = new_recording_path(LC_RESULTS_DIR, "test", suffix="")
    recorded = usd.recording_time(folder)  # Names the card files
    folder.mkdir()
    meta = {"config": config, "victim_pos": plan.victim, "source_start": plan.source_start}
    meta |= {"aside": plan.aside, "over": plan.over, "phases": [], "abort": None, "sd": {}}
    phases, records, source_radio = meta["phases"], [], None
    try:
        usd.check_variables(victim.cf, victim.name, usd.role_config("lc", "victim").variables)
        meta["sd"]["victim"] = {"start": usd.start_logging(victim.cf, victim.name, recorded)}
        t0 = time.time()
        time.sleep(flight["baseline_time"])
        phases.append(("still_before", t0, time.time()))
        logger.info(
            f"Load cell at {victim.radio_log.latest('loadcell.rate'):.0f} Hz, victim battery "
            f"{victim.radio_log.latest('pm.vbat'):.2f} V"
        )
        if spin:  # Alone first: the load cell measures the thrust at the command
            meta["spin"] = {"pwm": spin_pwm(config)} | config["spin"]
            t0 = time.time()
            victim.spin(meta["spin"]["pwm"])
            time.sleep(config["spin"]["spin_up_time"])
            phases.append(("spin_up", t0, time.time()))
            t0 = time.time()
            time.sleep(flight["baseline_time"])
            phases.append(("spin_alone", t0, time.time()))

        source_radio = RadioLog(source_name, report=("supervisor.info",))
        source_config = {"drone": config["source"] | {"pos_landing": [*plan.source_start[:2], 0.0]}}
        drone = connect(
            source_config,
            Control.state,
            log_variables=SOURCE_RADIO_VARIABLES,
            log_callback=source_radio,
        )
        source_console = usd.ConsoleCapture(drone.cf)
        # Ctrl+C in the air lands the source beside the victim instead of dropping it on the cell
        stop = threading.Event()

        def land_on_interrupt(signum: int, frame: object):
            logger.warning("Ctrl+C: landing the source (after the takeoff, if it is climbing)")
            stop.set()

        previous_handler = signal.signal(signal.SIGINT, land_on_interrupt)
        try:
            usd.check_variables(drone.cf, drone.name, usd.role_config("lc", "source").variables)
            meta["sd"]["source"] = {"start": usd.start_logging(drone.cf, drone.name, recorded)}
            logger.info(f"Source battery {source_radio.latest('pm.vbat'):.2f} V")
            t0 = time.time()
            takeoff(drone, plan.aside, 0.0, calibrate=True, ctrl_freq=flight["ctrl_freq"])
            phases.append(("takeoff", t0, time.time()))
            t0 = time.time()
            records, meta["abort"] = settle(drone, plan, config, stop)
            phases.append(("settle", t0, time.time()))
            if meta["abort"] is None:
                flown, t_zero, meta["abort"] = fly(drone, plan, config, stop)
                records += flown
                t_end = time.time()  # The phases an abort cut short end here
                phases.extend(
                    (name, t_zero + a, min(t_zero + b, t_end))
                    for name, a, b in plan.phases
                    if t_zero + a < t_end
                )
            if meta["abort"] is not None:
                logger.warning(f"Stopped: {meta['abort']}. Landing at the takeoff spot")
            t0 = time.time()
            drone.return_to_start(keep_height=True)
            phases.append(("landing", t0, time.time()))
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            if "source" in meta["sd"]:
                meta["sd"]["source"]["stop"] = usd.stop_logging(drone.cf, drone.name)
            meta["sd"]["source"] = meta["sd"].get("source", {}) | {
                "files": source_console.files(),
                "written": source_console.written(),
            }
            drone.close()

        if spin:
            t0 = time.time()
            time.sleep(flight["baseline_time"])
            phases.append(("spin_alone_after", t0, time.time()))
            t0 = time.time()
            victim.stop_spin()
            time.sleep(config["spin"]["spin_up_time"])
            phases.append(("spin_down", t0, time.time()))
        t0 = time.time()
        time.sleep(flight["baseline_time"])
        phases.append(("still_after", t0, time.time()))
    finally:
        if victim.spinning:
            victim.stop_spin()
        if "victim" in meta["sd"]:
            meta["sd"]["victim"]["stop"] = usd.stop_logging(victim.cf, victim.name)
        meta["sd"]["victim"] = meta["sd"].get("victim", {}) | {
            "files": victim.console.files(),
            "written": victim.console.written(),
        }
        victim.close()
        save(folder, meta, victim.radio_log, source_radio, records)

    for line in phase_summary(victim.radio_log.arrays(), phases):
        logger.info(f"Load cell, {line}")
    for role, sd in meta["sd"].items():
        logger.info(f"Copy {sd['files']} of the {role} card to {folder / 'sd' / role}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hover the source over the victim on the load cell"
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan and check, fly nothing")
    parser.add_argument("--spin", action="store_true", help="Run the victim's motors meanwhile")
    args = parser.parse_args()
    main(args.dry_run, args.spin)
