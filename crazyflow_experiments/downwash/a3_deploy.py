"""Fly one flight of block A3: the source passes over the hovering victim, nine legs.

Start the motion capture, an estimator node per drone and the rosbag recorder first, see the
downwash README. The script connects both drones, checks where they stand, their card variables
and batteries, starts SD logging, takes both off together from wherever they stand to their
positions, flies the legs from one loop, lands both in place and stops the logging. Everything the
conversion needs goes to `results/downwash/a3/flight<k>_<timestamp>/`: `meta.json` (config,
design rows, git hash, console lines with the SD file names, masses, abort reason, skipped legs,
leg flags, both timelines) and `records.pkl` (every tick of the takeoff and the flight). The cards
name their files after the flight, `vic_<timestamp>_SD` and `src_<timestamp>_SD`, on firmware with
`usd.fileDate` (see `usd.py`), and `usd.py collect <card> --block a3` copies them into the folder.

`--dry-run` prints the leg table and the closest approach of the two commanded paths, saves
`plots/downwash/a3_flight<k>_timeline.png` and stops before connecting. `--only-source` flies the
source's timeline alone, with the victim on the ground and unconnected (flight F0).

`--stand` flies the same legs over the victim clamped on the load cell instead, centred on its
position in the motion capture, see `a3_stand.py`: the victim is a `radio.Victim` that streams and
logs the load cell, and only the source flies. `--spin` runs the victim's motors at hover around
every pass, per leg on its battery. `--legs` flies a part of the flight, e.g. the legs a spin run
left for a refly. A stand run records to `results/downwash/a3/stand_flight<k>_<still|spin>_<time>/`
with `radio.npz` besides; `a3_stand_convert.py` converts it. With `--dry-run`, `--stand-pos` and
`--source-pos` stand in for the motion capture.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/a3_deploy.py --flight 1 \
        [--dry-run] [--only-source] [--design PATH]
    pixi run -e sim2real python crazyflow_experiments/downwash/a3_deploy.py --flight 1 --stand \
        [--spin] [--legs 3,5-7] [--dry-run --stand-pos X Y Z --source-pos X Y Z]
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import argparse
import json
import logging
import pickle
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash import (
    A1_RESULTS_DIR,
    A3_RESULTS_DIR,
    DOWNWASH_PLOTS_DIR,
    a1_stand,
    a3_stand,
    plots,
    usd,
)
from crazyflow_experiments.downwash.a3_design import (
    DESIGN_PATH,
    check_design,
    flights,
    load_config,
    read_design,
)
from crazyflow_experiments.downwash.timeline import build_flight, victim_hover_point
from crazyflow_experiments.utils.paths import REPO_ROOT, new_recording_path

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from numpy.typing import NDArray

    from crazyflow_experiments.deploy.crazyflie import Drone
    from crazyflow_experiments.downwash.pair import Flight, Takeoff
    from crazyflow_experiments.downwash.timeline import Timeline

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("crazyflow_experiments.downwash.a3_deploy")  # Not __name__, the __main__
logging.getLogger("crazyflow_experiments").setLevel(logging.INFO)  # Flight progress and leg lines

FIRMWARE_ALERTS = ("[FAIL]", "Locked", "Crashed", "free falling", "tumbled")  # Console lines
MIN_SPACING = 1.0  # m, horizontal distance between the two drones standing before takeoff
PATH_CHECK_FREQ = 100.0  # Hz, rate the commanded paths are compared at in the dry run


@dataclass(frozen=True)
class Block:
    """What differs between the blocks that fly through `fly` and `fly_stand`.

    Attributes:
        name: Name in the recording's meta.json, e.g. "A3"; the stand's runs add "_stand".
        results_dir: Where the recordings go; None for A3's, which tests redirect by patching
            `A3_RESULTS_DIR`.
        free_stem: Folder stem of a free flight, followed by the flight's number.
        stand_stem: Folder stem of a run on the stand, followed by the number and the mode.
        build: Builds the victim's and the source's timeline of the rows, see `pair.run_flight`.
        plan_stand: Plans a run on the stand from the config, the rows, the stand's and the
            source's positions, the items to fly and whether the victim spins: the config, rows
            and spin schedule of the run.
        items_flag: Command line flag naming the items a refly hint lists.
        hooks: The keyword arguments of `pair.run_flight` that the block sets, from the rows.
    """

    name: str
    results_dir: Path | None
    free_stem: str
    stand_stem: str
    build: Callable
    plan_stand: Callable
    items_flag: str
    hooks: Callable[[Sequence[Mapping]], dict]


def leg_distances(rows: Sequence[Mapping], victim: Timeline, source: Timeline) -> list[float]:
    """Closest approach of the two commanded paths during each leg's pass."""
    distances = []
    for row in rows:
        times = source.leg_times(int(row["leg"]))
        t = np.linspace(times["pass_start"], times["pass_end"], 2001)
        gap = np.linalg.norm(source.state(t)[0] - victim.state(t)[0], axis=1)
        distances.append(float(gap.min()))
    return distances


def dry_run(rows: Sequence[Mapping], cfg: Mapping, flight: int, prefix: str = "a3") -> Path:
    """Log the leg table and the path distances of one flight and plot its timeline."""
    victim, source = build_flight(rows, cfg)
    logger.info(f"Flight {flight}: {len(rows)} legs, {source.duration:.1f} s after takeoff")
    logger.info(
        "leg cell  v[m/s] dz[m] dir  start [m]              end [m]               "
        "leg start  pass  cross  pass end  leg end [s]  closest [m]"
    )
    for row, closest in zip(rows, leg_distances(rows, victim, source)):
        times = source.leg_times(int(row["leg"]))
        start = source.state(times["pass_start"])[0]
        end = source.state(times["pass_end"])[0]
        logger.info(
            f"{int(row['leg']):3d} {int(row['cell']):4d}  {row['speed']:6.2f} {row['height_sep']:5.2f}"
            f" {int(row['direction']):+3d}  {np.array2string(start, precision=2):22s}"
            f" {np.array2string(end, precision=2):22s} {times['start']:8.2f} {times['pass_start']:6.2f}"
            f" {times['cross']:6.2f} {times['pass_end']:8.2f} {times['end']:8.2f}  {closest:11.3f}"
        )
    t = np.append(np.arange(0.0, source.duration, 1 / PATH_CHECK_FREQ), source.duration)
    p_vic, p_src = victim.state(t)[0], source.state(t)[0]
    gap = np.linalg.norm(p_src - p_vic, axis=1)
    logger.info(
        f"Closest approach of the commanded paths: {gap.min():.3f} m at t = {t[gap.argmin()]:.2f} s"
    )
    margin = np.array([cfg["safety"]["x_margin"], cfg["safety"]["x_margin"], 0.0])
    lo, hi = np.asarray(cfg["safety"]["arena_min"]), np.asarray(cfg["safety"]["arena_max"])
    for name, pos in (("victim", p_vic), ("source", p_src)):
        room = np.minimum(pos - lo, hi - pos).min(axis=0) - margin
        logger.info(
            f"Arena room of the {name} beyond the margin: x {room[0]:.2f} m,"
            f" y {room[1]:.2f} m, z {room[2]:.2f} m"
        )
    fig, axes = plots.figure(3, row_height=2.0)
    for row in rows:
        times = source.leg_times(int(row["leg"]))
        for ax in axes[:, 0]:
            ax.axvspan(times["pass_start"], times["pass_end"], color=plots.SHADE, lw=0)
    for ax, axis, label in ((axes[0, 0], 0, "x [m]"), (axes[1, 0], 2, "z [m]")):
        ax.plot(t, p_vic[:, axis], color=plots.VICTIM, lw=plots.LINE_WIDTH, label="victim")
        ax.plot(t, p_src[:, axis], color=plots.SOURCE, lw=plots.LINE_WIDTH, label="source")
        ax.set_ylabel(label)
    axes[2, 0].plot(t, gap, color=plots.INK_SECONDARY, lw=plots.LINE_WIDTH)
    axes[2, 0].set_ylabel("distance [m]")
    axes[2, 0].set_ylim(bottom=0.0)
    for row in rows:
        cross = source.leg_times(int(row["leg"]))["cross"]
        axes[2, 0].annotate(
            f"Δz {row['height_sep']:.2f}",
            (cross, row["height_sep"]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=7,
            color=plots.INK_SECONDARY,
        )
    axes[2, 0].set_xlabel("time from the flight's timeline zero [s]")
    plots.legend(axes[0, 0], loc="upper right", ncols=2)
    axes[0, 0].set_title(f"A3 flight {flight}: commanded paths, passes shaded", loc="left")
    return plots.save(fig, DOWNWASH_PLOTS_DIR / f"{prefix}_flight{flight}_timeline.png")


def git_info() -> dict:
    """Commit and dirty state of the repository the script runs from."""

    def git(*args: str) -> str:
        cmd = ["git", "-C", str(REPO_ROOT), *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()

    return {"hash": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def check_placement(drones: Mapping[str, Drone], cfg: Mapping):
    """Both drones stand inside the arena margin and at least `MIN_SPACING` apart.

    They may stand anywhere else: the takeoff climbs both straight up before either moves level.
    """
    safety = cfg["safety"]
    lo = np.asarray(safety["arena_min"][:2], dtype=float) + safety["x_margin"]
    hi = np.asarray(safety["arena_max"][:2], dtype=float) - safety["x_margin"]
    spots = {
        name: np.asarray(drone.obs()["pos"], dtype=float)[:2] for name, drone in drones.items()
    }
    for name, spot in spots.items():
        if np.any(spot < lo) or np.any(spot > hi):
            raise RuntimeError(f"The {name} stands at {spot.round(2)}, outside the arena margin")
    if len(spots) == 2:
        spacing = float(np.linalg.norm(spots["victim"] - spots["source"]))
        if spacing < MIN_SPACING:
            raise RuntimeError(
                f"The drones stand {spacing:.2f} m apart, less than {MIN_SPACING} m: victim at"
                f" {spots['victim'].round(2)}, source at {spots['source'].round(2)}"
            )
    logger.info(
        "Standing at " + ", ".join(f"{name} {spot.round(2)}" for name, spot in spots.items())
    )


def check_batteries(drones: Mapping[str, Drone], cfg: Mapping) -> dict[str, float]:
    """Battery voltage of every drone, each at least `vbat_min_start`."""
    vbat = {
        name: float(usd.read_once(drone, ["pm.vbat"])["pm.vbat"]) for name, drone in drones.items()
    }
    logger.info(f"Batteries: {', '.join(f'{name} {v:.2f} V' for name, v in vbat.items())}")
    low = {name: v for name, v in vbat.items() if v < cfg["safety"]["vbat_min_start"]}
    if low:
        raise RuntimeError(f"Batteries below {cfg['safety']['vbat_min_start']} V: {low}")
    return vbat


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def each(items: Iterable, action: Callable):
    """Apply `action` to every item, also to those after one that fails, then raise the failure."""
    items = list(items)
    if not items:
        return
    try:
        action(items[0])
    finally:
        each(items[1:], action)


def outcome_meta(
    error: BaseException | None,
    takeoff: Takeoff | None,
    flight: Flight | None,
    drones: Iterable[str],
    usd_times: Mapping[str, dict],
    consoles: Mapping[str, usd.ConsoleCapture],
) -> dict:
    """The meta.json entries known only after the flight."""
    reason = takeoff.error if takeoff is not None else None
    if flight is not None:
        reason = flight.abort_reason
    if reason is None and error is not None:
        reason = f"exception: {error!r}"
    meta = {
        "t_wall_stop": time.time(),
        "error": None if error is None else repr(error),
        "abort_reason": reason,
        "usd": {
            name: {
                **usd_times[name],
                "variables": list(usd.role_config("a3", name).variables),
                "files": consoles[name].files() if name in consoles else [],
                "skipped_variables": consoles[name].skipped_variables() if name in consoles else [],
                "console": consoles[name].lines if name in consoles else [],
            }
            for name in drones
        },
    }
    if flight is not None:
        first = flight.records[0] if flight.records else None
        meta |= {
            "skipped_legs": flight.skipped,
            "vbat_close_legs": flight.vbat,
            "leg_stats": flight.leg_stats,
            "leg_flags": flight.leg_flags,
            "t_wall_zero": first["t_wall"] - first["t_perf"] if first else None,
            "timeline": {
                "victim": flight.victim_timeline.table(),
                "source": flight.source_timeline.table(),
            },
        }
    return meta


def write_outputs(folder: Path, meta: dict, takeoff: Takeoff | None, flight: Flight | None):
    """Write meta.json and records.pkl of a flight."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    records = {
        "takeoff": takeoff.records if takeoff is not None else [],
        "flight": flight.records if flight is not None else [],
    }
    with open(folder / "records.pkl", "wb") as f:
        pickle.dump(records, f)
    logger.info(f"Wrote {folder / 'meta.json'} and {folder / 'records.pkl'}")


def fly(
    rows: Sequence[Mapping],
    cfg: dict,
    flight_index: int,
    only_source: bool,
    design: Path,
    block: Block | None = None,
):
    """Fly one flight and record it, see the module docstring."""
    block = A3 if block is None else block
    # Imported here, so the dry run needs neither the radio nor ROS
    from crazyflow_experiments.downwash.pair import (
        connect_pair,
        ctrl_c_ends_flight,
        land_pair,
        run_flight,
        takeoff_pair,
    )

    folder = new_recording_path(
        block.results_dir or A3_RESULTS_DIR, f"{block.free_stem}{flight_index}", suffix=""
    )
    recorded = usd.recording_time(folder)  # Names both card files after the flight
    _, source_plan = block.build(rows, cfg)
    targets = {"victim": victim_hover_point(cfg), "source": source_plan.state(0.0)[0]}
    meta = {
        "block": block.name,
        "flight": flight_index,
        "only_source": only_source,
        "created": datetime.now().astimezone().isoformat(),
        "argv": sys.argv,
        "hostname": socket.gethostname(),
        "git": git_info(),
        "config": cfg,
        "design_file": str(design),
        "design_rows": list(rows),
        "masses": {role: cfg["drones"][role]["mass_kg"] for role in ("victim", "source")},
        "targets": {role: target.tolist() for role, target in targets.items()},
        "t_wall_start": time.time(),
    }
    logger.info(f"Flight {flight_index}: recording to {folder}")
    victim, source = connect_pair(cfg, only_source)
    drones = {"victim": victim, "source": source} if victim is not None else {"source": source}
    consoles, usd_times = {}, {name: {} for name in drones}
    takeoff = flight = None
    airborne = landed = False
    try:
        consoles = {name: usd.ConsoleCapture(drone.cf) for name, drone in drones.items()}
        check_placement(drones, cfg)
        for name, drone in drones.items():
            usd.check_variables(drone.cf, drone.name, usd.role_config("a3", name).variables)
        meta["vbat_start"] = check_batteries(drones, cfg)
        for name, drone in drones.items():
            usd_times[name]["t_wall_start"] = usd.start_logging(drone.cf, drone.name, recorded)
        airborne = True
        with ctrl_c_ends_flight():  # Ctrl+C lands both and keeps the records of the legs flown
            takeoff = takeoff_pair(victim, source, targets, cfg)
            if takeoff.error is None:
                flight = run_flight(victim, source, rows, cfg, **block.hooks(rows))
        land_pair(victim, source, cfg)
        landed = True
    finally:
        # Each step runs even if an earlier one failed: land, close the SD files, keep the data,
        # and only then cut the motors. A failure is raised once every step was tried.
        error = sys.exception()
        try:
            if airborne and not landed:
                logger.error(f"Landing both after {error!r}")
                land_pair(victim, source, cfg)
        finally:
            try:

                def stop(item: tuple[str, Drone]):
                    usd_times[item[0]]["t_wall_stop"] = usd.stop_logging(item[1].cf, item[1].name)

                each(drones.items(), stop)
            finally:
                try:
                    meta |= outcome_meta(error, takeoff, flight, drones, usd_times, consoles)
                    write_outputs(folder, meta, takeoff, flight)
                finally:
                    each(drones.values(), lambda drone: drone.close())
    flown = [
        leg for leg, flags in (flight.leg_flags if flight else {}).items() if "skipped" not in flags
    ]
    logger.info(f"Legs flown: {flown}, abort: {meta['abort_reason']}")
    for name, info in meta["usd"].items():
        logger.info(f"SD file of the {name}: {info['files'] or 'unknown, see its card'}")
        for line in info["console"]:
            if any(word in line for word in FIRMWARE_ALERTS):
                logger.warning(f"Console of the {name}: {line}")


def card_meta(
    times: Mapping[str, dict], consoles: Mapping[str, usd.ConsoleCapture], blocks: Mapping[str, str]
) -> dict:
    """The card entries of a stand run's meta.json, under "usd" as in A3."""
    return {
        role: {
            **times.get(role, {}),
            "block": blocks[role],
            "variables": list(usd.role_config(blocks[role], role).variables),
            "files": console.files(),
            "skipped_variables": console.skipped_variables(),
            "console": console.lines,
        }
        for role, console in consoles.items()
    }


def fly_stand(
    rows: Sequence[Mapping],
    cfg: dict,
    flight_index: int,
    design: Path,
    mode: str,
    legs: Sequence[int] | None,
    block: Block | None = None,
):
    """Fly one flight, or some of its legs, over the victim on the stand and record it.

    Still air with the motors off first, then the source's takeoff, the legs with the victim
    following its schedule (`a3_stand.VictimSchedule`), the landing clear of the stand and still
    air again. The victim's card logs throughout, the source's card from its connection.
    """
    block = A3 if block is None else block
    # Imported here, so the dry run needs neither the radio nor ROS
    import rclpy
    from drone_estimators.ros_nodes.ros2_connector import ROSConnector

    from crazyflow_experiments.downwash.lc_test_deploy import still_position
    from crazyflow_experiments.downwash.pair import (
        connect_pair,
        ctrl_c_ends_flight,
        land_source_clear,
        run_flight,
        takeoff_pair,
    )
    from crazyflow_experiments.downwash.radio import Victim

    stand = cfg["stand"]
    a1_cfg = a1_stand.load_config()
    hover = stand_hover_cmd(cfg, a1_cfg)
    drones = cfg["drones"]
    names = {role: f"cf{drones[role]['id']}" for role in ("victim", "source")}
    if not rclpy.ok():
        rclpy.init()
    mocap = ROSConnector(tf_names=list(names.values()), timeout=10.0)
    stand_pos = still_position(mocap, names["victim"])
    source_pos = still_position(mocap, names["source"])
    mocap.close()
    stand_cfg, stand_rows, schedule = block.plan_stand(
        cfg, rows, stand_pos, source_pos, legs, mode == "spin"
    )
    log_schedule(schedule, stand, mode == "spin")
    _, source_plan = block.build(stand_rows, stand_cfg)
    folder = new_recording_path(
        block.results_dir or A3_RESULTS_DIR, f"{block.stand_stem}{flight_index}_{mode}", suffix=""
    )
    recorded = usd.recording_time(folder)  # Names both card files after the run
    folder.mkdir()
    blocks = {"victim": "lc", "source": "a3"}
    meta = {
        "block": f"{block.name}_stand",
        "flight": flight_index,
        "mode": mode,
        "legs": [int(row["leg"]) for row in stand_rows],
        "only_source": True,
        "created": datetime.now().astimezone().isoformat(),
        "argv": sys.argv,
        "hostname": socket.gethostname(),
        "git": git_info(),
        "config": stand_cfg,
        "design_file": str(design),
        "design_rows": stand_rows,
        "stand_pos": stand_pos,
        "source_start": source_pos,
        "load_cell": a1_cfg["load_cell"],
        "victim_mass": a1_stand.victim_mass(a1_cfg),
        "hover": hover,
        "phases": [],
        "t_wall_start": time.time(),
    }
    logger.info(f"Stand run, {mode}, legs {meta['legs']}: recording to {folder}")
    victim = Victim(
        drones["victim"]["id"],
        drones["victim"]["channel"],
        drones["victim"]["radio"],
        a1_cfg["load_cell"],
    )
    plan = a3_stand.VictimSchedule(victim, schedule, stand, spin=mode == "spin")
    source = takeoff = flight = None
    consoles, times = {"victim": victim.console}, {"victim": {}, "source": {}}
    airborne = landed = False
    try:
        usd.check_variables(victim.cf, victim.name, usd.role_config("lc", "victim").variables)
        time.sleep(2 * a3_stand.VBAT_RECENT)  # A few status packets with the battery
        vbat = victim.radio_log.latest("pm.vbat")
        meta["vbat_victim_start"] = vbat
        logger.info(f"Victim battery {vbat:.2f} V at rest")
        if mode == "spin" and vbat < stand["vbat_rest_min"]:
            raise RuntimeError(
                f"The victim's battery is at {vbat:.2f} V, below {stand['vbat_rest_min']} V"
            )
        times["victim"]["t_wall_start"] = usd.start_logging(victim.cf, victim.name, recorded)
        t0 = time.time()
        time.sleep(stand["baseline_time"])
        meta["phases"].append(("still_before", t0, time.time()))
        _, source = connect_pair(stand_cfg, only_source=True)
        consoles["source"] = usd.ConsoleCapture(source.cf)
        usd.check_variables(source.cf, source.name, usd.role_config("a3", "source").variables)
        meta["vbat_start"] = check_batteries({"source": source}, stand_cfg)
        times["source"]["t_wall_start"] = usd.start_logging(source.cf, source.name, recorded)
        airborne = True
        t0 = time.time()
        with ctrl_c_ends_flight():  # Ctrl+C ends the legs, the source lands clear of the stand
            takeoff = takeoff_pair(None, source, {"source": source_plan.state(0.0)[0]}, stand_cfg)
            if takeoff.error is None:
                flight = run_flight(
                    None,
                    source,
                    stand_rows,
                    stand_cfg,
                    fixed_victim=stand_pos,
                    on_start=plan.start,
                    **block.hooks(stand_rows),
                )
        plan.stop()
        waypoints = a3_stand.landing_waypoints(source.obs()["pos"], stand_pos, stand_cfg)
        land_source_clear(source, waypoints, stand_cfg)
        landed = True
        meta["phases"].append(("flight", t0, time.time()))
        times["source"]["t_wall_stop"] = usd.stop_logging(source.cf, source.name)
        t0 = time.time()
        time.sleep(stand["baseline_time"])
        meta["phases"].append(("still_after", t0, time.time()))
    finally:
        # Each step runs even if an earlier one failed: the victim's motors off, the source down,
        # the cards closed, the data kept, the links closed. A failure is raised at the end.
        error = sys.exception()
        try:
            plan.stop()
            if airborne and not landed:
                logger.error(f"Landing the source after {error!r}")
                waypoints = a3_stand.landing_waypoints(source.obs()["pos"], stand_pos, stand_cfg)
                land_source_clear(source, waypoints, stand_cfg)
        finally:
            try:
                if source is not None and "t_wall_stop" not in times["source"]:
                    times["source"]["t_wall_stop"] = usd.stop_logging(source.cf, source.name)
                if "t_wall_start" in times["victim"]:
                    times["victim"]["t_wall_stop"] = usd.stop_logging(victim.cf, victim.name)
            finally:
                try:
                    meta |= stand_outcome(error, takeoff, flight, plan, times, consoles, blocks)
                    write_outputs(folder, meta, takeoff, flight)
                    np.savez(
                        folder / "radio.npz",
                        **{f"victim/{k}": v for k, v in victim.radio_log.arrays().items()},
                    )
                finally:
                    if source is not None:
                        source.close()
                    victim.close()
    logger.info(f"Legs flown: {sorted(meta.get('leg_flags', {}))}, abort: {meta['abort_reason']}")
    if refly := meta["victim_schedule"]["refly_legs"]:
        logger.warning(
            f"Legs {refly} did not spin through their pass: refly them on a fresh battery with "
            f"--flight {flight_index} --stand --spin {block.items_flag} {','.join(map(str, refly))}"
        )
    for role, info in meta["usd"].items():
        logger.info(f"SD file of the {role}: {info['files'] or 'unknown, see its card'}")
        for line in info["console"]:
            if any(word in line for word in FIRMWARE_ALERTS):
                logger.warning(f"Console of the {role}: {line}")


def stand_outcome(
    error: BaseException | None,
    takeoff: Takeoff | None,
    flight: Flight | None,
    plan: a3_stand.VictimSchedule,
    times: Mapping[str, dict],
    consoles: Mapping[str, usd.ConsoleCapture],
    blocks: Mapping[str, str],
) -> dict:
    """The meta.json entries of a stand run known only after it."""
    meta = outcome_meta(error, takeoff, flight, (), {}, {})
    meta["usd"] = card_meta(times, consoles, blocks)
    meta["victim_schedule"] = plan.report()
    if flight is not None:
        meta["timeline"]["victim"] = None  # The victim does not fly
    return meta


def stand_hover_cmd(cfg: Mapping, a1_cfg: Mapping) -> dict:
    """The configured hover command against A1: its weight, the thrust cap and the steady windows.

    Raises if the victim's weight lies above the thrust cap of the stand; warns if the latest A1
    block puts the hover command more than `a3_stand.HOVER_CMD_TOL` away from the configured one.
    """
    from crazyflow_experiments.downwash.a1_convert import read_steady

    mass = a1_stand.victim_mass(dict(a1_cfg))
    weight = mass * a1_cfg["g"]
    cap, reason = a1_stand.thrust_cap(dict(a1_cfg), mass)
    if weight > cap:
        raise ValueError(f"The victim's weight {weight:.3f} N exceeds the thrust cap ({reason})")
    info = {"hover_cmd": cfg["stand"]["hover_cmd"], "weight_N": weight, "cap_N": cap}
    blocks = sorted(A1_RESULTS_DIR.glob("all_*/steady.csv"))
    if blocks:
        cmd, derivation = a3_stand.hover_command(read_steady(blocks[-1]), weight)
        info |= {"a1_block": blocks[-1].parent.name, "a1_hover_cmd": cmd} | derivation
        if abs(cmd - info["hover_cmd"]) > a3_stand.HOVER_CMD_TOL * cmd:
            logger.warning(
                f"[stand] hover_cmd {info['hover_cmd']} differs from {cmd} of A1's "
                f"{blocks[-1].parent.name}; update it knowingly"
            )
    return info


def plan_stand(
    cfg: Mapping,
    rows: Sequence[Mapping],
    stand_pos: NDArray,
    source_pos: NDArray,
    legs: Sequence[int] | None,
) -> tuple[dict, list[dict], list[a3_stand.LegSpin]]:
    """The stand's config, rows and spin schedule of a flight, checked against every limit."""
    rows = a3_stand.select_legs(rows, legs)
    flown = {int(row["leg"]) for row in a3_stand.stand_rows(rows, cfg)}
    if not flown:
        raise ValueError(
            f"No leg of this flight is at a separation the stand flies, [stand] height_seps "
            f"{cfg['stand']['height_seps']}: fly the near flights (even numbers) on the stand"
        )
    if legs is not None and (other := sorted(set(legs) - flown)):
        raise ValueError(
            f"Legs {other} fly at a separation the stand skips, [stand] height_seps "
            f"{cfg['stand']['height_seps']}"
        )
    stand_cfg, stand_rows = a3_stand.stand_flight(cfg, rows, stand_pos)
    check_design(stand_rows, stand_cfg)
    a3_stand.check_source_start(source_pos, stand_pos, cfg["stand"]["clearance"])
    _, source_plan = build_flight(stand_rows, stand_cfg)
    schedule = a3_stand.spin_schedule(source_plan, stand_rows, cfg["stand"])
    return stand_cfg, stand_rows, schedule


A3 = Block(
    name="A3",
    results_dir=None,
    free_stem="flight",
    stand_stem="stand_flight",
    build=build_flight,
    plan_stand=lambda cfg, rows, stand_pos, source_pos, legs, spin: plan_stand(
        cfg, rows, stand_pos, source_pos, legs
    ),
    items_flag="--legs",
    hooks=lambda rows: {},
)


def log_schedule(schedule: Sequence[a3_stand.LegSpin], stand: Mapping, spin: bool):
    """The victim's windows of every leg and the battery budget of a spin run."""
    logger.info(
        "leg  tare   arm   cmd  baseline  pass  cross  pass end  motors off  spin [s] (timeline s)"
    )
    for leg in schedule:
        logger.info(
            f"{leg.leg:3d} {leg.tare_start:6.1f} {leg.arm_at:5.1f} {leg.spin_on:5.1f} "
            f"{leg.baseline_start:8.1f} {leg.pass_start:6.1f} {leg.cross:6.1f} {leg.pass_end:8.1f} "
            f"{leg.spin_off:10.1f} {leg.spin_time:9.1f}"
        )
    budget = a3_stand.battery_budget(schedule, stand)
    logger.info(
        f"Battery budget: {budget['spin_s_total']:.0f} s of spin over {len(schedule)} legs, "
        f"{max(budget['spin_s_per_leg'].values()):.1f} s at most per leg; one battery gives about "
        f"{budget['battery_spin_s']:.0f} s, so {budget['legs_per_battery']} legs per battery and "
        f"{budget['batteries']} batteries for these legs"
        + ("" if spin else " (still run: the motors stay off)")
    )


def _parse_point(values: Sequence[float] | None) -> NDArray | None:
    return None if values is None else np.asarray(values, dtype=float)


def main():
    """Command line: fly or dry-run one flight of the design."""
    parser = argparse.ArgumentParser(description="Fly one flight of the downwash block A3.")
    parser.add_argument("--flight", type=int, required=True, help="Flight of the design, 1 to 3")
    parser.add_argument("--dry-run", action="store_true", help="Plan only, connect nothing")
    parser.add_argument("--only-source", action="store_true", help="Leave the victim on the ground")
    parser.add_argument("--design", type=Path, default=DESIGN_PATH, help="Design csv")
    parser.add_argument("--stand", action="store_true", help="Victim clamped on the load cell")
    parser.add_argument("--spin", action="store_true", help="With --stand: spin it at hover")
    parser.add_argument("--legs", type=a3_stand.parse_legs, help="Fly only these legs, e.g. 3,5-7")
    parser.add_argument("--stand-pos", type=float, nargs=3, help="Dry run: the victim on the stand")
    parser.add_argument("--source-pos", type=float, nargs=3, help="Dry run: the source's spot")
    args = parser.parse_args()
    if (args.spin or args.legs or args.stand_pos) and not args.stand:
        raise ValueError("--spin, --legs and --stand-pos go with --stand")
    flying = ("source",) if args.stand or args.only_source else ("victim", "source")
    cfg = load_config(require_masses=not args.dry_run, mass_roles=flying)
    rows_all = read_design(args.design)
    check_design(rows_all, cfg)
    grouped = flights(rows_all)
    if args.flight not in grouped:
        raise ValueError(f"{args.design} has no flight {args.flight}, only {sorted(grouped)}")
    rows = grouped[args.flight]
    if args.stand:
        mode = "spin" if args.spin else "still"
        stand_pos, source_pos = _parse_point(args.stand_pos), _parse_point(args.source_pos)
        if args.dry_run:
            if stand_pos is None or source_pos is None:
                raise ValueError("A dry run of --stand needs --stand-pos and --source-pos")
            stand_cfg, stand_rows, schedule = plan_stand(
                cfg, rows, stand_pos, source_pos, args.legs
            )
            logger.info(f"Stand run, {mode}: victim at {stand_pos}, source at {source_pos}")
            logger.info(f"Hover command: {stand_hover_cmd(cfg, a1_stand.load_config())}")
            logger.info(f"Saved {dry_run(stand_rows, stand_cfg, args.flight, 'a3_stand')}")
            log_schedule(schedule, cfg["stand"], args.spin)
            return
        fly_stand(rows, cfg, args.flight, args.design, mode, args.legs)
        return
    if args.dry_run:
        logger.info(f"Saved {dry_run(rows, cfg, args.flight)}")
        return
    fly(rows, cfg, args.flight, args.only_source, args.design)


if __name__ == "__main__":
    main()
