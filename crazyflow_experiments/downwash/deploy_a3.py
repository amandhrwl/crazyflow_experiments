"""Fly one flight of block A3: the source passes over the hovering victim, nine legs.

Start the motion capture, an estimator node per drone and the rosbag recorder first, see the
downwash README. The script connects both drones, checks where they stand, their card variables
and batteries, starts SD logging, takes both off together from wherever they stand to their
positions, flies the legs from one loop, lands both in place and stops the logging. Everything the conversion needs goes to
`results/downwash/a3/flight<k>_<timestamp>/`: `meta.json` (config, design rows, git hash, console
lines with the SD file names, masses, abort reason, skipped legs, leg flags, both timelines) and
`records.pkl` (every tick of the takeoff and the flight).

`--dry-run` prints the leg table and the closest approach of the two commanded paths, saves
`plots/downwash/a3_flight<k>_timeline.png` and stops before connecting. `--only-source` flies the
source's timeline alone, with the victim on the ground and unconnected (flight F0).

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/deploy_a3.py --flight 1 \
        [--dry-run] [--only-source] [--design PATH]
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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash import A3_RESULTS_DIR, DOWNWASH_PLOTS_DIR, plots, usd
from crazyflow_experiments.downwash.design import (
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

    from crazyflow_experiments.deploy.crazyflie import Drone
    from crazyflow_experiments.downwash.pair import Flight, Takeoff
    from crazyflow_experiments.downwash.timeline import Timeline

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("crazyflow_experiments.downwash.deploy_a3")  # Not __name__, the __main__
logging.getLogger("crazyflow_experiments").setLevel(logging.INFO)  # Flight progress and leg lines

FIRMWARE_ALERTS = ("[FAIL]", "Locked", "Crashed", "free falling", "tumbled")  # Console lines
MIN_SPACING = 1.0  # m, horizontal distance between the two drones standing before takeoff
PATH_CHECK_FREQ = 100.0  # Hz, rate the commanded paths are compared at in the dry run


def leg_distances(rows: Sequence[Mapping], victim: Timeline, source: Timeline) -> list[float]:
    """Closest approach of the two commanded paths during each leg's pass."""
    distances = []
    for row in rows:
        times = source.leg_times(int(row["leg"]))
        t = np.linspace(times["pass_start"], times["pass_end"], 2001)
        gap = np.linalg.norm(source.state(t)[0] - victim.state(t)[0], axis=1)
        distances.append(float(gap.min()))
    return distances


def dry_run(rows: Sequence[Mapping], cfg: Mapping, flight: int) -> Path:
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
            f"Δz {row['height_sep']:.1f}",
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
    return plots.save(fig, DOWNWASH_PLOTS_DIR / f"a3_flight{flight}_timeline.png")


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
                "variables": list(usd.role_config(name).variables),
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


def fly(rows: Sequence[Mapping], cfg: dict, flight_index: int, only_source: bool, design: Path):
    """Fly one flight and record it, see the module docstring."""
    # Imported here, so the dry run needs neither the radio nor ROS
    from crazyflow_experiments.downwash.pair import (
        connect_pair,
        ctrl_c_ends_flight,
        land_pair,
        run_flight,
        takeoff_pair,
    )

    folder = new_recording_path(A3_RESULTS_DIR, f"flight{flight_index}", suffix="")
    _, source_plan = build_flight(rows, cfg)
    targets = {"victim": victim_hover_point(cfg), "source": source_plan.state(0.0)[0]}
    meta = {
        "block": "A3",
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
        consoles = {name: usd.capture_console(drone) for name, drone in drones.items()}
        check_placement(drones, cfg)
        for name, drone in drones.items():
            usd.check_variables(drone, usd.role_config(name).variables)
        meta["vbat_start"] = check_batteries(drones, cfg)
        for name, drone in drones.items():
            usd_times[name]["t_wall_start"] = usd.start_usd_logging(drone)
        airborne = True
        with ctrl_c_ends_flight():  # Ctrl+C lands both and keeps the records of the legs flown
            takeoff = takeoff_pair(victim, source, targets, cfg)
            if takeoff.error is None:
                flight = run_flight(victim, source, rows, cfg)
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
                    usd_times[item[0]]["t_wall_stop"] = usd.stop_usd_logging(item[1])

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


def main():
    """Command line: fly or dry-run one flight of the design."""
    parser = argparse.ArgumentParser(description="Fly one flight of the downwash block A3.")
    parser.add_argument("--flight", type=int, required=True, help="Flight of the design, 1 to 3")
    parser.add_argument("--dry-run", action="store_true", help="Plan only, connect nothing")
    parser.add_argument("--only-source", action="store_true", help="Leave the victim on the ground")
    parser.add_argument("--design", type=Path, default=DESIGN_PATH, help="Design csv")
    args = parser.parse_args()
    cfg = load_config(require_masses=not args.dry_run)
    rows_all = read_design(args.design)
    check_design(rows_all, cfg)
    grouped = flights(rows_all)
    if args.flight not in grouped:
        raise ValueError(f"{args.design} has no flight {args.flight}, only {sorted(grouped)}")
    rows = grouped[args.flight]
    if args.dry_run:
        logger.info(f"Saved {dry_run(rows, cfg, args.flight)}")
        return
    fly(rows, cfg, args.flight, args.only_source, args.design)


if __name__ == "__main__":
    main()
