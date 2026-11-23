"""Fly block A2, the static map: the source holds still at a grid of points over the victim.

Start the motion capture and an estimator node per flying drone first, see the downwash README. The
design is `a2_design.py`'s: per height a reference, offsets along x and a last reference, flown far to
near, `a2.dwell_s` at each point. `--condition free` flies the victim hovering and the source over
it, `--condition stand` the source over the victim clamped on the load cell (`--spin` runs the
victim's motors at hover for one part, `--parts` picks the parts, see `a2_stand.py`). The flight, the
SD logging, the landing and the recording are those of `a3_deploy.py`, which this script drives
through its `Block`: everything the conversion needs goes to
`results/downwash/a2/{free_flight,stand_flight}<k>_<mode>_<timestamp>/`.

`--dry-run` prints the item table and the closest approach of the commanded paths, saves a plot and
stops before connecting; on the stand, `--stand-pos` and `--source-pos` stand in for the motion
capture.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/a2_deploy.py --condition stand \
        [--spin] [--parts 4,5] [--dry-run --stand-pos X Y Z --source-pos X Y Z]
    pixi run -e sim2real python crazyflow_experiments/downwash/a2_deploy.py --condition free \
        [--parts 0-3] [--only-source] [--dry-run]
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash import (
    A2_RESULTS_DIR,
    DOWNWASH_PLOTS_DIR,
    a1_stand,
    a2_design,
    a2_stand,
    a3_deploy,
    a3_stand,
)
from crazyflow_experiments.downwash.timeline import build_static_flight, victim_hover_point

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

logger = logging.getLogger("crazyflow_experiments.downwash.a2_deploy")  # Not __name__, the __main__
logging.getLogger("crazyflow_experiments").setLevel(logging.INFO)

STILL_STD = 0.02  # m, spread of the source's position in a dwell's used seconds that is flagged
VICTIM_WARN_DROP = 0.10  # m the free victim sinks below its hover point that is worth a look


def item_summary(
    records: Sequence[Mapping], row: Mapping, times: Mapping, cfg: Mapping
) -> tuple[dict, list[str]]:
    """Numbers of one flown item from its records, for the line printed after it.

    The used seconds are those of the dwell after `a2.discard_s`. The source's position is the
    motion capture's, against the point the item asks for.

    Args:
        records: Records of the flight; those of the item are picked by the source's leg label.
        row: The item's design row.
        times: The item's boundaries in the source timeline.
        cfg: The experiment config.

    Returns:
        The numbers and the item's flags: `off_target` (the mean position in the plane is farther
        from the point than `a2.off_target_tol`), `not_still` (its spread exceeds `STILL_STD`),
        `not_flown` and `short` (no used seconds, e.g. an abort in the item).
    """
    a2 = cfg["a2"]
    item = int(row["leg"])
    recs = [r for r in records if r["source"]["leg"] == item and r["source"]["phase"] == "dwell"]
    if not recs:
        return {}, ["not_flown"]
    t = np.array([r["t_perf"] for r in recs])
    use = t >= t[0] + a2["discard_s"]
    if use.sum() < 10:
        return {}, ["short"]
    pos = np.array([r["source"]["pos"] for r in recs])[use]
    target = np.array([float(row["x_src"]), float(row["y_src"]), float(row["z_src"])])
    error, spread = pos.mean(axis=0) - target, pos.std(axis=0)
    stats = {"error": error.tolist(), "std": spread.tolist(), "n_used": int(use.sum())}
    flags = []
    if float(np.linalg.norm(error[:2])) > a2["off_target_tol"]:
        flags.append("off_target")
    if float(np.linalg.norm(spread[:2])) > STILL_STD:
        flags.append("not_still")
    if recs[0]["victim"] is not None:
        p_vic = victim_hover_point(cfg)
        vic = np.array([r["victim"]["pos"] for r in recs])[use]
        stats["victim_dz"] = float(np.mean(vic[:, 2] - p_vic[2]))
        stats["victim_dxy"] = float(np.linalg.norm(np.mean(vic[:, :2] - p_vic[:2], axis=0)))
        if -stats["victim_dz"] > VICTIM_WARN_DROP:
            flags.append("victim_sinks")
    return stats, flags


def log_item(row: Mapping, stats: Mapping, flags: Sequence[str], skipped: bool):
    """The line printed after an item: its point, how far the source held from it, the victim."""
    head = (
        f"Item {int(row['leg']):2d} (part {int(row['part'])}, Δz {row['height_sep']:.2f} m, "
        f"Δx {row['dx']:+.2f} m, {row['kind']})"
    )
    if skipped or not stats:
        logger.warning(f"{head}: {'skipped' if skipped else ', '.join(flags)}")
        return
    ex, ey, ez = (100 * v for v in stats["error"])
    parts = [
        f"source off target x {ex:+.1f} y {ey:+.1f} z {ez:+.1f} cm",
        f"spread {100 * float(np.linalg.norm(stats['std'][:2])):.1f} cm",
    ]
    if "victim_dz" in stats:
        parts.append(
            f"victim dz {100 * stats['victim_dz']:+.1f} cm, dxy {100 * stats['victim_dxy']:.1f} cm"
        )
    logger.info(f"{head}: {', '.join(parts)}" + (f"  [{', '.join(flags)}]" if flags else ""))


A2 = a3_deploy.Block(
    name="A2",
    results_dir=A2_RESULTS_DIR,
    free_stem="free_flight",
    stand_stem="stand_flight",
    build=build_static_flight,
    plan_stand=a2_stand.plan_stand,
    items_flag="--parts",
    hooks=lambda rows: {
        "build": build_static_flight,
        "summarise": item_summary,
        "log": log_item,
        "close_legs": [],  # No battery gate: the victim hovers a few minutes at most
    },
)


def dry_run(rows: Sequence[Mapping], cfg: Mapping, name: str) -> Path:
    """Log the item table and the path distances of a run and plot its timeline."""
    victim, source = build_static_flight(rows, cfg)
    logger.info(f"{len(rows)} items, {source.duration:.1f} s after takeoff")
    logger.info(
        "item part  Δz[m]  Δx[m] kind       point [m]                dwell start  used  end [s]"
    )
    for row in rows:
        point = np.array([row["x_src"], row["y_src"], row["z_src"]])
        logger.info(
            f"{int(row['leg']):4d} {int(row['part']):4d} {row['height_sep']:6.2f} {row['dx']:+6.2f} "
            f"{row['kind']:10s} {np.array2string(point, precision=2):24s} {row['t_dwell_start']:10.1f} "
            f"{row['t_use_start']:6.1f} {row['t_end']:6.1f}"
        )
    t = np.append(np.arange(0.0, source.duration, 0.01), source.duration)
    p_vic, p_src = victim.state(t)[0], source.state(t)[0]
    gap = np.linalg.norm(p_src - p_vic, axis=1)
    logger.info(
        f"Closest approach of the commanded paths: {gap.min():.3f} m at t = {t[gap.argmin()]:.2f} s"
    )
    margin = np.array([cfg["safety"]["x_margin"], cfg["safety"]["x_margin"], 0.0])
    lo, hi = np.asarray(cfg["safety"]["arena_min"]), np.asarray(cfg["safety"]["arena_max"])
    room = np.minimum(p_src - lo, hi - p_src).min(axis=0) - margin
    logger.info(
        f"Arena room of the source beyond the margin: x {room[0]:.2f} m, y {room[1]:.2f} m, z {room[2]:.2f} m"
    )
    return a2_design.plot_design(rows, cfg, DOWNWASH_PLOTS_DIR / f"a2_{name}_timeline.png")


def _point(values: Sequence[float] | None) -> NDArray | None:
    return None if values is None else np.asarray(values, dtype=float)


def main():
    """Command line: fly or dry-run one run of the static map."""
    parser = argparse.ArgumentParser(description="Fly block A2, the static map.")
    parser.add_argument("--condition", choices=("free", "stand"), required=True)
    parser.add_argument(
        "--spin", action="store_true", help="Stand: run the victim's motors at hover"
    )
    parser.add_argument("--parts", type=a3_stand.parse_legs, help="Parts to fly, e.g. 4,5 or 0-3")
    parser.add_argument("--flight", type=int, default=1, help="Number in the recording's name")
    parser.add_argument(
        "--only-source", action="store_true", help="Free: leave the victim on the ground"
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only, connect nothing")
    parser.add_argument("--design", type=Path, default=a2_design.DESIGN_PATH, help="Design csv")
    parser.add_argument(
        "--stand-pos", type=float, nargs=3, help="Dry run, stand: the victim's position"
    )
    parser.add_argument(
        "--source-pos", type=float, nargs=3, help="Dry run, stand: the source's spot"
    )
    args = parser.parse_args()
    stand = args.condition == "stand"
    if (args.spin or args.stand_pos or args.source_pos) and not stand:
        raise ValueError("--spin, --stand-pos and --source-pos go with --condition stand")
    flying = ("source",) if stand or args.only_source else ("victim", "source")
    cfg = a2_design.load_config(require_masses=not args.dry_run, mass_roles=flying)
    rows = a2_design.read_design(args.design)
    if stand:
        mode = "spin" if args.spin else "still"
        if args.dry_run:
            stand_pos, source_pos = _point(args.stand_pos), _point(args.source_pos)
            if stand_pos is None or source_pos is None:
                raise ValueError(
                    "A dry run of --condition stand needs --stand-pos and --source-pos"
                )
            stand_cfg, items, schedule = a2_stand.plan_stand(
                cfg, rows, stand_pos, source_pos, args.parts, args.spin
            )
            logger.info(f"Stand run, {mode}: victim at {stand_pos}, source at {source_pos}")
            logger.info(f"Hover command: {a3_deploy.stand_hover_cmd(cfg, a1_stand.load_config())}")
            logger.info(f"Saved {dry_run(items, stand_cfg, f'stand_{mode}')}")
            a3_deploy.log_schedule(schedule, cfg["stand"], args.spin)
            return
        a3_deploy.fly_stand(rows, cfg, args.flight, args.design, mode, args.parts, block=A2)
        return
    items = a2_design.retime(a2_design.select_parts(rows, args.parts), cfg)
    a2_design.check_design(items, cfg)
    if args.dry_run:
        logger.info(f"Saved {dry_run(items, cfg, 'free')}")
        return
    a3_deploy.fly(items, cfg, args.flight, args.only_source, args.design, block=A2)


if __name__ == "__main__":
    main()
