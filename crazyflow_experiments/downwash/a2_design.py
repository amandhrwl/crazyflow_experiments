"""Item table of block A2: the static map of the wake under a source that holds still.

The source hovers at a grid of points over the victim, `a2.dwell_s` at each, and the wake force on
the victim is measured as a function of the height separation and the offset along x. Per height,
far to near, the points are a reference at `a2.reference_dx` (outside the jet: the tare of the stand,
the bias of the free flight, the drift check), the offsets `a2.offsets_x`, and a last reference. The
offsets are cut into the parts of `a2.part_sizes`, each led by a reference, so that a spin run of
the stand, whose victim battery gives about 45 s of hover, flies one part from its own reference:
[ref, 4 offsets] and [ref, 3 offsets, ref]. Between heights the source changes z at the reference
point, never above the victim.

Every item is a row: `leg` (also `item`) is its index in the whole design, `part` its part, and
`x_src, y_src, z_src` are absolute, from the victim's hover point; the stand and the part subsets
re-place and re-time the rows (`place`, `retime`). Times are seconds from the flight's timeline
zero: `t_start` of the move into the point, `t_dwell_start`, `t_use_start` (after the discarded
`a2.discard_s`) and `t_end`.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/a2_design.py [write|plot]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash import A2_RESULTS_DIR, CONFIG_PATH, DOWNWASH_PLOTS_DIR, plots
from crazyflow_experiments.downwash.a3_design import (
    MOTOR_DISTANCE,
    S0,
    SEPARATION_MARGIN,
    S,
    arrival_time,
    footprint_half_width,
)
from crazyflow_experiments.downwash.a3_design import load_config as load_a3_config
from crazyflow_experiments.downwash.timeline import (
    CEILING_MARGIN,
    build_static_flight,
    move_duration,
    pre_leg_time,
    victim_hover_point,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import ArrayLike

logger = logging.getLogger("crazyflow_experiments.downwash.a2_design")  # Not __name__, the __main__

DESIGN_PATH = A2_RESULTS_DIR / "design_a2.csv"
COLUMNS = (
    "item",
    "leg",
    "flight",
    "index_in_flight",
    "part",
    "height_sep",
    "kind",
    "dx",
    "dy",
    "x_src",
    "y_src",
    "z_src",
    "move_time",
    "t_start",
    "t_dwell_start",
    "t_use_start",
    "t_end",
    "r_half",
    "arrival_s",
    "footprint_half",
)
INT_COLUMNS = ("item", "leg", "flight", "index_in_flight", "part")
STR_COLUMNS = ("kind",)
CHECK_FREQ = 100.0  # Hz, rate the commanded paths are checked at
REFERENCE_MARGIN = 0.1  # m the reference lies outside the widest footprint
WAKE_MARGIN = 0.1  # s the discarded part of a dwell exceeds the wake's arrival by


def jet_half_width(depth: float) -> float:
    """Radius at which the jet's speed is half its centerline value `depth` below the rotors, in m.

    From the jet fit of `a3_design.jet_speed`: its profile falls to a half at `xi = 1`.
    """
    sigma = depth / MOTOR_DISTANCE - S0
    return MOTOR_DISTANCE * S * sigma


def load_config(
    path: Path = CONFIG_PATH,
    require_masses: bool = True,
    mass_roles: Sequence[str] = ("victim", "source"),
) -> dict:
    """Load the experiment config and check the `[a2]` section.

    Args:
        path: The config file.
        require_masses: Require the weighed masses of `mass_roles`.
        mass_roles: The drones whose mass is required, those that fly.

    Returns:
        The config, as `a3_design.load_config` returns it.

    Raises:
        ValueError: The parts do not cover the offsets, the discarded part of a dwell is shorter
            than the wake takes to arrive, or the reference lies inside the widest footprint.
    """
    cfg = load_a3_config(path, require_masses, mass_roles)
    a2, model = cfg["a2"], cfg["drones"]["model"]
    if sum(a2["part_sizes"]) != len(a2["offsets_x"]):
        raise ValueError(
            f"part_sizes {a2['part_sizes']} do not cover {len(a2['offsets_x'])} offsets"
        )
    if not 0 < a2["discard_s"] < a2["dwell_s"]:
        raise ValueError("discard_s has to lie inside dwell_s")
    widest = max(cfg["a3"]["height_seps"])
    if arrival_time(widest, model) + WAKE_MARGIN > a2["discard_s"]:
        raise ValueError(
            f"The wake needs {arrival_time(widest, model):.2f} s to reach the victim at "
            f"{widest} m: raise discard_s"
        )
    if abs(a2["reference_dx"]) < footprint_half_width(widest, model) + REFERENCE_MARGIN:
        raise ValueError("The reference dwells lie inside the wake footprint")
    return cfg


def design_heights(cfg: Mapping) -> list[float]:
    """The free flight's height separations, far to near."""
    return sorted((float(dz) for dz in cfg["a3"]["height_seps"]), reverse=True)


def build_design(cfg: Mapping) -> list[dict]:
    """All items of the free flight, placed and timed from the victim's hover point."""
    a2, model = cfg["a2"], cfg["drones"]["model"]
    sizes = list(a2["part_sizes"])
    offsets = [float(x) for x in a2["offsets_x"]]
    rows: list[dict] = []
    for height_index, height in enumerate(design_heights(cfg)):
        first = 0
        for k, size in enumerate(sizes):
            part = height_index * len(sizes) + k
            points = [("reference", a2["reference_dx"])]
            points += [("point", dx) for dx in offsets[first : first + size]]
            if k == len(sizes) - 1:
                points.append(("reference", a2["reference_dx"]))
            first += size
            for kind, dx in points:
                rows.append(
                    {
                        "part": part,
                        "height_sep": height,
                        "kind": kind,
                        "dx": float(dx),
                        "dy": 0.0,
                        "r_half": jet_half_width(height),
                        "arrival_s": arrival_time(height, model),
                        "footprint_half": footprint_half_width(height, model),
                    }
                )
    return place(rows, victim_hover_point(cfg), cfg)


def place(rows: Sequence[Mapping], origin: ArrayLike, cfg: Mapping) -> list[dict]:
    """Rows with the points over `origin`, the victim's position, and retimed from the takeoff.

    Args:
        rows: Items with `height_sep`, `dx` and `dy`, in flight order.
        origin: Position of the victim as (3,): its hover point, or the stand's.
        cfg: The experiment config.

    Returns:
        Copies with `x_src, y_src, z_src` set, and the times, see `retime`.
    """
    origin = np.asarray(origin, dtype=float)
    placed = []
    for index, row in enumerate(rows):
        row = dict(row)
        row |= {
            "x_src": origin[0] + float(row["dx"]),
            "y_src": origin[1] + float(row["dy"]),
            "z_src": origin[2] + float(row["height_sep"]),
        }
        row.setdefault("item", index)
        row.setdefault("leg", row["item"])
        placed.append(row)
    return retime(placed, cfg)


def retime(rows: Sequence[Mapping], cfg: Mapping) -> list[dict]:
    """Rows renumbered in flight order and timed from the takeoff: the first needs no move.

    Args:
        rows: Placed items in flight order, e.g. the items of some parts.
        cfg: The experiment config.
    """
    a2 = cfg["a2"]
    t, previous, timed = pre_leg_time(cfg), None, []
    for index, row in enumerate(rows):
        row = dict(row)
        pos = np.array([float(row["x_src"]), float(row["y_src"]), float(row["z_src"])])
        move = 0.0 if previous is None else move_duration(previous, pos, cfg["a3"])
        dwell_start = t + move
        row |= {
            "flight": 1,
            "index_in_flight": index,
            "move_time": move,
            "t_start": t,
            "t_dwell_start": dwell_start,
            "t_use_start": dwell_start + a2["discard_s"],
            "t_end": dwell_start + a2["dwell_s"],
        }
        t, previous = row["t_end"], pos
        timed.append(row)
    return timed


def select_parts(rows: Sequence[Mapping], parts: Sequence[int] | None) -> list[dict]:
    """The items of the given parts in flight order, all of them for None; not retimed."""
    if parts is None:
        return [dict(row) for row in rows]
    known = {int(row["part"]) for row in rows}
    if missing := sorted(set(parts) - known):
        raise ValueError(f"Parts {missing} are not in this design, it has {sorted(known)}")
    return [dict(row) for row in rows if int(row["part"]) in parts]


def check_design(rows: Sequence[Mapping], cfg: Mapping):
    """Validate the items: geometry, order and both commanded paths at 100 Hz.

    Every point and both paths lie inside the arena with `x_margin` in the plane and
    `CEILING_MARGIN` below the ceiling, wherever the paths come within `xy_guard` horizontally the
    source is above the separation abort with a margin, the references lie outside the widest
    footprint, and a change of height happens between two references at the same x and y.

    Raises:
        ValueError: A check fails.
    """
    safety = cfg["safety"]
    widest = max(float(row["height_sep"]) for row in rows)
    for row in rows:
        if row["kind"] == "reference" and abs(float(row["dx"])) < footprint_half_width(
            widest, cfg["drones"]["model"]
        ):
            raise ValueError(f"Item {row['leg']}: the reference lies inside the footprint")
    ordered = sorted(rows, key=lambda row: int(row["index_in_flight"]))
    for prev, row in zip(ordered, ordered[1:]):
        if float(prev["height_sep"]) != float(row["height_sep"]):
            same_spot = math.isclose(float(prev["x_src"]), float(row["x_src"]), abs_tol=1e-9)
            same_spot &= math.isclose(float(prev["y_src"]), float(row["y_src"]), abs_tol=1e-9)
            if not (prev["kind"] == row["kind"] == "reference" and same_spot):
                raise ValueError(
                    f"Items {prev['leg']} to {row['leg']}: height changes off the reference"
                )
    margin = np.array([safety["x_margin"], safety["x_margin"], 0.0])
    lo = np.asarray(safety["arena_min"], dtype=float) + margin
    hi = (
        np.asarray(safety["arena_max"], dtype=float) - margin - np.array([0.0, 0.0, CEILING_MARGIN])
    )
    victim, source = build_static_flight(rows, cfg)
    t = np.append(np.arange(0.0, source.duration, 1 / CHECK_FREQ), source.duration)
    p_vic, p_src = victim.state(t)[0], source.state(t)[0]
    for name, pos in (("victim", p_vic), ("source", p_src)):
        outside = np.any((pos < lo - 1e-9) | (pos > hi + 1e-9), axis=1)
        if outside.any():
            raise ValueError(
                f"The {name} leaves the arena at t = {t[int(np.argmax(outside))]:.2f} s"
            )
    close = np.linalg.norm(p_src[:, :2] - p_vic[:, :2], axis=1) < safety["xy_guard"]
    too_low = close & (p_src[:, 2] - p_vic[:, 2] < safety["min_vertical_sep"] + SEPARATION_MARGIN)
    if too_low.any():
        raise ValueError(f"The source flies too low at t = {t[int(np.argmax(too_low))]:.2f} s")


def write_design(rows: Sequence[Mapping], path: Path) -> Path:
    """Write the rows as csv, floats rounded to the µm and µs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: int(row[k])
                    if k in INT_COLUMNS
                    else row[k]
                    if k in STR_COLUMNS
                    else round(float(row[k]), 6)
                    for k in COLUMNS
                }
            )
    return path


def read_design(path: Path = DESIGN_PATH) -> list[dict]:
    """Read a design csv back into typed rows."""
    with open(path, newline="") as f:
        return [
            {
                k: int(v) if k in INT_COLUMNS else v if k in STR_COLUMNS else float(v)
                for k, v in row.items()
            }
            for row in csv.DictReader(f)
        ]


def plot_design(rows: Sequence[Mapping], cfg: Mapping, path: Path) -> Path:
    """Source x and z over the flight with the dwells shaded and the victim's hover point."""
    _, source = build_static_flight(rows, cfg)
    t = np.linspace(0.0, source.duration, int(source.duration * 20) + 1)
    pos = source.state(t)[0]
    p_vic = victim_hover_point(cfg)
    fig, axes = plots.figure(2, 1, width=12.0, row_height=2.6)
    for ax, axis, label in ((axes[0, 0], 0, "x [m]"), (axes[1, 0], 2, "z [m]")):
        for row in rows:
            color = plots.GRID if row["kind"] == "reference" else plots.SHADE
            ax.axvspan(row["t_dwell_start"], row["t_end"], color=color, lw=0)
        ax.axhline(p_vic[axis], color=plots.VICTIM, lw=plots.LINE_WIDTH, label="victim")
        ax.plot(t, pos[:, axis], color=plots.SOURCE, lw=plots.LINE_WIDTH, label="source")
        ax.set_ylabel(label)
    axes[1, 0].set_xlabel(
        "time from the flight's timeline zero [s]; dwells shaded, references darker"
    )
    axes[0, 0].set_title("A2 design: the source's dwells over the victim", loc="left")
    plots.legend(axes[0, 0], loc="upper right", ncols=2)
    return plots.save(fig, path)


def main():
    """Command line: write the design csv, or plot it."""
    parser = argparse.ArgumentParser(description="Item table of the downwash block A2.")
    parser.add_argument("command", nargs="?", choices=("write", "plot"), default="write")
    args = parser.parse_args()
    cfg = load_config(require_masses=False)
    if args.command == "plot":
        rows = read_design()
        logger.info(f"Saved {plot_design(rows, cfg, DOWNWASH_PLOTS_DIR / 'a2_design.png')}")
        return
    rows = build_design(cfg)
    check_design(rows, cfg)
    write_design(rows, DESIGN_PATH)
    _, source = build_static_flight(rows, cfg)
    parts = len({row["part"] for row in rows})
    logger.info(
        f"{len(rows)} items in {parts} parts at {design_heights(cfg)} m, {source.duration:.1f} s "
        f"after takeoff, written to {DESIGN_PATH}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
