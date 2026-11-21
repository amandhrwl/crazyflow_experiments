"""Leg table of block A3: flyover passes of the source over the hovering victim.

The nine cells, three height separations by three speeds, are flown once per flight in three
flights. Flight 1 is the shakedown order, from the far wake to the close passes: height separation
descending, speed ascending within each height. Flights 2 and 3 are shuffles seeded with `seed`.
Every leg flies from -x to +x; after the wait at its end point the source flies back to the next
start point through the return lane, `return_lane` to the side of the victim, clear of its wake.

The fastest speed `v_max` follows from the arena: the source must fit the whole pass, ramps and
settling included, inside the x limits with `x_margin` to spare.

Times are seconds from the flight's timeline zero, the start of the joint loop after takeoff and
calibration. `settled_half_s` is the half-width of the settled window around `t_cross`, and
`settled_end_ext_offset_s` the end of the extended settled window after `t_cross`, where the wake
that arrives `arrival_s` after the source has still been seen.

`--makeup LEG ...` writes one more flight with legs of `--design` (the block by default) to fly
again, e.g. those whose conversion failed on a gap in an SD log. The legs keep their numbers,
cells and repeats, and the flight is numbered after the design's last one; fly it with
`a3_deploy.py --flight <its number> --design <the written csv>`.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/a3_design.py --config a3 [--seed 0]
    pixi run -e sim2real python crazyflow_experiments/downwash/a3_design.py --config a3 \
        --single 1.0 0.5
    pixi run -e sim2real python crazyflow_experiments/downwash/a3_design.py --makeup 4 13 22
    pixi run -e sim2real python crazyflow_experiments/downwash/a3_design.py plot [--design PATH]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import tomllib
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.dynamics import load_params
from scipy.optimize import brentq

from crazyflow_experiments.downwash import A3_RESULTS_DIR, CONFIG_PATH, DOWNWASH_PLOTS_DIR, plots
from crazyflow_experiments.downwash.timeline import (
    CEILING_MARGIN,
    approach_duration,
    build_flight,
    pre_leg_time,
    victim_hover_point,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

logger = logging.getLogger("crazyflow_experiments.downwash.a3_design")  # Not __name__, the __main__

DESIGN_PATH = A3_RESULTS_DIR / "design_a3.csv"
COLUMNS = (
    "leg",
    "flight",
    "index_in_flight",
    "cell",
    "speed",
    "height_sep",
    "lateral_offset",
    "direction",
    "repeat",
    "half",
    "D",
    "x_start",
    "x_end",
    "z_src",
    "move_time",
    "t_leg_start",
    "t_pass_start",
    "t_cross",
    "t_pass_end",
    "t_leg_end",
    "settled_half_s",
    "arrival_s",
    "needed_half_m",
    "settled_end_ext_offset_s",
)
INT_COLUMNS = ("leg", "flight", "index_in_flight", "cell", "direction", "repeat")

MIN_V_MAX = 1.2  # m/s, below it the three speeds no longer span the sim's range; ask first
SETTLED_EXT_MARGIN = 0.15  # s, the sim pipeline's margin after the wake arrival

SEPARATION_MARGIN = 0.05  # m, the commanded paths keep this much above the separation abort
CHECK_FREQ = 100.0  # Hz, rate the commanded paths are checked at

# Jet model of afterwash for the design columns `arrival_s` and `needed_half_m`.
# Source: afterwash `make_design.py`, with the constants of `afterwash/utils/fluid_dynamics.py`.
# The afterwash source was not available when this was written, so the constants and the two
# functions are rebuilt from the afterwash notes: the jet fit of Bauersfeld et al.
# (arXiv:2403.13321, Eq. (9)), the centerline travel time and the two-slot shift of afterwash's
# buffer. They reproduce the A3 numbers, 0.09 / 0.20 / 0.35 s arrival at 0.4 / 0.8 / 1.2 m and
# 0.116 s at 0.5 m, and the footprints to within 0.015 m. [VERIFY] replace by the verbatim copy.
BD = 10.11  # Centerline decay constant of the unified fit
S = 0.07668  # Growth of the jet half-width per motor distance
S0 = -5.817  # Virtual origin in motor distances, above the rotor plane
MOTOR_DISTANCE = 0.1  # m, between opposite motors of the cf21B, 2 sqrt(2) L
PROPELLER_RADIUS = 0.0275  # m
AIR_DENSITY = 1.225  # kg/m^3
SLOT_FREQ = 100.0  # Hz, slot rate of afterwash's buffer, which delays every arrival by two slots
FOOTPRINT_FRACTION = 0.05  # The footprint ends where the victim's rotors see 5 % of the jet


@cache
def model_params(model: str) -> dict:
    """Crazyflow's first-principles parameters of a drone model, as numpy arrays."""
    return load_params("first_principles", model)


def gravity(model: str) -> float:
    """Magnitude of crazyflow's gravity vector in m/s^2."""
    return float(np.linalg.norm(model_params(model)["gravity_vec"]))


def hover_inflow(model: str) -> float:
    """Induced velocity at the rotor disks in hover in m/s, momentum theory on all four disks."""
    thrust = float(np.asarray(model_params(model)["mass"])) * gravity(model)
    area = 4 * math.pi * PROPELLER_RADIUS**2
    return math.sqrt(thrust / (2 * AIR_DENSITY * area))


def _sigma(depth: float | NDArray) -> float | NDArray:
    """Axial distance from the jet's virtual origin in motor distances."""
    return np.asarray(depth) / MOTOR_DISTANCE - S0


def jet_speed(depth: float, radius: NDArray, inflow: float) -> NDArray:
    """Speed of the steady jet at `depth` below the rotors and `radius` from its axis, in m/s."""
    sigma = _sigma(depth)
    xi = np.asarray(radius) / MOTOR_DISTANCE / (S * sigma)
    return inflow * BD / sigma / (1 + (math.sqrt(2) - 1) * xi**2) ** 2


def arrival_time(height_sep: float, model: str = "cf21B_500") -> float:
    """Delay T(Δz) after which the source's jet reaches the victim's rotor plane, in s.

    A parcel on the centerline travels with the centerline speed `inflow * BD / sigma`, which
    integrates to `D / (2 BD u) (sigma(Δz)^2 - sigma(0)^2)`. afterwash's buffer adds two slots.

    Args:
        height_sep: Height of the source's rotors above the victim's in m.
        model: Crazyflow drone model, for the hover inflow.
    """
    travel = MOTOR_DISTANCE / (2 * BD * hover_inflow(model)) * (_sigma(height_sep) ** 2 - S0**2)
    return float(travel) + 2 / SLOT_FREQ


def footprint_half_width(height_sep: float, model: str = "cf21B_500") -> float:
    """Horizontal distance from the jet axis at which the wake on the victim fades, in m.

    The wake on the victim is the jet speed summed over its four rotors. The footprint ends where
    that sum falls to `FOOTPRINT_FRACTION` of its value with the victim on the axis.

    Args:
        height_sep: Height of the source's rotors above the victim's in m.
        model: Crazyflow drone model, for the arm length and the hover inflow.
    """
    arm = float(np.asarray(model_params(model)["L"]))
    rotors = arm * np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    inflow = hover_inflow(model)

    def wake(offset: float) -> float:
        radius = np.linalg.norm(rotors + np.array([offset, 0.0]), axis=1)
        return float(jet_speed(height_sep, radius, inflow).sum())

    on_axis = wake(0.0)
    return brentq(lambda d: wake(d) / on_axis - FOOTPRINT_FRACTION, 0.0, 10.0)


def drag_per_speed(model: str = "cf21B_500") -> float:
    """Horizontal deceleration per m/s of speed from the linear drag of the model, in 1/s.

    0.50 1/s for cf21B_500; the source's cruise tilt in F0 (18 Nov) gave 0.47 1/s.
    """
    params = model_params(model)
    return abs(float(np.asarray(params["drag_matrix"])[0, 0])) / float(np.asarray(params["mass"]))


def cruise_tilt(speed: float, model: str = "cf21B_500") -> float:
    """Tilt of the source in cruise at `speed`, where its thrust balances the linear drag, in rad."""
    return math.atan(drag_per_speed(model) * speed / gravity(model))


def needed_half(height_sep: float, speed: float, model: str = "cf21B_500") -> float:
    """Half-length of cruise the wake needs: the footprint, shifted back by the source's tilt."""
    tilt = cruise_tilt(speed, model)
    return footprint_half_width(height_sep, model) + height_sep * math.tan(tilt)


def pass_half(speed: float, a3: Mapping) -> float:
    """Half-length of a pass: the settled window, the settling distance and half a ramp."""
    return a3["cruise_half"] + speed * (a3["settle_time"] + a3["ramp"] / 2)


def pass_duration(speed: float, a3: Mapping) -> float:
    """Duration of a pass, ramps included."""
    return 2 * pass_half(speed, a3) / speed + a3["ramp"]


def x_limit(cfg: Mapping) -> float:
    """Room along x on the tighter side of the victim's hover point."""
    x_vic = victim_hover_point(cfg)[0]
    safety = cfg["safety"]
    return min(safety["arena_max"][0] - x_vic, x_vic - safety["arena_min"][0])


def max_speed(cfg: Mapping) -> float:
    """Fastest speed whose pass fits the arena with `x_margin`, floored to 0.1 m/s."""
    a3 = cfg["a3"]
    room = x_limit(cfg) - cfg["safety"]["x_margin"] - a3["cruise_half"]
    return math.floor(room / (a3["settle_time"] + a3["ramp"] / 2) * 10 + 1e-9) / 10


def load_config(path: Path = CONFIG_PATH, require_masses: bool = True) -> dict:
    """Load the experiment config, append `v_max` to the speeds and validate it.

    Args:
        path: The config file.
        require_masses: Require the weighed masses. Planning works without them, flying and the
            conversion do not.

    Returns:
        The config, with `a3.v_max` set and appended to `a3.speeds`.
    """
    cfg = tomllib.loads(Path(path).read_text())
    a3 = cfg["a3"]
    v_max = max_speed(cfg)
    if v_max < MIN_V_MAX:
        raise ValueError(f"v_max = {v_max} m/s from the arena is below {MIN_V_MAX} m/s")
    if any(speed >= v_max for speed in a3["speeds"]):
        raise ValueError(f"The configured speeds {a3['speeds']} must stay below v_max = {v_max}")
    a3["v_max"] = v_max
    a3["speeds"] = [*a3["speeds"], v_max]
    check_config(cfg, require_masses)
    return cfg


def check_config(cfg: Mapping, require_masses: bool = True):
    """Validate the static parts of the config, see `load_config`."""
    a3, safety, drones = cfg["a3"], cfg["safety"], cfg["drones"]
    model = drones["model"]
    for height_sep in a3["height_seps"]:
        arrival = arrival_time(height_sep, model)
        if arrival + SETTLED_EXT_MARGIN > a3["settle_time"]:
            raise ValueError(
                f"Δz = {height_sep} m: the wake arrives {arrival:.3f} s late, so the extended "
                f"settled window would outlast the cruise; raise settle_time"
            )
    clearance = max(
        safety["xy_guard"], *(footprint_half_width(dz, model) for dz in a3["height_seps"])
    )
    if abs(a3["return_lane"]) <= clearance:
        raise ValueError(
            f"return_lane = {a3['return_lane']} m would fly the source back within {clearance:.3f} m"
            " of the victim, inside its wake footprint or the separation guard"
        )
    z_top = drones["victim"]["hover_z"] + max(a3["height_seps"])
    if z_top > safety["arena_max"][2] - CEILING_MARGIN + 1e-9:
        raise ValueError(f"The source would fly at {z_top} m, too close to the arena ceiling")
    for role in ("victim", "source"):
        if not 0.5 < drones[role]["thrust_scale"] < 1.5:
            raise ValueError(
                f"drones.{role}.thrust_scale = {drones[role]['thrust_scale']} is no calibration "
                "of crazyflow's rpm2thrust, expected close to 1"
            )
    if require_masses:
        for role in ("victim", "source"):
            if drones[role]["mass_kg"] <= 0:
                raise ValueError(f"Weigh the {role} and fill in drones.{role}.mass_kg")


def design_cells(cfg: Mapping) -> list[tuple[float, float]]:
    """The cells as (height separation, speed), in the shakedown order of flight 1."""
    a3 = cfg["a3"]
    return [(dz, v) for dz in sorted(a3["height_seps"], reverse=True) for v in sorted(a3["speeds"])]


def flight_orders(n_cells: int, repeats: int, seed: int) -> list[list[int]]:
    """Cell order of every flight: the shakedown order, then seeded shuffles.

    Args:
        n_cells: Number of cells.
        repeats: Number of flights, one repeat of every cell each.
        seed: Seed of the shuffles.

    Returns:
        The orders.
    """
    rng = np.random.default_rng(seed)
    shuffles = [rng.permutation(n_cells).tolist() for _ in range(repeats - 1)]
    return [list(range(n_cells)), *shuffles]


def _flight_rows(
    order: Sequence[int],
    cells: Sequence[tuple[float, float]],
    flight: int,
    first_leg: int,
    cfg: dict,
) -> list[dict]:
    """Rows of one flight, timed from its timeline zero."""
    a3 = cfg["a3"]
    model = cfg["drones"]["model"]
    p_vic = victim_hover_point(cfg)
    t, prev_end, rows = pre_leg_time(cfg), None, []
    for index, cell in enumerate(order):
        height_sep, speed = cells[cell]
        direction = 1  # Every pass flies -x to +x, the source flies back in the return lane
        half, duration = pass_half(speed, a3), pass_duration(speed, a3)
        y, z = p_vic[1] + a3["lateral_offset"], p_vic[2] + height_sep
        start = np.array([p_vic[0] - half, y, z])
        end = np.array([p_vic[0] + half, y, z])
        move = 0.0 if prev_end is None else approach_duration(prev_end, start, p_vic, a3)
        t_pass_start = t + move + a3["hover_time"]
        t_pass_end = t_pass_start + duration
        settled_half = a3["cruise_half"] / speed
        arrival = arrival_time(height_sep, model)
        rows.append(
            {
                "leg": first_leg + index,
                "flight": flight,
                "index_in_flight": index,
                "cell": cell,
                "speed": speed,
                "height_sep": height_sep,
                "lateral_offset": a3["lateral_offset"],
                "direction": direction,
                "repeat": flight - 1,
                "half": half,
                "D": duration,
                "x_start": float(start[0]),
                "x_end": float(end[0]),
                "z_src": z,
                "move_time": move,
                "t_leg_start": t,
                "t_pass_start": t_pass_start,
                "t_cross": t_pass_start + duration / 2,
                "t_pass_end": t_pass_end,
                "t_leg_end": t_pass_end + a3["pause_time"] + a3["wait_time"],
                "settled_half_s": settled_half,
                "arrival_s": arrival,
                "needed_half_m": needed_half(height_sep, speed, model),
                "settled_end_ext_offset_s": min(
                    settled_half + arrival + SETTLED_EXT_MARGIN, settled_half + a3["settle_time"]
                ),
            }
        )
        t, prev_end = rows[-1]["t_leg_end"], end
    return rows


def build_design(
    cfg: dict, seed: int | None = None, single: tuple[float, float] | None = None
) -> tuple[list[dict], int | None]:
    """All legs of the block, or a single leg for a quick test.

    Args:
        cfg: The config from `load_config`.
        seed: First seed of the search, `a3.seed` by default.
        single: (speed, height separation) of a one-leg flight instead of the block.

    Returns:
        The rows and the seed that was used, None for a single leg.
    """
    a3 = cfg["a3"]
    if single is not None:
        speed, height_sep = single
        cells, orders, seed_used = [(height_sep, speed)], [[0]], None
    else:
        cells = design_cells(cfg)
        seed_used = a3["seed"] if seed is None else seed
        orders = flight_orders(len(cells), a3["repeats"], seed_used)
    rows: list[dict] = []
    for flight, order in enumerate(orders, start=1):
        rows += _flight_rows(order, cells, flight, len(rows), cfg)
    return rows, seed_used


def build_makeup(cfg: dict, design_rows: Sequence[Mapping], legs: Sequence[int]) -> list[dict]:
    """One more flight that repeats legs of a design, e.g. those a gap in an SD log failed.

    Each leg keeps its number, cell and repeat, so its conversion replaces the failed one. The
    flight is numbered after the last flight of `design_rows`, flies the legs in the given order and
    is timed like every other flight, sync hops and still-air waits included.

    Args:
        cfg: The config from `load_config`.
        design_rows: The design the legs come from.
        legs: Numbers of the legs to repeat.

    Returns:
        The rows of the make-up flight.
    """
    by_leg = {int(row["leg"]): row for row in design_rows}
    if unknown := [leg for leg in legs if leg not in by_leg]:
        raise ValueError(f"Legs {unknown} are not in the design")
    if len(set(legs)) != len(legs):
        raise ValueError(f"Legs {list(legs)} repeat a leg")
    originals = [by_leg[leg] for leg in legs]
    cells = [(float(row["height_sep"]), float(row["speed"])) for row in originals]
    flight = max(int(row["flight"]) for row in design_rows) + 1
    rows = _flight_rows(range(len(cells)), cells, flight, 0, cfg)
    for row, original in zip(rows, originals):
        row |= {key: int(original[key]) for key in ("leg", "cell", "repeat")}
    return rows


def flights(rows: Sequence[Mapping]) -> dict[int, list[Mapping]]:
    """The rows grouped by flight, each in flight order."""
    grouped = defaultdict(list)
    for row in sorted(rows, key=lambda row: (int(row["flight"]), int(row["index_in_flight"]))):
        grouped[int(row["flight"])].append(row)
    return dict(grouped)


def check_design(rows: Sequence[Mapping], cfg: Mapping):
    """Validate every leg: geometry, wake fit and both commanded paths at 100 Hz.

    Both drones stay inside the arena with `x_margin` along x and y and `CEILING_MARGIN` below the
    ceiling, the wake footprint fits the settled window, and wherever the two commanded paths come
    within `xy_guard` horizontally the source is above the separation abort with a margin.
    """
    a3, safety = cfg["a3"], cfg["safety"]
    for row in rows:
        speed = float(row["speed"])
        if speed > a3["v_max"] + 1e-9:
            raise ValueError(f"Leg {row['leg']}: {speed} m/s exceeds v_max = {a3['v_max']}")
        if not math.isclose(float(row["half"]), pass_half(speed, a3), abs_tol=1e-6):
            raise ValueError(f"Leg {row['leg']}: half does not follow from the config profile")
        if float(row["needed_half_m"]) >= a3["cruise_half"]:
            raise ValueError(f"Leg {row['leg']}: the wake footprint exceeds cruise_half")
        if float(row["arrival_s"]) + SETTLED_EXT_MARGIN > a3["settle_time"]:
            raise ValueError(f"Leg {row['leg']}: the extended settled window outlasts the cruise")
    margin = np.array([safety["x_margin"], safety["x_margin"], 0.0])
    lo = np.asarray(safety["arena_min"], dtype=float) + margin
    hi = (
        np.asarray(safety["arena_max"], dtype=float) - margin - np.array([0.0, 0.0, CEILING_MARGIN])
    )
    for flight, flight_rows in flights(rows).items():
        victim, source = build_flight(flight_rows, cfg)
        t = np.append(np.arange(0.0, source.duration, 1 / CHECK_FREQ), source.duration)
        p_vic, p_src = victim.state(t)[0], source.state(t)[0]
        for name, pos in (("victim", p_vic), ("source", p_src)):
            outside = np.any((pos < lo - 1e-9) | (pos > hi + 1e-9), axis=1)
            if outside.any():
                k = int(np.argmax(outside))
                raise ValueError(
                    f"Flight {flight}: the {name} leaves the arena at t = {t[k]:.2f} s"
                )
        close = np.linalg.norm(p_src[:, :2] - p_vic[:, :2], axis=1) < safety["xy_guard"]
        too_low = close & (
            p_src[:, 2] - p_vic[:, 2] < safety["min_vertical_sep"] + SEPARATION_MARGIN
        )
        if too_low.any():
            k = int(np.argmax(too_low))
            raise ValueError(f"Flight {flight}: the source passes too low at t = {t[k]:.2f} s")


def write_design(rows: Sequence[Mapping], path: Path) -> Path:
    """Write the rows as csv, floats rounded to the µm and µs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: int(row[k]) if k in INT_COLUMNS else round(float(row[k]), 6) for k in COLUMNS}
            )
    return path


def read_design(path: Path = DESIGN_PATH) -> list[dict]:
    """Read a design csv back into typed rows."""
    with open(path, newline="") as f:
        return [
            {k: int(v) if k in INT_COLUMNS else float(v) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def plot_design(rows: Sequence[Mapping], cfg: Mapping, path: Path) -> Path:
    """Source x and z over each flight, with the victim's hover point and the leg boundaries."""
    grouped = flights(rows)
    fig, axes = plots.figure(len(grouped), 2, width=12.0, row_height=2.4)
    p_vic = victim_hover_point(cfg)
    for (flight, flight_rows), (ax_x, ax_z) in zip(grouped.items(), axes):
        _, source = build_flight(flight_rows, cfg)
        t = np.linspace(0.0, source.duration, int(source.duration * 20) + 1)
        pos = source.state(t)[0]
        for ax, axis, label in ((ax_x, 0, "x [m]"), (ax_z, 2, "z [m]")):
            for row in flight_rows:
                ax.axvspan(row["t_pass_start"], row["t_pass_end"], color=plots.SHADE, lw=0)
                ax.axvline(row["t_leg_start"], color=plots.GRID, lw=0.8)
            ax.axhline(p_vic[axis], color=plots.VICTIM, lw=plots.LINE_WIDTH, label="victim")
            ax.plot(t, pos[:, axis], color=plots.SOURCE, lw=plots.LINE_WIDTH, label="source")
            ax.set_ylabel(label)
            ax.set_title(f"Flight {flight}, source {label[0]}", loc="left")
        for row in flight_rows:
            ax_z.annotate(
                f"{row['height_sep']:.1f}/{row['speed']:.1f}",
                (row["t_cross"], row["z_src"]),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7,
                color=plots.INK_SECONDARY,
            )
    for ax in axes[-1]:
        ax.set_xlabel("time from the flight's timeline zero [s]")
    plots.legend(axes[0, 0], loc="upper right", ncols=2)
    fig.suptitle("A3 design: passes shaded, labels Δz/v", color=plots.INK, fontsize=11)
    return plots.save(fig, path)


def main():
    """Command line: write the design csv, or plot a design."""
    parser = argparse.ArgumentParser(description="Leg table of the downwash block A3.")
    parser.add_argument("command", nargs="?", choices=("write", "plot"), default="write")
    parser.add_argument("--config", choices=("a3",), default="a3", help="Block of the design")
    parser.add_argument("--seed", type=int, default=None, help="First seed of the shuffle search")
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--single", type=float, nargs=2, metavar=("V", "DZ"), default=None)
    variant.add_argument(
        "--makeup",
        type=int,
        nargs="+",
        metavar="LEG",
        default=None,
        help="Legs of --design to fly again in one more flight",
    )
    parser.add_argument("--design", type=Path, default=None, help="Design csv to plot or make up")
    args = parser.parse_args()
    cfg = load_config(require_masses=False)
    a3 = cfg["a3"]
    if args.command == "plot":
        path = args.design or DESIGN_PATH
        rows = read_design(path)
        name = "a3_design" if path == DESIGN_PATH else path.stem
        logger.info(f"Saved {plot_design(rows, cfg, DOWNWASH_PLOTS_DIR / f'{name}.png')}")
        return
    seed = None
    if args.makeup is not None:
        rows = build_makeup(cfg, read_design(args.design or DESIGN_PATH), args.makeup)
        legs = "-".join(f"{leg:02d}" for leg in args.makeup)
        path = A3_RESULTS_DIR / f"design_makeup_f{rows[0]['flight']}_legs{legs}.csv"
    elif args.single is not None:
        rows, seed = build_design(cfg, single=args.single)
        speed, height_sep = args.single
        path = A3_RESULTS_DIR / f"design_single_v{speed:.2f}_dz{height_sep:.2f}.csv"
    else:
        rows, seed = build_design(cfg, seed=args.seed)
        path = DESIGN_PATH
    check_design(rows, cfg)
    write_design(rows, path)
    logger.info(f"v_max = {a3['v_max']} m/s, speeds {a3['speeds']}, seed {seed}")
    for flight, flight_rows in flights(rows).items():
        victim, _ = build_flight(flight_rows, cfg)
        cells = " ".join(str(r["cell"]) for r in flight_rows)
        logger.info(f"Flight {flight}: {victim.duration:.1f} s, cells {cells}")
    logger.info(f"Wrote {len(rows)} legs to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
