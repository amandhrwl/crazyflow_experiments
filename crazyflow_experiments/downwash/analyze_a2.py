"""The static map of block A2: the jet's width and decay, per height and victim condition.

Input are the ok items of the converted runs: `static_map_S0.csv` and `static_map_S1.csv` of
`a2_stand_convert.py --combine` (the load cell on the stand, motors off and at hover) and, for the
free flight, the `items.csv` of `a2_convert.py` (the victim's steady label). The force is the
vertical force on the victim against the baseline of its part, up positive, at the **measured**
offset of the source from the victim, never the nominal one.

Per height and condition the points of the row, the references excluded, are fitted with

    F_z(dx) = F0 exp(-(dx - mu)^2 / (2 sigma^2)) + c

and the row of the jet model, the sum of `a3_design.jet_speed` over the four rotors (afterwash is
not on this machine), by the same fit: the force averages the jet over the four rotors and their
disks, so its width is the jet's widened, and `sigma_model` is the fair comparison, not `r_half`.
`mu` is the offset of the jet's axis from the victim in x: the mirror test, F_z(+dx) against
F_z(-dx), whose failure is a sideways offset of the stand or of the motion capture frame. The
slope of sigma over the height and the peak `F0` against it follow.

`bv_S1` is the thrust loss coefficient of the victim's rotors, `-(S1 - S0) / sum(U_D f)` over the
points within 0.05 m of the axis, with the model's jet speed at the victim's four rotors for `U_D`,
the rotor thrust of the spin run for `f`, the S1 change of the wake force `dforce_wake_N` and the S0
force at the same item. It stands in for afterwash's twin until that is available.

The maps hold several series of `a2_design.py` (`series` column; rows without it are the `map`). The
fit goes along x and uses one row per height, so only series that hold their points along x and
do not share a height are fitted together: `--series` picks them, by default the map and the
profiles of the S0 and S1 runs. The repeat at a height an earlier series covers (`s0_repeat`) is
fitted on its own, `--series s0_repeat`, and the sideways scan (`s0_scan_y`) is not fitted. A pick
other than the default tags the output files with its series.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/analyze_a2.py \
        [--series map s0_profile s1_map] \
        [--free results/downwash/a2/free_flight1_<time>/items.csv]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit

from crazyflow_experiments.downwash import A2_RESULTS_DIR, DOWNWASH_PLOTS_DIR, plots
from crazyflow_experiments.downwash.a3_design import hover_inflow, jet_speed, model_params

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("crazyflow_experiments.downwash.analyze_a2")  # Not __name__
logger.setLevel(logging.INFO)

MODEL = "cf21B_500"
A2_PLOTS_DIR = DOWNWASH_PLOTS_DIR / "a2"
SUMMARY_PATH = A2_RESULTS_DIR / "a2_summary.csv"
MIN_POINTS = 5  # Points a row needs for the four-parameter fit
AXIS_BAND = 0.05  # m, points this close to the jet's axis carry the b_v estimate
CONDITIONS = {"S0": "fz_rel_N", "S1": "fz_rel_N", "F": "steady_rel_fz_N"}
DEFAULT_SERIES = ("map", "s0_profile", "s1_map")  # Along x, at heights that do not overlap


def gaussian(dx: NDArray, f0: float, sigma: float, mu: float, c: float) -> NDArray:
    """The row's model: a Gaussian of height `f0`, width `sigma`, centre `mu` on a level `c`."""
    return f0 * np.exp(-((dx - mu) ** 2) / (2 * sigma**2)) + c


def fit_row(dx: NDArray, force: NDArray) -> dict[str, float]:
    """Fit a row of the map; NaN for every parameter when it has too few points or does not fit.

    Args:
        dx: Measured offsets in m.
        force: The force at them, in N.

    Returns:
        `f0`, `sigma`, `mu`, `c` and the standard error of `sigma` and `mu`, and the point count.
    """
    nan = {k: math.nan for k in ("f0", "sigma", "mu", "c", "sigma_se", "mu_se")}
    nan["n"] = len(dx)
    if len(dx) < MIN_POINTS:
        return nan
    peak = force[np.argmax(np.abs(force - np.median(force)))] - np.median(force)
    start = [peak, 0.1, 0.0, float(np.median(force))]
    bounds = ([-np.inf, 0.01, -0.3, -np.inf], [np.inf, 1.0, 0.3, np.inf])
    popt, pcov = curve_fit(gaussian, dx, force, p0=start, bounds=bounds, maxfev=20000)
    f0, sigma, mu, c = popt
    errors = np.sqrt(np.diag(pcov))
    return {
        "f0": float(f0), "sigma": float(sigma), "mu": float(mu), "c": float(c),
        "sigma_se": float(errors[1]), "mu_se": float(errors[2]), "n": len(dx),
    }  # fmt: skip


def rotor_positions(model: str = MODEL) -> NDArray:
    """The four rotors' positions in the body's plane, (4, 2) in m."""
    arm = float(np.asarray(model_params(model)["L"]))
    return arm * np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])


def rotor_jet_speed(
    height_sep: float, dx: float | NDArray, dy: float = 0.0, model: str = MODEL
) -> NDArray:
    """The model jet's speed at the victim's four rotors, (..., 4) in m/s, source `dx`, `dy` aside."""
    dx = np.asarray(dx, dtype=float)
    offset = np.stack([dx, np.full_like(dx, dy)], axis=-1)[..., None, :]
    radius = np.linalg.norm(rotor_positions(model) + offset, axis=-1)
    return jet_speed(height_sep, radius, hover_inflow(model))


def model_row(height_sep: float, dx: NDArray) -> NDArray:
    """The model's wake force along a row, normalised to 1 on the axis (the sum over the rotors)."""
    total = rotor_jet_speed(height_sep, dx).sum(axis=-1)
    return total / rotor_jet_speed(height_sep, 0.0).sum(axis=-1)


def model_sigma(height_sep: float) -> float:
    """The width the Gaussian fit gives the model's row at a height."""
    grid = np.linspace(-0.5, 0.5, 101)
    return fit_row(grid, model_row(height_sep, grid))["sigma"]


def read_rows(path: Path) -> list[dict]:
    """The rows of a static map or items csv, numbers as floats and empty cells as NaN."""

    def parse(value: str) -> float | str:
        if value == "":
            return math.nan
        try:
            return float(value)
        except ValueError:
            return value

    with open(path, newline="") as f:
        return [{k: parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def filter_series(rows: Sequence[Mapping], series: Sequence[str]) -> list[dict]:
    """The rows of the given series; rows without a series column are the map's.

    Raises:
        ValueError: A kept point is offset along y, the fit is along x only.
    """
    kept = [dict(r) for r in rows if r.get("series", "map") in series]
    if any(abs(float(r.get("dy", 0.0))) > 1e-9 for r in kept if r["kind"] == "point"):
        raise ValueError(f"Series {list(series)} have points offset along y, the fit is along x")
    return kept


def summarise(maps: Mapping[str, Sequence[Mapping]]) -> list[dict]:
    """Fit every height and condition; one row each, with the model's width and the S1 loss.

    Args:
        maps: The ok items per condition ("S0", "S1", "F").

    Returns:
        The summary rows, by condition and height, near heights last.
    """
    rows = []
    for condition, items in maps.items():
        column = CONDITIONS[condition]
        for height in sorted({float(r["height_sep"]) for r in items}, reverse=True):
            pick = [r for r in items if float(r["height_sep"]) == height and r["kind"] == "point"]
            dx = np.array([r["dx_meas"] for r in pick], dtype=float)
            force = np.array([r[column] for r in pick], dtype=float)
            good = np.isfinite(dx) & np.isfinite(force)
            fit = fit_row(dx[good], force[good])
            row = {"condition": condition, "height_sep": height, **fit}
            row["sigma_model"] = model_sigma(height)
            row["sigma_ratio"] = fit["sigma"] / row["sigma_model"]
            row["n_nominal_offsets"] = len({float(r["dx"]) for r in pick})
            rows.append(row)
    return rows


def bv_S1(s0: Sequence[Mapping], s1: Sequence[Mapping]) -> dict[float, float]:
    """The thrust loss coefficient per height from the points near the axis, see the module docstring."""
    by_item = {(float(r["height_sep"]), float(r["dx"])): r for r in s0 if r["kind"] == "point"}
    out = {}
    for height in sorted({float(r["height_sep"]) for r in s1}):
        loss, load = [], []
        for r in s1:
            key = (height, float(r["dx"]))
            if float(r["height_sep"]) != height or r["kind"] != "point" or key not in by_item:
                continue
            if abs(float(r["dx_meas"])) > AXIS_BAND:
                continue
            speeds = rotor_jet_speed(height, float(r["dx_meas"]), float(r["dy_meas"]))
            thrust = float(r["thrust_rpm_N"]) / 4
            loss.append(float(r["dforce_wake_N"]) - float(by_item[key]["fz_rel_N"]))
            load.append(float(speeds.sum() * thrust))
        out[height] = -float(np.sum(loss) / np.sum(load)) if load else math.nan
    return out


def slope(summary: Sequence[Mapping], condition: str) -> tuple[float, float]:
    """The linear growth of the fitted width with the height, d sigma / d dz, and its offset."""
    pick = [r for r in summary if r["condition"] == condition and math.isfinite(r["sigma"])]
    if len(pick) < 2:
        return math.nan, math.nan
    a, b = np.polyfit([r["height_sep"] for r in pick], [r["sigma"] for r in pick], 1)
    return float(a), float(b)


def plot_rows(
    maps: Mapping[str, Sequence[Mapping]], summary: Sequence[Mapping], path: Path
) -> Path:
    """One panel per height: the force against the measured offset, the conditions overlaid."""
    heights = sorted(
        {float(r["height_sep"]) for items in maps.values() for r in items}, reverse=True
    )
    fig, axes = plots.figure(1, len(heights), width=3.6 * len(heights), row_height=3.2, sharex=True)
    grid = np.linspace(-0.4, 0.4, 161)
    for ax, height in zip(axes[0], heights):
        for color, (condition, items) in zip(plots.SERIES, maps.items()):
            pick = [r for r in items if float(r["height_sep"]) == height and r["kind"] == "point"]
            ax.plot(
                [r["dx_meas"] for r in pick],
                [1e3 * r[CONDITIONS[condition]] for r in pick],
                "o",
                color=color,
                ms=4,
                label=condition,
            )
            fit = next(
                (s for s in summary if s["condition"] == condition and s["height_sep"] == height),
                None,
            )
            if fit is not None and math.isfinite(fit["sigma"]):
                ax.plot(
                    grid,
                    1e3 * gaussian(grid, fit["f0"], fit["sigma"], fit["mu"], fit["c"]),
                    color=color,
                    lw=plots.LINE_WIDTH,
                )
        ax.set_title(f"Δz {height:.2f} m", loc="left")
        ax.set_xlabel("measured Δx [m]")
    axes[0, 0].set_ylabel("F_z against the baseline [mN]")
    plots.legend(axes[0, 0], loc="lower right")
    return plots.save(fig, path)


def plot_width(summary: Sequence[Mapping], path: Path) -> Path:
    """The fitted width and the peak against the height, with the model's width."""
    fig, axes = plots.figure(1, 2, width=10.0, row_height=3.4, sharex=False)
    conditions = sorted({s["condition"] for s in summary})
    for color, condition in zip(plots.SERIES, conditions):
        pick = sorted(
            (s for s in summary if s["condition"] == condition), key=lambda s: s["height_sep"]
        )
        axes[0, 0].plot(
            [s["height_sep"] for s in pick],
            [s["sigma"] for s in pick],
            "o-",
            color=color,
            label=condition,
        )
        axes[0, 1].plot(
            [s["height_sep"] for s in pick],
            [1e3 * s["f0"] for s in pick],
            "o-",
            color=color,
            label=condition,
        )
    heights = np.array(sorted({s["height_sep"] for s in summary}))
    axes[0, 0].plot(
        heights, [model_sigma(h) for h in heights], "--", color=plots.MUTED, label="jet model"
    )
    axes[0, 0].set_xlabel("height separation Δz [m]")
    axes[0, 0].set_ylabel("fitted width σ [m]")
    axes[0, 1].set_xlabel("height separation Δz [m]")
    axes[0, 1].set_ylabel("peak F_z at the axis [mN]")
    plots.legend(axes[0, 0], loc="upper left")
    return plots.save(fig, path)


def main():
    """Command line: summarise the maps found in `results/downwash/a2/`, and the free flight."""
    parser = argparse.ArgumentParser(description="Fit the static map of block A2.")
    parser.add_argument(
        "--free", type=Path, nargs="*", default=[], help="items.csv of free flights"
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=list(DEFAULT_SERIES),
        help="Series to fit, e.g. s0_repeat; the default fits the map and the S0, S1 profiles",
    )
    args = parser.parse_args()
    tag = "" if tuple(args.series) == DEFAULT_SERIES else "_" + "_".join(args.series)
    maps: dict[str, list[dict]] = {}
    for name in ("S0", "S1"):
        path = A2_RESULTS_DIR / f"static_map_{name}.csv"
        if path.is_file():
            if rows := filter_series(read_rows(path), args.series):
                maps[name] = rows
    free = [
        r
        for path in args.free
        for r in filter_series(read_rows(path), args.series)
        if r["status"] == "ok"
    ]
    if free:
        maps["F"] = free
    if not maps:
        raise FileNotFoundError(f"No static map in {A2_RESULTS_DIR}: convert and combine the runs")
    summary = summarise(maps)
    for row in summary:
        logger.info(
            f"{row['condition']} Δz {row['height_sep']:.2f}: n {row['n']}, F0 {1e3 * row['f0']:+.1f} mN,"
            f" sigma {row['sigma']:.3f} m (model {row['sigma_model']:.3f}), mu {1e3 * row['mu']:+.0f} mm"
        )
    if "S0" in maps and "S1" in maps:
        for height, value in bv_S1(maps["S0"], maps["S1"]).items():
            logger.info(f"bv_S1 at Δz {height:.2f}: {value:.3f} (jet model for U_D)")
    for condition in maps:
        logger.info(
            f"{condition}: sigma grows {slope(summary, condition)[0]:.3f} m per m of height"
        )
    summary_path = SUMMARY_PATH.with_stem(f"{SUMMARY_PATH.stem}{tag}")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in summary for k in r)))
        writer.writeheader()
        writer.writerows(summary)
    logger.info(f"Saved {plot_rows(maps, summary, A2_PLOTS_DIR / f'rows_by_height{tag}.png')}")
    logger.info(f"Saved {plot_width(summary, A2_PLOTS_DIR / f'jet_width{tag}.png')}")


if __name__ == "__main__":
    main()
