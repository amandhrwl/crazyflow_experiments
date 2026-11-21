"""Block A1: fit the rotor thrust curve to the steady windows and compare it with crazyflow's.

    pixi run -e sim2real python crazyflow_experiments/downwash/a1_fit.py \
        results/downwash/a1/all_<timestamp> [more all blocks] [--single <single block>]

Least squares on the window means, F = 4 c0 + c1 sum(rpm) + c2 sum(rpm^2), the form of crazyflow's
rpm2thrust: (a) with c0 = 0 as crazyflow, (b) with c0 free, and (s) crazyflow's curve times one
scale, the repo's `thrust_scale` (`hover/config.toml`, `downwash/config.toml`), which the hover
flights put at 0.90 to 0.92. Standard errors come from the fit and from a bootstrap over windows.

rho = f_fit / f_crazyflow over the measured rpm decides: within 3 % over 0.6 to 1.6 x hover thrust
per rotor keeps crazyflow's curve, otherwise the fit replaces it. A victim that rests on the load
cell never reaches hover, so the part of that band above its cap is reported as not measured.
Writes `results/downwash/a1/rpm2thrust.toml`, a section of `NOTES.md` and the plots in
`plots/downwash/a1/`.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import toml

from crazyflow_experiments.downwash import A1_PLOTS_DIR, A1_RESULTS_DIR
from crazyflow_experiments.downwash.a1_convert import (
    INK,
    INK_2,
    MUTED,
    SERIES,
    SURFACE,
    read_steady,
    style,
)
from crazyflow_experiments.downwash.a1_stand import N_MOTORS
from crazyflow_experiments.downwash.thrust import rotor_rpm, rotor_thrust

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

N_BOOT = 2000
RHO_TOL = 0.03  # |rho - 1| within this over the band keeps crazyflow's curve
STOP_TOL = 0.10  # The spec's stop-and-ask difference anywhere in the measured range
BAND = (0.6, 1.6)  # Of the hover thrust per rotor, where the decision is made
PER_MOTOR_TOL = 0.03  # Summed per-motor curves against the all-motor thrust
C0_T = 3.0  # |c0| / se above which a free offset counts
C0_RMSE_GAIN = 0.9  # and the free fit's rmse has to drop below this fraction of the other's
REACHED = {"up": "reached from below", "down": "reached from above", "first": "first of a block"}


@dataclass
class Fit:
    """A least squares fit of the window means.

    Attributes:
        name: "zero_offset", "free_offset" or "scale".
        coef: Rotor curve (c0, c1, c2) in N with rpm, crazyflow's form.
        se: Standard errors of `coef` from the fit.
        boot_se: Standard errors of `coef` from the bootstrap.
        rmse: Root mean square residual of the total thrust in N.
        r2: Coefficient of determination.
        residuals: Measured minus fitted total thrust per window in N.
        scale: The factor on crazyflow's curve, for the scale fit.
        scale_se: Its standard error.
    """

    name: str
    coef: NDArray
    se: NDArray
    boot_se: NDArray = field(default_factory=lambda: np.full(3, np.nan))
    rmse: float = np.nan
    r2: float = np.nan
    residuals: NDArray = field(default_factory=lambda: np.empty(0))
    scale: float = np.nan
    scale_se: float = np.nan


def least_squares(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Coefficients, their covariance and the residuals of y = x @ beta.

    The columns (a constant, rpm sums near 1e4 and squared ones near 1e9) are scaled to one first,
    or the normal matrix is too ill-conditioned to invert.
    """
    norm = np.abs(x).max(axis=0)
    norm[norm == 0] = 1.0
    scaled = x / norm
    beta, *_ = np.linalg.lstsq(scaled, y, rcond=None)
    residuals = y - scaled @ beta
    dof = max(len(y) - x.shape[1], 1)
    cov = residuals @ residuals / dof * np.linalg.inv(scaled.T @ scaled)
    return beta / norm, cov / np.outer(norm, norm), residuals


def fit_windows(
    force: NDArray,
    sum_rpm: NDArray,
    sum_rpm2: NDArray,
    cf_coef: NDArray,
    kind: str,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> Fit:
    """Fit the total thrust of the windows.

    Args:
        force: Mean total thrust of each window in N.
        sum_rpm: Sum over the motors of the mean rpm.
        sum_rpm2: Sum over the motors of the mean squared rpm.
        cf_coef: Crazyflow's rpm2thrust.
        kind: "zero_offset" (c0 = 0), "free_offset" or "scale" (crazyflow's curve times s).
        n_boot: Bootstrap resamples over windows.
        seed: Seed of the bootstrap.
    """
    force, sum_rpm, sum_rpm2 = (np.asarray(v, dtype=float) for v in (force, sum_rpm, sum_rpm2))

    n_params = {"zero_offset": 2, "free_offset": 3, "scale": 1}[kind]

    def solve(idx: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        f, s1, s2 = force[idx], sum_rpm[idx], sum_rpm2[idx]
        if kind == "zero_offset":
            beta, cov, res = least_squares(np.column_stack([s1, s2]), f)
            return np.r_[0.0, beta], np.r_[0.0, np.sqrt(np.diag(cov))], res
        if kind == "free_offset":
            x = np.column_stack([np.full_like(s1, N_MOTORS), s1, s2])
            beta, cov, res = least_squares(x, f)
            return beta, np.sqrt(np.diag(cov)), res
        if kind == "scale":
            cf = N_MOTORS * cf_coef[0] + cf_coef[1] * s1 + cf_coef[2] * s2
            beta, cov, res = least_squares(cf[:, None], f)
            return beta[0] * cf_coef, np.sqrt(cov[0, 0]) * np.abs(cf_coef), res
        raise ValueError(f"Unknown fit {kind!r}")

    everything = np.arange(len(force))
    coef, se, residuals = solve(everything)
    rng = np.random.default_rng(seed)
    if len(np.unique(sum_rpm)) < n_params:
        raise ValueError(f"{len(np.unique(sum_rpm))} distinct rpm, the {kind} fit needs {n_params}")
    draws = (rng.integers(0, len(force), len(force)) for _ in range(n_boot))
    # A resample with fewer distinct levels than parameters has no unique fit and is left out
    usable = (idx for idx in draws if len(np.unique(sum_rpm[idx])) >= n_params)
    boot = np.array([solve(idx)[0] for idx in usable])
    tss = float(((force - force.mean()) ** 2).sum())
    fit = Fit(
        kind,
        coef,
        se,
        boot_se=boot.std(axis=0),
        rmse=float(np.sqrt(np.mean(residuals**2))),
        r2=1 - float(residuals @ residuals) / tss if tss > 0 else np.nan,
        residuals=residuals,
    )
    if kind == "scale":
        fit.scale = float(coef[2] / cf_coef[2])
        fit.scale_se = float(se[2] / cf_coef[2])
    return fit


def rho(rpm: NDArray, coef: NDArray, cf_coef: NDArray) -> NDArray:
    """Fitted over crazyflow's rotor thrust at each rpm."""
    return rotor_thrust(rpm, coef) / rotor_thrust(rpm, cf_coef)


def band_check(
    coef: NDArray, cf_coef: NDArray, hover_rotor: float, rpm_range: tuple[float, float]
) -> dict:
    """Where rho stays within RHO_TOL over the decision band, and how much of it was measured.

    Args:
        coef: The fitted curve.
        cf_coef: Crazyflow's curve.
        hover_rotor: Hover thrust per rotor in N, m g / 4.
        rpm_range: Lowest and highest mean rotor speed of the windows.

    Returns:
        The band in rpm (where crazyflow's curve makes 0.6 and 1.6 x hover per rotor), the measured
        fraction of it, the largest |rho - 1| over the measured part and whether it stays within.
    """
    band = rotor_rpm(np.array(BAND) * hover_rotor, cf_coef)
    lo, hi = max(band[0], rpm_range[0]), min(band[1], rpm_range[1])
    coverage = max(0.0, (hi - lo) / (band[1] - band[0]))
    if hi <= lo:
        return {"band_rpm": band.tolist(), "coverage": 0.0, "max_dev": np.nan, "within": None}
    deviation = np.abs(rho(np.linspace(lo, hi, 200), coef, cf_coef) - 1)
    return {
        "band_rpm": band.tolist(),
        "measured_rpm": [lo, hi],
        "coverage": coverage,
        "max_dev": float(deviation.max()),
        "within": bool(deviation.max() <= RHO_TOL),
    }


def hysteresis(residuals: NDArray, direction: Sequence[str]) -> dict:
    """Mean residual of the windows reached from below against those reached from above."""
    direction = np.asarray(direction)
    out = {}
    for name in ("up", "down"):
        r = residuals[direction == name]
        out[name] = {
            "n": len(r),
            "mean_mN": float(r.mean() * 1e3) if len(r) else np.nan,
            "sem_mN": float(r.std(ddof=1) / np.sqrt(len(r)) * 1e3) if len(r) > 1 else np.nan,
        }
    out["difference_mN"] = out["up"]["mean_mN"] - out["down"]["mean_mN"]
    out["difference_se_mN"] = float(np.hypot(out["up"]["sem_mN"], out["down"]["sem_mN"]))
    return out


def per_motor_fits(rows: list[dict], all_coef: NDArray) -> dict[int, dict]:
    """Each motor's own curve from the single-motor block, c0 = 0.

    The three idling motors' thrust comes from the curve of block all and is subtracted.
    """
    out = {}
    for motor in range(1, N_MOTORS + 1):
        own = [r for r in rows if r["motor"] == motor]
        if len(own) < 3:
            continue
        others = np.array(
            [
                sum(
                    all_coef[0] + all_coef[1] * r[f"rpm{j}"] + all_coef[2] * r[f"rpm{j}_sq"]
                    for j in range(1, N_MOTORS + 1)
                    if j != motor
                )
                for r in own
            ]
        )
        force = np.array([r["F_mean"] for r in own]) - others
        x = np.column_stack([[r[f"rpm{motor}"] for r in own], [r[f"rpm{motor}_sq"] for r in own]])
        beta, cov, residuals = least_squares(x, force)
        out[motor] = {
            "coef": [0.0, *beta.tolist()],
            "se": [0.0, *np.sqrt(np.diag(cov)).tolist()],
            "rmse_mN": float(np.sqrt(np.mean(residuals**2)) * 1e3),
            "rpm": [r[f"rpm{motor}"] for r in own],
            "force": force.tolist(),
        }
    return out


def per_motor_closure(rows: list[dict], motors: dict[int, dict]) -> float:
    """Mean relative gap of the summed per-motor curves to the all-motor thrust of block all."""
    if len(motors) < N_MOTORS:
        return np.nan
    gaps = []
    for r in rows:
        total = sum(
            motors[k]["coef"][1] * r[f"rpm{k}"] + motors[k]["coef"][2] * r[f"rpm{k}_sq"]
            for k in range(1, N_MOTORS + 1)
        )
        gaps.append(total / r["F_mean"] - 1)
    return float(np.mean(gaps))


def implied_masses(reference: dict, coef: NDArray, cf_coef: NDArray, g: float) -> dict:
    """Masses the reference hovers imply with the fitted curve instead of crazyflow's.

    A hover whose rpm crazyflow's curve read as `implied_mass_g` was at the rpm where that curve
    makes it; the fitted curve makes rho times as much there.
    """
    out = {}
    for name, implied in reference.get("implied_mass_g", {}).items():
        rpm = float(rotor_rpm(implied * 1e-3 * g / N_MOTORS, cf_coef))
        out[name] = {
            "crazyflow_g": implied,
            "fitted_g": float(implied * rho(rpm, coef, cf_coef)),
            "weighed_g": reference.get("weighed_mass_g", {}).get(name),
            "rpm": rpm,
        }
    return out


def load_blocks(folders: Sequence[Path], keep_flagged: bool) -> tuple[list[dict], dict]:
    """Windows of the blocks, each tagged with its block, and the meta.json of the first."""
    rows, meta = [], None
    for folder in folders:
        folder = Path(folder)
        block_meta = json.loads((folder / "meta.json").read_text())
        meta = meta or block_meta
        for row in read_steady(folder / "steady.csv"):
            if row["ok"] or keep_flagged:
                rows.append(row | {"block": folder.name})
    if meta is None:
        raise ValueError("No block given")
    return rows, meta


def decide(fits: dict[str, Fit], checks: dict[str, dict]) -> dict:
    """Keep crazyflow's curve or replace it, see the module docstring."""
    a, b = fits["zero_offset"], fits["free_offset"]
    free_counts = abs(b.coef[0]) > C0_T * b.se[0] and b.rmse < C0_RMSE_GAIN * a.rmse
    chosen = "free_offset" if free_counts else "zero_offset"
    check = checks[chosen]
    if check["within"] is None:
        verdict = "undecided: none of the band 0.6-1.6 x hover per rotor was measured"
    elif check["within"]:
        verdict = "keep crazyflow's curve"
    else:
        verdict = "use the fitted curve"
    caveat = ""
    if check["coverage"] < 1:
        caveat = (
            f"only {check['coverage']:.0%} of the band was measured: a resting victim, the thrust "
            "cap or a block stopped early (battery) kept the sweep below its top"
        )
    return {"fit": chosen, "verdict": verdict, "caveat": caveat}


def fit(
    folders: Sequence[Path],
    single: Sequence[Path] = (),
    keep_flagged: bool = False,
    results_dir: Path = A1_RESULTS_DIR,
    plots_dir: Path = A1_PLOTS_DIR,
) -> dict:
    """Fit the blocks, decide, and write rpm2thrust.toml, the NOTES.md section and the plots.

    Returns:
        Everything written to rpm2thrust.toml.
    """
    rows, meta = load_blocks(folders, keep_flagged)
    rows = [r for r in rows if r["motor"] == 0]
    if len(rows) < 4:
        raise ValueError(f"{len(rows)} usable windows, at least 4 are needed")
    config = meta["config"]
    g, mass = config["g"], meta["mass_kg"]
    cf_coef = np.asarray(meta["crazyflow_rpm2thrust"], dtype=float)
    force = np.array([r["F_mean"] for r in rows])
    s1, s2 = np.array([r["sum_rpm"] for r in rows]), np.array([r["sum_rpm2"] for r in rows])
    fits = {
        kind: fit_windows(force, s1, s2, cf_coef, kind)
        for kind in ("zero_offset", "free_offset", "scale")
    }
    rotor = s1 / N_MOTORS
    rpm_range = (float(rotor.min()), float(rotor.max()))
    hover_rotor = mass * g / N_MOTORS
    hover_rpm_cf = float(rotor_rpm(hover_rotor, cf_coef))
    checks = {k: band_check(f.coef, cf_coef, hover_rotor, rpm_range) for k, f in fits.items()}
    decision = decide(fits, checks)
    chosen = fits[decision["fit"]]
    grid = np.linspace(*rpm_range, 200)
    max_dev = float(np.abs(rho(grid, chosen.coef, cf_coef) - 1).max())
    if max_dev > STOP_TOL:
        logger.warning(
            f"The fit differs from crazyflow's by up to {max_dev:.1%} over the measured rpm, more "
            "than the spec's 10 %. The hover calibrations (thrust_scale 0.90-0.92) predict about "
            "this much; check the mount and the tare before you trust either"
        )
    reference = config.get("reference", {})
    report = {
        "drone": f"cf{config['victim']['id']}",
        "model": config["victim"]["model"],
        "source": "A1 load cell sweep",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "blocks": [Path(f).name for f in folders],
        "mount": config["victim"]["mount"],
        "clamped": config["victim"]["clamped"],
        "mass_kg": mass,
        "n_windows": len(rows),
        "rpm_range": list(rpm_range),
        "crazyflow": {"rpm2thrust": cf_coef.tolist()},
        "decision": decision,
        "hover": {
            "thrust_per_rotor_N": hover_rotor,
            "rpm_crazyflow": hover_rpm_cf,
            "extrapolated": not rpm_range[0] <= hover_rpm_cf <= rpm_range[1],
            "rho": {k: float(rho(hover_rpm_cf, f.coef, cf_coef)) for k, f in fits.items()},
            "rpm_fitted": float(rotor_rpm(hover_rotor, chosen.coef)),
        },
        "thrust_scale": {
            "fitted": fits["scale"].scale,
            "se": fits["scale"].scale_se,
            "boot_se": float(fits["scale"].boot_se[2] / cf_coef[2]),
            "references": {k: v for k, v in reference.items() if k.startswith("thrust_scale")},
        },
        "implied_masses": implied_masses(reference, chosen.coef, cf_coef, g),
        "max_rho_deviation_measured": max_dev,
        "hysteresis": hysteresis(fits["zero_offset"].residuals, [r["direction"] for r in rows]),
        "drift_mN_per_min": float(
            np.polyfit([r["t_mid"] / 60 for r in rows], fits["zero_offset"].residuals * 1e3, 1)[0]
        ),
    }
    for kind, f in fits.items():
        report[kind] = {
            "rpm2thrust": f.coef.tolist(),
            "se": f.se.tolist(),
            "boot_se": f.boot_se.tolist(),
            "rmse_mN": f.rmse * 1e3,
            "r2": f.r2,
            "band": checks[kind],
        }
    if len(folders) > 1:
        report["per_block_scale"] = per_block_scales(rows, cf_coef)
    motors = {}
    if single:
        single_rows, _ = load_blocks(single, keep_flagged)
        motors = per_motor_fits([r for r in single_rows if r["motor"] > 0], chosen.coef)
        report["per_motor"] = {f"m{k}": v | {"rpm": [], "force": []} for k, v in motors.items()}
        report["per_motor_closure"] = per_motor_closure(rows, motors)
    results_dir.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(report, default=float))  # numpy scalars to plain numbers
    (results_dir / "rpm2thrust.toml").write_text(
        "# Written by a1_fit.py: the rotor thrust curve of block A1, f = c0 + c1 rpm + c2 rpm^2 "
        "in N.\n# With the verdict 'use the fitted curve' the sim twins and every thrust "
        "conversion use it,\n# or thrust_scale.fitted in the repo's thrust_scale convention.\n"
        + toml.dumps(clean)
    )
    append_notes(results_dir / "NOTES.md", report)
    plot_all(rows, fits, cf_coef, report, motors, plots_dir)
    logger.info(f"{decision['verdict']} ({decision['fit']}). {decision['caveat']}")
    logger.info(
        f"thrust_scale {fits['scale'].scale:.4f} +- {fits['scale'].scale_se:.4f}; rho at the "
        f"hover rpm {report['hover']['rho'][decision['fit']]:.3f}"
        f"{' (extrapolated)' if report['hover']['extrapolated'] else ''}"
    )
    return report


def per_block_scales(rows: list[dict], cf_coef: NDArray) -> dict:
    """The scale fit of each block alone, to see whether repeats agree within their errors."""
    out = {}
    for block in sorted({r["block"] for r in rows}):
        own = [r for r in rows if r["block"] == block]
        if len(own) < 3:
            continue
        f = fit_windows(
            np.array([r["F_mean"] for r in own]),
            np.array([r["sum_rpm"] for r in own]),
            np.array([r["sum_rpm2"] for r in own]),
            cf_coef,
            "scale",
            n_boot=200,
        )
        out[block] = {"scale": f.scale, "se": f.scale_se}
    return out


def append_notes(path: Path, report: dict):
    """Add a section with the fit and the decision to NOTES.md."""
    decision, hover, scale = report["decision"], report["hover"], report["thrust_scale"]
    lines = [
        f"\n## Fit {datetime.now().strftime('%Y-%m-%d %H:%M')} ({', '.join(report['blocks'])})\n",
        f"- {report['n_windows']} windows, rotor speeds {report['rpm_range'][0]:.0f} to "
        f"{report['rpm_range'][1]:.0f} rpm, mount {report['mount']}, "
        f"{'clamped' if report['clamped'] else 'resting'}, {report['mass_kg'] * 1e3:.2f} g",
    ]
    for kind in ("zero_offset", "free_offset", "scale"):
        f = report[kind]
        coef = ", ".join(f"{c:.4e}" for c in f["rpm2thrust"])
        se = ", ".join(f"{c:.1e}" for c in f["se"])
        lines.append(
            f"- {kind}: ({coef}) +- ({se}), rmse {f['rmse_mN']:.2f} mN, R2 {f['r2']:.5f}, "
            f"band coverage {f['band']['coverage']:.0%}, max |rho-1| {f['band']['max_dev']:.3f}"
        )
    lines += [
        f"- thrust_scale {scale['fitted']:.4f} +- {scale['se']:.4f} (bootstrap "
        f"{scale['boot_se']:.4f}); references {scale['references']}",
        f"- rho at the hover rpm {hover['rpm_crazyflow']:.0f}: "
        + ", ".join(f"{k} {v:.3f}" for k, v in hover["rho"].items())
        + (" (extrapolated)" if hover["extrapolated"] else ""),
        f"- implied masses: {report['implied_masses']}",
        f"- hysteresis up - down: {report['hysteresis']['difference_mN']:.2f} +- "
        f"{report['hysteresis']['difference_se_mN']:.2f} mN; residual drift "
        f"{report['drift_mN_per_min']:.2f} mN/min",
        f"- **Decision: {decision['verdict']}** ({decision['fit']}). {decision['caveat']}",
    ]
    if "per_motor_closure" in report:
        lines.append(
            f"- per-motor curves sum to block all within {report['per_motor_closure']:+.1%}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def plot_all(
    rows: list[dict],
    fits: dict[str, Fit],
    cf_coef: NDArray,
    report: dict,
    motors: dict[int, dict],
    directory: Path,
):
    """The figures of the fit, one file each."""
    directory.mkdir(parents=True, exist_ok=True)
    force = np.array([r["F_mean"] for r in rows])
    rotor = np.array([r["sum_rpm"] for r in rows]) / N_MOTORS
    f_cf = np.array([r["F_cf"] for r in rows])
    grid = np.linspace(
        0.9 * rotor.min(), 1.05 * max(rotor.max(), report["hover"]["rpm_crazyflow"]), 300
    )
    measured = (grid >= rotor.min()) & (grid <= rotor.max())
    chosen = fits[report["decision"]["fit"]]

    def figure(name: str, draw: Callable[[plt.Axes], None], title: str):
        fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
        style(ax)
        draw(ax)
        ax.set_title(title, loc="left", color=INK)
        fig.tight_layout()
        fig.savefig(directory / f"{name}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    def one_to_one(ax: plt.Axes):
        top = 1.05 * max(force.max(), f_cf.max()) * 1e3
        ax.plot([0, top], [0, top], color=MUTED, lw=1, ls="--", label="1:1")
        ax.plot(
            [0, top],
            [0, top * fits["scale"].scale],
            color=SERIES[1],
            lw=1.5,
            label=f"crazyflow x {fits['scale'].scale:.3f}",
        )
        ax.scatter(
            f_cf * 1e3,
            force * 1e3,
            s=30,
            color=SERIES[0],
            edgecolor=SURFACE,
            lw=1.5,
            zorder=3,
            label="windows",
        )
        ax.set_xlabel("Crazyflow's thrust of the measured rpm [mN]")
        ax.set_ylabel("Load cell thrust [mN]")
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=9)

    def rotor_curve(ax: plt.Axes):
        ax.plot(grid, rotor_thrust(grid, cf_coef) * 1e3, color=INK_2, lw=1.5, label="crazyflow")
        ax.plot(
            grid,
            rotor_thrust(grid, chosen.coef) * 1e3,
            color=SERIES[0],
            lw=1.5,
            label=f"fit ({chosen.name})",
        )
        ax.plot(
            grid,
            rotor_thrust(grid, fits["scale"].coef) * 1e3,
            color=SERIES[1],
            lw=1.5,
            ls="--",
            label="crazyflow x scale",
        )
        ax.scatter(
            rotor,
            force / N_MOTORS * 1e3,
            s=30,
            color=SERIES[0],
            edgecolor=SURFACE,
            lw=1.5,
            zorder=3,
            label="windows",
        )
        ax.axhline(report["hover"]["thrust_per_rotor_N"] * 1e3, color=MUTED, lw=1, ls=":")
        ax.annotate(
            "hover",
            (grid[0], report["hover"]["thrust_per_rotor_N"] * 1e3),
            color=MUTED,
            fontsize=8,
            va="bottom",
        )
        ax.set_xlabel("Mean rotor speed [rpm]")
        ax.set_ylabel("Thrust per rotor [mN]")
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=9)

    def ratio(ax: plt.Axes):
        ax.axhspan(1 - RHO_TOL, 1 + RHO_TOL, color="#efeee9", lw=0, zorder=0)
        for color, kind in zip(SERIES[:3], ("zero_offset", "free_offset", "scale"), strict=True):
            r = rho(grid, fits[kind].coef, cf_coef)
            ax.plot(grid[measured], r[measured], color=color, lw=1.5, label=kind.replace("_", " "))
            ax.plot(grid[~measured], r[~measured], color=color, lw=1, ls=":")
        ax.axvline(report["hover"]["rpm_crazyflow"], color=MUTED, lw=1, ls=":")
        ax.annotate(
            "hover (crazyflow)",
            (report["hover"]["rpm_crazyflow"], 1 + RHO_TOL),
            color=MUTED,
            fontsize=8,
            ha="right",
            va="bottom",
        )
        ax.set_xlabel("Rotor speed [rpm] (dotted: extrapolated)")
        ax.set_ylabel("rho = fitted / crazyflow")
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=9)

    figure("measured_vs_crazyflow", one_to_one, "Load cell against crazyflow's rpm2thrust")
    figure("rotor_curve", rotor_curve, "Thrust per rotor")
    figure("rho", ratio, "rho over the rotor speed, grey: within 3 %")

    fig, (by_rpm, by_time) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, facecolor=SURFACE)
    residual = fits["zero_offset"].residuals * 1e3
    direction = np.array([r["direction"] for r in rows])
    t_mid = np.array([r["t_mid"] for r in rows]) / 60
    for ax, x, label in (
        (by_rpm, rotor, "Mean rotor speed [rpm]"),
        (by_time, t_mid, "Time in the block [min]"),
    ):
        style(ax)
        for color, name in ((SERIES[0], "up"), (SERIES[1], "down"), (MUTED, "first")):
            pick = direction == name
            ax.scatter(
                x[pick],
                residual[pick],
                s=30,
                color=color,
                edgecolor=SURFACE,
                lw=1.5,
                label=REACHED[name],
            )
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel(label)
    by_rpm.set_ylabel("Residual of the zero-offset fit [mN]")
    by_rpm.legend(frameon=False, labelcolor=INK_2, fontsize=9)
    by_rpm.set_title("Residuals", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(directory / "residuals.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    if motors:

        def per_motor(ax: plt.Axes):
            ax.plot(
                grid,
                rotor_thrust(grid, chosen.coef) * 1e3,
                color=INK_2,
                lw=1.5,
                ls="--",
                label="block all",
            )
            for k, m in motors.items():
                ax.plot(
                    grid,
                    rotor_thrust(grid, m["coef"]) * 1e3,
                    color=SERIES[k - 1],
                    lw=1.5,
                    label=f"M{k}",
                )
                ax.scatter(
                    m["rpm"],
                    np.array(m["force"]) * 1e3,
                    s=30,
                    color=SERIES[k - 1],
                    edgecolor=SURFACE,
                    lw=1.5,
                    zorder=3,
                )
            ax.set_xlabel("Rotor speed [rpm]")
            ax.set_ylabel("Thrust of the motor [mN]")
            ax.legend(frameon=False, labelcolor=INK_2, fontsize=9)

        figure("per_motor", per_motor, "Per-motor curves from the single-motor block")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Fit rpm2thrust to the A1 blocks")
    parser.add_argument("folders", type=Path, nargs="+", help="Converted blocks all")
    parser.add_argument(
        "--single", type=Path, nargs="*", default=(), help="Converted blocks single"
    )
    parser.add_argument("--keep-flagged", action="store_true", help="Fit flagged windows as well")
    args = parser.parse_args()
    fit(args.folders, args.single, args.keep_flagged)
