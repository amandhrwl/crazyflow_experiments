"""The A1 fit recovers known thrust curves from synthetic windows and decides as specified."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import json  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import toml  # noqa: E402

from crazyflow_experiments.downwash import a1_convert, a1_fit  # noqa: E402
from crazyflow_experiments.downwash.a1_fit import (  # noqa: E402
    band_check,
    fit_windows,
    hysteresis,
    per_motor_closure,
    per_motor_fits,
    rho,
)
from tests.downwash import a1_fakes  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

CF = a1_fakes.CF_COEF
NOISE = 2e-3  # N on each window's total thrust


def windows(coef: np.ndarray, n: int = 30, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Total thrust, sum of rpm and sum of squared rpm of n windows from 8000 to 20000 rpm."""
    rng = np.random.default_rng(seed)
    rpm = np.linspace(8000, 20000, n)[:, None] * a1_fakes.MOTOR_SPREAD
    s1, s2 = rpm.sum(axis=1), (rpm**2).sum(axis=1)
    force = 4 * coef[0] + coef[1] * s1 + coef[2] * s2 + rng.normal(0, NOISE, n)
    return force, s1, s2


@pytest.mark.parametrize(
    ("kind", "coef"),
    [
        ("zero_offset", np.array([0.0, -2.0e-7, 4.1e-10])),
        ("free_offset", np.array([4e-3, -2.0e-7, 4.1e-10])),
    ],
)
def test_fit_recovers_known_coefficients(kind: str, coef: np.ndarray):
    """The thrust of the curve over the measured rpm comes back within three standard errors."""
    force, s1, s2 = windows(coef)
    fit = fit_windows(force, s1, s2, CF, kind, n_boot=300)
    grid = np.linspace(8000, 20000, 50)
    error = a1_fit.rotor_thrust(grid, fit.coef) - a1_fit.rotor_thrust(grid, coef)
    assert np.abs(error).max() < 3 * NOISE / 4 * 2
    assert fit.rmse == pytest.approx(NOISE, rel=0.4)
    assert fit.r2 > 0.999
    # The bootstrap agrees with the fit's standard errors to within a factor of two
    nonzero = fit.se > 0
    assert np.all(fit.boot_se[nonzero] / fit.se[nonzero] < 2)
    assert np.all(fit.boot_se[nonzero] / fit.se[nonzero] > 0.5)


def test_scale_fit_recovers_the_thrust_scale():
    """Crazyflow's curve times 0.9046, as the hover of cf3 gave, comes back as the scale."""
    force, s1, s2 = windows(0.9046 * CF, seed=3)
    fit = fit_windows(force, s1, s2, CF, "scale", n_boot=300)
    assert fit.scale == pytest.approx(0.9046, abs=3 * fit.scale_se)
    assert rho(15000.0, fit.coef, CF) == pytest.approx(fit.scale)


def test_band_check_keeps_a_close_curve_and_reports_the_unmeasured_band():
    """Within 3 % keeps crazyflow's; a resting victim's range covers only part of the band."""
    hover_rotor = 0.05 * 9.81 / 4
    band = band_check(1.01 * CF, CF, hover_rotor, (5000.0, 30000.0))
    assert band["within"] and band["coverage"] == pytest.approx(1.0)
    off = band_check(0.9 * CF, CF, hover_rotor, (5000.0, 30000.0))
    assert not off["within"] and off["max_dev"] == pytest.approx(0.1)
    hover_rpm = a1_fit.rotor_rpm(hover_rotor, CF)
    partial = band_check(CF, CF, hover_rotor, (5000.0, float(hover_rpm)))
    assert 0 < partial["coverage"] < 1
    nothing = band_check(CF, CF, hover_rotor, (1000.0, 2000.0))
    assert nothing["within"] is None and nothing["coverage"] == 0


def test_hysteresis_separates_up_and_down_steps():
    """A residual offset between windows reached from below and above shows as their difference."""
    residuals = np.r_[np.full(10, 1e-3), np.full(10, -1e-3)]
    out = hysteresis(residuals, ["up"] * 10 + ["down"] * 10)
    assert out["difference_mN"] == pytest.approx(2.0)


def test_per_motor_curves_subtract_the_idling_motors():
    """Each motor's own curve comes back from the single-motor windows, and they sum up."""
    scales = [0.88, 0.92, 0.90, 0.94]
    idle_rpm = 7300.0
    rows = []
    for motor, scale in enumerate(scales, start=1):
        for rpm in np.linspace(9000, 22000, 6):
            speeds = [rpm if m == motor else idle_rpm for m in range(1, 5)]
            force = sum(
                (scale if m == motor else 0.9) * a1_fit.rotor_thrust(speeds[m - 1], CF)
                for m in range(1, 5)
            )
            rows.append(
                {"motor": motor, "F_mean": force}
                | {f"rpm{m}": speeds[m - 1] for m in range(1, 5)}
                | {f"rpm{m}_sq": speeds[m - 1] ** 2 for m in range(1, 5)}
            )
    motors = per_motor_fits(rows, 0.9 * CF)
    for motor, scale in enumerate(scales, start=1):
        assert motors[motor]["coef"][2] / CF[2] == pytest.approx(scale, rel=1e-6)
    all_rows = [
        {"F_mean": sum(s * a1_fit.rotor_thrust(15000.0, CF) for s in scales)}
        | {f"rpm{m}": 15000.0 for m in range(1, 5)}
        | {f"rpm{m}_sq": 15000.0**2 for m in range(1, 5)}
    ]
    assert per_motor_closure(all_rows, motors) == pytest.approx(0.0, abs=1e-9)


def test_end_to_end_fit_of_a_faked_block_writes_the_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Convert and fit a faked block: the scale is recovered and rpm2thrust.toml written."""
    monkeypatch.setattr(a1_convert, "A1_PLOTS_DIR", tmp_path / "plots")
    folder = tmp_path / "all_fake"
    folder.mkdir()
    a1_fakes.write_block(folder)
    a1_convert.convert(folder)
    report = a1_fit.fit([folder], results_dir=tmp_path / "a1", plots_dir=tmp_path / "plots")
    assert report["thrust_scale"]["fitted"] == pytest.approx(a1_fakes.SCALE, abs=0.01)
    assert report["decision"]["verdict"] in (
        "use the fitted curve",
        "undecided: none of the band 0.6-1.6 x hover per rotor was measured",
    )
    written = toml.loads((tmp_path / "a1" / "rpm2thrust.toml").read_text())
    assert written["drone"] == "cf10" and "zero_offset" in written
    assert "Decision" in (tmp_path / "a1" / "NOTES.md").read_text()
    for name in ("measured_vs_crazyflow", "rotor_curve", "rho", "residuals"):
        assert (tmp_path / "plots" / f"{name}.png").is_file()
    assert json.loads(json.dumps(report, default=float))
