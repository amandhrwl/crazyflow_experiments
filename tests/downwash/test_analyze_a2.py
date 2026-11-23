"""The static map's fit: widths, the jet's axis, the model's row and the thrust loss, on known maps."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import math  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.downwash.analyze_a2 import (  # noqa: E402
    bv_S1,
    fit_row,
    gaussian,
    model_row,
    model_sigma,
    plot_rows,
    plot_width,
    rotor_jet_speed,
    slope,
    summarise,
)

if TYPE_CHECKING:
    from pathlib import Path

OFFSETS = [0.0, 0.05, -0.05, 0.10, -0.10, 0.20, -0.20]


def test_fit_recovers_width_axis_and_level():
    dx = np.linspace(-0.3, 0.3, 13)
    force = gaussian(dx, -0.12, 0.09, 0.012, 0.003)
    fit = fit_row(dx, force)
    assert fit["f0"] == pytest.approx(-0.12, rel=1e-4) and fit["sigma"] == pytest.approx(
        0.09, rel=1e-4
    )
    assert fit["mu"] == pytest.approx(0.012, abs=1e-5) and fit["c"] == pytest.approx(
        0.003, abs=1e-6
    )


def test_fit_survives_noise_and_needs_enough_points():
    rng = np.random.default_rng(3)
    dx = np.array(OFFSETS) + rng.normal(0.0, 0.004, len(OFFSETS))  # The measured, not the nominal
    force = gaussian(dx, -0.15, 0.08, 0.0, 0.0) + rng.normal(0.0, 0.002, len(dx))
    fit = fit_row(dx, force)
    assert fit["sigma"] == pytest.approx(0.08, rel=0.15) and abs(fit["mu"]) < 0.02
    few = fit_row(dx[:3], force[:3])
    assert math.isnan(few["sigma"]) and few["n"] == 3


def test_the_model_row_is_one_on_the_axis_and_falls_off_to_a_wider_jet_higher_up():
    grid = np.linspace(-0.4, 0.4, 81)
    row = model_row(0.5, grid)
    assert row[40] == pytest.approx(1.0) and row.max() == pytest.approx(1.0)
    assert np.allclose(row, row[::-1])  # Symmetric in x
    widths = [model_sigma(h) for h in (0.25, 0.5, 0.75, 1.0)]
    assert all(a < b for a, b in zip(widths, widths[1:]))  # The jet spreads with the height
    assert 0.03 < widths[0] < widths[-1] < 0.30


def test_the_jet_speed_at_the_rotors_is_lower_off_the_axis():
    on_axis = rotor_jet_speed(0.5, 0.0)
    assert on_axis.shape == (4,) and np.allclose(
        on_axis, on_axis[0]
    )  # Four rotors alike on the axis
    off = rotor_jet_speed(0.5, np.array([0.0, 0.1, 0.3]))
    assert off.shape == (3, 4) and off[0].sum() > off[1].sum() > off[2].sum()


def static_map(sigmas: dict[float, float], mu: float = 0.0, amplitude: float = -0.1) -> list[dict]:
    """Rows of a map with a known width per height, the references at their baseline."""
    rows = []
    for height, sigma in sigmas.items():
        for dx in OFFSETS:
            rows.append(
                {
                    "height_sep": height, "dx": dx, "dx_meas": dx + 0.01, "dy_meas": 0.0,
                    "kind": "point",
                    "fz_rel_N": float(gaussian(np.array([dx + 0.01]), amplitude, sigma, mu, 0.0)[0]),
                }
            )  # fmt: skip
        rows.append(
            {
                "height_sep": height,
                "dx": -1.0,
                "dx_meas": -0.99,
                "kind": "reference",
                "fz_rel_N": 0.0,
            }
        )
    return rows


def test_summarise_gives_one_row_per_height_without_the_references():
    maps = {"S0": static_map({0.5: 0.10, 0.25: 0.07}, mu=0.0)}
    summary = summarise(maps)
    assert [(r["condition"], r["height_sep"]) for r in summary] == [("S0", 0.5), ("S0", 0.25)]
    for row, sigma in zip(summary, (0.10, 0.07)):
        assert row["sigma"] == pytest.approx(sigma, rel=1e-3) and row["n"] == 7
        assert row["mu"] == pytest.approx(0.0, abs=1e-3)  # The measured offsets carry the 1 cm
        assert row["sigma_ratio"] == pytest.approx(sigma / model_sigma(row["height_sep"]), rel=1e-3)


def test_a_sideways_offset_of_the_jet_shows_as_the_axis_mu():
    summary = summarise({"S0": static_map({0.5: 0.10}, mu=0.02)})
    assert summary[0]["mu"] == pytest.approx(0.02, abs=2e-3)


def test_width_grows_linearly_with_height():
    heights = {0.25: 0.06, 0.5: 0.08, 0.75: 0.10, 1.0: 0.12}
    summary = summarise({"S0": static_map(heights)})
    growth, offset = slope(summary, "S0")
    assert growth == pytest.approx(0.08, rel=1e-2) and offset == pytest.approx(0.04, abs=2e-3)
    assert math.isnan(slope(summary, "S1")[0])


def test_thrust_loss_is_recovered_from_the_difference_of_the_spin_and_still_runs():
    """S1 wake force = S0 wake force - b sum(U_D f): the b of the model is what comes out."""
    b, thrust = 0.35, 0.133 * 4  # b, and the victim's rotor thrust sum at hover
    s0 = static_map({0.5: 0.10})
    s1 = []
    for row in s0:
        speeds = rotor_jet_speed(0.5, row["dx_meas"], 0.0)
        wake = row["fz_rel_N"] - (
            b * float(speeds.sum() * thrust / 4) if row["kind"] == "point" else 0.0
        )
        s1.append(row | {"dforce_wake_N": wake, "thrust_rpm_N": thrust})
    assert bv_S1(s0, s1)[0.5] == pytest.approx(b, rel=1e-6)


def test_plots_are_written(tmp_path: Path):
    maps = {"S0": static_map({0.5: 0.10, 0.25: 0.07}), "S1": static_map({0.5: 0.11, 0.25: 0.08})}
    summary = summarise(maps)
    assert plot_rows(maps, summary, tmp_path / "rows.png").is_file()
    assert plot_width(summary, tmp_path / "width.png").is_file()
