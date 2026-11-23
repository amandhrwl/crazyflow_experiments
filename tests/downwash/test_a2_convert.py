"""A2 free flight: the steady labels, and the conversion of a synthetic flight with a known wake."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from crazyflow_experiments.downwash import a2_convert  # noqa: E402
from crazyflow_experiments.downwash.a2_convert import steady_labels  # noqa: E402
from crazyflow_experiments.downwash.a2_deploy import A2  # noqa: E402
from crazyflow_experiments.downwash.a2_design import (  # noqa: E402
    build_design,
    load_config,
    retime,
    select_parts,
)
from crazyflow_experiments.downwash.a3_design import gravity, model_params  # noqa: E402
from tests.downwash.test_convert_sync import synthetic_flight  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

CFG = load_config(require_masses=False)
PARAMS = model_params(CFG["drones"]["model"])
ARM = float(np.asarray(PARAMS["L"]))
MIXING = np.asarray(PARAMS["mixing_matrix"], dtype=float)
MASS = 0.045
G = gravity(CFG["drones"]["model"])
AMPLITUDE = 0.03  # N, the wake's pull on the victim on the axis
WIDTH = 0.10  # m


def hover_thrust(n: int, extra: float = 0.0) -> np.ndarray:
    """Four equal rotor thrusts that carry the weight and `extra` N."""
    return np.tile((MASS * G + extra) / 4, (n, 4))


def test_a_level_unloaded_hover_has_no_wake():
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (5, 1))
    force, torque = steady_labels(quat, hover_thrust(5), MASS, G, ARM, MIXING)
    assert np.allclose(force, 0.0, atol=1e-12) and np.allclose(torque, 0.0, atol=1e-12)


def test_the_wake_pulling_down_is_what_the_rotors_carry_extra():
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
    force, _ = steady_labels(quat, hover_thrust(3, extra=0.03), MASS, G, ARM, MIXING)
    assert force == pytest.approx(np.tile([0.0, 0.0, -0.03], (3, 1)))  # F_z = m g - sum f


def test_a_tilt_is_balanced_by_a_horizontal_wake_force():
    """1 deg of pitch (nose down, +x) at hover thrust: the wake pushes the victim along -x by mg sin."""
    pitch, roll = np.deg2rad(1.0), np.deg2rad(2.0)
    quat = R.from_euler("xyz", [roll, pitch, 0.0]).as_quat()[None]
    thrust = hover_thrust(1) / (np.cos(pitch) * np.cos(roll))  # The rotors still carry the vertical
    force, _ = steady_labels(quat, thrust, MASS, G, ARM, MIXING)
    assert force[0, 0] == pytest.approx(-MASS * G * np.sin(pitch), rel=1e-6)
    assert force[0, 1] == pytest.approx(MASS * G * np.cos(pitch) * np.sin(roll), rel=1e-6)


def test_the_rotor_imbalance_is_the_torque_the_wake_has_to_provide():
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (1, 1))
    base = hover_thrust(1)
    for rotor, (tx, ty) in enumerate(
        [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    ):  # mixing_matrix rows 0, 1
        thrust = base.copy()
        thrust[0, rotor] += 0.01
        _, torque = steady_labels(quat, thrust, MASS, G, ARM, MIXING)
        # The rotors' torque is +tx f L about x and +ty f L about y; the wake provides the opposite
        assert torque[0, 0] == pytest.approx(-tx * 0.01 * ARM)
        assert torque[0, 1] == pytest.approx(-ty * 0.01 * ARM)


def read_items(path: Path) -> list[dict]:
    """The rows of an items.csv, numbers as floats and empty cells as NaN."""

    def parse(value: str) -> float | str | bool:
        if value in ("True", "False"):
            return value == "True"
        if value == "":
            return math.nan
        try:
            return float(value)
        except ValueError:
            return value

    with open(path, newline="") as f:
        return [{k: parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


@pytest.fixture(scope="module")
def flown(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Parts 6 and 7 (Δz 0.25 m) flown on the fake drones, the victim's rotors carrying the wake."""
    items = retime(select_parts(build_design(CFG), [6, 7]), CFG)

    def wake(tau: np.ndarray) -> np.ndarray:
        force = np.zeros_like(tau)
        for item in items:
            if item["kind"] == "point":
                inside = (tau >= item["t_dwell_start"]) & (tau < item["t_end"])
                force[inside] = -AMPLITUDE * math.exp(-((item["dx"] / WIDTH) ** 2))
        return force

    folder = tmp_path_factory.mktemp("free_flight1_synthetic")
    result = synthetic_flight(
        folder,
        rows=items,
        hooks=A2.hooks(items),
        wake=wake,
        compensate=True,
        converter=a2_convert.convert_flight,
    )
    rows = read_items(folder / "items.csv")
    return {"folder": folder, "rows": rows, "items": items, "summary": result["summary"]}


def test_every_item_converts(flown: dict):
    assert len(flown["rows"]) == len(flown["items"]) == 10
    assert {row["status"] for row in flown["rows"]} == {"ok"}
    assert (flown["folder"] / "sync.json").is_file() and (flown["folder"] / "checks.json").is_file()


def test_geometry_is_the_source_minus_the_victim(flown: dict):
    for row in flown["rows"]:
        assert row["dx_meas"] == pytest.approx(row["dx"], abs=2e-3)
        assert row["dy_meas"] == pytest.approx(0.0, abs=2e-3)
        assert row["dz_meas"] == pytest.approx(row["height_sep"], abs=2e-3)
        assert row["victim_dz"] == pytest.approx(0.0, abs=2e-3)


def test_both_labels_recover_the_wake_against_the_baseline(flown: dict):
    for row in flown["rows"]:
        expected = (
            -AMPLITUDE * math.exp(-((row["dx"] / WIDTH) ** 2)) if row["kind"] == "point" else 0.0
        )
        assert row["imu_rel_fz_N"] == pytest.approx(expected, abs=4e-3), row["leg"]
        assert row["steady_rel_fz_N"] == pytest.approx(expected, abs=4e-3), row["leg"]
        assert row["steady_minus_imu_fz_N"] == pytest.approx(0.0, abs=4e-3)
        assert abs(row["steady_rel_fx_N"]) < 4e-3 and abs(row["steady_rel_tx_Nm"]) < 1e-4


def test_the_rotors_carry_the_wake(flown: dict):
    on_axis = next(r for r in flown["rows"] if r["kind"] == "point" and r["dx"] == 0.0)
    assert on_axis["weight_ratio"] == pytest.approx(1 + AMPLITUDE / (MASS * G), abs=0.02)
    reference = next(r for r in flown["rows"] if r["kind"] == "reference")
    assert reference["weight_ratio"] == pytest.approx(1.0, abs=0.02)


def test_the_part_ending_on_a_reference_reports_the_baseline_drift(flown: dict):
    drift = [
        r
        for r in flown["rows"]
        if "ref_drift_steady_fz_N" in r and not math.isnan(r["ref_drift_steady_fz_N"])
    ]
    assert len(drift) == 1 and abs(drift[0]["ref_drift_steady_fz_N"]) < 4e-3
    checks = json.loads((flown["folder"] / "checks.json").read_text())
    assert max(checks["ref_drift_steady_max_N"]) < 4e-3
    assert checks["victim_min_dz_m"] == pytest.approx(0.0, abs=5e-3)
    assert all(entry["hop_lags_ok"] for entry in checks["sync"].values())


def test_a_flight_of_the_source_alone_has_its_geometry_only(tmp_path: Path):
    items = retime(select_parts(build_design(CFG), [7]), CFG)
    synthetic_flight(
        tmp_path,
        only_source=True,
        rows=items,
        hooks=A2.hooks(items),
        wake=lambda tau: np.zeros_like(tau),
        compensate=True,
        converter=a2_convert.convert_flight,
    )
    with open(tmp_path / "items.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5 and {r["status"] for r in rows} == {"ok"}
    assert "steady_fz_N" not in rows[0] and float(rows[1]["dz_meas"]) == pytest.approx(
        0.25, abs=2e-3
    )
