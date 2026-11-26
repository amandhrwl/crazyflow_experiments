"""The validator of LEGS_FORMAT §8 on a fixture dataset: S0, S1 with a card gap, a duplicate."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import json  # noqa: E402
import shutil  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.downwash.legs.export import (  # noqa: E402
    build_leg,
    resolve_duplicates,
    write_dataset,
)
from crazyflow_experiments.downwash.legs.validate import validate  # noqa: E402
from tests.downwash.legs import fakes  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Three legs: S0, S1 with a card gap away from the wake, and a later attempt of that S1 leg."""
    legs = []
    for mode, run_id, gap in (
        ("still", "stand_flight2_still_2026-11-22_20-17-53", None),
        ("spin", "stand_flight2_spin_2026-11-22_20-24-13", (-2.0, -1.9)),
        ("spin", "stand_flight2_spin_2026-11-22_20-26-54", None),
    ):
        run, row, spin, outcome = fakes.fake_run(mode, card_gap=gap, run_id=run_id, held=True)
        legs.append(build_leg(run, row, spin, outcome, fakes.fake_ctx()))
    resolve_duplicates(legs)
    out = tmp_path / "legs_test"
    write_dataset(out, legs, fakes.header(), fakes.STAND)
    return out


def test_fixture_passes(dataset: Path):
    assert validate(dataset) == {}
    meta = json.loads(next(dataset.glob("A3_S1_*20-24-13*/meta.json")).read_text())
    assert "superseded" in meta["quality"]["flags"] and "card_gap" in meta["quality"]["flags"]


def test_missing_leg_folder(dataset: Path):
    shutil.rmtree(next(dataset.glob("A3_S0_*")))
    assert "dataset" in validate(dataset)


def resave(folder: Path, **changes: np.ndarray):
    states = dict(np.load(folder / "states.npz"))
    states.update(changes)
    np.savez(folder / "states.npz", **states)


def test_moving_victim(dataset: Path):
    folder = next(dataset.glob("A3_S0_*"))
    quat = np.load(folder / "states.npz")["quat"].copy()
    quat[100:, 0] = [0.0, 0.2, 0.0, np.sqrt(1 - 0.04)]
    resave(folder, quat=quat)
    assert any("victim" in e for e in validate(dataset)[folder.name])


def test_thrust_off_its_curve(dataset: Path):
    folder = next(dataset.glob("A3_S1_*20-26-54*"))
    thrust = np.load(folder / "states.npz")["motor_thrust"].copy()
    thrust[:, 0] *= 1.01
    resave(folder, motor_thrust=thrust)
    assert any("rpm2thrust" in e for e in validate(dataset)[folder.name])


def test_nan_outside_the_card_gaps(dataset: Path):
    folder = next(dataset.glob("A3_S1_*20-26-54*"))
    fz = np.load(folder / "states.npz")["fz_total"].copy()
    fz[50] = np.nan
    resave(folder, fz_total=fz)
    assert any("NaN in fz_total" in e for e in validate(dataset)[folder.name])


def test_broken_chain_test(dataset: Path):
    arrays = dict(np.load(dataset / "chain_test.npz"))
    key = next(k for k in arrays if k.endswith("/impulse"))
    arrays[key] = arrays[key] * (1 + 1e-6)
    np.savez(dataset / "chain_test.npz", **arrays)
    assert any("chain_test" in e for e in validate(dataset)["dataset"])
