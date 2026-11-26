"""The stand chain of LEGS_FORMAT §6.3: unit area, centroid shifts, linearity, local NaN."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.downwash.legs.chain import (  # noqa: E402
    GRID_FREQ,
    centroid_shift,
    chain,
    chain_test,
    shift,
    stand_hash,
)

STANDS = {
    "second_order": {
        "dynamics": {"kind": "second_order", "f_n_hz": 20.0, "zeta": 0.6, "tau_s": 0.01},
        "processing": {"shift_s": 0.004, "filter": "none"},
    },
    "delay": {
        "dynamics": {"kind": "delay", "tau_eff_s": 0.027, "gain_ratio": 0.986, "source": "test"},
        "processing": {"shift_s": 0.0, "filter": "none"},
    },
    "delay_filtered": {
        "dynamics": {"kind": "delay", "tau_eff_s": 0.027, "gain_ratio": 1.0, "source": "test"},
        "processing": {
            "shift_s": 0.005,
            "filter": "butter",
            "order": 4,
            "fc_hz": 15.0,
            "zero_phase": True,
        },
    },
    "unknown": {"dynamics": {"kind": "unknown"}, "processing": {"shift_s": 0.0, "filter": "none"}},
}
T = np.arange(round(-1.0 * GRID_FREQ), round(3.0 * GRID_FREQ) + 1) / GRID_FREQ
IMPULSE = np.where(np.isclose(T, 0.0), GRID_FREQ, 0.0)  # Unit area


@pytest.mark.parametrize("name", list(STANDS))
def test_unit_area(name: str):
    y = chain(IMPULSE, STANDS[name])
    assert np.sum(y) / GRID_FREQ == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", list(STANDS))
def test_centroid_shift(name: str):
    y = chain(IMPULSE, STANDS[name])
    assert np.sum(T * y) / np.sum(y) == pytest.approx(centroid_shift(STANDS[name]), abs=1e-3)


def test_centroid_shift_formulas():
    assert centroid_shift(STANDS["delay"]) == pytest.approx(0.027)
    assert centroid_shift(STANDS["delay_filtered"]) == pytest.approx(0.022)
    assert centroid_shift(STANDS["unknown"]) == 0.0
    assert centroid_shift(STANDS["second_order"]) == pytest.approx(
        0.01 + 2 * 0.6 / (2 * np.pi * 20.0) - 0.004
    )


@pytest.mark.parametrize("name", list(STANDS))
def test_linearity(name: str):
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=len(T)), rng.normal(size=len(T))
    left = chain(2.5 * x - 0.7 * y, STANDS[name])
    right = 2.5 * chain(x, STANDS[name]) - 0.7 * chain(y, STANDS[name])
    np.testing.assert_allclose(left, right, atol=1e-12)


@pytest.mark.parametrize("name", list(STANDS))
def test_nan_stays_local(name: str):
    rng = np.random.default_rng(2)
    x = rng.normal(size=len(T))
    k = len(T) // 2
    x[k] = np.nan
    y = chain(x, STANDS[name])
    bad = np.flatnonzero(~np.isfinite(y))
    assert len(bad) > 0
    assert len(bad) < GRID_FREQ  # Within a second, not the rest of the leg
    assert np.isfinite(y[: k - 250]).all() and np.isfinite(y[k + 250 :]).all()


def test_delay_spreads_nan_to_two_samples():
    x = np.ones(200)
    x[100] = np.nan
    y = shift(x, 0.027)  # 13.5 samples
    assert np.flatnonzero(~np.isfinite(y)).tolist() == [113, 114]


def test_shift_holds_the_first_sample():
    x = np.arange(50, dtype=float) + 3.0
    y = shift(x, 0.01)  # 5 samples later
    np.testing.assert_allclose(y[:6], 3.0)
    np.testing.assert_allclose(y[10], x[5])


def test_chain_test_reproduces():
    stand = STANDS["delay"]
    key = stand_hash(stand)
    arrays = chain_test({key: stand})
    assert key == stand_hash(dict(stand))
    np.testing.assert_allclose(arrays[f"{key}/impulse"], chain(arrays["impulse"], stand), atol=0)
    np.testing.assert_allclose(arrays[f"{key}/rect"], chain(arrays["rect"], stand), atol=0)
    assert np.sum(arrays["rect"]) / GRID_FREQ == pytest.approx(0.3)
