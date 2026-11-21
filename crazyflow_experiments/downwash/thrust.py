"""Rotor thrust from the measured rpm, the one place the downwash blocks compute it.

Crazyflow's `rpm2thrust` of a drone model gives the thrust of one rotor as
`f = c0 + c1 rpm + c2 rpm^2` in N. Each physical drone scales it by its `thrust_scale` from the
config (`config.toml` of A3, `hover/config.toml`), the steady-hover thrust from the logged rpm
over m g, inverted; block A1 measures that curve on the load cell (`a1_fit.py`). A3's force
labels and A1's predictions and fits evaluate the curve here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def curve(rpm2thrust: ArrayLike, thrust_scale: float = 1.0) -> NDArray:
    """Coefficients (3,) of `f = c0 + c1 rpm + c2 rpm^2` in N per rotor, scaled.

    Args:
        rpm2thrust: Crazyflow's `rpm2thrust` of the model, of any shape with three entries.
        thrust_scale: The drone's `thrust_scale` from its config, 1 for crazyflow's own curve.
    """
    return thrust_scale * np.asarray(rpm2thrust, dtype=float).reshape(-1)


def model_curve(model: str, thrust_scale: float = 1.0) -> NDArray:
    """Crazyflow's thrust curve of a drone model, times the drone's `thrust_scale`, see `curve`."""
    from crazyflow.dynamics import load_params  # JAX, only where the curve is needed

    return curve(load_params("first_principles", model)["rpm2thrust"], thrust_scale)


def rotor_thrust(rpm: ArrayLike, coef: ArrayLike) -> NDArray:
    """Thrust of one rotor in N at `rpm`, f = c0 + c1 rpm + c2 rpm^2."""
    c0, c1, c2 = np.asarray(coef, dtype=float)
    rpm = np.asarray(rpm, dtype=float)
    return c0 + c1 * rpm + c2 * rpm**2


def rotor_rpm(thrust: ArrayLike, coef: ArrayLike) -> NDArray:
    """Rpm at which one rotor makes `thrust` in N, the positive root of the curve."""
    c0, c1, c2 = np.asarray(coef, dtype=float)
    thrust = np.asarray(thrust, dtype=float)
    return (-c1 + np.sqrt(c1**2 - 4 * c2 * (c0 - thrust))) / (2 * c2)
