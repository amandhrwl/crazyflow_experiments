"""Hover a Crazyflie in place with the Mellinger controller.

The hover is a zero-size ellipse, a trajectory that stays at its origin, so it flies through the
same takeoff, controller and tracking loop as the figure-8 experiment. `sim.py` flies it in the
Crazyflow simulator and `deploy.py` on the real drone, both configured by `config.toml`.
"""

from __future__ import annotations

from crazyflow_experiments.utils.paths import RESULTS_DIR

RECORDINGS_DIR = RESULTS_DIR / "hover"
