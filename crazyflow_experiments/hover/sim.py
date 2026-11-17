"""Hover with the Mellinger controller in the Crazyflow simulator.

The dry run of `deploy.py`: same config, takeoff and controller.

Run with
    pixi run -e figure8 python crazyflow_experiments/hover/sim.py
"""

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import logging
import pickle
from pathlib import Path

import numpy as np
import toml
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_points

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.control.interface import control_interface
from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.control.simulator import extract_obs
from crazyflow_experiments.control.takeoff import takeoff
from crazyflow_experiments.hover import RECORDINGS_DIR
from crazyflow_experiments.sim2real import ATTITUDE_FREQ, PHYSICS_FREQ
from crazyflow_experiments.sim2real.utils import generate_filename_from_config
from crazyflow_experiments.utils.errors import tracking_rmse
from crazyflow_experiments.utils.paths import new_recording_path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("crazyflow_experiments.hover.sim")  # Not __name__, which is __main__
logger.setLevel(logging.INFO)


def main(config: dict | None = None) -> None:
    """Simulate the configured hover and save its recording.

    Args:
        config: Experiment config, `config.toml` next to this file if None.
    """
    if config is None:
        config = toml.load(Path(__file__).parent / "config.toml")
    data = []

    ctrl_type = config["controller"]["type"]
    if ctrl_type != "mellinger":
        raise ValueError(f"The hover flies the Mellinger controller, not {ctrl_type}")
    ctrl_freq = config["controller"]["ctrl_freq"]
    model = config["drone"]["model"]
    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    pos_start, yaw_start = trajectory_config.start_pose(trajectory)

    sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone=model,
        dynamics=config["sim"]["physics"],
        control=control_interface(ctrl_type),
        freq=PHYSICS_FREQ,
        attitude_freq=ATTITUDE_FREQ,
        device="cpu",
    )
    sim.reset()

    # The controller assumes the weighed mass as on the drone, the simulated drone keeps crazyflow's
    mass = config["drone"]["mass"]
    controller = MellingerController(extract_obs(sim), trajectory, model, mass=mass)
    takeoff(sim, pos_start, yaw_start, ctrl_freq, render=config["sim"]["render"])

    controller._t_start = extract_obs(sim)["t"] - trajectory_config.start_time
    logger.info(f"Hovering at {pos_start} for {trajectory.duration} s")
    i = 0
    while True:
        obs = extract_obs(sim)
        action = controller(obs)
        sim.attitude_control(action[None, None, :])
        sim.step(sim.freq // ctrl_freq)

        obs["u"] = action
        data.append(obs | {"t": obs["t"] - controller._t_start})  # Recorded in trajectory time
        if controller.done(obs):
            break

        if (
            config["sim"]["render"]
            and ((i * config["sim"]["fps"]) % ctrl_freq) < config["sim"]["fps"]
        ):
            draw_points(sim, pos_start[None], [0, 1, 0, 1])
            sim.render()
        i += 1
    sim.close()

    t = np.array([d["t"] for d in data])
    pos = np.array([d["pos"] for d in data])
    logger.info(f"Hover RMSE: {tracking_rmse(t, pos, trajectory):.4f} m")

    path = new_recording_path(RECORDINGS_DIR, generate_filename_from_config(config))
    logger.info(f"Storing data to {path}")
    with open(path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
