"""Hover a real Crazyflie with the Mellinger controller.

Flies the same takeoff, controller and tracking loop as `figure8/deploy.py`. Start the motion
capture tracker and the estimator for `cf<id>` first, see the main readme.

The drone logs to its micro-SD card throughout. Each run records into
`results/hover/<name>_<timestamp>/`: `recording.pkl` (the tracked observations and actions),
`firmware.pkl` (the streamed firmware state) and `meta.json` (config, card file). On firmware with
`usd.fileDate` the card names its file after the run, `<stem>_<timestamp>_SD` with the stem of the
card's `config.txt`, and `usd.py collect <card> --block hover` copies it into `sd/drone/`.

Run with
    pixi run -e figure8 python crazyflow_experiments/hover/deploy.py
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"  # TODO remove once scipy uses the array API by default

import json
import logging
import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import toml

from crazyflow_experiments.control.config import TrajectoryConfig
from crazyflow_experiments.control.interface import control_interface
from crazyflow_experiments.control.mellinger_controller import MellingerController
from crazyflow_experiments.deploy.experiment import connect, track
from crazyflow_experiments.deploy.takeoff import takeoff
from crazyflow_experiments.downwash import usd
from crazyflow_experiments.hover import RECORDINGS_DIR
from crazyflow_experiments.sim2real.utils import generate_filename_from_config
from crazyflow_experiments.utils.errors import tracking_rmse
from crazyflow_experiments.utils.paths import new_recording_path

if TYPE_CHECKING:
    from cflib.crazyflie.log import LogConfig

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("crazyflow_experiments.hover.deploy")  # Not __name__, which is __main__
logging.getLogger("crazyflow_experiments").setLevel(logging.INFO)  # Takeoff and flight progress

# Firmware state streamed back during the flight, to see whether the drone is armed and the motors
# run. After arming, the brushless firmware waits up to supervisor.spinupTimeout (2 s) for every
# motor to idle between 1000 and 3500 rpm and disarms if one does not, see supervisor.c
FIRMWARE_LOG = [
    "supervisor.info",
    "pm.vbat",
    "pm.state",
    *(f"motor.m{i}" for i in range(1, 5)),
    *(f"motor.m{i}_rpm" for i in range(1, 5)),
]
# Bits of supervisor.info, lowest first, as documented in the firmware's supervisor.c
SUPERVISOR_FLAGS = (
    "can be armed",
    "armed",
    "auto arm",
    "can fly",
    "flying",
    "tumbled",
    "locked",
    "crashed",
    "high level flying",
    "high level trajectory finished",
    "high level disabled",
    "deck fault",
)


class FirmwareMonitor:
    """Reports the supervisor state on every change and the battery and motors once a second."""

    def __init__(self, period: float = 1.0):
        """Initialize the monitor.

        Args:
            period: Seconds between the battery and motor reports.
        """
        self.samples: list[dict] = []
        self._latest: dict = {}
        self._info: int | None = None
        self._last_report = 0.0
        self._period = period

    def __call__(self, timestamp: int, data: dict, logconf: LogConfig) -> None:
        """Receive one packet of streamed variables from cflib."""
        self._latest |= data
        self.samples.append({"t": timestamp / 1000.0} | data)
        info = self._latest.get("supervisor.info")
        if info is not None and info != self._info:
            self._info = info
            flags = [name for bit, name in enumerate(SUPERVISOR_FLAGS) if info >> bit & 1]
            logger.info(f"Supervisor {info:#06x}: {', '.join(flags) or 'no flags'}")
        now = time.perf_counter()
        if len(self._latest) == len(FIRMWARE_LOG) and now - self._last_report >= self._period:
            self._last_report = now
            s = self._latest
            motors = [s[f"motor.m{i}"] for i in range(1, 5)]
            rpms = [s[f"motor.m{i}_rpm"] for i in range(1, 5)]
            logger.info(
                f"vbat {s['pm.vbat']:.2f} V, pm.state {s['pm.state']}, motors {motors}, rpm {rpms}"
            )


def main(calibrate: bool = True, save: bool = True) -> None:
    """Take off, hover at the configured origin, return home and save the recording.

    Args:
        calibrate: Calibrate the drone orientation once it hovers after takeoff.
        save: Write the recording folder to `results/hover/`.
    """
    config = toml.load(Path(__file__).parent / "config.toml")
    model = config["drone"]["model"]
    if "cf" not in model:
        raise NotImplementedError(f"The hover only flies Crazyflies, not {model}")
    ctrl_type = config["controller"]["type"]
    if ctrl_type != "mellinger":
        raise ValueError(f"The hover flies the Mellinger controller, not {ctrl_type}")
    ctrl_freq = config["controller"]["ctrl_freq"]

    trajectory_config = TrajectoryConfig.from_config(config)
    trajectory = trajectory_config.build()
    pos_start, yaw_start = trajectory_config.start_pose(trajectory)

    initial_obs = {
        "pos": np.zeros(3),
        "quat": np.zeros(4),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "rpy": np.zeros(3),
        "t": time.perf_counter(),
    }
    controller = MellingerController(initial_obs, trajectory, model, mass=config["drone"]["mass"])

    # Reserved now, its timestamp names the card file
    folder = new_recording_path(RECORDINGS_DIR, generate_filename_from_config(config, True), "")
    monitor = FirmwareMonitor()
    drone = connect(
        config, control_interface(ctrl_type), log_variables=FIRMWARE_LOG, log_callback=monitor
    )
    console = usd.ConsoleCapture(drone.cf)
    card: dict = {}

    try:
        card["start"] = usd.start_logging(drone.cf, drone.name, usd.recording_time(folder))
        takeoff(drone, pos_start, yaw_start, calibrate=calibrate)
        logger.info(f"Hovering at {pos_start} for {trajectory.duration} s")
        data = track(drone, controller, ctrl_freq, trajectory_config.start_time)
        logger.info("Hover finished, returning home")
        drone.return_to_start()
    finally:  # Close the SD log file, then kill the drone
        if "start" in card:
            card["stop"] = usd.stop_logging(drone.cf, drone.name)
        card |= {"files": console.files(), "written": console.written()}
        drone.close()
    logger.info(f"Card file {card['files'] or 'not reported, see the card'}")

    if data:
        t = np.array([d["t"] for d in data])
        pos = np.array([d["pos"] for d in data])
        logger.info(f"Hover RMSE: {tracking_rmse(t, pos, trajectory):.4f} m")
    if save:
        folder.mkdir()
        logger.info(f"Storing data to {folder}")
        with open(folder / "recording.pkl", "wb") as f:
            pickle.dump(data, f)
        with open(folder / "firmware.pkl", "wb") as f:
            pickle.dump(monitor.samples, f)
        meta = {"config": config, "sd": {"drone": card}}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
