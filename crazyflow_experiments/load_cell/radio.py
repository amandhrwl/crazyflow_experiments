"""Radio side of the load cell test: log packets with their PC arrival time, and the victim's link.

The victim does not fly, so it is not a `deploy.crazyflie.Drone` with an estimator.
`Victim` reboots it like a `Drone`, sets the load cell calibration and sample rate and streams the
load cell over the radio. For the spinning test it arms the victim and runs its motors at a fixed
command on the load cell (`Victim.spin`).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cflib.crtp
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils.power_switch import PowerSwitch

from crazyflow_experiments.deploy import crazyflie as deploy_crazyflie
from crazyflow_experiments.load_cell.usd import ConsoleCapture, check_variables, set_param

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

TOC_CACHE = Path(deploy_crazyflie.__file__).parent / ".cache"  # Shared with `Drone`
LOAD_CELL_VARIABLES = ("loadcell.rawWeight", "loadcell.weight")
# Sample rate of the load cell, the battery and the supervisor's state
STATUS_VARIABLES = ("loadcell.rate", "pm.vbat", "supervisor.info")
STATUS_PERIOD_MS = 500
ARMED, CAN_FLY = 0x0002, 0x0008  # Bits of supervisor.info
ARM_TIMEOUT = 4.0  # s, the arming waits up to 2 s for the motors to idle in their rpm range
# s between the zero setpoints that keep the supervisor's watchdog from locking the spinning victim
SETPOINT_PERIOD = 0.05


def uri(radio: int, channel: int, drone_id: int) -> str:
    """Radio URI of a drone, with the address scheme of `deploy.crazyflie.Drone`."""
    return f"radio://{radio}/{channel}/2M/E7E7E7E7{drone_id:02X}"


class RadioLog:
    """Packets of cflib log blocks, each with its onboard tick and its PC arrival time.

    Passed as the data callback of log blocks. cflib calls it from its receive thread as soon as a
    packet is unpacked, so `time.time()` there is the arrival time.
    """

    def __init__(self, name: str, report: tuple[str, ...] = ()):
        """Initialize an empty log.

        Args:
            name: Name of the drone, for the reports.
            report: Integer flags whose every change is logged, e.g. `supervisor.info`.
        """
        self.name = name
        self._report = report
        self._reported: dict[str, float] = {}
        self._packets: dict[str, list[tuple[float, int, dict]]] = {}
        self._lock = threading.Lock()

    def __call__(self, timestamp: int, data: dict, logconf: LogConfig):
        """Store one packet of a log block."""
        t_pc = time.time()
        with self._lock:
            self._packets.setdefault(logconf.name, []).append((t_pc, timestamp, data))
        for variable in self._report:
            if variable in data and data[variable] != self._reported.get(variable):
                self._reported[variable] = data[variable]
                logger.info(f"{self.name} {variable}: {data[variable]:#06x}")

    def latest(self, variable: str) -> float:
        """The last value of a variable received in any block."""
        with self._lock:
            for packets in self._packets.values():
                if packets and variable in packets[-1][2]:
                    return packets[-1][2][variable]
        raise KeyError(f"{self.name}: no packet with {variable} yet")

    def arrays(self) -> dict[str, NDArray]:
        """Every block as arrays `<block>/t_pc`, `<block>/t_tick_ms` and `<block>/<variable>`."""
        out = {}
        with self._lock:
            for block, packets in self._packets.items():
                out[f"{block}/t_pc"] = np.array([p[0] for p in packets])
                out[f"{block}/t_tick_ms"] = np.array([p[1] for p in packets], dtype=np.int64)
                for variable in packets[0][2]:
                    out[f"{block}/{variable}"] = np.array([p[2][variable] for p in packets])
        return out


def start_block(
    cf: Crazyflie, name: str, variables: tuple[str, ...], period_ms: int, callback: RadioLog
) -> LogConfig:
    """Start streaming variables over the radio in one log block of at most 26 bytes."""
    block = LogConfig(name=name, period_in_ms=period_ms)
    for variable in variables:
        block.add_variable(variable)
    cf.log.add_config(block)
    block.data_received_cb.add_callback(callback)
    block.error_cb.add_callback(lambda conf, msg: logger.error(f"Log block {conf.name}: {msg}"))
    block.start()
    return block


class Victim:
    """Link to the victim on the load cell, which streams the load cell and logs to its card."""

    def __init__(self, drone_id: int, channel: int, radio: int, load_cell: dict):
        """Reboot and connect the victim, set up its load cell and start streaming it.

        Args:
            drone_id: ID of the victim.
            channel: Its radio channel.
            radio: Index of the Crazyradio.
            load_cell: The `[load_cell]` section of the config.
        """
        self.name = f"cf{drone_id}"
        self.radio_log = RadioLog(self.name, report=("supervisor.info",))
        self._feeding: threading.Thread | None = None
        self._stop_feeding = threading.Event()
        self.cf = self._connect(uri(radio, channel, drone_id))
        self.console = ConsoleCapture(self.cf)
        check_variables(self.cf, self.name, (*LOAD_CELL_VARIABLES, *STATUS_VARIABLES))
        # The calibration and the rate are not stored on the drone, the boot resets them
        set_param(self.cf, "loadcell.a", load_cell["a"])
        set_param(self.cf, "loadcell.b", load_cell["b"])
        set_param(self.cf, "loadcell.sampleRate", load_cell["sample_rate"])
        self._blocks = [
            start_block(
                self.cf,
                "loadcell",
                LOAD_CELL_VARIABLES,
                load_cell["radio_period_ms"],
                self.radio_log,
            ),
            start_block(self.cf, "status", STATUS_VARIABLES, STATUS_PERIOD_MS, self.radio_log),
        ]

    def spin(self, pwm: int):
        """Arm the victim and run its four motors at `pwm` on the load cell, until `stop_spin`.

        `motorPowerSet` overrides the commands of the controller, but the motors only run while the
        victim is armed. Spinning above idle, it counts as flying for the supervisor, which levels
        out after 0.5 s without a setpoint and stops the motors after 2 s, so a thread sends zero
        setpoints until the stop. A lost link stops the motors the same way.

        Args:
            pwm: Motor command, 0 to 65535, as the firmware's motor ratio.
        """
        self.cf.commander.send_setpoint(0, 0, 0, 0)  # Unlocks the commander
        self._stop_feeding.clear()
        self._feeding = threading.Thread(target=self._feed_setpoints, daemon=True)
        self._feeding.start()
        self.cf.supervisor.send_arming_request(True)
        t_end = time.monotonic() + ARM_TIMEOUT
        while (info := int(self.radio_log.latest("supervisor.info"))) & (ARMED | CAN_FLY) != (
            ARMED | CAN_FLY
        ):
            if time.monotonic() > t_end:
                self.stop_spin()
                raise RuntimeError(
                    f"{self.name} did not arm, supervisor.info {info:#06x}. A motor idling outside "
                    "its rpm range disarms it, see the console"
                )
            time.sleep(0.1)
        for motor in range(1, 5):
            set_param(self.cf, f"motorPowerSet.m{motor}", pwm)
        set_param(self.cf, "motorPowerSet.enable", 1)
        logger.info(f"{self.name} spinning at {pwm}")

    @property
    def spinning(self) -> bool:
        """Whether the victim was set spinning and not stopped yet."""
        return self._feeding is not None

    def stop_spin(self):
        """Disarm the victim, which stops its motors at once, and hand them back to the controller.

        Never raises, it also runs while a failed flight is cleaned up.
        """
        self.cf.supervisor.send_arming_request(False)
        self.cf.param.set_value("motorPowerSet.enable", 0)
        time.sleep(STATUS_PERIOD_MS / 1000 * 2)  # Two status packets, the second after the disarm
        self._stop_feeding.set()
        if self._feeding is not None:
            self._feeding.join()
            self._feeding = None
        if int(self.radio_log.latest("supervisor.info")) & ARMED:
            logger.warning(f"{self.name} still reports armed after the disarm, check its motors")
        logger.info(f"{self.name} motors stopped")

    def _feed_setpoints(self):
        """Send zero setpoints until `stop_spin`, the motor commands come from motorPowerSet."""
        while not self._stop_feeding.wait(SETPOINT_PERIOD):
            self.cf.commander.send_setpoint(0, 0, 0, 0)

    def close(self):
        """Stop streaming and close the link."""
        for block in self._blocks:
            block.stop()
        self.cf.close_link()

    def _connect(self, link_uri: str) -> Crazyflie:
        """Reboot the drone, so it reads its card's config.txt again, and connect to it."""
        cflib.crtp.init_drivers()
        power_switch = PowerSwitch(link_uri)
        power_switch.stm_power_cycle()
        power_switch.close()
        time.sleep(deploy_crazyflie.BOOT_TIME)

        cf = Crazyflie(rw_cache=str(TOC_CACHE))
        connected = mp.Event()
        cf.fully_connected.add_callback(lambda _: connected.set())
        cf.connection_lost.add_callback(lambda _, msg: logger.warning(f"Victim link lost: {msg}"))
        # A stalled connection setup never reports an error, only reopening the link helps, as in
        # `deploy.crazyflie.Drone`
        for _ in range(deploy_crazyflie.CONNECT_ATTEMPTS):
            cf.open_link(link_uri)
            if connected.wait(deploy_crazyflie.CONNECT_TIMEOUT):
                logger.info(f"Victim {self.name} connected to {link_uri}")
                return cf
            logger.warning(f"Connection setup of {self.name} stalled, reopening the link")
            cf.close_link()
        raise RuntimeError(f"{self.name} failed to connect to {link_uri}")
