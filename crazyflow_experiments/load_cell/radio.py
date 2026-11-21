"""Radio side of the load cell test: log packets with their PC arrival time, and the victim's link.

The victim does not fly, so it is not a `deploy.crazyflie.Drone` with an estimator.
`Victim` reboots it like a `Drone`, sets the load cell calibration and sample rate and streams the
load cell over the radio. For the spinning test it arms the victim and runs its motors at a fixed
command on the load cell (`Victim.spin`), block A1 of the downwash experiment at a sequence of
commands (`Victim.set_motors`).
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
    from collections.abc import Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

TOC_CACHE = Path(deploy_crazyflie.__file__).parent / ".cache"  # Shared with `Drone`
LOAD_CELL_VARIABLES = ("loadcell.rawWeight", "loadcell.weight")
# Sample rate of the load cell, the battery and the supervisor's state
STATUS_VARIABLES = ("loadcell.rate", "pm.vbat", "supervisor.info")
STATUS_PERIOD_MS = 500
CAN_ARM, ARMED, CAN_FLY = 0x0001, 0x0002, 0x0008  # Bits of supervisor.info
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

    def recent(self, variable: str, seconds: float) -> NDArray:
        """Values of a variable that arrived within the last `seconds`, in any block."""
        t_from = time.time() - seconds
        with self._lock:
            for packets in self._packets.values():
                if packets and variable in packets[-1][2]:
                    values = []
                    for t_pc, _, data in reversed(packets):  # Newest first, stop at the old ones
                        if t_pc < t_from:
                            break
                        values.append(data[variable])
                    return np.array(values[::-1])
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

    def __init__(
        self,
        drone_id: int,
        channel: int,
        radio: int,
        load_cell: dict,
        extra_blocks: Sequence[tuple[str, tuple[str, ...], int]] = (),
    ):
        """Reboot and connect the victim, set up its load cell and start streaming it.

        Args:
            drone_id: ID of the victim.
            channel: Its radio channel.
            radio: Index of the Crazyradio.
            load_cell: The `[load_cell]` section of the config.
            extra_blocks: Further radio log blocks as name, variables and period in ms.
        """
        self.name = f"cf{drone_id}"
        self.radio_log = RadioLog(self.name, report=("supervisor.info",))
        self._feeding: threading.Thread | None = None
        self._stop_feeding = threading.Event()
        self.cf = self._connect(uri(radio, channel, drone_id))
        self.console = ConsoleCapture(self.cf)
        extra = tuple(v for _, variables, _ in extra_blocks for v in variables)
        check_variables(self.cf, self.name, (*LOAD_CELL_VARIABLES, *STATUS_VARIABLES, *extra))
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
            *(
                start_block(self.cf, name, variables, period_ms, self.radio_log)
                for name, variables, period_ms in extra_blocks
            ),
        ]

    def spin(self, pwm: int | Sequence[int]):
        """Arm the victim and run its motors at `pwm` on the load cell, until `stop_spin`.

        Args:
            pwm: Motor command, 0 to 65535, as the firmware's motor ratio, one for all four motors
                or one per motor, see `set_motors`.
        """
        self.arm()
        self.set_motors(pwm)
        logger.info(f"{self.name} spinning at {pwm}")

    def arm(self):
        """Arm the victim and keep its supervisor fed with zero setpoints until `stop_spin`.

        `motorPowerSet` overrides the motor ratio in the firmware's `motorsSetRatio`, which
        `motorsStop` goes through as well (motors.c). So neither a disarm nor the supervisor's
        watchdogs stop motors under the override, only zero commands or `enable` 0 sent over the
        radio do, and a lost link leaves them at their last command until the power is cut. The
        victim is armed all the same: the arming checks that all four motors idle in their rpm
        range, and the motors-not-responding check that runs while it is armed catches a motor
        that stalls. Spinning above idle it counts as flying, which levels out after 0.5 s without
        a setpoint and locks after 2 s, so a thread sends zero setpoints until the stop.

        The firmware drops an arm request outside PreFlChecksPassed without a word, so the can-arm
        bit of `supervisor.info` is checked first.
        """
        if not (info := int(self.radio_log.latest("supervisor.info"))) & CAN_ARM:
            raise RuntimeError(
                f"{self.name} cannot arm, supervisor.info {info:#06x}. 0x0000 after the boot means "
                "the self-test failed (system.selftestPassed, e.g. an SD buffer too big for the "
                "heap); otherwise it is still leaving a flying or locked state"
            )
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

    def set_motors(self, pwm: int | Sequence[int]):
        """Command the motors of the armed victim through `motorPowerSet`.

        One command goes to all four motors in a single write, `m1` with `enable` 2, so they change
        together; four commands go to `m1` to `m4` with `enable` 1.

        Args:
            pwm: Motor command, 0 to 65535, for all motors or one per motor.
        """
        if not self.spinning:
            raise RuntimeError(f"{self.name} is not armed, call arm() first")
        if isinstance(pwm, int):
            set_param(self.cf, "motorPowerSet.m1", pwm)
            set_param(self.cf, "motorPowerSet.enable", 2)
            return
        if len(pwm) != 4:
            raise ValueError(f"Expected one command per motor, got {pwm}")
        for motor, command in enumerate(pwm, start=1):
            set_param(self.cf, f"motorPowerSet.m{motor}", int(command))
        set_param(self.cf, "motorPowerSet.enable", 1)

    @property
    def spinning(self) -> bool:
        """Whether the victim was armed and not stopped yet."""
        return self._feeding is not None

    def stop_spin(self):
        """Stop the motors and disarm the victim, then hand the motors back to the controller.

        The disarm goes first: it switches off the motors-not-responding check, which a motor
        spinning down through 1 to 500 rpm for 200 ms while armed would trip, locking the
        supervisor. Zero commands then stop the motors, the disarm alone does not under the
        override (see `arm`). Never raises, it also runs while a failed run is cleaned up.
        """
        self.cf.supervisor.send_arming_request(False)
        for motor in range(1, 5):
            self.cf.param.set_value(f"motorPowerSet.m{motor}", 0)
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
