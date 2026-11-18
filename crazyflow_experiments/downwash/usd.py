"""Onboard logging to the micro-SD card deck of both drones.

The firmware (`usddeck.c`) reads `config.txt` once at boot and opens the next free file `<name>NN`
each time `usd.logging` is set to 1. The file only gets its CRC32 when logging is stopped again
with `usd.logging = 0`. The variable lists of both cards live in `sd/config_victim.txt` and
`sd/config_source.txt`, and `write-config` copies one of them onto a card as its `config.txt`:

    pixi run -e sim2real python crazyflow_experiments/downwash/usd.py write-config /media/<card> \
        --role victim

`bench` logs on the ground for a flight's length and reports how many events the card dropped
while it stalled, without taking the card out:

    pixi run -e sim2real python crazyflow_experiments/downwash/usd.py bench --role source

`decode_usd` reads a log file back into numpy arrays. It is an own implementation of the format
(moved here from `sd_log_inspection.ipynb`), the firmware's GPL decoder `cfusdlog.py` is not used.

Only numpy and the standard library are imported at module level, so the notebook can import the
decoder without cflib or ROS. The drone helpers import cflib when they are called.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import struct
import threading
import time
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash import SD_CONFIG_DIR

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from crazyflow_experiments.deploy.crazyflie import Drone

logger = logging.getLogger(__name__)

ROLES = ("victim", "source")
MAX_VARIABLES = 20  # MAX_USD_LOG_VARIABLES_PER_EVENT, the firmware skips further lines silently
RPM_INVALID = 65535  # motor.mX_rpm without a reading
STOP_WAIT = 1.0  # s, the writer task closes the file in the background after the stop
LOGGING_TO = re.compile(r"Logging to:\s*(\S+)")  # Console line naming the file a start opens
SKIPPED_VARIABLE = "Skip log variable"  # Console warning for a variable the firmware drops
WROTE = re.compile(r"Wrote \d+ B to:\s*(\S+) \((\d+) of (\d+) events\)")  # Console line at a stop
BENCH_REPORT = 30.0  # s between progress lines of `bench`


@dataclass(frozen=True)
class UsdConfig:
    """Contents of a card's `config.txt`.

    Attributes:
        version: Format version of the config file.
        buffer: Size of the write buffer in bytes.
        name: Stem of the log files, the firmware appends the number.
        enable_on_startup: Whether logging starts at boot.
        event: The logged event, `on:fixedFrequency` for synchronous logging.
        frequency: Logging rate in Hz, a divisor of the 1 kHz stabilizer rate.
        mode: 1 logs synchronously with the stabilizer.
        variables: The logged variables in file order.
    """

    version: int
    buffer: int
    name: str
    enable_on_startup: bool
    event: str
    frequency: int
    mode: int
    variables: tuple[str, ...]


def config_path(role: str) -> Path:
    """Path of the stored `config.txt` of a role.

    Args:
        role: "victim" or "source".
    """
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}, expected one of {ROLES}")
    return SD_CONFIG_DIR / f"config_{role}.txt"


def read_config(path: Path) -> UsdConfig:
    """Parse and check a `config.txt`.

    The first seven lines are the header, every further line one variable. The firmware takes a
    variable line verbatim, so variable lines must not carry comments.

    Args:
        path: The config file.

    Returns:
        The parsed config.
    """
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError(f"{path}: a header of 7 lines and at least one variable expected")
    header = [line.split()[0] for line in lines[:7]]  # The firmware reads up to the first blank
    config = UsdConfig(
        version=int(header[0]),
        buffer=int(header[1]),
        name=header[2],
        enable_on_startup=bool(int(header[3])),
        event=header[4],
        frequency=int(header[5]),
        mode=int(header[6]),
        variables=tuple(lines[7:]),
    )
    if not config.event.startswith("on:"):
        raise ValueError(f"{path}: event line {config.event!r} must start with 'on:'")
    if config.frequency <= 0 or 1000 % config.frequency != 0:
        raise ValueError(f"{path}: frequency {config.frequency} Hz must divide the 1 kHz loop")
    if len(config.variables) > MAX_VARIABLES:
        raise ValueError(
            f"{path}: {len(config.variables)} variables, the firmware logs at most {MAX_VARIABLES}"
        )
    for name in config.variables:
        if not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", name):
            raise ValueError(f"{path}: {name!r} is not a group.name log variable")
    if len(set(config.variables)) != len(config.variables):
        raise ValueError(f"{path}: duplicate variables")
    return config


def role_config(role: str) -> UsdConfig:
    """The stored `config.txt` of a role, parsed."""
    return read_config(config_path(role))


def write_config(directory: Path, role: str) -> Path:
    """Copy the stored `config.txt` of a role onto a mounted card.

    Args:
        directory: Root of the mounted card.
        role: "victim" or "source".

    Returns:
        The written file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory, is the card mounted?")
    source = config_path(role)
    read_config(source)  # Refuse to write a config the firmware would misread
    target = directory / "config.txt"
    shutil.copyfile(source, target)
    logger.info(f"Wrote the {role} config to {target}")
    return target


def check_variables(drone: Drone, names: tuple[str, ...] | list[str]) -> None:
    """Check that every variable of a card config is in the drone's log TOC.

    The firmware skips a variable it does not know with only a console message, so a missing
    `motor.mX_rpm` has to be caught on the ground.

    Args:
        drone: A connected drone.
        names: The variables of its card config.
    """
    toc = drone.cf.log.toc
    missing = [name for name in names if toc.get_element_by_complete_name(name) is None]
    if missing:
        raise RuntimeError(f"Drone {drone.name}: {missing} not in the log TOC")


class ConsoleCapture:
    """Console text of one drone, split into lines as it arrives."""

    def __init__(self, drone: Drone):
        """Register the console callback of the drone.

        Args:
            drone: A connected drone.
        """
        self._complete: list[str] = []
        self._partial = ""
        self._lock = threading.Lock()
        drone.cf.console.receivedChar.add_callback(self._on_text)

    def _on_text(self, text: str):
        with self._lock:
            *complete, self._partial = (self._partial + text).split("\n")
            self._complete.extend(line.strip() for line in complete if line.strip())

    @property
    def lines(self) -> list[str]:
        """Every line received so far, the unfinished last line included."""
        with self._lock:
            partial = [self._partial.strip()] if self._partial.strip() else []
            return [*self._complete, *partial]

    def files(self) -> list[str]:
        """Names of the log files the card opened, from the `uSD: Logging to: <file>` lines."""
        return [m.group(1) for line in self.lines if (m := LOGGING_TO.search(line))]

    def skipped_variables(self) -> list[str]:
        """Console warnings about log variables the firmware dropped."""
        return [line for line in self.lines if SKIPPED_VARIABLE in line]

    def written(self) -> list[tuple[str, int, int]]:
        """File, events written and events requested of every closed file, from the `Wrote` lines.

        Requested events that did not fit into the buffer while the card was busy are dropped.
        """
        return [
            (m.group(1), int(m.group(2)), int(m.group(3)))
            for line in self.lines
            if (m := WROTE.search(line))
        ]


def capture_console(drone: Drone) -> ConsoleCapture:
    """Collect the console of a drone, for the SD file names and skipped variable warnings."""
    return ConsoleCapture(drone)


def read_once(drone: Drone, names: list[str], timeout: float = 2.0) -> dict[str, float]:
    """Read variables once over the radio with a temporary log block.

    Args:
        drone: A connected drone.
        names: Log variables, at most 26 bytes in total.
        timeout: Seconds to wait for the first sample.

    Returns:
        The first sample of every variable.
    """
    from cflib.crazyflie.log import LogConfig

    conf = LogConfig(name=f"read_once_{time.monotonic_ns()}", period_in_ms=100)
    for name in names:
        conf.add_variable(name)
    sample: dict[str, float] = {}
    received = threading.Event()

    def on_data(timestamp: int, data: dict, logconf: LogConfig):
        if not received.is_set():
            sample.update(data)
            received.set()

    conf.data_received_cb.add_callback(on_data)
    drone.cf.log.add_config(conf)
    conf.start()
    arrived = received.wait(timeout)
    conf.stop()
    conf.delete()
    if not arrived:
        raise TimeoutError(f"Drone {drone.name}: no sample of {names} within {timeout} s")
    return sample


def start_usd_logging(drone: Drone) -> float:
    """Start logging to the micro-SD card deck. Each start opens a new file on the card.

    Returns:
        The PC wall time at which logging was requested.
    """
    # Deck detected and config.txt read. canLog is set before the buffer is allocated, so a buffer
    # larger than the free heap only shows as "0 of N events" in the console after the stop
    for name in ("deck.bcUSD", "usd.canLog"):
        if int(drone.cf.param.get_value(name)) != 1:
            raise RuntimeError(f"{name} is not 1: SD deck not detected or config.txt not read")
    t_wall = time.time()
    drone.cf.param.set_value("usd.logging", 1)
    logger.info(f"SD card logging started on {drone.name}")
    return t_wall


def stop_usd_logging(drone: Drone) -> float:
    """Stop logging so the firmware flushes its buffer, writes the CRC and closes the file.

    Returns:
        The PC wall time at which the stop was requested.
    """
    t_wall = time.time()
    drone.cf.param.set_value("usd.logging", 0)
    time.sleep(STOP_WAIT)  # The writer task closes the file in the background
    logger.info(f"SD card logging stopped on {drone.name}")
    return t_wall


def decode_usd(path: Path) -> dict:
    """Decode a micro-SD deck log file into numpy arrays.

    The firmware writes, all little-endian: a header (byte `0xBC`, `uint16` format version,
    `uint16` number of event types), per event type its `uint16` id, name, `uint16` number of
    variables and one `"group.name(t)"` per variable with `t` a `struct` type character, then the
    records (`uint16` event id, timestamp in ms for version 1 and in µs for version 2, the packed
    values) and finally a CRC32 over everything before it. The CRC is only written when logging is
    stopped with `usd.logging = 0`; without it the decoder warns and keeps every complete record.

    Args:
        path: The log file, e.g. `vic00`.

    Returns:
        `{event name: {"t_us": timestamps in µs, variable name: values}, "_info": {...}}`. With
        `on:fixedFrequency` in config.txt the only event is "fixedFrequency".
    """
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 5 or data[0] != 0xBC:
        raise ValueError(f"{path}: not a uSD log file")
    version, n_types = struct.unpack_from("<HH", data, 1)
    if version not in (1, 2):
        raise ValueError(f"{path}: unsupported format version {version}")
    crc_ok = zlib.crc32(data[:-4]) == struct.unpack_from("<I", data, len(data) - 4)[0]
    if not crc_ok:
        warnings.warn(f"{path.name}: CRC mismatch, logging was probably not stopped cleanly")

    def read_string(i: int) -> tuple[str, int]:
        end = data.index(0, i)
        return data[i:end].decode(), end + 1

    # Header: event types and their variables
    idx, events = 5, {}
    for _ in range(n_types):
        (event_id,) = struct.unpack_from("<H", data, idx)
        name, idx = read_string(idx + 2)
        (n_vars,) = struct.unpack_from("<H", data, idx)
        idx += 2
        names, fmt = [], "<"
        for _ in range(n_vars):
            s, idx = read_string(idx)
            names.append(s[:-3])  # "acc.x(f)" -> "acc.x"
            fmt += s[-2]  # -> "f"
        events[event_id] = {
            "name": name,
            "names": names,
            "fmt": struct.Struct(fmt),
            "t": [],
            "rows": [],
        }

    # Records
    head = struct.Struct("<HI" if version == 1 else "<HQ")
    end = len(data) - 4 if crc_ok else len(data)
    truncated = False
    while idx + head.size <= end:
        event_id, timestamp = head.unpack_from(data, idx)
        if event_id not in events:
            raise ValueError(f"{path}: unknown event id {event_id} at byte {idx}")
        ev = events[event_id]
        if idx + head.size + ev["fmt"].size > end:
            truncated = True
            break
        ev["t"].append(timestamp * 1000 if version == 1 else timestamp)
        ev["rows"].append(ev["fmt"].unpack_from(data, idx + head.size))
        idx += head.size + ev["fmt"].size
    if crc_ok and idx != end:
        raise ValueError(f"{path}: {end - idx} bytes left over before the CRC")

    out = {}
    for ev in events.values():
        if not ev["t"]:
            continue
        columns = list(zip(*ev["rows"]))
        out[ev["name"]] = {"t_us": np.asarray(ev["t"], dtype=np.int64)}
        out[ev["name"]].update({n: np.asarray(c) for n, c in zip(ev["names"], columns)})
    out["_info"] = {
        "file": path.name,
        "version": version,
        "crc_ok": crc_ok,
        "truncated": truncated,
        "bytes": len(data),
    }
    return out


def quatcompress(q: ArrayLike) -> NDArray[np.uint64]:
    """Compress quaternions (N, 4) in (x, y, z, w) order like the firmware's `quatcompress.h`.

    The largest component is left out and restored from the unit norm. Its index takes the top two
    bits, each other component a sign bit and a 9 bit magnitude scaled by 1/sqrt(2).
    """
    q = np.atleast_2d(q)
    i_largest = np.argmax(np.abs(q), axis=1)
    negate = q[np.arange(len(q)), i_largest] < 0
    comp = i_largest.astype(np.uint64)
    for i in range(4):
        keep = i_largest != i
        negbit = ((q[:, i] < 0) ^ negate).astype(np.uint64)
        mag = (511 * np.abs(q[:, i]) / np.sqrt(0.5) + 0.5).astype(np.uint64)
        comp = np.where(keep, (comp << np.uint64(10)) | (negbit << np.uint64(9)) | mag, comp)
    return comp


def quatdecompress(comp: ArrayLike) -> NDArray[np.floating]:
    """Decompress `stateEstimateZ.quat` values into quaternions (N, 4) in (x, y, z, w) order.

    The angle error of the compression is at most about 0.2°.
    """
    c = np.asarray(comp).astype(np.uint64)
    mask = np.uint64(511)
    i_largest = (c >> np.uint64(30)).astype(int)
    q = np.zeros((len(c), 4))
    sum_squares = np.zeros(len(c))
    for i in range(3, -1, -1):
        keep = i_largest != i
        mag = (c & mask).astype(float)
        sign = np.where((c >> np.uint64(9)) & np.uint64(1), -1.0, 1.0)
        value = np.sqrt(0.5) * mag / 511 * sign
        q[keep, i] = value[keep]
        sum_squares[keep] += value[keep] ** 2
        c = np.where(keep, c >> np.uint64(10), c)
    q[np.arange(len(q)), i_largest] = np.sqrt(np.clip(1.0 - sum_squares, 0.0, 1.0))
    return q


def bench(role: str, duration: float) -> tuple[int, int]:
    """Log to a drone's card on the ground for `duration` and count the dropped events.

    The card stalls now and then while it writes; events that arrive while the buffer is full are
    dropped, which the firmware reports in its `Wrote` line at the stop. A failed buffer allocation
    shows as no event written at all. The motors spin up for the arming at the connection only.

    Args:
        role: The drone of the experiment config whose card is tested.
        duration: Seconds of logging, a flight's worth is about 300 s.

    Returns:
        Events written and events requested.
    """
    from crazyflow.control import Control  # The drone helpers need cflib and ROS

    from crazyflow_experiments.deploy.experiment import connect
    from crazyflow_experiments.downwash.design import load_config
    from crazyflow_experiments.downwash.pair import disarm, drone_config

    config = role_config(role)
    drone = connect(drone_config(load_config(require_masses=False), role), Control.state)
    try:
        disarm(drone)
        console = capture_console(drone)
        start_usd_logging(drone)
        t_end = time.monotonic() + duration
        while (left := t_end - time.monotonic()) > 0:
            time.sleep(min(left, BENCH_REPORT))
            logger.info(f"{max(t_end - time.monotonic(), 0.0):.0f} s of logging left")
        stop_usd_logging(drone)
    finally:
        drone.close()
    if not (written := console.written()):
        raise RuntimeError(f"No 'Wrote' line from {drone.name}'s card: {console.lines}")
    name, n_written, n_requested = written[-1]
    dropped = n_requested - n_written
    logger.info(
        f"{name}: {n_written} of {n_requested} events written with a {config.buffer} B buffer,"
        f" {dropped} dropped ({dropped / config.frequency:.2f} s of {config.frequency} Hz logging)"
    )
    if n_written == 0:
        raise RuntimeError(f"Nothing written: the {config.buffer} B buffer did not fit the heap")
    return n_written, n_requested


def main():
    """Command line: write a role's config onto a mounted card, or test a card on the ground."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write-config", help="Copy a role's config.txt onto a card")
    write.add_argument("directory", type=Path, help="Root of the mounted card")
    write.add_argument("--role", choices=ROLES, required=True)
    test = commands.add_parser("bench", help="Log on the ground and count the dropped events")
    test.add_argument("--role", choices=ROLES, required=True)
    test.add_argument("--seconds", type=float, default=300.0, help="Logging duration")
    args = parser.parse_args()
    if args.command == "write-config":
        write_config(args.directory, args.role)
    elif args.command == "bench":
        bench(args.role, args.seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
