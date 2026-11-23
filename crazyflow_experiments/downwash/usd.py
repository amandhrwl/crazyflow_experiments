"""Onboard logging to the micro-SD card decks of the downwash blocks.

The firmware (`usddeck.c`) reads `config.txt` once at boot and opens a new file each time
`usd.logging` is set to 1. Firmware with the `usd.fileDate` and `usd.fileTime` parameters (the
patched `usddeck.c` of `../crazyflie-firmware`) names it after the recording that `start_logging`
passes, `<stem>_YYYY-MM-DD_hh-mm-ss_SD` with the stem from line 3 of `config.txt` (`vic` or `src`)
and the timestamp of the recording folder; older firmware numbers it `<stem>NN` and
`start_logging` warns. The file only gets its CRC32 when logging is stopped again with
`usd.logging = 0`. Every experiment records the name the console reports in its `meta.json`,
and `sd_file` finds the file in `<recording>/sd/<role>/` by it, by the recording's timestamp, or
as the only file there.

Every card config lives in `sd/config_<block>_<role>.txt` (`CARD_CONFIGS`). `write-config` copies
one onto a mounted card, `collect` copies the files from a card, or a folder they were copied to,
into the recordings whose `meta.json` names them (A1, A3, the load cell test and the hover), and
`bench` logs on the ground for a flight's length and reports how many events the card dropped
while it stalled:

    pixi run -e sim2real python crazyflow_experiments/downwash/usd.py write-config /media/<card> \
        --block a3 --role victim
    pixi run -e sim2real python crazyflow_experiments/downwash/usd.py collect /media/<card> \
        --block lc
    pixi run -e sim2real python crazyflow_experiments/downwash/usd.py bench --role source

`decode_usd` reads a log file back into numpy arrays, every event of it, and `decode` the single
fixedFrequency event as one flat record. Both rest on one parser, an own implementation of the
format (moved here from `sd_log_inspection.ipynb`); the firmware's GPL decoder `cfusdlog.py` is
not used. The records carry the microsecond timer of the drone (`usecTimestamp`), not the
millisecond tick that stamps the radio log packets, see `sync.py`.

Only numpy and the standard library are imported at module level, so the notebook can import the
decoder without cflib or ROS. The drone helpers import cflib when they are called.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import struct
import threading
import time
import warnings
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash import (
    A1_RESULTS_DIR,
    A2_RESULTS_DIR,
    A3_RESULTS_DIR,
    LC_RESULTS_DIR,
    ROLES,
    SD_CONFIG_DIR,
)
from crazyflow_experiments.hover import RECORDINGS_DIR as HOVER_RESULTS_DIR

if TYPE_CHECKING:
    from cflib.crazyflie import Crazyflie
    from numpy.typing import ArrayLike, NDArray

    from crazyflow_experiments.deploy.crazyflie import Drone

logger = logging.getLogger(__name__)

# The card configs of every block: A1 logs the victim only, A3 and the load cell test both drones
CARD_CONFIGS = {"a1": ("victim",), "a3": ROLES, "lc": ROLES}
# Where `collect` puts card files, one folder per recording with its meta.json
RECORDINGS = {
    "a1": A1_RESULTS_DIR,
    "a3": A3_RESULTS_DIR,
    "a3_stand": A3_RESULTS_DIR,  # A3 on the stand records next to A3, its meta.json names both cards
    "a2": A2_RESULTS_DIR,  # A2 uses the A3 card configs; free and stand runs record together
    "a2_stand": A2_RESULTS_DIR,
    "lc": LC_RESULTS_DIR,
    "hover": HOVER_RESULTS_DIR,
}
STAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"  # Of `utils.paths.new_recording_path`, and of dated card files
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")  # A recording folder's timestamp
MAX_VARIABLES = 20  # MAX_USD_LOG_VARIABLES_PER_EVENT, the firmware skips further lines silently
MAX_NAME = 10  # Characters of the file stem the firmware keeps
RPM_INVALID = 65535  # motor.mX_rpm without a reading
STOP_WAIT = 1.0  # s, the writer task closes the file in the background after the stop
PARAM_TIMEOUT = 2.0  # s for a written parameter to read back
LOGGING_TO = re.compile(r"Logging to:\s*(\S+)")  # Console line naming the file a start opens
SKIPPED_VARIABLE = "Skip log variable"  # Console warning for a variable the firmware drops
WROTE = re.compile(r"Wrote \d+ B to:\s*(\S+) \((\d+) of (\d+) events\)")  # Console line at a stop
BENCH_REPORT = 30.0  # s between progress lines of `bench`


@dataclass(frozen=True)
class UsdConfig:
    """Contents of a card's `config.txt`.

    Attributes:
        version: Format version of the config file.
        buffer: Size of the write buffer in bytes, at most 8192 on the cf21B (heap).
        name: Stem of the log files, the firmware appends the number or the date.
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


def config_path(block: str, role: str) -> Path:
    """Path of the stored `config.txt` of a role in a block.

    Args:
        block: "a1", "a3" or "lc", see `CARD_CONFIGS`.
        role: "victim" or "source".
    """
    if role not in CARD_CONFIGS.get(block, ()):
        raise ValueError(f"No card config of role {role!r} in block {block!r}: {CARD_CONFIGS}")
    return SD_CONFIG_DIR / f"config_{block}_{role}.txt"


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
    if len(config.name) > MAX_NAME:
        raise ValueError(f"{path}: file stem {config.name!r} is longer than the firmware's 10")
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


def role_config(block: str, role: str) -> UsdConfig:
    """The stored `config.txt` of a role in a block, parsed."""
    return read_config(config_path(block, role))


def write_config(directory: Path, block: str, role: str) -> Path:
    """Copy the stored `config.txt` of a role in a block onto a mounted card.

    Args:
        directory: Root of the mounted card.
        block: "a1", "a3" or "lc".
        role: "victim" or "source".

    Returns:
        The written file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory, is the card mounted?")
    source = config_path(block, role)
    read_config(source)  # Refuse to write a config the firmware would misread
    target = directory / "config.txt"
    shutil.copyfile(source, target)
    logger.info(f"Wrote the {block} {role} config to {target}")
    return target


def check_variables(cf: Crazyflie, name: str, variables: tuple[str, ...] | list[str]) -> None:
    """Fail unless every variable of a card config is in the drone's log TOC.

    The firmware skips a variable it does not know with only a console message, so a missing
    `motor.mX_rpm`, or a firmware without the load cell driver, has to be caught on the ground.

    Args:
        cf: The connected drone.
        name: Its name, for the message.
        variables: The variables of its card config.
    """
    toc = cf.log.toc
    if missing := [v for v in variables if toc.get_element_by_complete_name(v) is None]:
        raise RuntimeError(f"{name}: {missing} not in the log TOC, is the firmware right?")


class ConsoleCapture:
    """Console text of one drone, split into lines as it arrives."""

    def __init__(self, cf: Crazyflie):
        """Register the console callback of a connected drone."""
        self._complete: list[str] = []
        self._partial = ""
        self._lock = threading.Lock()
        cf.console.receivedChar.add_callback(self._on_text)

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


def set_param(cf: Crazyflie, name: str, value: float):
    """Write a parameter and wait until it reads back, as the write confirmation arrives later."""
    cf.param.set_value(name, value)
    t_end = time.monotonic() + PARAM_TIMEOUT
    while not np.isclose(float(cf.param.get_value(name)), value, rtol=1e-6, atol=0.0):
        if time.monotonic() > t_end:
            raise RuntimeError(f"{name} reads {cf.param.get_value(name)}, not {value}")
        time.sleep(0.05)


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


def start_logging(cf: Crazyflie, name: str, recorded: datetime | None = None) -> float:
    """Start logging to the card, which opens a new file, and return the PC time of the request.

    Args:
        cf: The connected drone.
        name: Its name, for the messages.
        recorded: Time of the recording folder, which names the file on firmware that can. None
            leaves the naming alone, so the file is numbered.
    """
    # Deck detected and config.txt read. canLog is set before the buffer is allocated, so a buffer
    # larger than the free heap only shows as "0 of N events" in the console after the stop
    if int(cf.param.get_value("deck.bcUSD")) != 1:
        raise RuntimeError(
            f"{name}: deck.bcUSD is 0, the SD deck was not detected at boot. Reseat the deck with "
            "the drone off and power it on with the button, a reboot over the radio never helps."
        )
    if int(cf.param.get_value("usd.canLog")) != 1:
        raise RuntimeError(
            f"{name}: usd.canLog is 0, the card or its config.txt could not be read. Check that "
            "the card sits fully in the deck and holds config.txt, see usd.py write-config."
        )
    if recorded is not None:
        if cf.param.toc.get_element_by_complete_name("usd.fileDate") is None:
            logger.warning(f"{name}: its firmware numbers the card file instead of dating it")
        else:  # Read at the start of the run, the file is named before logging begins
            set_param(cf, "usd.fileDate", int(recorded.strftime("%Y%m%d")))
            set_param(cf, "usd.fileTime", int(recorded.strftime("%H%M%S")))
    t_wall = time.time()
    cf.param.set_value("usd.logging", 1)
    logger.info(f"SD card logging started on {name}")
    return t_wall


def stop_logging(cf: Crazyflie, name: str) -> float:
    """Stop logging so the firmware writes the CRC and closes the file, return the PC time."""
    t_wall = time.time()
    cf.param.set_value("usd.logging", 0)
    time.sleep(STOP_WAIT)  # The writer task closes the file in the background
    logger.info(f"SD card logging stopped on {name}")
    return t_wall


def collect(directory: Path, recordings: Path = LC_RESULTS_DIR) -> list[Path]:
    """Copy card files into the recordings whose `meta.json` names them, `<recording>/sd/<role>/`.

    A file already in its recording is left alone. A numbered name of older firmware recurs once
    the files on a card are deleted, so a file that several recordings name is skipped.

    Args:
        directory: A mounted card, or a folder the card files were copied to.
        recordings: The folder of the recordings.

    Returns:
        The copied files.
    """
    owners: dict[str, list[tuple[Path, str]]] = {}
    for meta_path in sorted(Path(recordings).glob("*/meta.json")):
        for role, card in cards(json.loads(meta_path.read_text())).items():
            for file_name in card.get("files", []):
                owners.setdefault(file_name, []).append((meta_path.parent, role))
    copied = []
    for path in sorted(p for p in Path(directory).iterdir() if p.is_file()):
        if len(found := owners.get(path.name, [])) > 1:
            folders = [folder.name for folder, _ in found]
            logger.warning(f"{path.name} is named by {folders}, copy it by hand")
        if len(found) != 1 or (target := found[0][0] / "sd" / found[0][1] / path.name).exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        logger.info(f"Copied {path.name} to {target.parent}")
        copied.append(target)
    return copied


def recording_time(folder: Path) -> datetime:
    """Time of a recording from the timestamp that ends its folder name, see `new_recording_path`.

    `start_logging` names the card files with it, so they carry the folder's timestamp.
    """
    if (match := STAMP.search(Path(folder).name)) is None:
        raise ValueError(f"{folder} does not end in a recording timestamp")
    return datetime.strptime(match.group(), STAMP_FORMAT)


def cards(meta: dict) -> dict[str, dict]:
    """The card entries of a recording's `meta.json` per role; A3 keeps them under "usd"."""
    return meta["sd"] if "sd" in meta else meta["usd"]


def sd_file(folder: Path, role: str, meta: dict) -> Path:
    """The card file of a role in a recording, `<recording>/sd/<role>/<file>`.

    The file the console reported in `meta.json` comes first, the last one if logging started more
    than once; that is the dated name on firmware that names the file, the numbered one otherwise
    and in recordings from before. Without a console report, the file dated with the recording's
    timestamp, `<stem>_<timestamp>_SD`, is taken, else the only file there.

    Args:
        folder: The recording.
        role: The role whose card file it is.
        meta: The recording's `meta.json`.
    """
    directory = Path(folder) / "sd" / role
    if names := cards(meta)[role].get("files", []):
        if not (path := directory / Path(names[-1]).name).is_file():
            raise FileNotFoundError(
                f"Copy {path.name} from the {role} card to {directory}/, see usd.py collect"
            )
        return path
    found = sorted(p for p in directory.iterdir() if p.is_file()) if directory.is_dir() else []
    if (stamp := STAMP.search(Path(folder).name)) is not None:
        found = [p for p in found if p.name.endswith(f"_{stamp.group()}_SD")] or found
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one card file of the {role} in {directory}: {found}")
    return found[0]


@dataclass
class _Event:
    """One event type of a log file and its records, as `_parse` reads them."""

    name: str
    names: list[str]
    fmt: struct.Struct
    t: list[int]
    rows: list[tuple]


def _parse(path: Path) -> tuple[list[_Event], dict]:
    """Read every event type and record of a log file.

    The firmware writes, all little-endian: a header (byte `0xBC`, `uint16` format version,
    `uint16` number of event types), per event type its `uint16` id, name, `uint16` number of
    variables and one `"group.name(t)"` per variable with `t` a `struct` type character, then the
    records (`uint16` event id, timestamp in ms for version 1 and in µs for version 2, the packed
    values) and finally a CRC32 over everything before it. The CRC is only written when logging is
    stopped with `usd.logging = 0`; without it the parser warns and keeps every complete record.

    Returns:
        The event types in file order with their timestamps in µs and value rows, and the file
        info.
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
        events[event_id] = _Event(name, names, struct.Struct(fmt), [], [])

    # Records
    head = struct.Struct("<HI" if version == 1 else "<HQ")
    end = len(data) - 4 if crc_ok else len(data)
    truncated = False
    while idx + head.size <= end:
        event_id, timestamp = head.unpack_from(data, idx)
        if event_id not in events:
            raise ValueError(f"{path}: unknown event id {event_id} at byte {idx}")
        ev = events[event_id]
        if idx + head.size + ev.fmt.size > end:
            truncated = True
            break
        ev.t.append(timestamp * 1000 if version == 1 else timestamp)
        ev.rows.append(ev.fmt.unpack_from(data, idx + head.size))
        idx += head.size + ev.fmt.size
    if crc_ok and idx != end:
        raise ValueError(f"{path}: {end - idx} bytes left over before the CRC")
    info = {
        "file": path.name,
        "version": version,
        "crc_ok": crc_ok,
        "truncated": truncated,
        "bytes": len(data),
    }
    return list(events.values()), info


def decode_usd(path: Path) -> dict:
    """Decode a micro-SD deck log file into numpy arrays, every event of it, see `_parse`.

    Args:
        path: The log file, e.g. `vic00`.

    Returns:
        `{event name: {"t_us": timestamps in µs, variable name: values}, "_info": {...}}`, without
        the events that have no record. With `on:fixedFrequency` in config.txt the only event is
        "fixedFrequency".
    """
    events, info = _parse(path)
    out = {}
    for ev in events:
        if not ev.t:
            continue
        columns = list(zip(*ev.rows))
        out[ev.name] = {"t_us": np.asarray(ev.t, dtype=np.int64)}
        out[ev.name].update({n: np.asarray(c) for n, c in zip(ev.names, columns)})
    out["_info"] = info
    return out


def decode(path: Path) -> dict[str, np.ndarray]:
    """Decode a log file of `on:fixedFrequency` logging into one flat record, see `_parse`.

    Args:
        path: The log file, e.g. `vic_2026-11-21_16-31-44_SD`.

    Returns:
        `{"t_us": timestamps in µs, variable name: values}` of the fixedFrequency event, with empty
        arrays for a file without records.
    """
    events, _ = _parse(path)
    if [ev.name for ev in events] != ["fixedFrequency"]:
        raise ValueError(
            f"{path}: expected the single event fixedFrequency, got {[ev.name for ev in events]}"
        )
    (ev,) = events
    columns = list(zip(*ev.rows)) if ev.rows else [()] * len(ev.names)
    return {"t_us": np.asarray(ev.t, dtype=np.int64)} | {
        n: np.asarray(c) for n, c in zip(ev.names, columns, strict=True)
    }


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
    """Log to an A3 drone's card on the ground for `duration` and count the dropped events.

    The card stalls now and then while it writes; events that arrive while the buffer is full are
    dropped, which the firmware reports in its `Wrote` line at the stop. A failed buffer allocation
    shows as no event written at all. The motors spin up for the arming at the connection only.

    Args:
        role: The drone of the A3 config whose card is tested.
        duration: Seconds of logging, a flight's worth is about 300 s.

    Returns:
        Events written and events requested.
    """
    from crazyflow.control import Control  # The drone helpers need cflib and ROS

    from crazyflow_experiments.deploy.experiment import connect
    from crazyflow_experiments.downwash.a3_design import load_config
    from crazyflow_experiments.downwash.pair import disarm, drone_config

    config = role_config("a3", role)
    drone = connect(drone_config(load_config(require_masses=False), role), Control.state)
    try:
        disarm(drone)
        console = ConsoleCapture(drone.cf)
        start_logging(drone.cf, drone.name)
        t_end = time.monotonic() + duration
        while (left := t_end - time.monotonic()) > 0:
            time.sleep(min(left, BENCH_REPORT))
            logger.info(f"{max(t_end - time.monotonic(), 0.0):.0f} s of logging left")
        stop_logging(drone.cf, drone.name)
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
    """Command line: write a config onto a card, collect card files, or bench test a card."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write-config", help="Copy a block's config.txt onto a card")
    write.add_argument("directory", type=Path, help="Root of the mounted card")
    write.add_argument("--block", choices=CARD_CONFIGS, required=True)
    write.add_argument("--role", choices=ROLES, required=True)
    gather = commands.add_parser("collect", help="Copy card files into their recordings")
    gather.add_argument("directory", type=Path, help="Mounted card or folder of card files")
    gather.add_argument("--block", choices=RECORDINGS, required=True, help="Whose recordings")
    test = commands.add_parser("bench", help="Log on the ground and count the dropped events")
    test.add_argument("--role", choices=ROLES, required=True)
    test.add_argument("--seconds", type=float, default=300.0, help="Logging duration")
    args = parser.parse_args()
    if args.command == "write-config":
        write_config(args.directory, args.block, args.role)
    elif args.command == "collect":
        collect(args.directory, RECORDINGS[args.block])
    else:
        bench(args.role, args.seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
