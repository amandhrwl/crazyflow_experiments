"""Onboard logging to the micro-SD card decks of the victim and the source.

The firmware (`usddeck.c`) reads `config.txt` once at boot and opens the next free file `<name>NN`
each time `usd.logging` is set to 1. The file only gets its CRC32 when logging is stopped again
with `usd.logging = 0`. The card configs live in `sd/config_<role>.txt`, `write-config` copies one
onto a mounted card:

    pixi run -e sim2real python crazyflow_experiments/load_cell/usd.py write-config /media/<card> \
        --role victim

The records carry the microsecond timer of the drone (`usecTimestamp`), not the millisecond tick
that stamps the radio log packets. Both count the same crystal but start at different times during
the boot, `sync.py` measures the offset between them.

Only numpy and the standard library are imported at module level, the decoder runs without cflib.
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

from crazyflow_experiments.load_cell import ROLES, SD_CONFIG_DIR

if TYPE_CHECKING:
    from cflib.crazyflie import Crazyflie

logger = logging.getLogger(__name__)

MAX_VARIABLES = 20  # MAX_USD_LOG_VARIABLES_PER_EVENT, the firmware skips further lines silently
STOP_WAIT = 1.0  # s, the writer task closes the file in the background after the stop
LOGGING_TO = re.compile(r"Logging to:\s*(\S+)")  # Console line naming the file a start opens
WROTE = re.compile(r"Wrote \d+ B to:\s*(\S+) \((\d+) of (\d+) events\)")  # Console line at a stop


@dataclass(frozen=True)
class UsdConfig:
    """Contents of a card's `config.txt`.

    Attributes:
        buffer: Size of the write buffer in bytes, at most 8192 on the cf21B (heap).
        name: Stem of the log files, the firmware appends the number.
        frequency: Logging rate in Hz, a divisor of the 1 kHz stabilizer rate.
        variables: The logged variables in file order.
    """

    buffer: int
    name: str
    frequency: int
    variables: tuple[str, ...]


def config_path(role: str) -> Path:
    """Path of the stored `config.txt` of "victim" or "source"."""
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}, expected one of {ROLES}")
    return SD_CONFIG_DIR / f"config_{role}.txt"


def read_config(path: Path) -> UsdConfig:
    """Parse and check a `config.txt`: a header of seven lines, then one variable per line.

    Args:
        path: The config file.
    """
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError(f"{path}: a header of 7 lines and at least one variable expected")
    header = [line.split()[0] for line in lines[:7]]  # The firmware reads up to the first blank
    if not header[4].startswith("on:"):
        raise ValueError(f"{path}: event line {header[4]!r} must start with 'on:'")
    config = UsdConfig(
        buffer=int(header[1]), name=header[2], frequency=int(header[5]), variables=tuple(lines[7:])
    )
    if config.frequency <= 0 or 1000 % config.frequency != 0:
        raise ValueError(f"{path}: frequency {config.frequency} Hz must divide the 1 kHz loop")
    if len(config.name) > 10:
        raise ValueError(f"{path}: file stem {config.name!r} is longer than the firmware's 10")
    if len(config.variables) > MAX_VARIABLES:
        raise ValueError(f"{path}: more than {MAX_VARIABLES} variables")
    for name in config.variables:
        if not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", name):
            raise ValueError(f"{path}: {name!r} is not a group.name log variable")
    return config


def role_config(role: str) -> UsdConfig:
    """The stored `config.txt` of a role, parsed."""
    return read_config(config_path(role))


def write_config(directory: Path, role: str) -> Path:
    """Copy the stored `config.txt` of a role onto a mounted card and return the written file."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory, is the card mounted?")
    read_config(config_path(role))  # Refuse to write a config the firmware would misread
    target = directory / "config.txt"
    shutil.copyfile(config_path(role), target)
    logger.info(f"Wrote the {role} config to {target}")
    return target


def check_variables(cf: Crazyflie, name: str, variables: tuple[str, ...]) -> None:
    """Fail unless every variable of a card config is in the drone's log TOC.

    The firmware skips a variable it does not know with only a console message, so a firmware
    without the load cell driver has to be caught on the ground.
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

    def written(self) -> list[tuple[str, int, int]]:
        """File, events written and events requested of every closed file."""
        return [
            (m.group(1), int(m.group(2)), int(m.group(3)))
            for line in self.lines
            if (m := WROTE.search(line))
        ]


def start_logging(cf: Crazyflie, name: str) -> float:
    """Start logging to the card, which opens a new file, and return the PC time of the request."""
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
    t_wall = time.time()
    cf.param.set_value("usd.logging", 1)
    logger.info(f"SD card logging started on {name}")
    return t_wall


def stop_logging(cf: Crazyflie, name: str) -> float:
    """Stop logging so the firmware writes the CRC and closes the file, return the PC time."""
    t_wall = time.time()
    cf.param.set_value("usd.logging", 0)
    time.sleep(STOP_WAIT)
    logger.info(f"SD card logging stopped on {name}")
    return t_wall


def decode(path: Path) -> dict[str, np.ndarray]:
    """Decode a micro-SD deck log file of `on:fixedFrequency` logging into numpy arrays.

    The firmware writes, all little-endian: a header (byte `0xBC`, `uint16` format version,
    `uint16` number of event types), per event type its `uint16` id, name, `uint16` number of
    variables and one `"group.name(t)"` per variable with `t` a `struct` type character, then the
    records (`uint16` event id, timestamp in ms for version 1 and in µs for version 2, the packed
    values) and finally a CRC32 over everything before it. Without the CRC, when logging was not
    stopped cleanly, the decoder warns and keeps every complete record.

    Args:
        path: The log file, e.g. `lcv00`.

    Returns:
        `{"t_us": timestamps in µs, variable name: values}` of the fixedFrequency event.
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
        events[event_id] = (name, names, struct.Struct(fmt))
    if [name for name, _, _ in events.values()] != ["fixedFrequency"]:
        raise ValueError(f"{path}: expected the single event fixedFrequency, got {events}")

    head = struct.Struct("<HI" if version == 1 else "<HQ")
    end = len(data) - 4 if crc_ok else len(data)
    _, names, values = next(iter(events.values()))
    t, rows = [], []
    while idx + head.size + values.size <= end:
        event_id, timestamp = head.unpack_from(data, idx)
        if event_id not in events:
            raise ValueError(f"{path}: unknown event id {event_id} at byte {idx}")
        t.append(timestamp * 1000 if version == 1 else timestamp)
        rows.append(values.unpack_from(data, idx + head.size))
        idx += head.size + values.size
    if crc_ok and idx != end:
        raise ValueError(f"{path}: {end - idx} bytes left over before the CRC")
    columns = list(zip(*rows)) if rows else [()] * len(names)
    return {"t_us": np.asarray(t, dtype=np.int64)} | {
        n: np.asarray(c) for n, c in zip(names, columns, strict=True)
    }


def main():
    """Command line: write a role's config onto a mounted card."""
    parser = argparse.ArgumentParser(description="Write a card config of the load cell test")
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write-config", help="Copy a role's config.txt onto a card")
    write.add_argument("directory", type=Path, help="Root of the mounted card")
    write.add_argument("--role", choices=ROLES, required=True)
    args = parser.parse_args()
    write_config(args.directory, args.role)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
