"""A load cell test recording faked from known clocks, to check that the sync recovers them.

The timing follows the firmware: the card samples on every other pass of the 1 kHz stabilizer
loop, which the IMU's own clock paces, right after the loop read the sensors. The radio log timer
runs every 10 ms of the tick and stamps a block a little after the tick's millisecond. The load
cell changes at its own 80 Hz, the accelerometer on every pass of the loop.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

EVENT_ID = 0xFFFF
PC_BOOT = 1.7e9  # s, PC time at which the drones booted
LATENCY_FLOOR = 1e-3  # s, shortest radio latency
STABILIZER_PERIOD = 1e-3 * (1 + 2e-4)  # s of tick per loop pass, the IMU's clock runs apart
A, B = 3.43e-4, 111.3  # Load cell calibration, g = A * raw + B
RAW_ZERO = -B / A  # Raw reading of zero force
DURATION = 60.0  # s of the recording
PHASES = [  # Name, start and end in s from the start of the recording
    ("still_before", 0.5, 5.5),
    ("takeoff", 6.0, 12.0),
    ("aside_1", 12.0, 17.0),
    ("move_in", 17.0, 21.0),
    ("over", 21.0, 31.0),
    ("move_out", 31.0, 35.0),
    ("aside_2", 35.0, 40.0),
    ("landing", 40.0, 47.0),
    ("still_after", 48.0, 53.0),
]
OVER_FORCE = -0.02  # N on the victim while the source holds over it
VICTIM_Z = 0.5  # m


@dataclass(frozen=True)
class Clock:
    """A drone's clocks: PC time = PC_BOOT + rate * tick, tick = sd_rate * SD timer + sd_offset."""

    rate: float
    sd_offset: float
    sd_rate: float = 1.0

    def pc(self, tick: NDArray) -> NDArray:
        """True PC time of onboard ticks."""
        return PC_BOOT + self.rate * tick

    def tick(self, pc: NDArray) -> NDArray:
        """Onboard tick at PC times."""
        return (pc - PC_BOOT) / self.rate

    def sd_us(self, tick: NDArray) -> NDArray:
        """SD timer in µs at onboard ticks, as the card stamps it."""
        return np.round((tick - self.sd_offset) / self.sd_rate * 1e6)


# Ten times the drift the cf21B's tick showed, to test the fit
VICTIM_CLOCK = Clock(rate=1.013, sd_offset=3.7e-3)
# Firmware from before the fix of the card timer's prescaler, like the first source
SOURCE_CLOCK = Clock(rate=0.9995, sd_offset=-1.2e-3, sd_rate=85 / 84)


def usd_file(variables: list[tuple[str, str]], t_us: NDArray, columns: list[NDArray]) -> bytes:
    """A version 2 card log of one fixedFrequency event, CRC included."""
    data = b"\xbc" + struct.pack("<HH", 2, 1)
    data += struct.pack("<H", EVENT_ID) + b"fixedFrequency\0" + struct.pack("<H", len(variables))
    data += b"".join(f"{name}({kind})".encode() + b"\0" for name, kind in variables)
    values = struct.Struct("<" + "".join(kind for _, kind in variables))
    for i, t in enumerate(t_us):
        data += struct.pack("<HQ", EVENT_ID, int(t)) + values.pack(*(c[i] for c in columns))
    return data + struct.pack("<I", zlib.crc32(data))


def stepped(changes: NDArray, values: NDArray, ticks: NDArray) -> NDArray:
    """A variable that takes `values` at the ticks `changes`, read at `ticks`."""
    return values[np.searchsorted(changes, ticks, side="right") - 1]


def loop_passes(clock: Clock, t0_pc: float) -> NDArray:
    """Onboard ticks of the stabilizer loop's passes over the recording."""
    n = int(DURATION / STABILIZER_PERIOD)
    return clock.tick(t0_pc) + 0.43e-3 + np.arange(n) * STABILIZER_PERIOD


def drone_logs(
    clock: Clock, rng: np.random.Generator, changes: NDArray, values: NDArray, t0_pc: float
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Card and radio records of a variable that takes `values` at the onboard ticks `changes`.

    Returns:
        The card's onboard ticks and values at 500 Hz, the radio stamps in ms, the radio values at
        100 Hz and their PC arrival times, in arrival order like cflib stores them.
    """
    sd_ticks = loop_passes(clock, t0_pc)[::2] + 20e-6
    tick0, tick1 = clock.tick(t0_pc), clock.tick(t0_pc + DURATION)
    stamps = np.arange(np.ceil(tick0 * 1000) + 100, np.floor(tick1 * 1000) - 100, 10)
    radio_ticks = stamps / 1000 + rng.uniform(0, 0.3e-3, len(stamps))
    arrival = clock.pc(radio_ticks) + LATENCY_FLOOR + rng.exponential(2e-3, len(radio_ticks))
    order = np.argsort(arrival, kind="stable")
    return (
        sd_ticks,
        stepped(changes, values, sd_ticks),
        stamps.astype(np.int64)[order],
        stepped(changes, values, radio_ticks)[order],
        arrival[order],
    )


def source_height(t_pc: NDArray, t0: float) -> NDArray:
    """Height of the source over PC time: up, over the victim and down again."""
    z = np.interp(t_pc - t0, [6.0, 10.0, 40.0, 47.0], [0.0, VICTIM_Z + 0.6, VICTIM_Z + 0.6, 0.0])
    return z.astype(np.float32)


def write_recording(folder: Path, seed: int = 0) -> None:
    """Write meta.json, radio.npz, flight.npz and both card files of a faked test."""
    rng = np.random.default_rng(seed)
    t0 = PC_BOOT + 30.0  # The recording starts 30 s after the boot

    # The load cell, at 80 Hz on its own oscillator, pushed down while the source is over
    tick0 = VICTIM_CLOCK.tick(t0)
    changes = tick0 - 1.0 + np.cumsum(rng.uniform(12.4e-3, 12.6e-3, int(62.0 / 12.5e-3)))
    t = VICTIM_CLOCK.pc(changes) - t0
    grams = np.where((t >= 21.0) & (t < 31.0), OVER_FORCE / 9.81 * 1e3, 0.0)
    raw = np.round(RAW_ZERO + (grams + rng.normal(0, 0.03, len(t))) / A).astype(np.int32)
    sd_ticks, v_sd, stamps, v_radio, arrival = drone_logs(VICTIM_CLOCK, rng, changes, raw, t0)
    t_us = VICTIM_CLOCK.sd_us(sd_ticks)
    (folder / "sd" / "victim").mkdir(parents=True)
    (folder / "sd" / "victim" / "lcv00").write_bytes(
        usd_file([("loadcell.rawWeight", "i")], t_us, [v_sd])
    )
    radio = {
        "victim/loadcell/t_pc": arrival,
        "victim/loadcell/t_tick_ms": stamps,
        "victim/loadcell/loadcell.rawWeight": v_radio,
        "victim/loadcell/loadcell.weight": A * v_radio + B,
    }

    # The source's accelerometer, new on every loop pass, and its height estimate
    changes = loop_passes(SOURCE_CLOCK, t0)
    acc = (9.81 + rng.normal(0, 0.05, len(changes))).astype(np.float32)
    sd_ticks, v_sd, stamps, v_radio, arrival = drone_logs(SOURCE_CLOCK, rng, changes, acc, t0)
    t_us = SOURCE_CLOCK.sd_us(sd_ticks)
    height = source_height(SOURCE_CLOCK.pc(sd_ticks), t0)
    (folder / "sd" / "source").mkdir(parents=True)
    (folder / "sd" / "source" / "lcs00").write_bytes(
        usd_file([("acc.z", "f"), ("stateEstimate.z", "f")], t_us, [v_sd, height])
    )
    radio |= {
        "source/cf_logger_0/t_pc": arrival,
        "source/cf_logger_0/t_tick_ms": stamps,
        "source/cf_logger_0/acc.z": v_radio,
    }
    np.savez(folder / "radio.npz", **radio)

    t_flight = t0 + np.arange(12.0, 40.0, 0.01)
    pos = np.zeros((len(t_flight), 3))
    pos[:, 2] = source_height(t_flight, t0)
    np.savez(folder / "flight.npz", t_pc=t_flight, pos=pos, setpoint=pos)
    meta = {
        "config": {"load_cell": {"a": A, "b": B}},
        "victim_pos": [0.0, 0.0, VICTIM_Z],
        "phases": [(name, t0 + a, t0 + b) for name, a, b in PHASES],
        "sd": {"victim": {"files": ["lcv00"]}, "source": {"files": ["lcs00"]}},
    }
    (folder / "meta.json").write_text(json.dumps(meta))
