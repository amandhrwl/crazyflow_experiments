"""Card configs and the log decoder of the load cell test."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import json  # noqa: E402
import logging  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.load_cell import ROLES, usd  # noqa: E402
from crazyflow_experiments.load_cell.convert import SYNC_VARIABLES  # noqa: E402
from tests.load_cell.fakes import usd_file  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("role", ROLES)
def test_card_configs_parse_and_log_the_sync_variable(role: str):
    """Each stored config is valid and logs the variable its drone also streams for the sync."""
    config = usd.role_config(role)
    assert config.frequency == 500
    assert config.buffer <= 8192  # A larger buffer fails the cf21B's boot self-test
    assert SYNC_VARIABLES[role][1] in config.variables


def test_victim_card_logs_the_load_cell_and_what_its_force_balance_needs():
    """The raw load cell reading, the motor rpm for the thrust, the accelerometer and the tilt.

    The force follows from the raw reading, so the firmware's converted weight is not logged.
    """
    variables = usd.role_config("victim").variables
    needed = ["loadcell.rawWeight", "acc.z", "stateEstimate.roll", "stateEstimate.pitch"]
    assert set(needed + [f"motor.m{i}_rpm" for i in range(1, 5)]) <= set(variables)
    assert "loadcell.weight" not in variables


def test_decoder_reads_back_what_the_firmware_writes(tmp_path: Path):
    """Timestamps and values of every type come back exactly."""
    t_us = np.array([1_000_000, 1_002_000, 1_004_000])
    raw = np.array([-324544, -324500, 2**31 - 1], dtype=np.int32)
    z = np.array([0.5, 0.25, -1.0], dtype=np.float32)
    path = tmp_path / "lcv00"
    path.write_bytes(
        usd_file([("loadcell.rawWeight", "i"), ("stateEstimate.z", "f")], t_us, [raw, z])
    )
    record = usd.decode(path)
    np.testing.assert_array_equal(record["t_us"], t_us)
    np.testing.assert_array_equal(record["loadcell.rawWeight"], raw)
    np.testing.assert_array_equal(record["stateEstimate.z"], z)


def test_decoder_keeps_the_records_of_a_file_without_crc(tmp_path: Path):
    """A file not closed by a stop has no CRC, the decoder warns and keeps every whole record."""
    t_us = np.arange(5) * 2000
    data = usd_file([("loadcell.rawWeight", "i")], t_us, [np.arange(5, dtype=np.int32)])
    path = tmp_path / "lcv01"
    path.write_bytes(data[:-4])
    with pytest.warns(UserWarning, match="CRC"):
        record = usd.decode(path)
    assert len(record["t_us"]) == 5


def test_write_config_refuses_an_unmounted_card(tmp_path: Path):
    """Writing to a missing card root fails instead of creating a directory."""
    with pytest.raises(FileNotFoundError):
        usd.write_config(tmp_path / "not_mounted", "victim")
    assert usd.read_config(usd.write_config(tmp_path, "source")) == usd.role_config("source")


class FakeParams:
    """The parameters of a drone with a working card, and its TOC."""

    def __init__(self, names: tuple[str, ...]):
        """Hold the deck's parameters and `names`."""
        self.values = {"deck.bcUSD": 1, "usd.canLog": 1, "usd.logging": 0} | dict.fromkeys(names, 0)
        self.toc = self

    def get_element_by_complete_name(self, name: str) -> str | None:
        """The TOC entry, None for a parameter the firmware lacks."""
        return name if name in self.values else None

    def set_value(self, name: str, value: float):
        """Write a parameter."""
        self.values[name] = value

    def get_value(self, name: str) -> float:
        """Read a parameter."""
        return self.values[name]


class FakeDrone:
    """A connected drone, only its parameters."""

    def __init__(self, names: tuple[str, ...] = ()):
        """Firmware with the parameters `names` besides the deck's."""
        self.param = FakeParams(names)


def test_start_logging_dates_the_card_file_after_the_recording():
    """Firmware that can name the file gets the time of the recording folder before the start."""
    drone = FakeDrone(("usd.fileDate", "usd.fileTime"))
    usd.start_logging(drone, "cf4", datetime(2026, 11, 20, 5, 26, 55))
    values = drone.param.values
    assert (values["usd.fileDate"], values["usd.fileTime"], values["usd.logging"]) == (
        20261120,
        52655,
        1,
    )


def test_start_logging_on_older_firmware_numbers_the_file(caplog: pytest.LogCaptureFixture):
    """Without the naming parameters the card still logs, under a numbered name."""
    drone = FakeDrone()
    with caplog.at_level(logging.WARNING):
        usd.start_logging(drone, "cf3", datetime(2026, 11, 20, 15, 26, 55))
    assert drone.param.values["usd.logging"] == 1
    assert "numbers the card file" in caplog.text


def test_collect_copies_each_card_file_into_its_recording(tmp_path: Path):
    """Files go to the recording and role that named them, a recurring numbered name stays."""
    recordings, card = tmp_path / "results", tmp_path / "card"
    card.mkdir()
    named = {
        "test_2026-11-20_15-26-55": {
            "victim": ["victim_2026-11-20_15-26-55_SD"],
            "source": ["src06"],
        },
        "test_2026-11-19_22-42-53": {"victim": ["lcv00"], "source": []},
        "test_2026-11-21_09-00-00": {"victim": ["lcv00"], "source": []},
    }
    for folder, sd in named.items():
        (recordings / folder).mkdir(parents=True)
        meta = {"sd": {role: {"files": files} for role, files in sd.items()}}
        (recordings / folder / "meta.json").write_text(json.dumps(meta))
    for name in ("victim_2026-11-20_15-26-55_SD", "src06", "lcv00", "config.txt"):
        (card / name).write_bytes(name.encode())
    copied = usd.collect(card, recordings)
    flight = recordings / "test_2026-11-20_15-26-55" / "sd"
    assert sorted(copied) == [
        flight / "source" / "src06",
        flight / "victim" / "victim_2026-11-20_15-26-55_SD",
    ]
    assert not list(recordings.glob("test_2026-11-*/sd/victim/lcv00"))
    assert usd.collect(card, recordings) == []  # Already there
