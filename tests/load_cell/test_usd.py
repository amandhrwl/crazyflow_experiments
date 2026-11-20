"""Card configs and the log decoder of the load cell test."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

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


def test_victim_card_logs_the_load_cell():
    """The victim's card records the raw load cell reading, the converted force derives from it."""
    assert "loadcell.rawWeight" in usd.role_config("victim").variables


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
