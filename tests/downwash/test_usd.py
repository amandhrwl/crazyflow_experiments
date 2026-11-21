"""SD card configs, logging, file collection, the log decoder and the quaternion compression."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import json  # noqa: E402
import logging  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from crazyflow_experiments.downwash import ROLES, usd  # noqa: E402
from crazyflow_experiments.downwash.lc_test_convert import SYNC_VARIABLES  # noqa: E402
from tests.downwash.sync_fakes import usd_file  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

VARIABLES = [("stateEstimate.z", "f"), ("stateEstimateZ.quat", "I"), ("motor.m1_rpm", "H")]


def make_log(rows: list[tuple], timestamps: list[int], version: int = 2) -> bytes:
    """A complete A3-like log file of `rows`, CRC included."""
    return usd_file(VARIABLES, timestamps, [list(c) for c in zip(*rows)], version=version)


@pytest.fixture
def log_rows() -> tuple[list[tuple], list[int]]:
    rng = np.random.default_rng(0)
    quats = usd.quatcompress(R.random(50, random_state=1).as_quat())
    rows = [
        (float(np.float32(z)), int(q), int(rpm))
        for z, q, rpm in zip(rng.uniform(0, 2, 50), quats, rng.integers(5000, 30000, 50))
    ]
    return rows, [1_000_000 + 1978 * i for i in range(50)]


def check_decoded(decoded: dict, rows: list[tuple], timestamps: list[int]):
    event = decoded["fixedFrequency"]
    np.testing.assert_array_equal(event["t_us"], timestamps)
    for i, (name, _) in enumerate(VARIABLES):
        np.testing.assert_array_equal(event[name], [row[i] for row in rows])


def test_decoder_round_trip(tmp_path: Path, log_rows: tuple):
    rows, timestamps = log_rows
    path = tmp_path / "vic00"
    path.write_bytes(make_log(rows, timestamps))
    decoded = usd.decode_usd(path)
    check_decoded(decoded, rows, timestamps)
    assert decoded["_info"] | {"bytes": 0} == {
        "file": "vic00",
        "version": 2,
        "crc_ok": True,
        "truncated": False,
        "bytes": 0,
    }


def test_decoder_reads_version_1_in_ms(tmp_path: Path, log_rows: tuple):
    rows, _ = log_rows
    timestamps_ms = [1000 + 2 * i for i in range(len(rows))]
    path = tmp_path / "old00"
    path.write_bytes(make_log(rows, timestamps_ms, version=1))
    check_decoded(usd.decode_usd(path), rows, [t * 1000 for t in timestamps_ms])


def test_decoder_keeps_the_records_of_a_cut_off_file(tmp_path: Path, log_rows: tuple):
    """Logging not stopped: no CRC, the last record cut in half."""
    rows, timestamps = log_rows
    data = make_log(rows, timestamps)[:-4]  # No CRC
    path = tmp_path / "src00"
    path.write_bytes(data[:-7])  # The last record is incomplete
    with pytest.warns(UserWarning, match="CRC"):
        decoded = usd.decode_usd(path)
    check_decoded(decoded, rows[:-1], timestamps[:-1])
    assert not decoded["_info"]["crc_ok"]
    assert decoded["_info"]["truncated"]


def test_decoder_keeps_every_record_without_crc(tmp_path: Path, log_rows: tuple):
    rows, timestamps = log_rows
    path = tmp_path / "src01"
    path.write_bytes(make_log(rows, timestamps)[:-4])
    with pytest.warns(UserWarning, match="CRC"):
        decoded = usd.decode_usd(path)
    check_decoded(decoded, rows, timestamps)
    assert not decoded["_info"]["truncated"]


def test_decoder_rejects_other_files(tmp_path: Path):
    path = tmp_path / "config.txt"
    path.write_text("1\n2048\nvic\n")
    with pytest.raises(ValueError, match="not a uSD log"):
        usd.decode_usd(path)


def test_flat_decoder_reads_back_what_the_firmware_writes(tmp_path: Path):
    """Timestamps and values of every type come back exactly as one flat record."""
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


def test_flat_decoder_keeps_the_records_of_a_file_without_crc(tmp_path: Path):
    """A file not closed by a stop has no CRC, the decoder warns and keeps every whole record."""
    t_us = np.arange(5) * 2000
    data = usd_file([("loadcell.rawWeight", "i")], t_us, [np.arange(5, dtype=np.int32)])
    path = tmp_path / "lcv01"
    path.write_bytes(data[:-4])
    with pytest.warns(UserWarning, match="CRC"):
        record = usd.decode(path)
    assert len(record["t_us"]) == 5


def test_quaternion_round_trip():
    quat = R.random(20_000, random_state=0).as_quat()
    back = usd.quatdecompress(usd.quatcompress(quat))
    np.testing.assert_allclose(np.linalg.norm(back, axis=1), 1.0, atol=1e-12)
    angle = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(quat * back, axis=1)), 0.0, 1.0)))
    assert angle.max() < 0.25  # About 0.2 deg at most, as quatcompress.h documents
    identity = usd.quatdecompress(usd.quatcompress(np.array([0.0, 0.0, 0.0, 1.0])))
    np.testing.assert_allclose(identity, [[0.0, 0.0, 0.0, 1.0]], atol=1e-12)


def test_every_card_config_parses_within_the_deck_limits():
    """Each block's stored configs parse and keep the 8192 B buffer the cf21B's heap allows.

    The stem of every card file comes from line 3: `vic` on the victim, `src` on the source, as the
    cards wrote `vic_2026-11-21_16-31-44_SD` and `src_2026-11-21_16-31-44_SD` on 21 Nov.
    """
    for block, roles in usd.CARD_CONFIGS.items():
        for role in roles:
            config = usd.role_config(block, role)
            assert config.buffer <= 8192, (block, role)
            assert config.frequency == 500, (block, role)
            assert config.name == {"victim": "vic", "source": "src"}[role], (block, role)


@pytest.mark.parametrize(("role", "name"), [("victim", "vic"), ("source", "src")])
def test_a3_card_configs(role: str, name: str):
    config = usd.role_config("a3", role)
    assert (config.version, config.buffer, config.name) == (1, 8192, name)
    assert (config.event, config.frequency, config.mode) == ("on:fixedFrequency", 500, 1)
    assert not config.enable_on_startup
    assert len(config.variables) == usd.MAX_VARIABLES
    assert {"stateEstimateZ.quat", "pm.vbatMV", "acc.z", "ctrltarget.z"} <= set(config.variables)
    assert {f"motor.m{i}_rpm" for i in range(1, 5)} <= set(config.variables)
    assert ("ctrlMel.cmd_thrust" in config.variables) == (role == "victim")
    assert ("ctrltarget.x" in config.variables) == (role == "source")


@pytest.mark.parametrize("role", ROLES)
def test_lc_card_configs_log_the_sync_variable(role: str):
    """Each load cell test config logs the variable its drone also streams for the sync."""
    config = usd.role_config("lc", role)
    assert config.frequency == 500
    assert config.buffer <= 8192  # A larger buffer fails the cf21B's boot self-test
    assert SYNC_VARIABLES[role][1] in config.variables


def test_lc_victim_card_logs_the_load_cell_and_what_its_force_balance_needs():
    """The raw load cell reading, the motor rpm for the thrust, the accelerometer and the tilt.

    The force follows from the raw reading, so the firmware's converted weight is not logged. The
    tilt comes from the compressed attitude quaternion, as on the A3 cards.
    """
    variables = usd.role_config("lc", "victim").variables
    needed = ["loadcell.rawWeight", "acc.z", "stateEstimateZ.quat"]
    assert set(needed + [f"motor.m{i}_rpm" for i in range(1, 5)]) <= set(variables)
    assert "loadcell.weight" not in variables


def test_write_config_copies_the_role(tmp_path: Path):
    target = usd.write_config(tmp_path, "a3", "victim")
    assert target == tmp_path / "config.txt"
    assert target.read_text() == usd.config_path("a3", "victim").read_text()
    with pytest.raises(FileNotFoundError):
        usd.write_config(tmp_path / "not_mounted", "a3", "source")
    with pytest.raises(ValueError, match="role"):
        usd.config_path("a3", "bystander")
    with pytest.raises(ValueError, match="role"):
        usd.config_path("a1", "source")  # A1 logs the victim only


def test_write_config_refuses_an_unmounted_card(tmp_path: Path):
    """Writing to a missing card root fails instead of creating a directory."""
    with pytest.raises(FileNotFoundError):
        usd.write_config(tmp_path / "not_mounted", "lc", "victim")
    written = usd.write_config(tmp_path, "lc", "source")
    assert usd.read_config(written) == usd.role_config("lc", "source")


def test_read_config_rejects_what_the_firmware_would_misread(tmp_path: Path):
    header = "1\n2048\nvic\n0\non:fixedFrequency\n500\n1\n"
    cases = {
        "too_many": header + "".join(f"acc.v{i}\n" for i in range(21)),
        "comment": header + "acc.x  # x\n",
        "frequency": header.replace("500", "300") + "acc.x\n",
        "duplicate": header + "acc.x\nacc.x\n",
    }
    for name, text in cases.items():
        path = tmp_path / f"{name}.txt"
        path.write_text(text)
        with pytest.raises(ValueError):
            usd.read_config(path)


def fake_drone(known: set[str]) -> tuple[SimpleNamespace, list[Callable]]:
    callbacks: list[Callable] = []
    toc = SimpleNamespace(get_element_by_complete_name=lambda name: name if name in known else None)
    console = SimpleNamespace(receivedChar=SimpleNamespace(add_callback=callbacks.append))
    cf = SimpleNamespace(log=SimpleNamespace(toc=toc), console=console)
    return SimpleNamespace(name="cf3", cf=cf), callbacks


def test_check_variables_finds_missing_ones():
    variables = usd.role_config("a3", "victim").variables
    drone, _ = fake_drone(set(variables))
    usd.check_variables(drone.cf, drone.name, variables)
    drone, _ = fake_drone(set(variables) - {"motor.m3_rpm"})
    with pytest.raises(RuntimeError, match="motor.m3_rpm"):
        usd.check_variables(drone.cf, drone.name, variables)


def test_console_capture_splits_lines():
    drone, callbacks = fake_drone(set())
    console = usd.ConsoleCapture(drone.cf)
    for chunk in ("SYS: ok\nuSD: Logg", "ing to: vic03\n", "Skip log variable foo.bar\nuSD: Wr"):
        callbacks[0](chunk)
    assert console.files() == ["vic03"]
    assert console.skipped_variables() == ["Skip log variable foo.bar"]
    assert console.lines == [
        "SYS: ok",
        "uSD: Logging to: vic03",
        "Skip log variable foo.bar",
        "uSD: Wr",
    ]


def test_console_capture_counts_the_dropped_events():
    """src12 (F0, 18 Nov) dropped 1213 events while its card stalled."""
    drone, callbacks = fake_drone(set())
    console = usd.ConsoleCapture(drone.cf)
    callbacks[0]("uSD: Logging to: src12\nSUP: Ready to fly\n")
    assert console.written() == []
    callbacks[0]("uSD: Wrote 11662894 B to: src12 (145782 of 146995 events)\n")
    assert console.written() == [("src12", 145782, 146995)]


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


def test_start_logging_without_a_recording_leaves_the_numbering(caplog: pytest.LogCaptureFixture):
    """A3 passes no recording: its files stay numbered `vicNN`, `srcNN`, without a warning."""
    drone = FakeDrone(("usd.fileDate", "usd.fileTime"))
    with caplog.at_level(logging.WARNING):
        usd.start_logging(drone, "cf3")
    values = drone.param.values
    assert (values["usd.fileDate"], values["usd.fileTime"], values["usd.logging"]) == (0, 0, 1)
    assert not caplog.text


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


def test_recording_time_is_the_timestamp_of_the_folder():
    """The deploy scripts name the card files with it, so file and folder carry the same time."""
    folder = Path("results/downwash/a3/flight2_2026-11-22_09-05-07")
    assert usd.recording_time(folder) == datetime(2026, 11, 22, 9, 5, 7)
    with pytest.raises(ValueError, match="timestamp"):
        usd.recording_time(Path("test_fake"))


def card_folder(tmp_path: Path, files: list[str]) -> Path:
    """A recording whose victim card files are `files`."""
    folder = tmp_path / "test_2026-11-22_10-00-00"
    (folder / "sd" / "victim").mkdir(parents=True)
    for name in files:
        (folder / "sd" / "victim" / name).write_bytes(b"\xbc")
    return folder


def test_sd_file_takes_the_name_the_console_reported(tmp_path: Path):
    """A dated name, a numbered one of older firmware, and the last of several starts."""
    folder = card_folder(tmp_path, ["vic_2026-11-22_10-00-00_SD", "vic03", "vic04"])
    for files, expected in (
        (["vic_2026-11-22_10-00-00_SD"], "vic_2026-11-22_10-00-00_SD"),
        (["vic03"], "vic03"),
        (["vic03", "vic04"], "vic04"),
    ):
        meta = {"sd": {"victim": {"files": files}}}
        assert usd.sd_file(folder, "victim", meta).name == expected
    with pytest.raises(FileNotFoundError, match="collect"):
        usd.sd_file(folder, "victim", {"sd": {"victim": {"files": ["vic05"]}}})


def test_sd_file_without_a_console_report_takes_the_dated_file(tmp_path: Path):
    """The recording's timestamp picks its file among others, the only file is taken as well."""
    meta = {"usd": {"victim": {"files": []}}}  # A3 keeps its card entries under "usd"
    folder = card_folder(tmp_path, ["vic_2026-11-21_16-31-44_SD", "vic_2026-11-22_10-00-00_SD"])
    assert usd.sd_file(folder, "victim", meta).name == "vic_2026-11-22_10-00-00_SD"
    only = card_folder(tmp_path / "only", ["vic07"])
    assert usd.sd_file(only, "victim", meta).name == "vic07"
    both = card_folder(tmp_path / "both", ["vic07", "vic08"])
    with pytest.raises(FileNotFoundError, match="one card file"):
        usd.sd_file(both, "victim", meta)


def test_collect_finds_a3_flights_by_their_usd_entries(tmp_path: Path):
    """A3's meta.json names its cards under "usd", collect copies them all the same."""
    recordings, card = tmp_path / "a3", tmp_path / "card"
    card.mkdir()
    flight = recordings / "flight1_2026-11-22_10-00-00"
    flight.mkdir(parents=True)
    meta = {"usd": {"victim": {"files": ["vic_2026-11-22_10-00-00_SD"]}, "source": {"files": []}}}
    (flight / "meta.json").write_text(json.dumps(meta))
    (card / "vic_2026-11-22_10-00-00_SD").write_bytes(b"\xbc")
    copied = usd.collect(card, recordings)
    assert copied == [flight / "sd" / "victim" / "vic_2026-11-22_10-00-00_SD"]
    assert usd.sd_file(flight, "victim", meta) == copied[0]
