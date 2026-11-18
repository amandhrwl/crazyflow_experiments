"""SD card configs, the log decoder and the quaternion compression of the downwash experiment."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import struct  # noqa: E402
import zlib  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from crazyflow_experiments.downwash import usd  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

VARIABLES = (("stateEstimate.z", "f"), ("stateEstimateZ.quat", "I"), ("motor.m1_rpm", "H"))
EVENT_ID = 0xFFFF


def make_log(rows: list[tuple], timestamps: list[int], version: int = 2) -> bytes:
    """A complete log file as the firmware writes it, CRC included."""
    data = b"\xbc" + struct.pack("<HH", version, 1)
    data += struct.pack("<H", EVENT_ID) + b"fixedFrequency\0" + struct.pack("<H", len(VARIABLES))
    data += b"".join(f"{name}({kind})".encode() + b"\0" for name, kind in VARIABLES)
    head = struct.Struct("<HI" if version == 1 else "<HQ")
    values = struct.Struct("<" + "".join(kind for _, kind in VARIABLES))
    for timestamp, row in zip(timestamps, rows):
        data += head.pack(EVENT_ID, timestamp) + values.pack(*row)
    return data + struct.pack("<I", zlib.crc32(data))


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


def test_quaternion_round_trip():
    quat = R.random(20_000, random_state=0).as_quat()
    back = usd.quatdecompress(usd.quatcompress(quat))
    np.testing.assert_allclose(np.linalg.norm(back, axis=1), 1.0, atol=1e-12)
    angle = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(quat * back, axis=1)), 0.0, 1.0)))
    assert angle.max() < 0.25  # About 0.2 deg at most, as quatcompress.h documents
    identity = usd.quatdecompress(usd.quatcompress(np.array([0.0, 0.0, 0.0, 1.0])))
    np.testing.assert_allclose(identity, [[0.0, 0.0, 0.0, 1.0]], atol=1e-12)


@pytest.mark.parametrize(("role", "name"), [("victim", "vic"), ("source", "src")])
def test_card_configs(role: str, name: str):
    config = usd.role_config(role)
    assert (config.version, config.buffer, config.name) == (1, 8192, name)
    assert (config.event, config.frequency, config.mode) == ("on:fixedFrequency", 500, 1)
    assert not config.enable_on_startup
    assert len(config.variables) == usd.MAX_VARIABLES
    assert {"stateEstimateZ.quat", "pm.vbatMV", "acc.z", "ctrltarget.z"} <= set(config.variables)
    assert {f"motor.m{i}_rpm" for i in range(1, 5)} <= set(config.variables)
    assert ("ctrlMel.cmd_thrust" in config.variables) == (role == "victim")
    assert ("ctrltarget.x" in config.variables) == (role == "source")


def test_write_config_copies_the_role(tmp_path: Path):
    target = usd.write_config(tmp_path, "victim")
    assert target == tmp_path / "config.txt"
    assert target.read_text() == usd.config_path("victim").read_text()
    with pytest.raises(FileNotFoundError):
        usd.write_config(tmp_path / "not_mounted", "source")
    with pytest.raises(ValueError, match="role"):
        usd.config_path("bystander")


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
    variables = usd.role_config("victim").variables
    drone, _ = fake_drone(set(variables))
    usd.check_variables(drone, variables)
    drone, _ = fake_drone(set(variables) - {"motor.m3_rpm"})
    with pytest.raises(RuntimeError, match="motor.m3_rpm"):
        usd.check_variables(drone, variables)


def test_console_capture_splits_lines():
    drone, callbacks = fake_drone(set())
    console = usd.capture_console(drone)
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
    console = usd.capture_console(drone)
    callbacks[0]("uSD: Logging to: src12\nSUP: Ready to fly\n")
    assert console.written() == []
    callbacks[0]("uSD: Wrote 11662894 B to: src12 (145782 of 146995 events)\n")
    assert console.written() == [("src12", 145782, 146995)]
