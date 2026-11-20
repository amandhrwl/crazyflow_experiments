"""The clock sync of the load cell test recovers the known clocks of a faked recording."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import json  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.load_cell import convert, usd  # noqa: E402
from crazyflow_experiments.load_cell.sync import (  # noqa: E402
    TICK_WRAP,
    fit_radio_clock,
    unwrap_ticks,
)
from tests.load_cell import fakes  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

SD_TOLERANCE = 0.5e-3  # s, card records against their true PC time


@pytest.fixture
def recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A faked recording, converted, with the quicklook kept out of the repository."""
    monkeypatch.setattr(convert, "QUICKLOOK_DIR", tmp_path / "plots")
    folder = tmp_path / "test_fake"
    folder.mkdir()
    fakes.write_recording(folder)
    convert.convert(folder)
    return folder


@pytest.mark.parametrize(
    ("role", "clock"), [("victim", fakes.VICTIM_CLOCK), ("source", fakes.SOURCE_CLOCK)]
)
def test_card_records_land_on_their_true_pc_time(recording: Path, role: str, clock: fakes.Clock):
    """Every card record maps to its true PC time, late by the floor of the radio latency.

    The floor cannot be told apart from a clock offset, so it stays in, the same for both drones.
    """
    data = np.load(recording / "synced.npz")
    sd = usd.decode(recording / "sd" / role / f"{'lcv' if role == 'victim' else 'lcs'}00")
    true = clock.pc(clock.sd_rate * sd["t_us"] / 1e6 + clock.sd_offset) - float(data["t_ref"])
    error = data[f"{role}_sd/t"] - true
    assert np.abs(error - fakes.LATENCY_FLOOR).max() < SD_TOLERANCE


def test_report_recovers_the_drift_and_matches_the_card_to_the_radio(recording: Path):
    """The fit finds each crystal's drift, and the shifted cards reproduce the radio values.

    The victim's card catches every load cell reading, the source's card every other
    accelerometer value, so half of the source's radio samples have no match on its card. The
    source's card timer runs at the rate of the firmware before the fix of its prescaler.
    """
    report = json.loads((recording / "sync.json").read_text())
    for role, clock, fraction in (
        ("victim", fakes.VICTIM_CLOCK, 1.0),
        ("source", fakes.SOURCE_CLOCK, 0.5),
    ):
        assert report[role]["drift_percent"] == pytest.approx((clock.rate - 1) * 100, abs=1e-3)
        assert report[role]["sd_rate"] == clock.sd_rate
        assert report[role]["match_fraction"] == pytest.approx(fraction, abs=0.05)
        halves = report[role]["sd_offset_halves_ms"]
        assert abs(halves[0] - halves[1]) < 0.5


def test_force_is_zeroed_on_still_air_and_shows_the_hold_over(recording: Path):
    """The force reads zero in the still air before and the faked push while the source is over."""
    phases = json.loads((recording / "sync.json").read_text())["load_cell"]["phases"]
    assert phases["still_before"]["mean_mN"] == pytest.approx(0.0, abs=0.05)
    assert phases["over"]["mean_mN"] == pytest.approx(fakes.OVER_FORCE * 1e3, abs=0.2)
    assert phases["aside_1"]["mean_mN"] == pytest.approx(0.0, abs=0.2)


def test_radio_clock_fit_runs_through_a_tick_wrap():
    """The 24 bit tick wraps after 4.66 h, the unwrapped time and the fit carry on through it."""
    tick = (TICK_WRAP - 5000 + np.arange(0, 10000, 10)) % TICK_WRAP
    unwrapped = unwrap_ticks(tick)
    assert np.all(np.diff(unwrapped) == pytest.approx(0.01))
    arrival = 100.0 + 1.002 * unwrapped + 1e-3
    clock, stats = fit_radio_clock(unwrapped, arrival)
    assert stats["drift_percent"] == pytest.approx(0.2, abs=1e-6)
    assert clock.pc_from_tick(unwrapped) == pytest.approx(arrival, abs=1e-9)


def test_a_card_from_another_flight_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A card record whose values never meet the radio's fails the sync instead of misaligning."""
    monkeypatch.setattr(convert, "QUICKLOOK_DIR", tmp_path / "plots")
    for name, seed in (("this", 0), ("other", 1)):
        (tmp_path / name).mkdir()
        fakes.write_recording(tmp_path / name, seed=seed)
    card = tmp_path / "this" / "sd" / "victim" / "lcv00"
    card.write_bytes((tmp_path / "other" / "sd" / "victim" / "lcv00").read_bytes())
    with pytest.raises(RuntimeError, match="reproduces only"):
        convert.convert(tmp_path / "this")
