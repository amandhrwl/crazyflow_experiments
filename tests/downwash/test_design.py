"""Leg table of the downwash block A3 and the config it is built from."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import math  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import pytest  # noqa: E402
import toml  # noqa: E402

from crazyflow_experiments.downwash import CONFIG_PATH  # noqa: E402
from crazyflow_experiments.downwash.a3_design import (  # noqa: E402
    arrival_time,
    build_design,
    build_makeup,
    check_design,
    cruise_tilt,
    design_cells,
    flight_orders,
    flights,
    footprint_half_width,
    load_config,
    needed_half,
    pass_duration,
    pass_half,
    read_design,
    write_design,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(require_masses=False)


@pytest.fixture(scope="module")
def design(cfg: dict) -> list[dict]:
    return build_design(cfg)[0]


def write_config(tmp_path: Path, **sections: dict) -> Path:
    """The repo config with some entries replaced, written to a temporary file."""
    config = toml.load(CONFIG_PATH)
    for section, values in sections.items():
        table = config
        for key in section.split("."):
            table = table[key]
        table.update(values)
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps(config))
    return path


def test_v_max_is_the_arenas_limit_capped_at_max_speed(cfg: dict, tmp_path: Path):
    a3 = cfg["a3"]
    assert a3["v_max"] == 2.0  # max_speed; the arena's own is floor((3.8 - 0.3 - 0.6) / 1.1) = 2.6
    assert a3["speeds"] == [0.5, 1.0, 1.5, 2.0]
    uncapped = load_config(write_config(tmp_path, a3={"max_speed": 9.0}), require_masses=False)
    assert uncapped["a3"]["v_max"] == 2.6
    assert uncapped["a3"]["speeds"] == [0.5, 1.0, 1.5, 2.6]


def test_pass_geometry_of_the_four_speeds(cfg: dict):
    a3 = cfg["a3"]
    expected = {0.5: (1.15, 5.6), 1.0: (1.70, 4.4), 1.5: (2.25, 4.0), 2.0: (2.80, 3.8)}
    for speed, (half, duration) in expected.items():
        assert pass_half(speed, a3) == pytest.approx(half)
        assert pass_duration(speed, a3) == pytest.approx(duration, abs=1e-3)


def test_jet_model_reproduces_the_a3_numbers():
    """The rebuilt afterwash model gives the arrival, footprint and tilt numbers of §2.2."""
    for height_sep, arrival in ((0.4, 0.09), (0.8, 0.20), (1.2, 0.35)):
        assert round(arrival_time(height_sep), 2) == arrival
    assert arrival_time(0.5) == pytest.approx(0.116, abs=5e-4)  # The first flyover's number
    for height_sep, footprint in ((0.4, 0.25), (0.8, 0.33), (1.2, 0.40)):
        assert footprint_half_width(height_sep) == pytest.approx(footprint, abs=0.015)
    for height_sep, needed in ((0.4, 0.28), (0.8, 0.38), (1.2, 0.49)):
        assert needed_half(height_sep, 1.4) == pytest.approx(needed, abs=0.015)
    for speed, tilt in ((0.5, 1.5), (1.0, 2.9), (1.4, 4.1)):
        assert math.degrees(cruise_tilt(speed)) == pytest.approx(tilt, abs=0.05)


def test_block_has_48_legs_in_six_flights_each_cell_once_per_repeat(cfg: dict, design: list[dict]):
    assert len(design) == 48
    assert [row["leg"] for row in design] == list(range(48))
    cells = design_cells(cfg)
    assert len(cells) == 16
    repeats_of_cell = defaultdict(list)
    for row in design:
        repeats_of_cell[row["cell"]].append(row["repeat"])
        assert (row["height_sep"], row["speed"]) == cells[row["cell"]]
    assert sorted(repeats_of_cell) == list(range(16))
    for cell, repeats in repeats_of_cell.items():
        assert sorted(repeats) == [0, 1, 2], f"cell {cell} is not flown once per repeat"
    assert sorted(flights(design)) == [1, 2, 3, 4, 5, 6]
    for flight, rows in flights(design).items():
        assert len(rows) == 8
        assert {row["repeat"] for row in rows} == {(flight - 1) // 2}


def test_every_flight_holds_whole_heights_the_stand_flies_every_second(
    cfg: dict, design: list[dict]
):
    """A far flight (1.0 and 0.75 m) then a near one (0.5 and 0.25 m) per repeat."""
    stand = set(cfg["stand"]["height_seps"])
    for flight, rows in flights(design).items():
        heights = {row["height_sep"] for row in rows}
        assert heights == ({0.5, 0.25} if flight % 2 == 0 else {1.0, 0.75})
        assert (heights <= stand) == (flight % 2 == 0)
        assert Counter(row["speed"] for row in rows) == {0.5: 2, 1.0: 2, 1.5: 2, 2.0: 2}


def test_flights_one_and_two_are_the_shakedown_order(design: list[dict]):
    first, second = flights(design)[1], flights(design)[2]
    order = [(row["height_sep"], row["speed"]) for row in [*first, *second]]
    assert order == sorted(order, key=lambda cell: (-cell[0], cell[1]))
    assert [row["height_sep"] for row in first] == [1.0] * 4 + [0.75] * 4
    assert [row["height_sep"] for row in second] == [0.5] * 4 + [0.25] * 4


def test_legs_fly_plus_x_and_chain(cfg: dict, design: list[dict]):
    """Every pass flies -x to +x, the source flies back in three moves through the return lane."""
    for rows in flights(design).values():
        for index, row in enumerate(rows):
            assert row["index_in_flight"] == index
            assert row["direction"] == 1
            assert row["x_start"] == pytest.approx(-row["half"])
            assert row["x_end"] == pytest.approx(row["half"])
            assert row["z_src"] == pytest.approx(
                cfg["drones"]["victim"]["hover_z"] + row["height_sep"]
            )
        assert rows[0]["move_time"] == 0.0
        for prev, row in zip(rows, rows[1:]):
            assert row["move_time"] >= 3 * cfg["a3"]["min_move_time"]
            assert row["t_leg_start"] == pytest.approx(prev["t_leg_end"])


def test_everything_inside_the_arena(cfg: dict, design: list[dict]):
    check_design(design, cfg)  # Every point of both commanded paths at 100 Hz
    safety = cfg["safety"]
    x_limit = safety["arena_max"][0] - safety["x_margin"]
    for row in design:
        assert max(abs(row["x_start"]), abs(row["x_end"])) <= x_limit
        assert row["z_src"] <= safety["arena_max"][2] - 0.1 + 1e-9
        assert row["needed_half_m"] < cfg["a3"]["cruise_half"]
        assert row["arrival_s"] + 0.15 <= cfg["a3"]["settle_time"]


def test_settled_windows(cfg: dict, design: list[dict]):
    a3 = cfg["a3"]
    for row in design:
        assert row["settled_half_s"] == pytest.approx(a3["cruise_half"] / row["speed"])
        cruise_end = row["t_pass_end"] - a3["ramp"] - row["t_cross"]
        assert row["settled_end_ext_offset_s"] <= cruise_end + 1e-9
        assert row["settled_end_ext_offset_s"] == pytest.approx(
            row["settled_half_s"] + row["arrival_s"] + 0.15
        )


def test_flight_orders(cfg: dict):
    seed = cfg["a3"]["seed"]
    orders = flight_orders(9, 3, seed)  # One flight per repeat
    assert orders[0] == list(range(9))
    assert all(sorted(order) == list(range(9)) for order in orders)
    assert flight_orders(9, 3, seed) == orders  # The seed reproduces the design
    split = flight_orders(16, 3, seed, 8)  # Each repeat in two flights of whole chunks
    assert split[:2] == [list(range(8)), list(range(8, 16))]
    for k, order in enumerate(split):
        first = 8 * (k % 2)
        assert sorted(order) == list(range(first, first + 8))
    assert split[2] != list(range(8)) and flight_orders(16, 3, seed, 8) == split
    with pytest.raises(ValueError, match="divide"):
        flight_orders(16, 3, seed, 6)


def test_csv_round_trip(tmp_path: Path, design: list[dict]):
    path = write_design(design, tmp_path / "design.csv")
    read = read_design(path)
    assert len(read) == len(design)
    for row, back in zip(design, read):
        assert set(back) == set(row)
        for key, value in row.items():
            assert back[key] == pytest.approx(value, abs=1e-6)


def test_single_leg_is_the_first_flyover(cfg: dict):
    rows, seed = build_design(cfg, single=(1.0, 0.5))
    assert seed is None
    assert len(rows) == 1
    row = rows[0]
    assert (row["half"], row["D"], row["z_src"]) == pytest.approx((1.7, 4.4, 1.3))
    assert row["arrival_s"] == pytest.approx(0.116, abs=5e-4)
    check_design(rows, cfg)


def test_makeup_flight_repeats_the_failed_legs(cfg: dict, design: list[dict]):
    """F0 (18 Nov) lost 2 of 9 legs to gaps in the source's SD log; they are flown again."""
    legs = [40, 4, 13]
    rows = build_makeup(cfg, design, legs)
    by_leg = {row["leg"]: row for row in design}
    assert [row["leg"] for row in rows] == legs
    assert [row["index_in_flight"] for row in rows] == [0, 1, 2]
    assert {row["flight"] for row in rows} == {7}
    for row in rows:
        original = by_leg[row["leg"]]
        for key in ("cell", "repeat", "speed", "height_sep", "half", "D", "x_start", "z_src"):
            assert row[key] == pytest.approx(original[key])
    assert rows[0]["t_leg_start"] == pytest.approx(design[0]["t_leg_start"])  # Same lead-in
    for prev, row in zip(rows, rows[1:]):
        assert row["t_leg_start"] == pytest.approx(prev["t_leg_end"])
    check_design(rows, cfg)
    assert build_makeup(cfg, rows, [4])[0]["flight"] == 8  # A make-up of the make-up
    with pytest.raises(ValueError, match="not in the design"):
        build_makeup(cfg, design, [48])
    with pytest.raises(ValueError, match="repeat a leg"):
        build_makeup(cfg, design, [4, 4])


def test_config_requires_masses_for_flying(tmp_path: Path):
    path = write_config(tmp_path, **{"drones.victim": {"mass_kg": 0.0}})
    with pytest.raises(ValueError, match="mass_kg"):
        load_config(path, require_masses=True)
    load_config(path, require_masses=False)
    filled = write_config(
        tmp_path, **{"drones.victim": {"mass_kg": 0.046}, "drones.source": {"mass_kg": 0.046}}
    )
    load_config(filled, require_masses=True)


def test_only_the_drones_that_fly_need_a_mass(tmp_path: Path):
    """The victim on the stand or on the ground has no free-flight mass, the source always one."""
    unweighed = write_config(
        tmp_path, **{"drones.victim": {"mass_kg": 0.0}, "drones.source": {"mass_kg": 0.046}}
    )
    load_config(unweighed, require_masses=True, mass_roles=("source",))
    with pytest.raises(ValueError, match="victim"):
        load_config(unweighed, require_masses=True)
    no_source = write_config(tmp_path, **{"drones.source": {"mass_kg": 0.0}})
    with pytest.raises(ValueError, match="source"):
        load_config(no_source, require_masses=True, mass_roles=("source",))


def test_config_requires_a_plausible_thrust_scale(tmp_path: Path):
    for scale in (0.0, 9.046):  # Unset, and a slipped decimal point
        path = write_config(tmp_path, **{"drones.source": {"thrust_scale": scale}})
        with pytest.raises(ValueError, match="thrust_scale"):
            load_config(path, require_masses=False)
    path = write_config(tmp_path, **{"drones.source": {"thrust_scale": 0.9046}})
    load_config(path, require_masses=False)


def test_config_rejects_legs_per_flight_that_do_not_divide_the_cells(tmp_path: Path):
    path = write_config(tmp_path, a3={"legs_per_flight": 6})
    with pytest.raises(ValueError, match="legs_per_flight"):
        load_config(path, require_masses=False)


def test_config_rejects_a_short_arena(tmp_path: Path):
    path = write_config(
        tmp_path, safety={"arena_min": [-2.0, -2.0, 0.0], "arena_max": [2.0, 2.0, 2.3]}
    )
    with pytest.raises(ValueError, match="v_max"):
        load_config(path, require_masses=False)


def test_config_rejects_a_low_ceiling(tmp_path: Path):
    path = write_config(tmp_path, safety={"arena_max": [3.5, 2.0, 1.8]})  # The 1.0 m pass at 1.8 m
    with pytest.raises(ValueError, match="ceiling"):
        load_config(path, require_masses=False)


def test_config_rejects_a_late_wake(tmp_path: Path):
    path = write_config(tmp_path, a3={"settle_time": 0.3})  # The wake arrives 0.27 s late at 1.0 m
    with pytest.raises(ValueError, match="settle_time"):
        load_config(path, require_masses=False)


def test_config_rejects_a_return_lane_in_the_wake(tmp_path: Path):
    path = write_config(tmp_path, a3={"return_lane": 0.4})  # Inside the 0.4 m separation guard
    with pytest.raises(ValueError, match="return_lane"):
        load_config(path, require_masses=False)


def test_cells_cover_the_grid(cfg: dict):
    cells = design_cells(cfg)
    assert Counter(dz for dz, _ in cells) == {1.0: 4, 0.75: 4, 0.5: 4, 0.25: 4}
    assert Counter(v for _, v in cells) == {0.5: 4, 1.0: 4, 1.5: 4, 2.0: 4}
