"""Block A2: the item table, its timeline, the stand's placement and spin schedule, item summary."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import toml  # noqa: E402

from crazyflow_experiments.downwash import CONFIG_PATH  # noqa: E402
from crazyflow_experiments.downwash.a2_deploy import item_summary  # noqa: E402
from crazyflow_experiments.downwash.a2_design import (  # noqa: E402
    build_design,
    check_design,
    design_heights,
    load_config,
    read_design,
    retime,
    select_parts,
    write_design,
)
from crazyflow_experiments.downwash.a2_stand import (  # noqa: E402
    plan_stand,
    spin_schedule,
    stand_flight,
)
from crazyflow_experiments.downwash.timeline import (  # noqa: E402
    build_static_flight,
    segment_dwell_times,
)

if TYPE_CHECKING:
    from pathlib import Path

STAND = np.array([0.466, 0.015, 1.521])  # The victim on the stand, 22 Nov
SOURCE = np.array([2.35, 0.02, 0.03])


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(require_masses=False)


@pytest.fixture(scope="module")
def design(cfg: dict) -> list[dict]:
    return build_design(cfg)


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


def test_grid_far_to_near_with_three_references_per_height(cfg: dict, design: list[dict]):
    assert design_heights(cfg) == [1.0, 0.75, 0.5, 0.25]
    assert len(design) == 4 * (1 + 4 + 1 + 3 + 1) == 40
    assert [row["leg"] for row in design] == list(range(40))
    assert [row["height_sep"] for row in design] == sorted(
        (row["height_sep"] for row in design), reverse=True
    )
    for part in range(8):
        rows = [row for row in design if row["part"] == part]
        assert rows[0]["kind"] == "reference"  # Every part is led by its own baseline
        points = [row["dx"] for row in rows if row["kind"] == "point"]
        assert points == (
            cfg["a2"]["offsets_x"][:4] if part % 2 == 0 else cfg["a2"]["offsets_x"][4:]
        )
        assert (rows[-1]["kind"] == "reference") == (
            part % 2 == 1
        )  # A height's last part ends on one
    assert all(
        row["dx"] == cfg["a2"]["reference_dx"] for row in design if row["kind"] == "reference"
    )


def test_times_and_positions(cfg: dict, design: list[dict]):
    a2, victim = cfg["a2"], cfg["drones"]["victim"]
    for prev, row in zip(design, design[1:]):
        assert row["t_start"] == pytest.approx(prev["t_end"])
    for row in design:
        assert row["t_use_start"] - row["t_dwell_start"] == pytest.approx(a2["discard_s"])
        assert row["t_end"] - row["t_dwell_start"] == pytest.approx(a2["dwell_s"])
        assert row["x_src"] == pytest.approx(victim["hover_xy"][0] + row["dx"])
        assert row["z_src"] == pytest.approx(victim["hover_z"] + row["height_sep"])
    assert design[0]["move_time"] == 0.0  # The takeoff ends at the first point
    assert all(row["move_time"] >= cfg["a3"]["min_move_time"] for row in design[1:])


def test_height_changes_only_at_the_reference(cfg: dict, design: list[dict]):
    check_design(design, cfg)
    broken = [dict(row) for row in design]
    broken[22] |= {"height_sep": 0.9, "z_src": broken[22]["z_src"] + 0.4}  # A point at a new height
    with pytest.raises(ValueError, match="off the reference"):
        check_design(broken, cfg)


def test_everything_inside_the_arena_and_clear_of_the_victim(cfg: dict, design: list[dict]):
    check_design(design, cfg)  # Both paths at 100 Hz
    victim, source = build_static_flight(design, cfg)
    t = np.linspace(0.0, source.duration, 4001)
    p_vic, p_src = victim.state(t)[0], source.state(t)[0]
    close = np.linalg.norm(p_src[:, :2] - p_vic[:, :2], axis=1) < cfg["safety"]["xy_guard"]
    assert (p_src[close, 2] - p_vic[close, 2]).min() >= 0.25 - 1e-9  # The lowest height, over it
    assert p_src[:, 2].max() <= 1.8 + 0.06  # The highest point plus a sync hop


def test_csv_round_trip(tmp_path: Path, design: list[dict]):
    read = read_design(write_design(design, tmp_path / "design.csv"))
    assert len(read) == len(design)
    for row, back in zip(design, read):
        assert set(back) == set(row)
        for key, value in row.items():
            assert back[key] == (
                value if isinstance(value, str) else pytest.approx(value, abs=1e-6)
            )


def test_config_rejects_what_does_not_fit(tmp_path: Path):
    for bad in ({"part_sizes": [4, 4]}, {"discard_s": 0.2}, {"reference_dx": -0.3}):
        with pytest.raises(ValueError, match="part_sizes|discard_s|reference|wake"):
            load_config(write_config(tmp_path, a2=bad), require_masses=False)


def test_the_static_flight_is_at_rest_at_every_dwell(cfg: dict, design: list[dict]):
    victim, source = build_static_flight(design, cfg)
    assert victim.duration == pytest.approx(source.duration)
    table = source.table()
    for row in design:
        times = segment_dwell_times(table, int(row["leg"]))
        assert times["end"] - times["dwell_start"] == pytest.approx(cfg["a2"]["dwell_s"])
        t = np.linspace(
            times["dwell_start"], times["end"] - 1e-6, 50
        )  # The end starts the next move
        pos, vel, acc = source.state(t)
        assert np.allclose(pos, [row["x_src"], row["y_src"], row["z_src"]])
        assert np.allclose(vel, 0.0) and np.allclose(acc, 0.0)
        assert source.label(times["dwell_start"] + 0.1) == ("dwell", int(row["leg"]))
    assert segment_dwell_times(table, 5)["dwell_start"] == pytest.approx(design[5]["t_dwell_start"])


def test_a_skipped_item_holds_where_the_source_is(cfg: dict, design: list[dict]):
    _, source = build_static_flight(design, cfg, skipped=[3])
    times = segment_dwell_times(source.table(), 3)
    pos = source.state(0.5 * (times["start"] + times["end"]))[0]
    assert np.allclose(pos, [design[2]["x_src"], design[2]["y_src"], design[2]["z_src"]])
    nominal = build_static_flight(design, cfg)[1]
    assert source.duration == pytest.approx(nominal.duration)


def test_the_stand_flies_the_near_heights_over_its_own_position(cfg: dict, design: list[dict]):
    stand_cfg, items = stand_flight(cfg, design, STAND, None)
    assert sorted({row["height_sep"] for row in items}) == [0.25, 0.5]
    assert [row["part"] for row in items if row["kind"] == "reference"][0] == 4
    for row in items:
        assert row["x_src"] == pytest.approx(STAND[0] + row["dx"])
        assert row["z_src"] == pytest.approx(STAND[2] + row["height_sep"])
    assert [row["index_in_flight"] for row in items] == list(range(20))
    assert items[0]["t_dwell_start"] == pytest.approx(retime(items, stand_cfg)[0]["t_dwell_start"])
    assert cfg["drones"]["victim"]["hover_z"] == 0.8  # The config itself is untouched
    assert stand_cfg["a3"]["height_seps"] == [
        0.25,
        0.5,
    ]  # The takeoff climbs to the stand's own 0.5 m
    with pytest.raises(ValueError, match="skips"):
        stand_flight(cfg, design, STAND, [0])


def test_a_part_alone_starts_at_its_first_reference(cfg: dict, design: list[dict]):
    stand_cfg, items = stand_flight(cfg, design, STAND, [5])
    assert [row["part"] for row in items] == [5] * 5
    assert [row["kind"] for row in items] == ["reference", "point", "point", "point", "reference"]
    assert items[0]["move_time"] == 0.0 and items[0]["x_src"] == pytest.approx(STAND[0] - 1.0)
    check_design(items, stand_cfg)
    assert len(select_parts(design, [0, 1])) == 10


def test_a_spin_run_covers_one_part_with_its_own_tare(cfg: dict, design: list[dict]):
    stand = cfg["stand"]
    for part in (4, 5, 6, 7):
        stand_cfg, items, schedule = plan_stand(cfg, design, STAND, SOURCE, [part], True)
        (entry,) = schedule
        assert entry.leg == part
        # Motors off for the tare, arming, settling, and the first reference is the baseline
        assert entry.tare_start == pytest.approx(entry.arm_at - stand["tare_s"])
        assert entry.spin_on == pytest.approx(entry.arm_at + stand["arm_lead_s"])
        assert entry.baseline_start - entry.spin_on == pytest.approx(stand["spin_settle_s"])
        first_use = items[0]["t_use_start"]
        assert entry.baseline_start == pytest.approx(first_use)
        assert entry.tare_start >= 0.0
        assert entry.spin_off == pytest.approx(items[-1]["t_end"] + stand["spin_after_s"])
        assert entry.spin_time < stand["battery_spin_s"]
    with pytest.raises(ValueError, match="one part"):
        plan_stand(cfg, design, STAND, SOURCE, [4, 5], True)
    _, items, still = plan_stand(cfg, design, STAND, SOURCE, None, False)
    assert len(items) == 20 and len(still) == 1  # A still run only stamps its windows


def test_the_source_must_stand_clear_of_the_stand(cfg: dict, design: list[dict]):
    with pytest.raises(ValueError, match="clearance"):
        plan_stand(cfg, design, STAND, STAND + [0.2, 0.2, -1.5], [4], True)
    stand_cfg, items = stand_flight(cfg, design, STAND, [4])
    _, source = build_static_flight(items, stand_cfg)
    spin_schedule(source, items, cfg["stand"], cfg["a2"]["discard_s"], spin=True)


def records(
    item: int, error: np.ndarray, spread: float = 0.002, victim_dz: float | None = None
) -> list[dict]:
    """Records of one item: a move tick, then 3 s of dwell at 100 Hz around target + error."""
    rng = np.random.default_rng(0)
    out = []
    for k in range(350):
        dwell = k >= 50
        out.append(
            {
                "t_perf": 10.0 + k / 100,
                "source": {
                    "leg": item,
                    "phase": "dwell" if dwell else "move",
                    "pos": np.array([0.1, 0.0, 2.0]) + error + rng.normal(0.0, spread, 3),
                },
                "victim": None
                if victim_dz is None
                else {"pos": np.array([0.0, 0.0, 0.8 + victim_dz])},
            }
        )
    return out


def test_item_summary_reports_the_hold_error_and_flags(cfg: dict):
    row = {"leg": 7, "x_src": 0.1, "y_src": 0.0, "z_src": 2.0}
    stats, flags = item_summary(records(7, np.array([0.01, -0.01, 0.048])), row, {}, cfg)
    assert stats["error"] == pytest.approx([0.01, -0.01, 0.048], abs=1e-3)
    assert stats["n_used"] == 200  # 2 of the 3 s: the first is discarded
    assert flags == []
    _, flags = item_summary(records(7, np.array([0.08, 0.0, 0.0])), row, {}, cfg)
    assert flags == ["off_target"]
    _, flags = item_summary(records(7, np.zeros(3), spread=0.03), row, {}, cfg)
    assert "not_still" in flags
    stats, flags = item_summary(records(7, np.zeros(3), victim_dz=-0.12), row, {}, cfg)
    assert stats["victim_dz"] == pytest.approx(-0.12) and "victim_sinks" in flags
    assert item_summary(records(8, np.zeros(3)), row, {}, cfg) == ({}, ["not_flown"])
    assert item_summary(records(7, np.zeros(3))[:60], row, {}, cfg) == ({}, ["short"])
