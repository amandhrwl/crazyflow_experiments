"""A2 on the stand: the static map from the load cell, on a synthetic run with known answers."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import csv  # noqa: E402
import math  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from crazyflow_experiments.downwash import a2_stand_convert  # noqa: E402
from crazyflow_experiments.downwash.a2_design import build_design, load_config  # noqa: E402
from crazyflow_experiments.downwash.a2_stand import stand_flight  # noqa: E402
from crazyflow_experiments.downwash.a2_stand_convert import item_rows, mean_over  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

CFG = load_config(require_masses=False)
STAND = np.array([0.466, 0.015, 1.521])
T_ZERO = 1000.0
G = 9.81
OFFSET = np.array([0.02, 0.0, 0.048])  # How far the source holds from its point
DIP = 0.100  # N, the wake force at the jet's axis
WIDTH = 0.100  # m
THRUST_STEP = 0.004  # N more rotor thrust at the item on the axis
BASE = 0.53  # N, the victim's thrust at hover


def wake(dx_meas: float) -> float:
    return -DIP * math.exp(-((dx_meas / WIDTH) ** 2))


def synthetic_run(mode: str, gap_item: int | None = None, part: int = 4) -> tuple[dict, list[dict]]:
    """One part on the stand: a load cell with a 120 g zero, the dips and a drifting baseline."""
    _, items = stand_flight(CFG, build_design(CFG), STAND, [part])
    t = np.arange(T_ZERO + 4.0, T_ZERO + items[-1]["t_end"] + 3.0, 1 / 500)
    force, thrust = np.zeros_like(t), np.zeros_like(t)
    spinning = mode == "spin"
    for item in items:
        inside = (t >= T_ZERO + item["t_dwell_start"]) & (t < T_ZERO + item["t_end"])
        if item["kind"] == "point":
            dx_meas = item["dx"] + OFFSET[0]
            force[inside] += wake(dx_meas)
            if spinning and item["dx"] == 0.0:
                thrust[inside] += THRUST_STEP
                force[inside] += THRUST_STEP
    if spinning:
        thrust += BASE
        force += BASE
    force += 0.001 * (t - t[0]) / (t[-1] - t[0])  # 1 mN of drift over the run
    keep = np.ones_like(t, dtype=bool)
    if gap_item is not None:
        item = items[gap_item]
        keep &= ~(
            (t > T_ZERO + item["t_use_start"] + 0.5) & (t < T_ZERO + item["t_use_start"] + 0.9)
        )
    t_src = np.arange(T_ZERO, T_ZERO + items[-1]["t_end"] + 5.0, 0.01)
    pos = np.zeros((len(t_src), 3))
    for item in items:
        at = (t_src >= T_ZERO + item["t_start"]) & (t_src < T_ZERO + item["t_end"])
        pos[at] = [item["x_src"], item["y_src"], item["z_src"]]
    run = {
        "meta": {"stand_pos": STAND.tolist()},
        "t": t[keep],
        "weight_g": (120.0 + force / G * 1e3)[keep],
        "rpm": np.tile(np.where(spinning, 18000.0, 0.0), (len(t), 4))[keep],
        "thrust_rpm": thrust[keep],
        "vbat": np.full(len(t), 3.9)[keep],
        "t_zero": T_ZERO,
        "src_t": t_src,
        "src_pos": pos + OFFSET,
        "src_vel": np.zeros_like(pos),
    }
    return run, items


def rows_by_item(rows: list[dict]) -> dict[int, dict]:
    return {int(row["leg"]): row for row in rows}


def test_mean_over_skips_missing_readings_and_finds_gaps():
    t = np.arange(0.0, 1.0, 0.01)
    x = np.stack([np.full(100, 2.0), np.where(t < 0.5, np.nan, 4.0)], axis=1)
    mean, std, n, gap = mean_over(t, x, 0.2, 0.8)
    assert mean == pytest.approx([2.0, 4.0]) and std == pytest.approx([0.0, 0.0]) and n == 61
    assert gap == pytest.approx(0.01)
    assert np.isnan(mean_over(t, x, 0.1, 0.3)[0][1])  # No reading in the window
    keep = (t < 0.4) | (t > 0.6)
    assert mean_over(t[keep], x[keep], 0.2, 0.8)[3] == pytest.approx(0.21, abs=0.011)


def test_still_run_gives_the_wake_force_against_the_first_reference():
    run, items = synthetic_run("still")
    rows = rows_by_item(item_rows(run, items, {}, "still", last_t=1e9))
    assert {row["status"] for row in rows.values()} == {"ok"}
    first = rows[int(items[0]["leg"])]
    assert first["fz_rel_N"] == pytest.approx(0.0, abs=1e-12)
    for item in items[1:]:
        row = rows[int(item["leg"])]
        expected = wake(item["dx"] + OFFSET[0]) if item["kind"] == "point" else 0.0
        assert row["fz_rel_N"] == pytest.approx(expected, abs=2e-3)  # The 1 mN drift is in it
    assert rows[int(items[1]["leg"])]["fz_rel_N"] == pytest.approx(-DIP * math.exp(-0.04), abs=2e-3)


def test_the_geometry_is_measured_not_nominal():
    run, items = synthetic_run("still")
    rows = rows_by_item(item_rows(run, items, {}, "still", last_t=1e9))
    for item in items:
        row = rows[int(item["leg"])]
        assert row["dx_meas"] == pytest.approx(item["dx"] + OFFSET[0], abs=1e-6)
        assert row["dz_meas"] == pytest.approx(item["height_sep"] + OFFSET[2], abs=1e-6)
        assert row["dx_err"] == pytest.approx(OFFSET[0], abs=1e-6)
        assert row["dz_err"] == pytest.approx(OFFSET[2], abs=1e-6)
        assert row["dy_meas"] == pytest.approx(0.0, abs=1e-6)
        assert row["pos_std_xy"] == pytest.approx(0.0, abs=1e-9)


def test_spin_run_splits_the_rotor_thrust_from_the_wake():
    run, items = synthetic_run("spin")
    rows = rows_by_item(item_rows(run, items, {4: True}, "spin", last_t=1e9))
    on_axis = rows[int(next(i for i in items if i["kind"] == "point" and i["dx"] == 0.0)["leg"])]
    assert on_axis["dthrust_rpm_N"] == pytest.approx(THRUST_STEP, abs=1e-9)
    assert on_axis["fz_rel_N"] == pytest.approx(wake(OFFSET[0]) + THRUST_STEP, abs=2e-3)
    assert on_axis["dforce_wake_N"] == pytest.approx(wake(OFFSET[0]), abs=2e-3)
    assert on_axis["rpm_mean"] == pytest.approx(18000.0) and on_axis["rpm_valid"] == 1.0
    assert on_axis["thrust_rpm_N"] == pytest.approx(BASE + THRUST_STEP, abs=1e-9)
    assert on_axis["vbat_V"] == pytest.approx(3.9)


def test_the_last_reference_reports_the_drift_of_the_tare():
    run, items = synthetic_run("still", part=5)  # [ref, 3 points, ref]
    assert items[-1]["kind"] == "reference"
    rows = item_rows(run, items, {}, "still", last_t=1e9)
    t0, t1 = run["t"][0], run["t"][-1]

    def drift(item: dict) -> float:  # The injected 1 mN ramp, at the middle of the used seconds
        return 1e-3 * (T_ZERO + 0.5 * (item["t_use_start"] + item["t_end"]) - t0) / (t1 - t0)

    assert rows[-1]["ref_drift_N"] == pytest.approx(drift(items[-1]) - drift(items[0]), abs=1e-5)
    assert rows[-1]["ref_drift_N"] > 0.4e-3
    assert all(math.isnan(row["ref_drift_N"]) for row in rows[:-1])  # Only references report one
    only_points = item_rows(
        *synthetic_run("still")[:1], synthetic_run("still")[1], {}, "still", 1e9
    )
    assert all(math.isnan(row["ref_drift_N"]) for row in only_points)  # Part 4 has no trailing one


def test_status_of_gaps_unspun_parts_and_unflown_items():
    run, items = synthetic_run("spin", gap_item=2)
    rows = rows_by_item(item_rows(run, items, {4: True}, "spin", last_t=1e9))
    assert rows[int(items[2]["leg"])]["status"] == "card_gap"
    assert rows[int(items[1]["leg"])]["status"] == "ok"
    unspun = item_rows(run, items, {4: False}, "spin", last_t=1e9)
    assert {row["status"] for row in unspun} == {"not_spun"}
    cut = rows_by_item(item_rows(run, items, {4: True}, "spin", last_t=items[2]["t_end"] - 1.0))
    assert [cut[int(item["leg"])]["status"] for item in items[2:]] == ["not_flown"] * 3
    assert cut[int(items[1]["leg"])]["status"] == "ok"
    # A part whose first reference stalled has no baseline
    run_gap, items = synthetic_run("spin", gap_item=0)
    rows = item_rows(run_gap, items, {4: True}, "spin", last_t=1e9)
    assert rows[0]["status"] == "card_gap"
    assert {row["status"] for row in rows[1:]} == {"no_baseline"}


def test_combine_keeps_the_ok_items_of_each_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(a2_stand_convert, "A2_RESULTS_DIR", tmp_path)
    folders = []
    for mode, status in (("still", "ok"), ("still", "card_gap"), ("spin", "ok")):
        folder = tmp_path / f"stand_flight1_{mode}_{len(folders)}"
        folder.mkdir()
        rows = [
            {"run": folder.name, "mode": mode, "status": status, "height_sep": "0.5", "dx": "0.05"}
        ]
        a2_stand_convert.write_rows(folder / "items.csv", rows)
        folders.append(folder)
    maps = a2_stand_convert.combine(folders)
    assert [len(maps["S0"]), len(maps["S1"])] == [1, 1]
    with open(tmp_path / "static_map_S0.csv", newline="") as f:
        assert [r["mode"] for r in csv.DictReader(f)] == ["still"]
