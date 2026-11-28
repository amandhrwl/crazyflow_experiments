"""A2 on the stand: the static map's items placed over the victim clamped on the load cell.

The source holds still at the items of `a2_design` over the stand's victim, `height_sep` above its
motion capture position, and the victim reads the load cell. In a spin run the victim runs its
motors at hover through `motorPowerSet` for one *part* of the design, a reference, three or four
offsets and, in the second part of a height, the last reference: about 25 s with the arming, which
a full battery carries (A1: 4.0 to 3.4 V within about a minute). The part's first reference is the
spin-alone baseline, its tare the motors-off seconds before the arming. A still run flies any parts
with the motors off; its tare is the references. This module holds what the deploy script, the
conversion and the tests share and needs no radio, ROS or motion capture: the placement and the
spin schedule; the victim's thread and the battery budget are A3's (`a3_stand`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crazyflow_experiments.downwash import a2_design, a3_stand
from crazyflow_experiments.downwash.timeline import build_static_flight, segment_dwell_times

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

    from crazyflow_experiments.downwash.timeline import Timeline


def stand_rows(rows: Sequence[Mapping], cfg: Mapping) -> list[dict]:
    """The rows the stand flies: the map at `[stand] height_seps`, every `[[a2.series]]` item."""
    kept = {int(r["leg"]) for r in a3_stand.stand_rows(rows, cfg)}
    return [
        dict(row)
        for row in rows
        if row.get("series", a2_design.MAP_SERIES) != a2_design.MAP_SERIES
        or int(row["leg"]) in kept
    ]


def stand_flight(
    cfg: Mapping, rows: Sequence[Mapping], stand_pos: ArrayLike, parts: Sequence[int] | None
) -> tuple[dict, list[dict]]:
    """Config and items of a flight over the victim on the stand.

    Only the map's items at the stand's `height_seps` and those of the series are kept, of the given parts, placed `height_sep`
    above the stand's position and retimed from the takeoff. See `a3_stand.stand_config`.

    Args:
        cfg: The A3 config, with its `[stand]` and `[a2]` sections.
        rows: The items of the design, see `a2_design.build_design`.
        stand_pos: Position of the victim on the stand as (3,), its marker frame.
        parts: Parts to fly, all of the stand's for None.

    Returns:
        The config and the items, both copies.

    Raises:
        ValueError: No item is at a height the stand flies, or a part is not.
    """
    kept = stand_rows(rows, cfg)
    if not kept:
        raise ValueError(
            f"No item is at a height of [stand] height_seps {cfg['stand']['height_seps']}"
        )
    known = sorted({int(row["part"]) for row in kept})
    if parts is not None and (other := sorted(set(parts) - set(known))):
        raise ValueError(f"Parts {other} fly at a height the stand skips, its parts are {known}")
    stand_cfg = a3_stand.stand_config(cfg, stand_pos)
    return stand_cfg, a2_design.place(a2_design.select_parts(kept, parts), stand_pos, stand_cfg)


def spin_schedule(
    source: Timeline, rows: Sequence[Mapping], stand: Mapping, discard_s: float, spin: bool
) -> list[a3_stand.LegSpin]:
    """The victim's windows over the flight, one `LegSpin` per part of a spin run.

    A part's tare, arming and settling fall before its first reference's used seconds, so that the
    reference is the spin-alone baseline; the motors stop `spin_after_s` after its last dwell. The
    entry's `leg` is the part, `baseline_start` where the baseline's used seconds start, `pass_start`
    their end and `cross`, `pass_end` and `leg_end` the end of the part's last dwell.

    A spin run flies one part: its battery carries no more. A still run has one entry for all of
    its parts, which only stamps the windows of the first.

    Args:
        source: The source's timeline of the flight, see `timeline.build_static_flight`.
        rows: The items of the flight.
        stand: The `[stand]` config.
        discard_s: Seconds of each dwell that are not used, `a2.discard_s`.
        spin: Whether the motors run.

    Returns:
        The windows in flight order.

    Raises:
        ValueError: A spin run covers several parts, the tare would start before t = 0 or the
            motors would run past the timeline.
    """
    table = source.table()
    parts = sorted({int(row["part"]) for row in rows})
    if spin and len(parts) > 1:
        raise ValueError(f"A spin run flies one part, its battery carries no more; got {parts}")
    ordered = sorted(rows, key=lambda row: int(row["index_in_flight"]))
    first, last = ordered[0], ordered[-1]
    use_start = segment_dwell_times(table, int(first["leg"]))["dwell_start"] + discard_s
    first_end = segment_dwell_times(table, int(first["leg"]))["end"]
    end = segment_dwell_times(table, int(last["leg"]))["end"]
    spin_on = use_start - stand["spin_settle_s"]
    arm_at = spin_on - stand["arm_lead_s"]
    tare_start = arm_at - stand["tare_s"]
    spin_off = end + stand["spin_after_s"]
    if tare_start < 0.0:
        raise ValueError(f"The tare would start {-tare_start:.2f} s before t = 0")
    if spin_off > source.duration:
        raise ValueError(f"The motors would run {spin_off - source.duration:.2f} s past the flight")
    return [
        a3_stand.LegSpin(
            leg=int(first["part"]),
            tare_start=tare_start,
            arm_at=arm_at,
            spin_on=spin_on,
            baseline_start=use_start,
            pass_start=first_end,
            cross=end,
            pass_end=end,
            spin_off=spin_off,
            leg_end=end,
        )
    ]


def plan_stand(
    cfg: Mapping,
    rows: Sequence[Mapping],
    stand_pos: NDArray,
    source_pos: NDArray,
    parts: Sequence[int] | None,
    spin: bool,
) -> tuple[dict, list[dict], list[a3_stand.LegSpin]]:
    """The stand's config, items and spin schedule of a run, checked against every limit.

    Args:
        cfg: The A3 config, with its `[stand]` and `[a2]` sections.
        rows: The items of the design.
        stand_pos: Position of the victim on the stand as (3,).
        source_pos: Where the source stands on the floor as (3,).
        parts: Parts to fly, all of the stand's for None.
        spin: Whether the victim's motors run.

    Returns:
        The stand's config, its items and the victim's windows.
    """
    stand_cfg, stand_items = stand_flight(cfg, rows, stand_pos, parts)
    a2_design.check_design(stand_items, stand_cfg)
    a3_stand.check_source_start(source_pos, stand_pos, cfg["stand"]["clearance"])
    _, source = build_static_flight(stand_items, stand_cfg)
    schedule = spin_schedule(source, stand_items, cfg["stand"], cfg["a2"]["discard_s"], spin)
    return stand_cfg, stand_items, schedule
