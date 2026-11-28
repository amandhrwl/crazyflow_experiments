"""Block A1: put a sweep onto PC time, turn the load cell into thrust and cut the steady windows.

Copy the card file into the block folder first (`--collect` does it from a mounted card or a
folder of card files), then run

    pixi run -e sim2real python crazyflow_experiments/downwash/a1_convert.py \
        results/downwash/a1/<block>_<timestamp> [--calibration <calib_run.toml>] [--collect <card>]

The events are stamped with the PC clock, so even case S, where the card holds both the load cell
and the rpm, maps the card onto PC time: the tick through the arrival times of the radio packets
and the card timer through the load cell reading it logs both ways (`sync.py`, which
also finds the 85/84 card timer of firmware before bitcraze #1684). Case R1 without a card uses
the radio blocks alone, on the tick.

The thrust is F = force_sign * g / 1000 * (w - w_tare(t)) with w = a * rawWeight + b and the tare
interpolated linearly between the tare windows. Each level gives one steady window from
`discard_s` after its start to its end; `steady.csv` holds their means, the flags and the
crazyflow thrust of their rpm, `sync.json` the clock, the tare drift and the sync pulses, and the
timeline goes to `plots/downwash/a1/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from crazyflow_experiments.downwash import A1_PLOTS_DIR, A1_RESULTS_DIR, usd
from crazyflow_experiments.downwash.a1_stand import N_MOTORS
from crazyflow_experiments.downwash.sync import (
    LOAD_CELL_SYNC,
    fit_radio_clock,
    sync_drone,
    unwrap_ticks,
)
from crazyflow_experiments.downwash.usd import RPM_INVALID, sd_file

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

SD_GAP = 0.010  # s, longer card gaps (card stalls) are counted and their edge samples dropped
STABILIZER_PERIOD = 1e-3  # s, the loop in which the DShot telemetry updates the rpm
# Case R1 with a card: the card has no load cell, the rpm syncs it (it changes every loop pass)
R1_SYNC = ("motors", "motor.m1_rpm", STABILIZER_PERIOD)
RPM_STD_TOL = 0.02  # Relative rpm standard deviation over a window that flags it
# Fraction of rpm values (65535, no telemetry) over a window that flags it. Single dropouts are
# normal on cf1 (37 of 114000 values in the sweep of 21 Nov) and already left out of the means.
RPM_INVALID_TOL = 0.01
TREND_TOL = 0.01  # Force change over a window, relative to its mean, that flags it
MIN_SAMPLE_FRACTION = 0.5  # Of the samples a window should hold at the card or radio rate
MIN_CORRELATION = 0.99  # Level means of the force against crazyflow's thrust of their rpm
TARE_DRIFT_TOL = 0.01  # Of the hover thrust, tare drift over a block that is reported
PULSE_BEFORE = 0.3  # s of baseline before a sync pulse
PULSE_PLATEAU = 0.1  # s at the end of a pulse taken as its plateau
WINDOW_KINDS = ("ramp", "center", "level")
# Light chart tokens and the first four categorical slots, as the load cell quicklook
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND, BAND_2 = "#e1e0d9", "#c3c2b7", "#efeee9", "#e6eef9"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def read_events(path: Path) -> list[dict]:
    """The steps of a block from events.csv, numbers converted."""
    ints = ("step", "level_id", "motor", "repeat", "m1", "m2", "m3", "m4")
    floats = ("duration", "t_start", "t_end", "predicted_N", "measured_N")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(
                row
                | {k: int(row[k]) for k in ints}
                | {k: float(row[k]) if row[k] else np.nan for k in floats}
            )
    return rows


def gap_mask(t: NDArray, max_gap: float = SD_GAP) -> tuple[NDArray[np.bool_], NDArray]:
    """Samples away from card stalls, and the start index and length of every gap in s.

    The sample on either side of a gap longer than `max_gap` is dropped, the samples further
    away are kept: a stall drops records, it does not corrupt the ones written.
    """
    dt = np.diff(t)
    gaps = np.flatnonzero(dt > max_gap)
    valid = np.ones(len(t), dtype=bool)
    valid[gaps] = False
    valid[gaps + 1] = False
    return valid, np.column_stack([gaps, dt[gaps]]) if len(gaps) else np.empty((0, 2))


def tare_line(
    t: NDArray, weight: NDArray, valid: NDArray, windows: Sequence[tuple[float, float]]
) -> tuple[NDArray, list[dict]]:
    """The zero of the load cell over time, linear between the means of the tare windows.

    Before the first and after the last tare window it holds their value.

    Args:
        t: Sample times in s.
        weight: Load cell reading in g.
        valid: Samples to use.
        windows: Start and end of each tare window in s, after its discard.

    Returns:
        The tare at every sample in g, and each window's mid time, mean and standard deviation.
    """
    tares = []
    for t0, t1 in windows:
        inside = valid & (t >= t0) & (t < t1)
        if inside.sum() < 2:
            raise ValueError(f"The tare window {t0:.1f}-{t1:.1f} s holds no load cell samples")
        tares.append(
            {"t_mid": (t0 + t1) / 2, "mean_g": weight[inside].mean(), "std_g": weight[inside].std()}
        )
    if not tares:
        raise ValueError("No tare window in the block, every block starts with one")
    mids = np.array([w["t_mid"] for w in tares])
    means = np.array([w["mean_g"] for w in tares])
    return np.interp(t, mids, means), tares


def window_stats(
    signals: dict[str, tuple[NDArray, NDArray, NDArray]],
    t0: float,
    t1: float,
    coef: NDArray,
    rates: dict[str, float],
) -> dict:
    """Means, spreads and flags of one steady window.

    Args:
        signals: Per signal ("force" in N, "rpm" with one column per motor, "vbat" in V) its
            times, values and valid samples. Invalid rpm (65535) are NaN.
        t0: Start of the window in s.
        t1: End of the window in s.
        coef: Crazyflow's rpm2thrust.
        rates: Sample rate in Hz of each signal, for the expected sample count.

    Returns:
        The row of steady.csv without the step's own columns.
    """
    t, force, valid = signals["force"]
    inside = valid & (t >= t0) & (t < t1)
    f, tf = force[inside], t[inside]
    t_rpm, rpm, valid_rpm = signals["rpm"]
    rows_rpm = rpm[valid_rpm & (t_rpm >= t0) & (t_rpm < t1)]
    t_v, vbat, valid_v = signals["vbat"]
    v = vbat[valid_v & (t_v >= t0) & (t_v < t1)]
    duration = t1 - t0
    out = {"n": int(inside.sum()), "n_rpm": len(rows_rpm)}
    flags = []
    if out["n"] < MIN_SAMPLE_FRACTION * rates["force"] * duration or len(rows_rpm) < 2:
        flags.append("few_samples")
    if out["n"] >= 2:
        slope = float(np.polyfit(tf - tf.mean(), f, 1)[0])
        mean_f = float(f.mean())
        trend = abs(slope * duration) / max(abs(mean_f), 1e-9)
        out |= {"F_mean": mean_f, "F_std": float(f.std()), "F_slope": slope, "F_trend": trend}
        if trend > TREND_TOL:
            flags.append("force_trending")
    else:
        out |= {"F_mean": np.nan, "F_std": np.nan, "F_slope": np.nan, "F_trend": np.nan}
    n_invalid = int(np.isnan(rows_rpm).sum()) if len(rows_rpm) else 0
    if n_invalid > RPM_INVALID_TOL * max(rows_rpm.size, 1):
        flags.append("rpm_invalid")
    means, stds, squares = [], [], []
    for i in range(N_MOTORS):
        column = rows_rpm[:, i] if len(rows_rpm) else np.array([np.nan])
        means.append(float(np.nanmean(column)) if np.isfinite(column).any() else np.nan)
        stds.append(float(np.nanstd(column)) if np.isfinite(column).any() else np.nan)
        squares.append(float(np.nanmean(column**2)) if np.isfinite(column).any() else np.nan)
        out |= {f"rpm{i + 1}": means[i], f"rpm{i + 1}_std": stds[i], f"rpm{i + 1}_sq": squares[i]}
    relative = np.array(stds) / np.maximum(np.array(means), 1.0)
    if np.nanmax(relative, initial=0.0) > RPM_STD_TOL:
        flags.append("rpm_unsteady")
    c0, c1, c2 = coef
    out["sum_rpm"] = float(np.sum(means))
    out["sum_rpm2"] = float(np.sum(squares))
    out["F_cf"] = N_MOTORS * c0 + c1 * out["sum_rpm"] + c2 * out["sum_rpm2"]
    out["vbat"] = float(v.mean()) if len(v) else np.nan
    gaps = signals.get("gaps")
    if gaps is not None:
        t_gap, length = gaps[0], gaps[1]
        hit = (t_gap >= t0) & (t_gap < t1)
        out["n_gaps"] = int(hit.sum())
        out["longest_gap_ms"] = float(length[hit].max() * 1e3) if hit.any() else 0.0
        if hit.any():
            flags.append("sd_gap")
    else:
        out |= {"n_gaps": 0, "longest_gap_ms": 0.0}
    return out | {"flags": "|".join(flags), "ok": not flags}


def directions(rows: list[dict]) -> list[str]:
    """Whether each window's total command rose or fell from the window before (hysteresis).

    The single-motor block compares windows of the same motor; a motor's first is "first".
    """
    out, previous = [], {}
    for row in rows:
        total = sum(row[f"m{i}"] for i in range(1, N_MOTORS + 1))
        before = previous.get(row["motor"])
        out.append(
            "first"
            if before is None
            else "up"
            if total > before
            else "down"
            if total < before
            else "same"
        )
        previous[row["motor"]] = total
    return out


def edge_time(t: NDArray, x: NDArray, t_cmd: float, duration: float) -> float:
    """When a signal crosses halfway from its level before a pulse to its plateau, in s.

    Args:
        t: Sample times in s.
        x: The signal.
        t_cmd: Time of the command step in s.
        duration: Length of the pulse in s.
    """
    before = (t >= t_cmd - PULSE_BEFORE) & (t < t_cmd)
    plateau = (t >= t_cmd + duration - PULSE_PLATEAU) & (t < t_cmd + duration)
    if before.sum() < 2 or plateau.sum() < 2:
        return np.nan
    base, top = x[before].mean(), x[plateau].mean()
    half = base + (top - base) / 2
    rising = np.sign(top - base)
    after = np.flatnonzero((t >= t_cmd - 0.05) & (t < t_cmd + duration))
    above = rising * (x[after] - half) > 0
    if not above.any() or above[0]:
        return np.nan
    k = after[np.argmax(above)]
    # Linear between the last sample below and the first above the half level
    return float(t[k - 1] + (half - x[k - 1]) / (x[k] - x[k - 1]) * (t[k] - t[k - 1]))


def pulse_checks(
    events: list[dict], cmd: tuple[NDArray, NDArray], signals: dict, t_ref: float
) -> list[dict]:
    """Edges of the command, the rpm and the force at every sync pulse.

    The force lags the rpm by the load cell's conversion and filter (about one or two samples at
    80 SPS). After a correct sync the lag is the same at the start and the end of a block.
    """
    t_cmd_log, m1 = cmd
    t_force, force, _ = signals["force"]
    t_rpm, rpm, _ = signals["rpm"]
    rpm_mean = np.nanmean(rpm, axis=1)
    out = []
    for row in events:
        if row["kind"] != "pulse" or row["skipped"]:
            continue
        t_start, duration = row["t_start"] - t_ref, row["duration"]
        # The command as the drone logged it: the first sample at the pulse level
        near = np.flatnonzero((t_cmd_log >= t_start - 0.2) & (m1 >= row["m1"]))
        t_cmd = float(t_cmd_log[near[0]]) if len(near) else t_start
        rpm_edge = edge_time(t_rpm, rpm_mean, t_cmd, duration)
        force_edge = edge_time(t_force, force, t_cmd, duration)
        out.append(
            {
                "step": row["step"],
                "t_cmd": t_cmd,
                "event_to_cmd_ms": (t_cmd - t_start) * 1e3,
                "rpm_edge_ms": (rpm_edge - t_cmd) * 1e3,
                "force_edge_ms": (force_edge - t_cmd) * 1e3,
                "force_minus_rpm_ms": (force_edge - rpm_edge) * 1e3,
            }
        )
    return out


def check_rises(rows: list[dict]) -> float:
    """Fail unless the force rises with the rpm over the level means.

    A falling force is a reversed sign (force_sign, or the cell's output, which reversed once
    after 20 Nov) or a loose mount.
    """
    used = [r for r in rows if np.isfinite(r["F_mean"]) and np.isfinite(r["F_cf"])]
    if len(used) < 3:
        logger.warning("Fewer than three windows, the rise of the force is not checked")
        return np.nan
    corr = float(np.corrcoef([r["F_mean"] for r in used], [r["F_cf"] for r in used])[0, 1])
    if corr < MIN_CORRELATION:
        hint = (
            "it falls as the rpm rise: check force_sign and the calibration's sign"
            if corr < 0
            else "check the mount and the flags"
        )
        raise RuntimeError(f"Force against crazyflow's thrust correlates {corr:.3f}; {hint}")
    return corr


def card_signals(
    sd: dict[str, NDArray], t: NDArray, a: float, b: float
) -> tuple[dict, NDArray, NDArray]:
    """Signals of the card record on PC time: weight in g, rpm, commands and battery."""
    valid, gaps = gap_mask(t)
    rpm = np.column_stack([sd[f"motor.m{i}_rpm"] for i in range(1, N_MOTORS + 1)]).astype(float)
    rpm[rpm == RPM_INVALID] = np.nan
    weight = a * sd["loadcell.rawWeight"].astype(float) + b if "loadcell.rawWeight" in sd else None
    signals = {
        "rpm": (t, rpm, valid),
        "vbat": (t, sd["pm.vbatMV"].astype(float) / 1e3, valid),
        "gaps": (t[gaps[:, 0].astype(int)] if len(gaps) else np.empty(0), gaps[:, 1]),
    }
    return signals, weight, np.asarray(sd["motor.m1"], dtype=float)


def convert(folder: Path, calibration: Path | None = None) -> dict:
    """Convert one block: sync, thrust, steady windows, checks and the timeline plot.

    Args:
        folder: The block, `results/downwash/a1/<block>_<timestamp>`.
        calibration: A calibscale.py file whose `a` and `b` replace the ones of the run.

    Returns:
        The contents of sync.json.
    """
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    config = meta["config"]
    events = read_events(folder / "events.csv")
    radio = {k.removeprefix("victim/"): v for k, v in np.load(folder / "radio.npz").items()}
    cal = config["load_cell"] if calibration is None else tomllib.loads(calibration.read_text())
    a, b = float(cal["a"]), float(cal["b"])
    g, sign = config["g"], config["load_cell"]["force_sign"]
    coef = np.asarray(meta["crazyflow_rpm2thrust"], dtype=float)
    discard = config["sweep"]["discard_s"]
    ran = [e for e in events if not e["skipped"]]
    t_ref = ran[0]["t_start"]
    report: dict = {"t_ref": t_ref, "case": meta["case"], "block": meta["block"]}
    report["calibration"] = {"a": a, "b": b, "from": str(calibration or "the run's config")}

    files = meta["sd"].get("victim", {}).get("files", [])
    card_dir = folder / "sd" / "victim"
    has_card = bool(files) or (card_dir.is_dir() and any(card_dir.iterdir()))
    if meta["case"] == "S" and not has_card:
        raise FileNotFoundError(f"No card file in {card_dir}, copy it in (--collect)")
    if has_card:
        sd = usd.decode(sd_file(folder, "victim", meta))
        sd_period = 1 / usd.role_config("a1", "victim").frequency
        sync = LOAD_CELL_SYNC if "loadcell.rawWeight" in sd else R1_SYNC
        clock, report["clock"] = sync_drone(radio, "victim", sd, sd_period, sync=sync)
        t_card = clock.pc_from_sd(sd["t_us"]) - t_ref
        signals, weight, m1 = card_signals(sd, t_card, a, b)
        command_log = (t_card, m1)
    else:  # R1 without a card: everything over the radio, on the tick
        clock, report["clock"] = fit_radio_clock(
            unwrap_ticks(radio["loadcell/t_tick_ms"]), radio["loadcell/t_pc"]
        )
        weight = None
        t_mot = clock.pc_from_tick(unwrap_ticks(radio["motors/t_tick_ms"])) - t_ref
        rpm = np.column_stack(
            [radio[f"motors/motor.m{i}_rpm"] for i in range(1, N_MOTORS + 1)]
        ).astype(float)
        rpm[rpm == RPM_INVALID] = np.nan
        t_status = clock.pc_from_tick(unwrap_ticks(radio["status/t_tick_ms"])) - t_ref
        ones = np.ones
        signals = {
            "rpm": (t_mot, rpm, ones(len(t_mot), dtype=bool)),
            "vbat": (t_status, radio["status/pm.vbat"].astype(float), ones(len(t_status), bool)),
        }
        command_log = (t_mot, radio["motors/motor.m1"].astype(float))
    if weight is None:  # The load cell from the radio (R1)
        t_lc = clock.pc_from_tick(unwrap_ticks(radio["loadcell/t_tick_ms"])) - t_ref
        weight = a * radio["loadcell/loadcell.rawWeight"].astype(float) + b
        valid_lc = np.ones(len(t_lc), dtype=bool)
        force_rate = 1000 / config["load_cell"]["radio_period_ms"]
    else:
        t_lc, valid_lc = signals["rpm"][0], signals["rpm"][2]
        force_rate = 1 / sd_period

    tare_windows = [
        (e["t_start"] - t_ref + discard, e["t_end"] - t_ref)
        for e in ran
        if e["kind"] == "tare" and e["t_end"] - e["t_start"] > discard
    ]
    tare, tares = tare_line(t_lc, weight, valid_lc, tare_windows)
    force = sign * g * 1e-3 * (weight - tare)
    signals["force"] = (t_lc, force, valid_lc)
    hover_g = meta["mass_kg"] * 1e3
    drift = tares[-1]["mean_g"] - tares[0]["mean_g"]
    report["tare"] = {
        "windows": tares,
        "drift_g": drift,
        "drift_of_hover": drift / hover_g,
        "noise_std_g": float(np.median([w["std_g"] for w in tares])),
    }
    if abs(drift) > TARE_DRIFT_TOL * hover_g:
        logger.warning(f"Tare drifted {drift:+.3f} g over the block, more than 1 % of hover")

    rates = {"force": force_rate}
    windows = [e for e in ran if e["kind"] in WINDOW_KINDS]
    rows = []
    for event, direction in zip(windows, directions(windows), strict=True):
        t0, t1 = event["t_start"] - t_ref + discard, event["t_end"] - t_ref
        own = {k: event[k] for k in ("step", "kind", "level_id", "motor", "repeat")}
        own |= {f"m{i}": event[f"m{i}"] for i in range(1, N_MOTORS + 1)}
        stats = window_stats(signals, t0, t1, coef, rates) if t1 > t0 else None
        if stats is None:
            continue
        rows.append(
            own
            | {"direction": direction, "t_start": t0, "t_end": t1, "t_mid": (t0 + t1) / 2}
            | stats
        )
    if meta["block"] != "bench":
        report["correlation"] = check_rises(rows)
    report["pulses"] = pulse_checks(events, command_log, signals, t_ref)
    report["windows"] = {
        "n": len(rows),
        "ok": sum(r["ok"] for r in rows),
        "flags": sorted({f for r in rows for f in r["flags"].split("|") if f}),
    }

    write_steady(folder / "steady.csv", rows)
    np.savez(
        folder / "synced.npz",
        t_ref=np.array(t_ref),
        force_t=t_lc,
        force=force,
        tare_g=tare,
        rpm_t=signals["rpm"][0],
        rpm=signals["rpm"][1],
        cmd_t=command_log[0],
        cmd_m1=command_log[1],
    )
    (folder / "sync.json").write_text(json.dumps(report, indent=2, default=float))
    timeline(folder.name, signals, command_log, events, t_ref, A1_PLOTS_DIR)
    for pulse in report["pulses"]:
        logger.info(
            f"Pulse at {pulse['t_cmd']:.1f} s: rpm edge {pulse['rpm_edge_ms']:.1f} ms, force "
            f"edge {pulse['force_edge_ms']:.1f} ms after the command"
        )
    logger.info(
        f"{report['windows']['ok']} of {len(rows)} windows ok, tare drift {drift:+.3f} g, "
        f"correlation {report.get('correlation', float('nan')):.4f}"
    )
    return report


def write_steady(path: Path, rows: list[dict]):
    """Write steady.csv, one row per window."""
    if not rows:
        raise ValueError("No steady window in the block")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_steady(path: Path) -> list[dict]:
    """steady.csv with its numbers and flags converted back."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out = {}
            for key, value in row.items():
                if key in ("kind", "direction", "flags"):
                    out[key] = value
                elif key == "ok":
                    out[key] = value == "True"
                elif key in ("step", "level_id", "motor", "repeat", "n", "n_rpm", "n_gaps"):
                    out[key] = int(value)
                elif key.startswith("m") and key[1:].isdigit():
                    out[key] = int(value)
                else:
                    out[key] = float(value)
            rows.append(out)
    return rows


def style(ax: plt.Axes):
    """The recessive grid and axes of the load cell quicklook."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def timeline(
    name: str,
    signals: dict,
    command_log: tuple[NDArray, NDArray],
    events: list[dict],
    t_ref: float,
    directory: Path,
):
    """The raw timeline of a block: command, the four rpm and the thrust, tares and windows."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, facecolor=SURFACE)
    cmd_ax, rpm_ax, force_ax = axes
    for ax in axes:
        style(ax)
        for e in events:
            if e["skipped"]:
                continue
            t0, t1 = e["t_start"] - t_ref, e["t_end"] - t_ref
            if e["kind"] == "tare":
                ax.axvspan(t0, t1, color=BAND, linewidth=0, zorder=0)
            elif e["kind"] in WINDOW_KINDS:
                ax.axvspan(t0 + 0.0, t1, color=BAND_2, linewidth=0, zorder=0, alpha=0.5)
    t_cmd, m1 = command_log
    cmd_ax.step(t_cmd, m1, where="post", color=SERIES[0], lw=1.5)
    cmd_ax.set_ylabel("motor.m1 command")
    cmd_ax.set_title(f"{name}: command, rpm and thrust", loc="left", color=INK)
    t_rpm, rpm, _ = signals["rpm"]
    for i in range(N_MOTORS):
        rpm_ax.plot(t_rpm, rpm[:, i], color=SERIES[i], lw=1, label=f"M{i + 1}")
    rpm_ax.set_ylabel("Rotor speed [rpm]")
    rpm_ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, ncols=4, loc="upper left")
    t_force, force, _ = signals["force"]
    force_ax.plot(t_force, force * 1e3, color=SERIES[0], lw=1)
    force_ax.set_ylabel("Thrust on the load cell [mN], up +")
    force_ax.set_xlabel("Time [s] (grey: tares, blue: steady windows)")
    fig.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}_timeline.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Convert a block of A1")
    parser.add_argument("folder", type=Path, help="results/downwash/a1/<block>_<timestamp>")
    parser.add_argument(
        "--calibration", type=Path, help="calibscale.py file whose a and b replace the run's"
    )
    parser.add_argument(
        "--collect", type=Path, help="Mounted card or folder of card files to copy in first"
    )
    args = parser.parse_args()
    if args.collect is not None:
        usd.collect(args.collect, recordings=A1_RESULTS_DIR)
    convert(args.folder, args.calibration)
