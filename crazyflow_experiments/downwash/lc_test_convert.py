"""Put the SD logs, the radio logs and the motion capture of a load cell test onto the PC clock.

Copy the card files into the recording first, `usd.py collect <card> --block lc` puts them there:

    results/load_cell/test_<timestamp>/sd/victim/vic_<timestamp>_SD
    results/load_cell/test_<timestamp>/sd/source/src_<timestamp>_SD

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/lc_test_convert.py \
        results/load_cell/test_<timestamp> [--calibration <calib_run.toml>]

Writes `synced.npz` and `sync.json` into the recording, every time in s from the start of the
still air before the flight, and a quicklook to `plots/load_cell/`. See `sync.py` for the clocks.
The force uses the load cell calibration the test flew with, from `meta.json`. `--calibration`
takes `a` and `b` from a calibration file of `calibscale.py` instead, for a test flown with an
outdated one; `sync.json` notes which calibration the force comes from.
"""

from __future__ import annotations

import argparse
import json
import logging
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crazyflow_experiments.downwash import LC_PLOTS_DIR, usd
from crazyflow_experiments.downwash.sync import ACC_SYNC, LOAD_CELL_SYNC, sync_drone, unwrap_ticks
from crazyflow_experiments.downwash.usd import sd_file

logger = logging.getLogger(__name__)

G = 9.81
# Per drone, its 100 Hz radio block, the variable it also logs to its card and how often that
# changes in step with the card's sampling, zero for a variable that changes at any time, see
# `sync.py` for the choice
SYNC_VARIABLES = {"victim": LOAD_CELL_SYNC, "source": ACC_SYNC}
# Light chart tokens, as in the load cell calibration plot
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#efeee9"
SERIES = ("#2a78d6", "#eb6834")  # Card, radio
LABELED_PHASES = ("still_before", "spin_alone", "aside_1", "over", "aside_2", "still_after")


def phase_forces(t: np.ndarray, force: np.ndarray, phases: list) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of the force in every phase, in mN."""
    out = {}
    for name, t0, t1 in phases:
        inside = (t >= t0) & (t < t1)
        if inside.any():
            out[name] = {"mean_mN": force[inside].mean() * 1e3, "std_mN": force[inside].std() * 1e3}
    return out


def quicklook(data: dict[str, np.ndarray], phases: list, title: str, path: Path):
    """Force on the victim, height of the source and a close look at the load cell sync."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), facecolor=SURFACE)
    force_ax, height_ax, zoom_ax = axes
    height_ax.sharex(force_ax)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(INK_2)
        ax.yaxis.label.set_color(INK_2)
    for ax in (force_ax, height_ax):
        for name, t0, t1 in phases:
            if name == "over":
                ax.axvspan(t0, t1, color=BAND, linewidth=0, zorder=0)
    for name, t0, t1 in phases:
        if name in LABELED_PHASES:
            force_ax.annotate(
                name.replace("_", " "),
                ((t0 + t1) / 2, 1.0),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=8,
                color=MUTED,
            )

    force_ax.plot(data["victim_sd/t"], data["victim_sd/force_z"] * 1e3, color=SERIES[0], lw=1)
    force_ax.set_ylabel("Force on the victim [mN], up +")
    force_ax.set_title(f"{title}: load cell from the victim's card", loc="left", color=INK, pad=16)

    if "mocap/t" in data:
        height_ax.plot(
            data["mocap/t"], data["mocap/height"], color=SERIES[0], lw=2, label="motion capture"
        )
    height_ax.plot(
        data["source_sd/t"], data["source_sd/height"], color=SERIES[1], lw=1, label="its card"
    )
    height_ax.set_ylabel("Source above the victim [m]")
    height_ax.set_xlabel("Time [s]")
    height_ax.legend(frameon=False, labelcolor=INK_2, fontsize=9)

    over = [(t0, t1) for name, t0, t1 in phases if name == "over"]
    t_mid = np.mean(over[0]) if over else np.median(data["victim_radio/t"])
    sd_in = np.abs(data["victim_sd/t"] - t_mid) < 0.15
    radio_in = np.abs(data["victim_radio/t"] - t_mid) < 0.15
    zoom_ax.step(
        data["victim_sd/t"][sd_in],
        data["victim_sd/force_z"][sd_in] * 1e3,
        where="post",
        color=SERIES[0],
        lw=2,
        label="card, 500 Hz",
    )
    zoom_ax.plot(
        data["victim_radio/t"][radio_in],
        data["victim_radio/force_z"][radio_in] * 1e3,
        "o",
        color=SERIES[1],
        ms=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label="radio, 100 Hz, at its onboard stamp",
    )
    zoom_ax.set_title("Synced: every radio sample sits on the card's step", loc="left", color=INK)
    zoom_ax.set_ylabel("Force [mN]")
    zoom_ax.set_xlabel("Time [s]")
    zoom_ax.legend(frameon=False, labelcolor=INK_2, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def convert(folder: Path, calibration: Path | None = None) -> dict:
    """Sync one recording, write `synced.npz`, `sync.json` and the quicklook.

    Args:
        folder: The recording, `results/load_cell/test_<timestamp>`.
        calibration: A calibration file of `calibscale.py` whose `a` and `b` replace the ones the
            test flew with.

    Returns:
        The contents of `sync.json`.
    """
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    radio = dict(np.load(folder / "radio.npz"))
    phases = meta["phases"]
    t_ref = phases[0][1]  # Start of the still air before the flight
    phases = [(name, t0 - t_ref, t1 - t_ref) for name, t0, t1 in phases]
    victim_pos = np.asarray(meta["victim_pos"])
    flown = meta["config"]["load_cell"]
    cal = flown if calibration is None else tomllib.loads(Path(calibration).read_text())
    a, b = cal["a"], cal["b"]

    data, report = {"t_ref": np.array(t_ref), "victim_pos": victim_pos}, {"t_ref": t_ref}
    for role in ("victim", "source"):
        sd = usd.decode(sd_file(folder, role, meta))
        sd_period = 1 / usd.role_config("lc", role).frequency
        own = {k.removeprefix(f"{role}/"): v for k, v in radio.items() if k.startswith(role)}
        clock, report[role] = sync_drone(own, role, sd, sd_period, SYNC_VARIABLES[role])
        data[f"{role}_sd/t"] = clock.pc_from_sd(sd["t_us"]) - t_ref
        data |= {f"{role}_sd/{k}": v for k, v in sd.items() if k != "t_us"}
        block = SYNC_VARIABLES[role][0]
        t_tick = unwrap_ticks(own[f"{block}/t_tick_ms"])
        data[f"{role}_radio/t"] = clock.pc_from_tick(t_tick) - t_ref  # When it was sampled
        data[f"{role}_radio/arrival"] = own[f"{block}/t_pc"] - t_ref
        for key, values in own.items():
            if key.startswith(f"{block}/") and key != f"{block}/t_pc":
                data[f"{role}_radio/{key.removeprefix(f'{block}/')}"] = values
        logger.info(f"{role}: {json.dumps(report[role])}")

    # The force from the raw reading in g of upward force, zeroed on the still air before
    weights = {r: a * data[f"{r}/loadcell.rawWeight"] + b for r in ("victim_sd", "victim_radio")}
    _, t0, t1 = phases[0]
    at_rest = (data["victim_sd/t"] >= t0) & (data["victim_sd/t"] < t1)
    tare = weights["victim_sd"][at_rest].mean()
    report["load_cell"] = {
        "calibration": {"a": a, "b": b, "from": str(calibration or "meta.json")},
        "zero_g": tare,
        "noise_std_g": weights["victim_sd"][at_rest].std(),
    }
    for record, weight in weights.items():
        data[f"{record}/force_z"] = (weight - tare) * 1e-3 * G
    report["load_cell"]["phases"] = phase_forces(
        data["victim_sd/t"], data["victim_sd/force_z"], phases
    )
    # A flight stopped on its first tick, right after the takeoff, left no records
    if (flight_path := folder / "flight.npz").is_file():
        flight = dict(np.load(flight_path))
        data["mocap/t"] = flight["t_pc"] - t_ref
        data["mocap/pos"], data["mocap/setpoint"] = flight["pos"], flight["setpoint"]
        data["mocap/height"] = flight["pos"][:, 2] - victim_pos[2]
    else:
        logger.warning(f"No flight records, the source stopped before its holds: {meta['abort']}")
    data["source_sd/height"] = data["source_sd/stateEstimate.z"] - victim_pos[2]

    np.savez(folder / "synced.npz", **data)
    (folder / "sync.json").write_text(json.dumps(report, indent=2))
    quicklook(data, phases, folder.name, LC_PLOTS_DIR / f"{folder.name}.png")
    for name, force in report["load_cell"]["phases"].items():
        logger.info(f"{name:>12}: {force['mean_mN']:+7.2f} mN, std {force['std_mN']:.2f} mN")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Sync the logs of a load cell test")
    parser.add_argument("folder", type=Path, help="results/load_cell/test_<timestamp>")
    parser.add_argument(
        "--calibration", type=Path, help="calibscale.py file whose a and b replace the flown ones"
    )
    args = parser.parse_args()
    convert(args.folder, args.calibration)
