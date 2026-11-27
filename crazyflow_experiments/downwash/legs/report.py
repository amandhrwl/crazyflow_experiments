"""Checks and figures of an exported dataset for its notes (export Steps 1b.5, 4.5, 8, 9).

- Step 9: the cell table of `notebooks/a3_cell_runs.ipynb` (cell 14) recomputed from the exported
  legs, against the same table computed from the viewer files the notebook read
  (`results/downwash/a3_evening22_states/`); then the table with the new `fz_wake` and the tail.
  Only with `--states`: the viewer files exist for the 22 Nov evening session alone, so a dataset
  of other runs skips the step unless matching viewer files are made for them.
- The old against the new wake label per S1 cell: peak and centroid.
- Step 1b.5: Welch spectra of the load cell's new samples in `window_pre`, S0 and S1.
- Step 4.5: the source's rotor thrust against its mocap thrust over the cruise.
- Step 8: one quicklook per cell, and the old against the new label of a slow and a fast S1 leg.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/legs/report.py \
        ../crazyflow_experiments_data/legs/legs_v1 \
        [--states ../crazyflow_experiments_data/results/downwash/a3_evening22_states] --out report.md
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from crazyflow_experiments.downwash.legs import GRID_FREQ, G
from crazyflow_experiments.downwash.legs.source_sd import mocap_thrust

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

DT = 1 / GRID_FREQ
PEAK_SOS = butter(4, 10.0, fs=GRID_FREQ, output="sos")  # As the notebook and a3_stand_convert
NOTEBOOK_WINDOW = (-0.25, 1.0)  # s, the notebook's peak window
CRUISE_HALF_M = 0.6  # m of cruise either side of the crossing (`[a3] cruise_half`)
PEAK_DECIMALS = 1
DELAY_DECIMALS = 3
RESONANCE_NEIGHBOURHOOD_HZ = 5.0
RESONANCE_RATIO = 2.0  # Power density over the neighbourhood's median that counts as a peak


def notebook_peak(t: NDArray, y: NDArray, window: tuple = NOTEBOOK_WINDOW) -> tuple[float, float]:
    """The notebook's peak: minimum of y low-passed at 10 Hz (NaN bridged) in the window."""
    ok = np.isfinite(y)
    i = np.arange(len(y))
    s = sosfiltfilt(PEAK_SOS, np.interp(i, i[ok], y[ok]))
    s[~ok | (t < window[0]) | (t > window[1])] = np.nan
    k = int(np.nanargmin(s))
    return float(s[k]), float(t[k])


def centroid(t: NDArray, y: NDArray, window: tuple) -> float:
    """Centroid of the 10 Hz low-passed signal over the window."""
    ok = np.isfinite(y)
    i = np.arange(len(y))
    s = sosfiltfilt(PEAK_SOS, np.interp(i, i[ok], y[ok]))
    w = (t >= window[0]) & (t <= window[1])
    return float(np.sum(t[w] * s[w]) / np.sum(s[w]))


def load_legs(folder: Path) -> list[dict]:
    """Manifest rows with their meta and arrays."""
    with open(folder / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["meta"] = json.loads((folder / row["path"] / "meta.json").read_text())
        row["states"] = dict(np.load(folder / row["path"] / "states.npz"))
    return rows


def old_wake(states: Mapping, meta: Mapping) -> NDArray:
    """The existing `dforce_wake`: the rpm thrust change subtracted without the chain."""
    t, thrust = states["t"], states["thrust_sum"]
    w0, w1 = meta["derived"]["window_pre"]
    pre = (t >= w0) & (t < w1) & np.isfinite(thrust)
    return states["fz_total"] - (thrust - thrust[pre].mean())


def notebook_set(states_dir: Path) -> list[tuple[str, str, float, float, Path]]:
    """(leg_id, mode, dz, v, viewer folder) of every leg the notebook read."""
    out = []
    for folder in sorted(states_dir.glob("dz*/v*/*")):
        run, leg = folder.name.rsplit("_leg", 1)
        mode = "spin" if "_spin_" in run else "still"
        condition = "S1" if mode == "spin" else "S0"
        dz = float(folder.parent.parent.name.removeprefix("dz"))
        v = float(folder.parent.name.removeprefix("v"))
        out.append((f"A3_{condition}_{run}_leg{int(leg):02d}", mode, dz, v, folder))
    return out


def fmt(values: Sequence[float], decimals: int, scale: float = 1.0) -> str:
    """Format mean ± sample std as the notebook prints it."""
    x = np.asarray(values, dtype=float) * scale
    return f"{x.mean():.{decimals}f} ± {x.std(ddof=1):.{decimals}f}"


def step9(legs: Sequence[dict], states_dir: Path) -> tuple[str, dict]:
    """The regression table: notebook viewer files against the exported legs, and the new label."""
    by_id = {row["leg_id"]: row for row in legs}
    per_leg, missing = [], []
    for lid, mode, dz, v, folder in notebook_set(states_dir):
        if lid not in by_id:
            missing.append(lid)
            continue
        ref = np.load(folder / "states.npz")
        t_ref, y_ref = ref["t"], ref["force_wake"][:, 0, 0]
        p_ref, d_ref = notebook_peak(t_ref, y_ref)
        row = by_id[lid]
        st, meta = row["states"], row["meta"]
        y_old = old_wake(st, meta) if mode == "spin" else st["fz_total"]
        p_old, d_old = notebook_peak(st["t"], y_old)
        y_new = st["fz_wake"] if mode == "spin" else st["fz_total"]
        p_new, d_new = notebook_peak(st["t"], y_new)
        per_leg.append(
            {
                "leg_id": lid,
                "mode": mode,
                "dz": dz,
                "v": v,
                "ref_peak": p_ref,
                "ref_delay": d_ref,
                "old_peak": p_old,
                "old_delay": d_old,
                "new_peak": p_new,
                "new_delay": d_new,
                "tail": meta["quality"]["tail_level_N"],
            }
        )
    cells = sorted({(r["dz"], r["v"]) for r in per_leg})
    lines = [
        "| Δz | v | spin peak (mN) notebook | export | spin delay (s) notebook | export "
        "| still peak (mN) notebook | export | still delay (s) notebook | export |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    mismatches = []
    for dz, v in cells:
        cols = []
        for mode in ("spin", "still"):
            sel = [r for r in per_leg if (r["dz"], r["v"], r["mode"]) == (dz, v, mode)]
            for key, dec, scale in (("peak", PEAK_DECIMALS, 1e3), ("delay", DELAY_DECIMALS, 1.0)):
                ref = fmt([r[f"ref_{key}"] for r in sel], dec, scale)
                new = fmt([r[f"old_{key}"] for r in sel], dec, scale)
                cols += [ref, new]
                if ref != new:
                    mismatches.append(f"{mode} Δz {dz} v {v} {key}: {ref} vs {new}")
        lines.append(f"| {dz} | {v} | " + " | ".join(cols) + " |")
    table_new = [
        "| Δz | v | n spin | spin peak old (mN) | spin peak new (mN) | spin delay old (s) "
        "| spin delay new (s) | tail spin (mN) | n still | still peak (mN) | tail still (mN) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for dz, v in cells:
        sp = [r for r in per_leg if (r["dz"], r["v"], r["mode"]) == (dz, v, "spin")]
        sl = [r for r in per_leg if (r["dz"], r["v"], r["mode"]) == (dz, v, "still")]
        tail_sp = np.array([r["tail"] for r in sp], dtype=float) * 1e3
        tail_sl = np.array([r["tail"] for r in sl], dtype=float) * 1e3
        table_new.append(
            f"| {dz} | {v} | {len(sp)} | {fmt([r['old_peak'] for r in sp], 1, 1e3)} "
            f"| {fmt([r['new_peak'] for r in sp], 1, 1e3)} | {fmt([r['old_delay'] for r in sp], 3)} "
            f"| {fmt([r['new_delay'] for r in sp], 3)} | {tail_sp.mean():+.1f} ± {tail_sp.std(ddof=1):.1f} "
            f"| {len(sl)} | {fmt([r['new_peak'] for r in sl], 1, 1e3)} "
            f"| {tail_sl.mean():+.1f} ± {tail_sl.std(ddof=1):.1f} |"
        )
    worst = {
        "peak_mN": max(abs(r["old_peak"] - r["ref_peak"]) for r in per_leg) * 1e3,
        "delay_ms": max(abs(r["old_delay"] - r["ref_delay"]) for r in per_leg) * 1e3,
    }
    text = (
        f"Legs: {len(per_leg)} of the notebook's set found in the export"
        + (f", missing: {missing}" if missing else "")
        + f". Largest per-leg difference against the notebook's viewer files: "
        f"{worst['peak_mN']:.2f} mN, {worst['delay_ms']:.1f} ms.\n\n"
        + "\n".join(lines)
        + "\n\n"
        + (
            "Mismatches at the table's rounding: " + "; ".join(mismatches)
            if mismatches
            else "All cells match at the table's rounding."
        )
        + "\n\nWith the new `fz_wake` (chain: 27 ms delay), and the tail level (mean label over the "
        "last 0.5 s of each leg):\n\n" + "\n".join(table_new) + "\n"
    )
    return text, {"per_leg": per_leg, "mismatches": mismatches, "worst": worst, "missing": missing}


def old_vs_new(legs: Sequence[dict]) -> str:
    """Peak and centroid of the old and the new wake label per S1 cell, usable legs."""
    rows = [r for r in legs if r["condition"] == "S1" and r["meta"]["quality"]["usable"]]
    cells = sorted({(float(r["dz_nominal"]), float(r["v_nominal"])) for r in rows})
    lines = [
        "| Δz | v | n | peak old (mN) | peak new (mN) | Δpeak (%) | centroid new − old (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for dz, v in cells:
        sel = [r for r in rows if (float(r["dz_nominal"]), float(r["v_nominal"])) == (dz, v)]
        p_old, p_new, d_c = [], [], []
        for r in sel:
            st, meta = r["states"], r["meta"]
            window = tuple(meta["quality"]["peak_window"])
            y_old, y_new = old_wake(st, meta), st["fz_wake"]
            p_old.append(notebook_peak(st["t"], y_old, window)[0])
            p_new.append(notebook_peak(st["t"], y_new, window)[0])
            d_c.append(centroid(st["t"], y_new, window) - centroid(st["t"], y_old, window))
        p_old, p_new = np.array(p_old) * 1e3, np.array(p_new) * 1e3
        rel = (p_new - p_old) / np.abs(p_old) * 100
        lines.append(
            f"| {dz} | {v} | {len(sel)} | {p_old.mean():.1f} | {p_new.mean():.1f} "
            f"| {rel.mean():+.1f} (max {np.abs(rel).max():.1f}) | {np.mean(d_c) * 1e3:+.1f} |"
        )
    return "\n".join(lines) + "\n"


def resonance(legs: Sequence[dict]) -> tuple[str, Path | None, dict]:
    """Welch spectra of the new load cell samples in window_pre, averaged over the usable legs."""
    spectra = {}
    for condition in ("S0", "S1"):
        psds, rates = [], []
        for r in legs:
            if r["condition"] != condition or not r["meta"]["quality"]["usable"]:
                continue
            st, meta = r["states"], r["meta"]
            w0, w1 = meta["derived"]["window_pre"]
            sel = st["fz_new"] & (st["t"] >= w0) & (st["t"] < w1)
            label = st["fz_wake"] if condition == "S1" else st["fz_total"]
            x, tk = label[sel], st["t"][sel]
            if len(x) < 64:
                continue
            fs = (len(tk) - 1) / (tk[-1] - tk[0])
            f, p = welch(x - x.mean(), fs=fs, nperseg=64)
            psds.append(np.interp(np.linspace(0, 39.5, 80), f, p))
            rates.append(fs)
        if psds:
            spectra[condition] = (
                np.linspace(0, 39.5, 80),
                np.mean(psds, axis=0),
                len(psds),
                np.mean(rates),
            )
    lines = []
    for condition, (f, p, n, fs) in spectra.items():
        # A peak stands out of its neighbourhood: twice the median of the density within 5 Hz
        k, _ = find_peaks(p)
        peaks = []
        for i in k:
            near = np.abs(f - f[i]) <= RESONANCE_NEIGHBOURHOOD_HZ
            ratio = p[i] / np.median(p[near])
            if 0.5 < f[i] < 40 and ratio > RESONANCE_RATIO:
                peaks.append(f"{f[i]:.1f} Hz (×{ratio:.1f} the median within 5 Hz)")
        lines.append(
            f"- {condition}: {n} legs, new samples at {fs:.1f} Hz; peaks: "
            + (", ".join(peaks) if peaks else "none (no density over twice its neighbourhood)")
        )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for condition, (f, p, n, _) in spectra.items():
        ax.semilogy(f, np.sqrt(p) * 1e3, label=f"{condition} (n={n})")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude density (mN/√Hz)")
    ax.set_title(
        "Load cell in window_pre, new samples only (80 Hz: meaningful below 40 Hz)", fontsize=9
    )
    ax.legend()
    fig.tight_layout()
    return "\n".join(lines) + "\n", fig, spectra


def source_thrust(legs: Sequence[dict]) -> str:
    """The source's rotor thrust against its mocap thrust over the cruise, legs with its card."""
    ratios = []
    for r in legs:
        meta = r["meta"]
        if "source_thrust_from_mocap" in meta["quality"]["flags"]:
            continue
        st = r["states"]
        t, pos = st["t"], st["pos"][:, 1]
        mass = meta["sim_params"]["drone/mass"][1]
        mocap = mocap_thrust(t, pos, t, mass, G)
        v = float(r["v_nominal"])
        cruise = np.abs(t) <= CRUISE_HALF_M / v
        rpm_thrust = st["motor_thrust"][:, 1].sum(axis=1)
        ok = cruise & np.isfinite(rpm_thrust) & np.isfinite(mocap)
        if ok.sum() < 50:
            continue
        ratios.append((r["leg_id"], v, float(np.mean(rpm_thrust[ok]) / np.mean(mocap[ok]))))
    x = np.array([q[2] for q in ratios])
    within = np.mean(np.abs(x - 1) <= 0.05)
    by_v = {
        v: np.array([q[2] for q in ratios if q[1] == v]) for v in sorted({q[1] for q in ratios})
    }
    lines = [
        f"rpm thrust / mocap thrust over the cruise (|x| ≤ 0.6 m), {len(x)} legs with the source card: "
        f"median {np.median(x):.3f}, range {x.min():.3f} to {x.max():.3f}, "
        f"{within:.0%} within 5 %.",
        "",
        "| v (m/s) | n | median | min | max |",
        "|---|---|---|---|---|",
    ]
    for v, values in by_v.items():
        lines.append(
            f"| {v} | {len(values)} | {np.median(values):.3f} | {values.min():.3f} | {values.max():.3f} |"
        )
    return "\n".join(lines) + "\n"


def cell_quicklooks(legs: Sequence[dict], out: Path) -> list[Path]:
    """One figure per cell: S0 and S1 labels of all attempts, window_pre and the peak marked."""
    paths = []
    cells = sorted({(float(r["dz_nominal"]), float(r["v_nominal"])) for r in legs})
    for dz, v in cells:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
        for ax, condition in zip(axes, ("S0", "S1")):
            sel = [
                r
                for r in legs
                if r["condition"] == condition
                and (float(r["dz_nominal"]), float(r["v_nominal"])) == (dz, v)
            ]
            for i, r in enumerate(sel):
                st, q = r["states"], r["meta"]["quality"]
                label = st["fz_wake"] if condition == "S1" else st["fz_total"]
                usable = q["usable"]
                color = f"C{i}" if usable else "0.6"
                name = r["leg_id"].split("_", 2)[2]
                ax.plot(
                    st["t"],
                    label * 1e3,
                    lw=0.6,
                    color=color,
                    alpha=0.85 if usable else 0.6,
                    ls="-" if usable else "--",
                    label=name + ("" if usable else " (not usable)"),
                )
                if q["peak_time_s"] is not None:
                    ax.plot(q["peak_time_s"], q["peak_fz_N"] * 1e3, "o", color=color, ms=5)
                w0, w1 = r["meta"]["derived"]["window_pre"]
                ax.axvspan(w0, w1, color=color, alpha=0.05)
            ax.axvline(0, color="k", lw=0.7)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylabel(f"{condition} {'fz_wake' if condition == 'S1' else 'fz_total'} (mN)")
            ax.legend(fontsize=6, frameon=False, loc="lower right")
            ax.set_xlim(-2.5, 4.0)
        axes[0].set_title(
            f"Δz {dz} m, v {v} m/s: labels (unfiltered), window_pre shaded, peak (10 Hz) marked",
            fontsize=9,
        )
        axes[-1].set_xlabel("time after the crossing (s)")
        fig.tight_layout()
        path = out / f"quicklook_dz{dz:.2f}_v{v:.2f}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def old_new_figure(legs: Sequence[dict], leg_ids: Sequence[str], out: Path) -> Path:
    """The old and the new wake label of the given S1 legs, raw and 10 Hz."""
    rows = [r for r in legs if r["leg_id"] in leg_ids]
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 3.2 * len(rows)))
    for ax, r in zip(np.atleast_1d(axes), rows):
        st, meta = r["states"], r["meta"]
        t = st["t"]
        old, new = old_wake(st, meta), st["fz_wake"]
        for y, name, color in (
            (st["fz_total"], "fz_total", "0.6"),
            (old, "old (raw thrust)", "C1"),
            (new, "new (chain 27 ms)", "C0"),
        ):
            ok = np.isfinite(y)
            i = np.arange(len(y))
            ax.plot(t, y * 1e3, color=color, lw=0.4, alpha=0.5)
            ax.plot(
                t,
                sosfiltfilt(PEAK_SOS, np.interp(i, i[ok], y[ok])) * 1e3,
                color=color,
                lw=1.5,
                label=name,
            )
        ax.set_xlim(-1.0, 2.0)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_ylabel("mN")
        ax.set_title(f"{r['leg_id']} (Δz {r['dz_nominal']}, v {r['v_nominal']})", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    np.atleast_1d(axes)[-1].set_xlabel("time after the crossing (s)")
    fig.tight_layout()
    path = out / "quicklook_old_vs_new_fz_wake.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def flag_table(legs: Sequence[dict]) -> str:
    """Flag and hard-fail counts per cell, and the hard-failed legs with their reasons."""
    cells = sorted({(r["condition"], float(r["dz_nominal"]), float(r["v_nominal"])) for r in legs})
    flags = sorted({f for r in legs for f in r["meta"]["quality"]["flags"]})
    lines = [
        "| cond | Δz | v | legs | usable | hard_fail | " + " | ".join(flags) + " |",
        "|---|---|---|---|---|---|" + "---|" * len(flags),
    ]
    for c, dz, v in cells:
        sel = [
            r
            for r in legs
            if (r["condition"], float(r["dz_nominal"]), float(r["v_nominal"])) == (c, dz, v)
        ]
        counts = [sum(f in r["meta"]["quality"]["flags"] for r in sel) for f in flags]
        lines.append(
            f"| {c} | {dz} | {v} | {len(sel)} | {sum(r['meta']['quality']['usable'] for r in sel)} "
            f"| {sum(r['meta']['quality']['hard_fail'] for r in sel)} | "
            + " | ".join(str(x) for x in counts)
            + " |"
        )
    failed = [
        f"- `{r['leg_id']}`: {r['meta']['quality']['hard_fail_reason']}"
        for r in legs
        if r["meta"]["quality"]["hard_fail"]
    ]
    return "\n".join(lines) + "\n\nHard-failed legs:\n\n" + "\n".join(failed) + "\n"


def report(
    folder: Path, states_dir: Path | None, slow_fast: Sequence[str] | None = None
) -> tuple[str, dict]:
    """Every section as markdown, figures written into the dataset folder.

    Without `states_dir` the Step 9 regression against the viewer files is skipped.
    """
    legs = load_legs(folder)
    if states_dir is None:
        text9, result9 = (
            "Skipped: no viewer files (`--states`) for these runs.\n",
            {"mismatches": []},
        )
    else:
        text9, result9 = step9(legs, states_dir)
    spectra_text, fig, _ = resonance(legs)
    fig.savefig(folder / "quicklook_resonance.png", dpi=120)
    plt.close(fig)
    cell_quicklooks(legs, folder)
    if slow_fast is None:
        usable_s1 = [r for r in legs if r["condition"] == "S1" and r["meta"]["quality"]["usable"]]
        slow = next(r["leg_id"] for r in usable_s1 if float(r["v_nominal"]) == 0.5)
        fast = next(r["leg_id"] for r in usable_s1 if float(r["v_nominal"]) == 2.0)
        slow_fast = (slow, fast)
    old_new_figure(legs, slow_fast, folder)
    sections = {
        "step9": text9,
        "old_vs_new": old_vs_new(legs),
        "resonance": spectra_text,
        "source_thrust": source_thrust(legs),
        "flags": flag_table(legs),
        "old_new_legs": ", ".join(f"`{x}`" for x in slow_fast),
    }
    text = "\n".join(f"## {k}\n\n{v}" for k, v in sections.items())
    return text, {"step9": result9, "sections": sections}


def main():
    """Command line: write the report's markdown and the figures."""
    parser = argparse.ArgumentParser(description="Report on an exported LEGS_FORMAT dataset.")
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--states", type=Path, help="Viewer files of the runs (a3_evening22_states); else no Step 9"
    )
    parser.add_argument("--out", type=Path, required=True, help="Markdown output")
    args = parser.parse_args()
    text, result = report(args.folder, args.states)
    args.out.write_text(text)
    if result["step9"]["mismatches"]:
        print("Step 9 mismatches:", *result["step9"]["mismatches"], sep="\n  ")  # noqa: T201
    print(f"report -> {args.out}")  # noqa: T201


if __name__ == "__main__":
    main()
