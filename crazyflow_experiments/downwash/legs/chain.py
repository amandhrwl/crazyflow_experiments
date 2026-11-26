"""The stand's measurement chain `C[x] = P[G * x]` of LEGS_FORMAT §6.3 on the 500 Hz grid.

`G` is the stand's physical response from `rig.stand.dynamics`, `P` the processing the code applied
afterwards from `rig.stand.processing`. Both are linear and time-invariant:

- `second_order`: `scipy.signal.cont2discrete` (bilinear) of `w_n^2 / (s^2 + 2 zeta w_n s + w_n^2)`
  and `lfilter` from rest, then the delay `tau_s`;
- `delay`: `G = exp(-s tau_eff)`, a pure delay (`gain_ratio` is recorded, not applied);
- `unknown`: `G = 1`;
- processing: a shift by `-shift_s`, then a Butterworth low-pass (`filtfilt` if zero phase) or
  `filter = "none"`.

Delays and shifts interpolate linearly on the grid and hold the first (last) sample beyond the
start (end). NaN in the input stays local: a delay spreads it to the two neighbours it interpolates
from, the recursive filters to the samples their impulse response reaches above `NAN_REACH_TOL` of its area (the error of
treating the NaN as 0 beyond that is below that share of the sample).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import butter, cont2discrete, filtfilt, lfilter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike, NDArray

GRID_FREQ = 500.0  # Hz, the grid of LEGS_FORMAT §4
DYNAMICS_KINDS = ("second_order", "delay", "unknown")
NAN_REACH_TOL = 1e-6  # Share of an impulse response's area below which its tail no longer counts
CHAIN_TEST_SPAN = (-1.0, 3.0)  # s, grid of chain_test.npz
CHAIN_TEST_RECT_S = 0.3  # s, width of the rectangular test pulse


def shift(x: ArrayLike, delay: float, fs: float = GRID_FREQ) -> NDArray:
    """`y(t) = x(t - delay)` by linear interpolation on the grid.

    Values before the start are held at the first sample, values after the end at the last.

    Args:
        x: Signal on the grid (N,).
        delay: Delay in s, positive moves the signal later.
        fs: Grid rate in Hz.
    """
    x = np.asarray(x, dtype=float)
    position = np.clip(np.arange(len(x)) - delay * fs, 0, len(x) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(x) - 1)
    w = position - lower
    # A neighbour of weight 0 must not turn the sample NaN, so the upper term counts where w > 0 only
    return x[lower] * (1 - w) + np.where(w > 0, x[upper] * w, 0.0)


def _reach(impulse_response: NDArray) -> int:
    """Samples after which an impulse response's tail holds less than `NAN_REACH_TOL` of it."""
    magnitude = np.abs(impulse_response)
    tail = np.cumsum(magnitude[::-1])[::-1]
    return int(np.count_nonzero(tail > NAN_REACH_TOL * magnitude.sum()))


def _nan_mask(bad: NDArray, before: int, after: int) -> NDArray:
    """`bad` dilated by `before` samples to the left and `after` to the right."""
    out = bad.copy()
    for k in np.flatnonzero(bad):
        out[max(k - before, 0) : k + after + 1] = True
    return out


def second_order(x: ArrayLike, f_n_hz: float, zeta: float, fs: float = GRID_FREQ) -> NDArray:
    """Unit-gain second-order low-pass, bilinear discretization, `lfilter` from rest.

    NaN samples are treated as 0 for the filter and mark the samples the response reaches as NaN.
    """
    x = np.asarray(x, dtype=float)
    w_n = 2 * math.pi * f_n_hz
    b, a, _ = cont2discrete(([w_n**2], [1.0, 2 * zeta * w_n, w_n**2]), 1 / fs, method="bilinear")
    b = np.asarray(b, dtype=float).ravel()
    bad = ~np.isfinite(x)
    y = lfilter(b, a, np.where(bad, 0.0, x))
    if bad.any():
        impulse = lfilter(b, a, np.eye(1, max(len(x), 1))[0])
        y[_nan_mask(bad, 0, _reach(impulse))] = np.nan
    return y


def lowpass(
    x: ArrayLike, order: int, fc_hz: float, zero_phase: bool, fs: float = GRID_FREQ
) -> NDArray:
    """Butterworth low-pass, `filtfilt` if zero phase else `lfilter` from rest, NaN kept local."""
    x = np.asarray(x, dtype=float)
    b, a = butter(order, fc_hz, fs=fs)
    bad = ~np.isfinite(x)
    clean = np.where(bad, 0.0, x)
    y = filtfilt(b, a, clean) if zero_phase else lfilter(b, a, clean)
    if bad.any():
        reach = _reach(lfilter(b, a, np.eye(1, max(len(x), 1))[0]))
        y[_nan_mask(bad, reach if zero_phase else 0, reach)] = np.nan
    return y


def apply_dynamics(x: ArrayLike, dynamics: Mapping, fs: float = GRID_FREQ) -> NDArray:
    """`G * x` for one `rig.stand.dynamics` entry."""
    kind = dynamics["kind"]
    if kind == "second_order":
        return shift(
            second_order(x, dynamics["f_n_hz"], dynamics["zeta"], fs), dynamics["tau_s"], fs
        )
    if kind == "delay":
        return shift(x, dynamics["tau_eff_s"], fs)
    if kind == "unknown":
        return np.asarray(x, dtype=float).copy()
    raise ValueError(f"dynamics.kind {kind!r} is not one of {DYNAMICS_KINDS}")


def apply_processing(x: ArrayLike, processing: Mapping, fs: float = GRID_FREQ) -> NDArray:
    """`P[x]` for one `rig.stand.processing` entry: the shift by -shift_s, then the filter."""
    y = shift(x, -float(processing.get("shift_s", 0.0)), fs)
    match processing.get("filter", "none"):
        case "none":
            return y
        case "butter":
            return lowpass(
                y, processing["order"], processing["fc_hz"], processing["zero_phase"], fs
            )
        case other:
            raise ValueError(f"processing.filter {other!r} is not 'none' or 'butter'")


def chain(x: ArrayLike, stand: Mapping, fs: float = GRID_FREQ) -> NDArray:
    """`C[x] = P[G * x]` for a `rig.stand` entry of LEGS_FORMAT §5.5."""
    return apply_processing(apply_dynamics(x, stand["dynamics"], fs), stand["processing"], fs)


def centroid_shift(stand: Mapping) -> float:
    """The centroid shift in s the chain should produce, for the tests and the notes.

    `second_order` moves the centroid by `tau + 2 zeta / w_n`, `delay` by `tau_eff`, `unknown` by
    0, and the processing's shift by `-shift_s`; a zero-phase filter adds nothing.
    """
    dynamics, processing = stand["dynamics"], stand["processing"]
    match dynamics["kind"]:
        case "second_order":
            base = dynamics["tau_s"] + 2 * dynamics["zeta"] / (2 * math.pi * dynamics["f_n_hz"])
        case "delay":
            base = dynamics["tau_eff_s"]
        case "unknown":
            base = 0.0
        case other:
            raise ValueError(f"dynamics.kind {other!r} is not one of {DYNAMICS_KINDS}")
    return base - float(processing.get("shift_s", 0.0))


def stand_hash(stand: Mapping) -> str:
    """Key of a `rig.stand` in chain_test.npz: sha256 of its sorted JSON, 16 hex digits."""
    text = json.dumps(stand, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def chain_test(stands: Mapping[str, Mapping], fs: float = GRID_FREQ) -> dict[str, NDArray]:
    """The arrays of chain_test.npz: a unit impulse and a 0.3 s pulse at t = 0, through each chain.

    The impulse has unit area (1/dt on the sample at t = 0), the pulse height 1 on [0, 0.3) s.

    Args:
        stands: `rig.stand` entries keyed by `stand_hash`.
        fs: Grid rate in Hz.

    Returns:
        `t`, `impulse`, `rect` and per stand `<hash>/impulse`, `<hash>/rect`, `<hash>/stand` (its
        JSON).
    """
    k = np.arange(round(CHAIN_TEST_SPAN[0] * fs), round(CHAIN_TEST_SPAN[1] * fs) + 1)
    t = k / fs
    impulse = np.where(k == 0, fs, 0.0)
    rect = ((k >= 0) & (k < round(CHAIN_TEST_RECT_S * fs))).astype(float)
    out = {"t": t, "impulse": impulse, "rect": rect}
    for key, stand in stands.items():
        out[f"{key}/impulse"] = chain(impulse, stand, fs)
        out[f"{key}/rect"] = chain(rect, stand, fs)
        out[f"{key}/stand"] = np.array(json.dumps(stand, sort_keys=True, default=float))
    return out
