"""Clocks of the load cell test: every onboard clock is mapped onto the PC clock.

Each drone has two clocks, the millisecond tick that stamps its radio log packets and the
microsecond timer that stamps its SD records. The PC stamps the arrival of every radio packet and
the motion capture with its wall clock (`time.time()`), which is the common time base.

The tick maps onto PC time through the arrival times of the radio packets (`fit_radio_clock`). A
packet arrives after it was stamped, late by a radio latency that varies but has a floor, so the
line under the arrival times maps the tick. Its slope absorbs the drift of the drone's crystal,
about 0.1 % on the cf21B.

The SD timer and the tick count the same crystal, so they differ by an offset, set during the
boot, and on older firmware by a fixed rate: before bitcraze/crazyflie-firmware#1684 (September
2026) the timer's prescaler divided by 85 instead of 84, so the card's microseconds run 1.19 %
long (`SD_RATES`). A variable that the drone logs both to the card and over the radio measures
them (`match_offset`): mapped by the right rate and offset, the card record shows the values the
radio did.
The variable has to change out of step with the radio log timer, or every radio sample sits at the
same point between two changes and any shift within that gap fits. The victim matches its load
cell reading, which the load cell's own oscillator clocks, the source its accelerometer, which
changes on every tick of the 1 kHz stabilizer loop that the IMU clocks. The source's height
estimate would not do: the Kalman filter predicts every 10 ms of the tick, like the radio logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

TICK_WRAP = 2**24  # ms, the radio log packets carry the tick in 3 bytes
PEAK_HALF_WIDTH = 4e-3  # s, shifts either side of the best that the peak fit uses
# Tick seconds per SD timer second: firmware with the fix of the timer's prescaler and before it
SD_RATES = (1.0, 85 / 84)


@dataclass(frozen=True)
class ClockMap:
    """PC time of a drone's onboard clocks.

    Attributes:
        t0_tick: Onboard tick in s at the reference point.
        t0_pc: PC time in s at the reference point.
        rate: PC seconds per onboard second.
        sd_offset: Tick minus the SD timer in s, after `sd_rate`.
        sd_rate: Tick seconds per SD timer second, see `SD_RATES`.
    """

    t0_tick: float
    t0_pc: float
    rate: float
    sd_offset: float = 0.0
    sd_rate: float = 1.0

    def pc_from_tick(self, t_tick: ArrayLike) -> NDArray[np.floating]:
        """PC time of onboard tick times in s."""
        return self.t0_pc + self.rate * (np.asarray(t_tick, dtype=float) - self.t0_tick)

    def pc_from_sd(self, t_us: ArrayLike) -> NDArray[np.floating]:
        """PC time of SD record timestamps in µs."""
        t_sd = np.asarray(t_us, dtype=float) / 1e6
        return self.pc_from_tick(self.sd_rate * t_sd + self.sd_offset)


def unwrap_ticks(t_ms: ArrayLike) -> NDArray[np.floating]:
    """Radio log timestamps in ms, in arrival order, as a continuous onboard time in s."""
    return np.unwrap(np.asarray(t_ms, dtype=float), period=TICK_WRAP) / 1000


def fit_radio_clock(
    t_tick: ArrayLike, t_pc: ArrayLike, window: float = 2.0, iterations: int = 5
) -> tuple[ClockMap, dict[str, float]]:
    """Map the onboard tick onto PC time from the arrival times of the radio packets.

    A least squares line through all packets starts the fit. Then, in every `window` of onboard
    time, the packet that arrived soonest relative to the current line is kept, and the line is
    fitted again through those, until it runs along the floor of the latency.

    Args:
        t_tick: Onboard tick of the packets in s, see `unwrap_ticks`.
        t_pc: PC arrival time of the packets in s.
        window: Onboard seconds per kept packet.
        iterations: Refits along the floor.

    Returns:
        The clock map, and the drift of the onboard clock in percent and the median and 95th
        percentile of the latency above the floor in s.
    """
    t_tick, t_pc = np.asarray(t_tick, dtype=float), np.asarray(t_pc, dtype=float)
    t0_tick, t0_pc = t_tick.min(), t_pc.min()  # The packets are in arrival order, not stamp order
    x, y = t_tick - t0_tick, t_pc - t0_pc  # Centered, the PC time alone is 1.7e9 s
    bins = (x // window).astype(int)
    # A part-filled window at either end holds too few packets to reach the floor, and a single
    # point off the floor at the end of the line tilts it by milliseconds over the flight
    _, index, counts = np.unique(bins, return_inverse=True, return_counts=True)
    full = counts[index] >= counts.max() / 2
    if len(np.unique(bins[full])) < 3:
        raise ValueError(f"{x.max():.1f} s of radio log are too short for a clock fit")
    xf, yf, bins = x[full], y[full], bins[full]
    rate, offset = np.polyfit(xf, yf, 1)
    for _ in range(iterations):
        residual = yf - (rate * xf + offset)
        order = np.lexsort((residual, bins))  # By window, soonest first
        floor = order[np.r_[True, bins[order][1:] != bins[order][:-1]]]
        rate, offset = np.polyfit(xf[floor], yf[floor], 1)
    latency = y - (rate * x + offset)
    clock = ClockMap(t0_tick=t0_tick, t0_pc=t0_pc + offset, rate=rate)
    stats = {
        "drift_percent": (rate - 1.0) * 100,
        "latency_median": float(np.median(latency)),
        "latency_p95": float(np.percentile(latency, 95)),
    }
    return clock, stats


def held(t: NDArray, values: NDArray, t_query: NDArray) -> tuple[NDArray, NDArray[np.bool_]]:
    """Values of a record held from each sample to the next, at `t_query`.

    Returns:
        The held values and where `t_query` lies within the record.
    """
    i = np.searchsorted(t, t_query, side="right") - 1
    inside = (i >= 0) & (t_query <= t[-1])
    return values[np.clip(i, 0, len(t) - 1)], inside


def match_fraction(
    t_sd: NDArray, v_sd: NDArray, t_tick: NDArray, v_radio: NDArray, offsets: NDArray
) -> NDArray:
    """Fraction of the radio values that the card record, shifted by each offset, reproduces."""
    counts = np.empty(len(offsets))
    for k, offset in enumerate(offsets):
        values, inside = held(t_sd, v_sd, t_tick - offset)
        counts[k] = np.count_nonzero(inside & (values == v_radio))
    return counts / len(v_radio)


def coarse_offsets(search: float) -> NDArray:
    """The 1 ms grid of offsets over `search` either way."""
    return np.arange(-search, search + 5e-4, 1e-3)


def match_rate(
    t_us: ArrayLike, v_sd: ArrayLike, t_tick: ArrayLike, v_radio: ArrayLike, search: float = 1.0
) -> float:
    """The entry of `SD_RATES` at which the card reproduces the most radio values.

    At the wrong rate the card slides past the radio by 12 ms every second and matches only by
    chance, a few percent of the radio values.

    Args:
        t_us: SD record timestamps in µs.
        v_sd: SD record values.
        t_tick: Onboard tick of the radio samples in s.
        v_radio: Radio values.
        search: Largest offset in s, either way.
    """
    t_sd, v_sd = np.asarray(t_us, dtype=float) / 1e6, np.asarray(v_sd)
    t_tick, v_radio = np.asarray(t_tick, dtype=float), np.asarray(v_radio)
    offsets = coarse_offsets(search)
    best = [match_fraction(r * t_sd, v_sd, t_tick, v_radio, offsets).max() for r in SD_RATES]
    return SD_RATES[int(np.argmax(best))]


def match_offset(
    t_sd: ArrayLike,
    v_sd: ArrayLike,
    t_tick: ArrayLike,
    v_radio: ArrayLike,
    card_lag: float,
    search: float = 1.0,
) -> tuple[float, float]:
    """Offset of the tick over the SD timer, from a variable the drone logs to both.

    The card record, held from sample to sample and shifted by a candidate offset, is read at the
    radio stamps, and the offset at which the most radio values come back exactly wins, first on a
    1 ms grid over `search`, then on a 0.05 ms grid around the best. A held card record shows a
    change only at its next sample, so the best shift reads the card late by `card_lag`, which the
    result corrects. The offset is the one of the tick stamps, which `fit_radio_clock` maps as well,
    so the time between the sampling of a radio block and its stamp cancels.

    Args:
        t_sd: SD record times in s, in tick seconds (scaled by the card's `SD_RATES` entry).
        v_sd: SD record values, logged with the same type as over the radio.
        t_tick: Onboard tick of the radio samples in s.
        v_radio: Radio values.
        card_lag: Mean delay of a change to the card record in s: half the card period for a
            variable that changes at any time, (card period - change period) / 2 for one that
            changes in step with the card's sampling.
        search: Largest offset in s, either way.

    Returns:
        The offset in s, the tick minus the SD timer, and the fraction of radio samples that match
        at it.
    """
    t_sd, v_sd = np.asarray(t_sd, dtype=float), np.asarray(v_sd)
    t_tick, v_radio = np.asarray(t_tick, dtype=float), np.asarray(v_radio)
    coarse = coarse_offsets(search)
    best = coarse[np.argmax(match_fraction(t_sd, v_sd, t_tick, v_radio, coarse))]
    fine = best + np.arange(-PEAK_HALF_WIDTH - 2e-3, PEAK_HALF_WIDTH + 2e-3 + 2.5e-5, 5e-5)
    score = match_fraction(t_sd, v_sd, t_tick, v_radio, fine)
    # The score falls off the best shift symmetrically but slowly, a few percent over a card
    # period, so the vertex of a parabola through the peak finds it better than the noisy maximum
    shift = fine[score == score.max()].mean()
    for _ in range(3):
        peak = np.abs(fine - shift) <= PEAK_HALF_WIDTH
        curvature, slope, _ = np.polyfit(fine[peak] - shift, score[peak], 2)
        if curvature >= 0:
            raise RuntimeError("The match score has no peak, the variable never changes")
        shift -= slope / (2 * curvature)
    return float(shift + card_lag), float(score.max())
