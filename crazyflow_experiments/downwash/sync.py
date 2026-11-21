"""Clocks of the downwash blocks: every onboard clock is mapped onto the PC clock.

Each drone has two clocks, the millisecond tick that stamps its radio log packets and the
microsecond timer that stamps its SD records. The PC stamps the arrival of every radio packet, the
motion capture and the setpoints it sends with its wall clock (`time.time()`), which is the common
time base. Two methods map the card onto it.

Through the radio (`sync_drone`, block A1 and the load cell test): the tick maps onto PC time
through the arrival times of the radio packets (`fit_radio_clock`). A
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

Through the setpoint echo (`fit_clock`, block A3): the card logs the setpoint the drone received,
which the PC sent, so the card timer maps straight onto PC time with an offset and a drift, the
85/84 of older firmware included.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

TICK_WRAP = 2**24  # ms, the radio log packets carry the tick in 3 bytes
PEAK_HALF_WIDTH = 4e-3  # s, shifts either side of the best that the peak fit uses
# Tick seconds per SD timer second: firmware with the fix of the timer's prescaler and before it
SD_RATES = (1.0, 85 / 84)
STABILIZER_PERIOD = 1e-3  # s, the loop that samples the IMU and triggers the card
# Radio block, variable and change period of a sync through the radio, see `sync_drone`: the
# victim's load cell reading, streamed by `radio.Victim`, changes at any time, the accelerometer in
# step with the stabilizer loop
LOAD_CELL_SYNC = ("loadcell", "loadcell.rawWeight", 0.0)
ACC_SYNC = ("cf_logger_0", "acc.z", STABILIZER_PERIOD)
# Fraction of radio samples the shifted card record must reproduce. The card at 500 Hz catches
# every load cell reading, but only every other value of the accelerometer, which changes at 1 kHz
MIN_MATCH = 0.3
MAX_OFFSET_SPREAD = 1e-3  # s, offsets of the two halves of a flight that disagree more fail
SD_GAP = 0.010  # s, card gaps longer than this are counted
# The setpoint echo of block A3, see `fit_clock`
PRIOR_WINDOW = 2.0  # s, the clock offset lies this close to the one the logging start gives
COARSE_BIN = 0.01  # s, histogram bin of the coarse clock offset
# s of setpoint changes the coarse offset comes from: 1.3 % drift moves the offset 13 ms over it
COARSE_SPAN = 1.0
MATCH_WINDOW = 0.03  # s, an echo lies this close to its setpoint under the current clock map
MIN_MATCHES = 20


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


def sync_drone(
    radio: dict,
    role: str,
    sd: dict[str, np.ndarray],
    sd_period: float,
    sync: tuple[str, str, float],
) -> tuple[ClockMap, dict]:
    """Map a drone's tick and card timer onto PC time through its radio log.

    Args:
        radio: The drone's radio blocks, `<block>/t_pc`, `<block>/t_tick_ms` and
            `<block>/<variable>`, without a role prefix.
        role: "victim" or "source", for the messages.
        sd: Its decoded card record, see `usd.decode`.
        sd_period: Period of the card record in s.
        sync: Radio block, variable and change period of the sync, e.g. `LOAD_CELL_SYNC`.

    Returns:
        The clock map and the numbers that check it.
    """
    block, variable, change_period = sync
    card_lag = (sd_period - change_period) / 2
    t_tick = unwrap_ticks(radio[f"{block}/t_tick_ms"])
    clock, stats = fit_radio_clock(t_tick, radio[f"{block}/t_pc"])
    v_sd, v_radio = sd[variable], radio[f"{block}/{variable}"]
    sd_rate = match_rate(sd["t_us"], v_sd, t_tick, v_radio)
    t_sd = sd_rate * sd["t_us"] / 1e6
    offset, fraction = match_offset(t_sd, v_sd, t_tick, v_radio, card_lag)
    if fraction < MIN_MATCH:
        raise RuntimeError(
            f"{role}: the card reproduces only {fraction:.0%} of the radio {variable} at its best "
            "offset. Is it the card file of this flight?"
        )
    # The offset is a constant of the boot, so both halves of the flight have to agree on it
    half = len(t_tick) // 2
    halves = [
        match_offset(t_sd, v_sd, t_tick[part], v_radio[part], card_lag)[0]
        for part in (slice(None, half), slice(half, None))
    ]
    if (spread := abs(halves[1] - halves[0])) > MAX_OFFSET_SPREAD:
        raise RuntimeError(f"{role}: the halves disagree on the card offset by {spread * 1e3} ms")
    gaps = np.diff(t_sd)
    if sd_rate != 1.0:
        logger.warning(f"{role}: its firmware predates the fix of the card timer, 85/84 applied")
    stats |= {
        "sync_variable": variable,
        "sd_rate": sd_rate,
        "sd_offset_ms": offset * 1e3,
        "sd_offset_halves_ms": [h * 1e3 for h in halves],
        "match_fraction": fraction,
        "sd_gaps_over_10ms": int(np.count_nonzero(gaps > SD_GAP)),
        "sd_longest_gap_ms": float(gaps.max() * 1e3),
    }
    return replace(clock, sd_offset=offset, sd_rate=sd_rate), stats


@dataclass
class ClockFit:
    """Map of an onboard clock onto PC time, `t_pc = offset + (1 + drift) t_onboard`.

    Attributes:
        offset: PC time at onboard time 0 in s.
        drift: Relative rate error of the onboard clock.
        residual_std: Standard deviation of the matched echoes about the fit in s.
        n_matches: Number of matched echoes in the fit.
    """

    offset: float
    drift: float
    residual_std: float
    n_matches: int

    def to_pc(self, t_onboard: NDArray) -> NDArray:
        """PC time of onboard times."""
        return self.offset + (1.0 + self.drift) * np.asarray(t_onboard, dtype=float)


def change_events(t: NDArray, values: NDArray) -> tuple[NDArray, NDArray]:
    """Times and new values (N, k) where any column of `values` changes."""
    values = np.asarray(values).reshape(len(t), -1)
    index = np.nonzero(np.any(values[1:] != values[:-1], axis=1))[0] + 1
    return np.asarray(t)[index], values[index]


def fit_clock(
    pc_t: NDArray, pc_mm: NDArray, onboard_t: NDArray, onboard_mm: NDArray, prior_offset: float
) -> ClockFit:
    """Fit the onboard clock to PC time from the setpoint echo.

    Every change of the onboard setpoint is a setpoint the PC sent with the same value in mm. The
    onboard clocks run about 1.3 % slow against the PC (F1, 18 Nov: 13400 ppm on cf3, 12900 ppm on
    cf4), so a single offset holds for about two seconds only. The coarse offset is therefore the
    densest bin of the offsets between equal values near `prior_offset` over the first
    `COARSE_SPAN` of onboard changes. Then the matched span doubles each round until it covers the
    log: each onboard change in it is matched to the nearest sent change of its value under the
    current map, and offset and drift are refitted by least squares with outliers beyond five
    robust standard deviations dropped. Each round extrapolates the last fit to twice its span,
    well inside `MATCH_WINDOW`, so the victim, whose setpoint changes only at the takeoff and the
    sync hops, is matched across its whole flight as well.

    Args:
        pc_t: Send times of the setpoints in PC time (N,).
        pc_mm: Sent setpoints in mm (N, k), the axes the drone echoes.
        onboard_t: Onboard times of the logged setpoint (M,).
        onboard_mm: Logged setpoint in mm (M, k).
        prior_offset: Rough offset, PC time at onboard time 0, e.g. from the logging start.

    Returns:
        The fitted map.
    """
    pc_t, pc_mm = change_events(pc_t, pc_mm)
    on_t, on_mm = change_events(onboard_t, onboard_mm)
    if len(on_t) == 0:
        raise RuntimeError("The logged setpoint never changes")
    on_values = list(map(tuple, on_mm))
    sent = defaultdict(list)
    for t, value in zip(pc_t, map(tuple, pc_mm)):
        sent[value].append(t)
    sent = {value: np.array(times) for value, times in sent.items()}
    early = on_t <= on_t[0] + COARSE_SPAN
    candidates = [
        times - t
        for t, value in zip(on_t[early], [v for v, e in zip(on_values, early) if e])
        if (times := sent.get(value)) is not None
    ]
    offsets = np.concatenate(candidates) if candidates else np.array([])
    offsets = offsets[np.abs(offsets - prior_offset) < PRIOR_WINDOW]
    if len(offsets) < MIN_MATCHES:
        raise RuntimeError(f"Only {len(offsets)} setpoint echoes near the logging start")
    bins, counts = np.unique(np.round((offsets - prior_offset) / COARSE_BIN), return_counts=True)
    offset, slope = prior_offset + bins[np.argmax(counts)] * COARSE_BIN, 1.0
    span = COARSE_SPAN
    while True:
        in_span = on_t <= on_t[0] + span
        pairs = _match_echoes(on_t[in_span], on_values, sent, offset, slope)
        if len(pairs) < MIN_MATCHES:
            raise RuntimeError(f"Only {len(pairs)} setpoint echoes matched")
        offset, slope, residual = _robust_line(*pairs.T)
        if in_span.all():
            break
        span *= 2
    return ClockFit(
        offset=offset,
        drift=slope - 1,
        residual_std=float(np.std(residual)),
        n_matches=len(residual),
    )


def _match_echoes(
    on_t: NDArray, on_values: Sequence[tuple], sent: Mapping, offset: float, slope: float
) -> NDArray:
    """Pairs (onboard time, send time) of the onboard changes with a sent twin under the map."""
    pairs = []
    for t, value in zip(on_t, on_values):
        if (times := sent.get(value)) is None:
            continue
        predicted = offset + slope * t
        nearest = times[np.argmin(np.abs(times - predicted))]
        if abs(nearest - predicted) < MATCH_WINDOW:
            pairs.append((t, nearest))
    return np.array(pairs).reshape(-1, 2)


def _robust_line(x: NDArray, y: NDArray) -> tuple[float, float, NDArray]:
    """Least-squares line `y = offset + slope x` without outliers beyond five robust std.

    Returns:
        Offset, slope and the residuals of the points kept.
    """
    x0, y0 = x.mean(), y.mean()
    keep = np.ones(len(x), dtype=bool)
    for _ in range(10):
        slope, intercept = np.polyfit(x[keep] - x0, y[keep] - y0, 1)
        residual = (y - y0) - (intercept + slope * (x - x0))
        center = np.median(residual[keep])
        scale = max(1.4826 * np.median(np.abs(residual[keep] - center)), 1e-4)
        new_keep = np.abs(residual - center) < 5 * scale
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return float(y0 + intercept - slope * x0), float(slope), residual[keep]
