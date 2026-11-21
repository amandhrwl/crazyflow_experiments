"""Segments, the pass profile and the flight timelines of the downwash experiment."""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import math  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.integrate import quad  # noqa: E402

from crazyflow_experiments.downwash.a3_design import (  # noqa: E402
    build_design,
    flights,
    load_config,
    pass_half,
)
from crazyflow_experiments.downwash.timeline import (  # noqa: E402
    CEILING_MARGIN,
    Hold,
    Hop,
    Move,
    Pass,
    Timeline,
    build_flight,
    pass_profile,
    victim_hover_point,
)

RAMP = 1.0
PROFILES = [(0.5, 2.3), (1.0, 3.4), (1.4, 4.28)]  # (speed, length) of the A3 passes


def sim_pass_profile(
    start: np.ndarray, end: np.ndarray, speed: float, ramp: float, freq: float
) -> np.ndarray:
    """Reference pass positions at `freq`, from the profile of the sim's `pass_profile`.

    Stand-in transcribed from the formula in A3_core_passes.md §2.3, because afterwash was not
    available when this was written. [VERIFY] replace by the verbatim copy of afterwash's
    `pass_profile`.
    """
    length = float(np.linalg.norm(end - start))
    duration = length / speed + ramp
    t = np.arange(int(round(duration * freq)) + 1) / freq
    s = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti < ramp:
            s[i] = speed / 2 * (ti - ramp / math.pi * math.sin(math.pi * ti / ramp))
        elif ti <= duration - ramp:
            s[i] = speed * ramp / 2 + speed * (ti - ramp)
        else:
            rest = duration - ti
            s[i] = length - speed / 2 * (rest - ramp / math.pi * math.sin(math.pi * rest / ramp))
    return start + s[:, None] * (end - start) / length


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(require_masses=False)


@pytest.fixture(scope="module")
def design(cfg: dict) -> list[dict]:
    return build_design(cfg)[0]


@pytest.fixture(scope="module")
def timelines(cfg: dict, design: list[dict]) -> dict[int, tuple[Timeline, Timeline]]:
    return {flight: build_flight(rows, cfg) for flight, rows in flights(design).items()}


@pytest.mark.parametrize(("speed", "length"), PROFILES)
def test_pass_profile_shape(speed: float, length: float):
    duration = length / speed + RAMP
    s, s_dot, _ = pass_profile([0.0, duration], length, speed, RAMP)
    assert s == pytest.approx([0.0, length], abs=1e-12)
    assert s_dot == pytest.approx([0.0, 0.0], abs=1e-12)
    for t in (RAMP, duration - RAMP):  # The speed is continuous where the ramps meet the cruise
        before, after = pass_profile([t - 1e-9, t + 1e-9], length, speed, RAMP)[1]
        assert before == pytest.approx(after, abs=1e-6)
    t = np.linspace(RAMP, duration - RAMP, 101)
    assert pass_profile(t, length, speed, RAMP)[1] == pytest.approx(np.full(101, speed))
    t = np.linspace(0.0, duration, 200_001)
    max_acc = np.abs(pass_profile(t, length, speed, RAMP)[2]).max()
    assert max_acc == pytest.approx(math.pi * speed / (2 * RAMP), rel=1e-6)


@pytest.mark.parametrize(("speed", "length"), PROFILES)
def test_pass_profile_integrates_its_speed(speed: float, length: float):
    """The distance is the integral of the speed, an independent check of the closed form."""
    duration = length / speed + RAMP

    def rate(t: float) -> float:
        return float(pass_profile(t, length, speed, RAMP)[1])

    for t in np.linspace(0.0, duration, 23):
        expected = quad(rate, 0.0, t, points=[RAMP, duration - RAMP], epsabs=1e-12)[0]
        assert float(pass_profile(t, length, speed, RAMP)[0]) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(("speed", "length"), PROFILES)
def test_pass_matches_sim_pass_profile(speed: float, length: float):
    start, end = np.array([-length / 2, 0.0, 1.8]), np.array([length / 2, 0.0, 1.8])
    segment = Pass(start, end, speed, RAMP)
    reference = sim_pass_profile(start, end, speed, RAMP, freq=500.0)
    t = np.arange(len(reference)) / 500.0
    np.testing.assert_allclose(segment.evaluate(t)[0], reference, atol=1e-6)


def test_segments_rest_at_both_ends():
    a, b = np.array([0.0, 0.0, 1.0]), np.array([1.0, -0.5, 1.4])
    segments = [Hold(a, 2.0), Move(a, b, 2.5), Hop(a, 0.05, 1.0), Pass(a, b, 0.5, RAMP)]
    for segment in segments:
        pos, vel, _ = segment.evaluate(np.array([0.0, segment.duration]))
        np.testing.assert_allclose(vel, 0.0, atol=1e-12)
        np.testing.assert_allclose(pos[0], segment.start, atol=1e-12)
        np.testing.assert_allclose(pos[1], segment.end, atol=1e-12)


def test_derivatives_match_at_100hz(timelines: dict):
    """Within every segment the numerical derivatives of pos and vel match vel and acc.

    The acceleration of a move or a hop jumps at its ends, so the derivatives are taken inside the
    segment a sample falls in, one-sided at its ends.
    """
    h = 1e-6
    for victim, source in timelines.values():
        for timeline in (victim, source):
            t = np.arange(0.0, timeline.duration, 0.01)
            index = timeline._index(t)
            for i, segment in enumerate(timeline.segments):
                local = t[index == i] - timeline.starts[i]
                if not len(local):
                    continue
                hi = np.minimum(local + h, segment.duration)
                lo = np.maximum(local - h, 0.0)
                pos_hi, vel_hi, _ = segment.evaluate(hi)
                pos_lo, vel_lo, _ = segment.evaluate(lo)
                _, vel, acc = segment.evaluate(local)
                width = (hi - lo)[:, None]
                np.testing.assert_allclose((pos_hi - pos_lo) / width, vel, atol=1e-3)
                np.testing.assert_allclose((vel_hi - vel_lo) / width, acc, atol=1e-3)


def test_position_and_velocity_continuous_at_boundaries(timelines: dict):
    for victim, source in timelines.values():
        for timeline in (victim, source):
            for boundary in timeline.starts[1:]:
                before = timeline.state(boundary - 1e-9)
                after = timeline.state(boundary)
                np.testing.assert_allclose(before[0], after[0], atol=1e-6)
                np.testing.assert_allclose(before[1], after[1], atol=1e-6)


def test_cruise_covers_the_settled_window(cfg: dict):
    """At constant speed from `settle_time` after the ramp until |x| leaves `cruise_half`."""
    a3 = cfg["a3"]
    for speed in a3["speeds"]:
        half = pass_half(speed, a3)
        segment = Pass(np.array([-half, 0.0, 1.8]), np.array([half, 0.0, 1.8]), speed, a3["ramp"])
        t_settled = a3["ramp"] + a3["settle_time"]
        t = np.linspace(t_settled, segment.duration - t_settled, 1001)
        pos, vel, _ = segment.evaluate(t)
        assert pos[0, 0] == pytest.approx(-a3["cruise_half"], abs=1e-9)
        assert pos[-1, 0] == pytest.approx(a3["cruise_half"], abs=1e-9)
        np.testing.assert_allclose(vel[:, 0], speed, atol=1e-12)


def test_design_times_match_the_timeline(design: list[dict], timelines: dict):
    for flight, rows in flights(design).items():
        victim, source = timelines[flight]
        assert victim.duration == pytest.approx(source.duration)
        for row in rows:
            times = source.leg_times(row["leg"])
            assert times["start"] == pytest.approx(row["t_leg_start"], abs=1e-6)
            assert times["pass_start"] == pytest.approx(row["t_pass_start"], abs=1e-6)
            assert times["cross"] == pytest.approx(row["t_cross"], abs=1e-6)
            assert times["pass_end"] == pytest.approx(row["t_pass_end"], abs=1e-6)
            assert times["end"] == pytest.approx(row["t_leg_end"], abs=1e-6)
            first = "move" if row["move_time"] > 0 else "hover"
            assert source.label(row["t_leg_start"] + 1e-6) == (first, row["leg"])
            assert source.label(row["t_pass_start"] + 1e-6) == ("ramp_up", row["leg"])
            assert source.label(row["t_cross"]) == ("cruise", row["leg"])
            assert source.label(row["t_pass_end"] - 1e-6) == ("ramp_down", row["leg"])
            assert source.label(row["t_pass_end"] + 1e-6) == ("pause", row["leg"])
            assert victim.label(row["t_cross"]) == ("hold", row["leg"])
            pos = source.state(row["t_cross"])[0]
            assert pos[0] == pytest.approx(0.0, abs=1e-9)
            assert pos[2] == pytest.approx(row["z_src"])


def test_victim_holds_its_hover_point_through_the_legs(
    cfg: dict, design: list[dict], timelines: dict
):
    p_vic = victim_hover_point(cfg)
    for flight, rows in flights(design).items():
        victim, _ = timelines[flight]
        t = np.linspace(rows[0]["t_leg_start"], rows[-1]["t_leg_end"], 5001)
        pos, vel, _ = victim.state(t)
        np.testing.assert_allclose(pos, np.tile(p_vic, (len(t), 1)), atol=1e-12)
        np.testing.assert_allclose(vel, 0.0, atol=1e-12)


def test_hops_stay_below_the_ceiling(cfg: dict, timelines: dict):
    """The victim hops up; the source dips where a rise would break the ceiling margin."""
    ceiling = cfg["safety"]["arena_max"][2] - CEILING_MARGIN
    height = cfg["sync"]["hop_height"]
    for victim, source in timelines.values():
        for timeline in (victim, source):
            for segment in timeline.segments:
                if not isinstance(segment, Hop):
                    continue
                assert abs(segment.height) == height
                z = segment.evaluate(np.linspace(0.0, segment.duration, 101))[0][:, 2]
                assert z.max() <= ceiling + 1e-9
                if segment.pos[2] + height <= ceiling:
                    assert segment.height > 0


def test_source_flies_back_clear_of_the_victim(cfg: dict, design: list[dict], timelines: dict):
    """Outside the passes the source keeps the return lane's distance from the victim."""
    a3 = cfg["a3"]
    clearance = min(a3["return_lane"], *(row["half"] for row in design))
    for flight, rows in flights(design).items():
        victim, source = timelines[flight]
        t = np.linspace(0.0, source.duration, 40_001)
        p_src, p_vic = source.state(t)[0], victim.state(t)[0]
        labels = [source.label(ti)[0] for ti in t]
        passing = np.array([label in ("ramp_up", "cruise", "ramp_down") for label in labels])
        dxy = np.linalg.norm(p_src[:, :2] - p_vic[:, :2], axis=1)
        assert dxy[~passing].min() >= clearance - 1e-9
        for row in rows[1:]:  # Each flyback reaches the lane and crosses the victim's x in it
            times = source.leg_times(row["leg"])
            back = (t >= times["start"]) & (t < times["hover_start"])
            assert p_src[back, 1].max() == pytest.approx(p_vic[0, 1] + a3["return_lane"])
            assert p_src[back, 0].min() == pytest.approx(row["x_start"])


def test_skipped_leg_holds_and_the_next_flies_back(cfg: dict, design: list[dict]):
    rows = flights(design)[1]
    nominal_victim, nominal_source = build_flight(rows, cfg)
    skipped_leg = rows[6]["leg"]  # The first close pass of flight 1
    victim, source = build_flight(rows, cfg, skipped=[skipped_leg])
    assert victim.duration == pytest.approx(source.duration)
    nominal = nominal_source.leg_times(skipped_leg)
    times = source.leg_times(skipped_leg)
    assert times["start"] == pytest.approx(nominal["start"])
    assert times["end"] == pytest.approx(nominal["end"])
    assert math.isnan(times["pass_start"])
    t = np.linspace(times["start"], times["end"], 1001)
    pos, vel, _ = source.state(t)
    np.testing.assert_allclose(pos, np.tile(pos[0], (len(t), 1)), atol=1e-12)
    np.testing.assert_allclose(vel, 0.0, atol=1e-12)
    for row in rows[7:]:  # Unchanged passes, each from its own start point
        times = source.leg_times(row["leg"])
        start = source.state(times["pass_start"])[0]
        assert start[0] == pytest.approx(row["x_start"], abs=1e-9)
    # Outside the passes the source never comes within xy_guard of the victim
    t = np.linspace(0.0, source.duration, 20_001)
    p_src, p_vic = source.state(t)[0], victim.state(t)[0]
    labels = [source.label(ti)[0] for ti in t]
    passing = np.array([label in ("ramp_up", "cruise", "ramp_down") for label in labels])
    close = np.linalg.norm(p_src[:, :2] - p_vic[:, :2], axis=1) < cfg["safety"]["xy_guard"]
    assert not (close & ~passing).any()


def test_timeline_holds_outside_its_range(timelines: dict):
    _, source = next(iter(timelines.values()))
    first, last = source.segments[0].start, source.segments[-1].end
    for t, expected in ((-5.0, first), (source.duration + 5.0, last)):
        pos, vel, acc = source.state(t)
        np.testing.assert_allclose(pos, expected, atol=1e-12)
        np.testing.assert_allclose(vel, 0.0, atol=1e-12)


def test_timeline_rejects_gaps():
    a, b = np.zeros(3), np.ones(3)
    with pytest.raises(ValueError, match="does not start"):
        Timeline([Hold(a, 1.0), Hold(b, 1.0)])
