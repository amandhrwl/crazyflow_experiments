"""Setpoints of both drones over one A3 flight, built from analytic segments.

A segment is a hold, a cosine move between two points, the sync hop or a flyover pass. Each gives
its position, velocity and acceleration in closed form and is at rest at both ends, so a `Timeline`
of segments is continuous in position and velocity. The acceleration of a move or a hop jumps at
its ends, as their cosine profiles do.

One flight is: hold 2 s, victim hop, hold 1 s, source hop, hold 1 s, the legs, victim hop, hold
1 s, source hop, hold 1 s. The victim holds its hover point throughout; the source moves to each
leg's start point, hovers, passes, pauses at the end point and waits for still air. Every pass flies
from -x to +x, so the source flies back to the next start point through the return lane, offset
sideways from the victim, and never flies over it outside a pass. Both timelines start at t = 0,
after the joint takeoff and calibration, and have the same duration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

PHASES = ("hold", "hop", "move", "hover", "ramp_up", "cruise", "ramp_down", "pause", "wait")
NO_LEG = -1  # Leg index outside the legs
FIRST_HOLD = 2.0  # s, hold before the first hop
HOP_GAP = 1.0  # s, hold after every hop
CEILING_MARGIN = 0.1  # m, every setpoint stays this far below the arena ceiling


def pass_profile(
    t: ArrayLike, length: float, speed: float, ramp: float
) -> tuple[NDArray, NDArray, NDArray]:
    """Distance, speed and acceleration along a pass with the sim's `pass_profile`.

    Cosine ramps of `ramp` seconds lead into and out of a cruise at `speed`, so the pass takes
    `length / speed + ramp` seconds and starts and ends at rest. Outside the pass the distance holds
    at 0 and `length`.

    Args:
        t: Time since the pass started in s.
        length: Length of the pass in m, at least `speed * ramp`.
        speed: Cruise speed in m/s.
        ramp: Duration of each ramp in s.

    Returns:
        The distance s, its rate and its acceleration, each shaped like `t`.
    """
    duration = length / speed + ramp
    t = np.clip(np.asarray(t, dtype=float), 0.0, duration)
    rest = duration - t
    w = math.pi / ramp
    up, down = t < ramp, t > duration - ramp
    s = np.where(
        up,
        0.5 * speed * (t - np.sin(w * t) / w),
        np.where(
            down,
            length - 0.5 * speed * (rest - np.sin(w * rest) / w),
            0.5 * speed * ramp + speed * (t - ramp),
        ),
    )
    s_dot = np.where(
        up,
        0.5 * speed * (1 - np.cos(w * t)),
        np.where(down, 0.5 * speed * (1 - np.cos(w * rest)), speed),
    )
    s_ddot = np.where(
        up,
        0.5 * speed * w * np.sin(w * t),
        np.where(down, -0.5 * speed * w * np.sin(w * rest), 0.0),
    )
    return s, s_dot, s_ddot


@dataclass(frozen=True, eq=False)
class Hold:
    """Rest at one point.

    Attributes:
        pos: The point as (3,).
        duration: Seconds held.
        phase: Label of the hold, e.g. "hover", "pause" or "wait" inside a leg.
        leg: Leg the hold belongs to, or `NO_LEG`.
    """

    pos: NDArray
    duration: float
    phase: str = "hold"
    leg: int = NO_LEG

    @property
    def start(self) -> NDArray:
        """Position at the start."""
        return self.pos

    @property
    def end(self) -> NDArray:
        """Position at the end."""
        return self.pos

    def evaluate(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Position, velocity and acceleration as (N, 3) at the local times `t` (N,)."""
        return np.tile(self.pos, (len(t), 1)), np.zeros((len(t), 3)), np.zeros((len(t), 3))

    def phase_at(self, t: float) -> str:
        """Phase label at local time `t`."""
        return self.phase


@dataclass(frozen=True, eq=False)
class Move:
    """Cosine blend from `p_a` to `p_b`, at rest at both ends.

    Attributes:
        p_a: Start point as (3,).
        p_b: End point as (3,).
        duration: Seconds of the move.
        leg: Leg the move belongs to, or `NO_LEG`.
        phase: Label of the move.
    """

    p_a: NDArray
    p_b: NDArray
    duration: float
    leg: int = NO_LEG
    phase: str = "move"

    @property
    def start(self) -> NDArray:
        """Position at the start."""
        return self.p_a

    @property
    def end(self) -> NDArray:
        """Position at the end."""
        return self.p_b

    def evaluate(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Position, velocity and acceleration as (N, 3) at the local times `t` (N,)."""
        w = math.pi / self.duration
        delta = self.p_b - self.p_a
        blend = 0.5 * (1 - np.cos(w * t))
        rate = 0.5 * w * np.sin(w * t)
        accel = 0.5 * w**2 * np.cos(w * t)
        return self.p_a + blend[:, None] * delta, rate[:, None] * delta, accel[:, None] * delta

    def phase_at(self, t: float) -> str:
        """Phase label at local time `t`."""
        return self.phase


@dataclass(frozen=True, eq=False)
class Hop:
    """Vertical cosine bump of `height` over `duration`, back at `pos` and at rest at its end.

    Attributes:
        pos: Point the hop starts and ends at as (3,).
        height: Height of the bump in m.
        duration: Seconds of the hop.
        leg: Leg the hop belongs to, or `NO_LEG`.
        phase: Label of the hop.
    """

    pos: NDArray
    height: float
    duration: float
    leg: int = NO_LEG
    phase: str = "hop"

    @property
    def start(self) -> NDArray:
        """Position at the start."""
        return self.pos

    @property
    def end(self) -> NDArray:
        """Position at the end."""
        return self.pos

    def evaluate(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Position, velocity and acceleration as (N, 3) at the local times `t` (N,)."""
        w = 2 * math.pi / self.duration
        up = np.array([0.0, 0.0, 1.0])
        z = 0.5 * self.height * (1 - np.cos(w * t))
        z_dot = 0.5 * self.height * w * np.sin(w * t)
        z_ddot = 0.5 * self.height * w**2 * np.cos(w * t)
        return self.pos + z[:, None] * up, z_dot[:, None] * up, z_ddot[:, None] * up

    def phase_at(self, t: float) -> str:
        """Phase label at local time `t`."""
        return self.phase


@dataclass(frozen=True, eq=False)
class Pass:
    """Straight flyover pass from `p_start` to `p_end` with the profile of `pass_profile`.

    Attributes:
        p_start: Start point as (3,).
        p_end: End point as (3,).
        speed: Cruise speed in m/s.
        ramp: Duration of each ramp in s.
        leg: Leg the pass belongs to.
        phase: Label of the whole pass in the segment table; `phase_at` splits it into the ramps
            and the cruise.
    """

    p_start: NDArray
    p_end: NDArray
    speed: float
    ramp: float
    leg: int = NO_LEG
    phase: str = "pass"

    def __post_init__(self):
        """Check that the pass is long enough for both ramps."""
        if self.length < self.speed * self.ramp:
            raise ValueError(f"Pass of {self.length:.3f} m is too short for its ramps")

    @property
    def length(self) -> float:
        """Length of the pass in m."""
        return float(np.linalg.norm(self.p_end - self.p_start))

    @property
    def duration(self) -> float:
        """Seconds of the pass, ramps included."""
        return self.length / self.speed + self.ramp

    @property
    def start(self) -> NDArray:
        """Position at the start."""
        return self.p_start

    @property
    def end(self) -> NDArray:
        """Position at the end."""
        return self.p_end

    def evaluate(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Position, velocity and acceleration as (N, 3) at the local times `t` (N,)."""
        s, s_dot, s_ddot = pass_profile(t, self.length, self.speed, self.ramp)
        u = (self.p_end - self.p_start) / self.length
        return self.p_start + s[:, None] * u, s_dot[:, None] * u, s_ddot[:, None] * u

    def phase_at(self, t: float) -> str:
        """Phase label at local time `t`: ramp_up, cruise or ramp_down."""
        if t < self.ramp:
            return "ramp_up"
        if t > self.duration - self.ramp:
            return "ramp_down"
        return "cruise"


Segment = Hold | Move | Hop | Pass


class Timeline:
    """Segments played one after another from t = 0."""

    def __init__(self, segments: Sequence[Segment]):
        """Chain the segments; segments of zero duration are dropped.

        Args:
            segments: The segments in order. Each must start where the previous one ended.
        """
        self.segments = tuple(s for s in segments if s.duration > 0)
        if not self.segments:
            raise ValueError("A timeline needs at least one segment")
        for prev, seg in zip(self.segments, self.segments[1:]):
            if not np.allclose(prev.end, seg.start, atol=1e-9):
                raise ValueError(f"{seg} does not start where {prev} ends")
        durations = np.array([s.duration for s in self.segments])
        self.ends = np.cumsum(durations)
        self.starts = self.ends - durations
        self.duration = float(self.ends[-1])

    def _index(self, t: NDArray) -> NDArray:
        """Segment of every time; a boundary belongs to the segment it starts."""
        return np.clip(np.searchsorted(self.ends, t, side="right"), 0, len(self.segments) - 1)

    def state(self, t: float | ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
        """Position, velocity and acceleration at time `t`.

        Before the start and after the end the timeline holds its first and last point at rest.

        Args:
            t: Time in s, a scalar or an array (N,).

        Returns:
            Position, velocity and acceleration as (3,) for a scalar time, (N, 3) otherwise.
        """
        times = np.atleast_1d(np.asarray(t, dtype=float))
        index = self._index(times)
        pos, vel, acc = (np.empty((len(times), 3)) for _ in range(3))
        for i in np.unique(index):
            mask = index == i
            seg = self.segments[i]
            local = np.clip(times[mask] - self.starts[i], 0.0, seg.duration)
            pos[mask], vel[mask], acc[mask] = seg.evaluate(local)
        if np.ndim(t) == 0:
            return pos[0], vel[0], acc[0]
        return pos, vel, acc

    def label(self, t: float) -> tuple[str, int]:
        """Phase and leg at time `t`."""
        i = int(self._index(np.array([t]))[0])
        seg = self.segments[i]
        return seg.phase_at(min(max(t - self.starts[i], 0.0), seg.duration)), seg.leg

    def table(self) -> list[dict]:
        """One row per segment, for the flight's meta.json and the conversion."""
        return [
            {
                "kind": type(seg).__name__.lower(),
                "phase": seg.phase,
                "leg": seg.leg,
                "t_start": float(t0),
                "t_end": float(t1),
                "start": [float(x) for x in seg.start],
                "end": [float(x) for x in seg.end],
            }
            for seg, t0, t1 in zip(self.segments, self.starts, self.ends)
        ]

    def leg_times(self, leg: int) -> dict[str, float]:
        """Boundaries of one leg, see `segment_leg_times`."""
        return segment_leg_times(self.table(), leg)


def segment_leg_times(table: Sequence[Mapping], leg: int) -> dict[str, float]:
    """Boundaries of one leg in a segment table of `Timeline.table`.

    Args:
        table: The segment table of a source timeline, e.g. from a flight's meta.json.
        leg: Global leg index.

    Returns:
        `start` and `end` of the leg, `hover_start`, `pass_start`, `cross` (the middle of the
        pass), `pass_end` and `pause_end`. The pass times are NaN for a skipped leg.
    """
    rows = [row for row in table if row["leg"] == leg]
    if not rows:
        raise KeyError(f"Leg {leg} is not in the timeline")
    by_phase = {row["phase"]: row for row in rows}
    nan = float("nan")
    times = {"start": rows[0]["t_start"], "end": rows[-1]["t_end"]}
    times["hover_start"] = by_phase["hover"]["t_start"] if "hover" in by_phase else nan
    times["pause_end"] = by_phase["pause"]["t_end"] if "pause" in by_phase else nan
    passes = by_phase.get("pass", {"t_start": nan, "t_end": nan})
    times["pass_start"], times["pass_end"] = passes["t_start"], passes["t_end"]
    times["cross"] = 0.5 * (times["pass_start"] + times["pass_end"])
    return times


def victim_hover_point(cfg: Mapping) -> NDArray:
    """Point the victim hovers at."""
    victim = cfg["drones"]["victim"]
    return np.array([*victim["hover_xy"], victim["hover_z"]], dtype=float)


def takeoff_timelines(
    grounds: Mapping[str, ArrayLike], targets: Mapping[str, ArrayLike], cfg: Mapping
) -> dict[str, Timeline]:
    """Paths of the drones from where they stand to their targets, all of the same duration.

    Climb: each drone rises straight up over `takeoff.duration`, the victim to its hover height,
    the source to the transfer height of the highest pass. Transfer: both fly level to above their
    targets, the source at least the largest height separation above the victim, so their paths
    may cross. Descent: the source sinks onto its start point, `half` of a pass away from the
    victim. Then both hold for `takeoff.hover_time`.

    Args:
        grounds: Where each drone stands as (3,), by role.
        targets: Where each drone goes as (3,), by role: the victim's hover point and the start
            point of the source's first leg.
        cfg: The experiment config.

    Returns:
        The timeline of every drone, by role.
    """
    a3, takeoff = cfg["a3"], cfg["takeoff"]
    z_transfer = cfg["drones"]["victim"]["hover_z"] + max(a3["height_seps"])
    points = {}
    for name, ground in grounds.items():
        ground, target = np.asarray(ground, dtype=float), np.asarray(targets[name], dtype=float)
        z = z_transfer if name == "source" else target[2]
        points[name] = (ground, np.array([*ground[:2], z]), np.array([*target[:2], z]), target)
    transfer = max(move_duration(p[1], p[2], a3) for p in points.values())
    descent = max(move_duration(p[2], p[3], a3) for p in points.values())
    return {
        name: Timeline(
            [
                Move(ground, climbed, takeoff["duration"], phase="climb"),
                Move(climbed, above, transfer, phase="transfer"),
                Move(above, target, descent, phase="descent"),
                Hold(target, takeoff["hover_time"], "hover"),
            ]
        )
        for name, (ground, climbed, above, target) in points.items()
    }


def move_duration(p_a: ArrayLike, p_b: ArrayLike, a3: Mapping) -> float:
    """Duration of a move: `min_move_time`, or longer at `move_speed`; 0 when there is no move."""
    distance = float(np.linalg.norm(np.asarray(p_b, dtype=float) - np.asarray(p_a, dtype=float)))
    if distance < 1e-9:
        return 0.0
    return max(a3["min_move_time"], distance / a3["move_speed"])


def approach(
    pos: ArrayLike,
    start: ArrayLike,
    p_vic: NDArray,
    a3: Mapping,
    leg: int = NO_LEG,
    min_duration: float = 0.0,
) -> list[Segment]:
    """Moves of the source from `pos` to a leg's start point, at rest between them.

    A source on the same side of the victim along x moves straight. One on the other side flies
    back through the return lane: sideways out to `return_lane` from the victim's y, along x to the
    start point while changing height, and sideways in to the start point.

    Args:
        pos: Where the source is as (3,).
        start: Start point of the leg as (3,).
        p_vic: Hover point of the victim as (3,).
        a3: The `[a3]` config.
        leg: Leg the moves belong to.
        min_duration: The moves are stretched to take at least this long.

    Returns:
        The moves, or a hold at `start` when the source is already there.
    """
    pos, start = np.asarray(pos, dtype=float), np.asarray(start, dtype=float)
    points = [pos, start]
    if (pos[0] - p_vic[0]) * (start[0] - p_vic[0]) < 0:
        y = p_vic[1] + a3["return_lane"]
        points = [pos, np.array([pos[0], y, pos[2]]), np.array([start[0], y, start[2]]), start]
    durations = [move_duration(a, b, a3) for a, b in zip(points, points[1:])]
    total = sum(durations)
    if total == 0.0:
        return [Hold(start, min_duration, "move", leg)]
    scale = max(1.0, min_duration / total)
    return [
        Move(a, b, duration * scale, leg)
        for a, b, duration in zip(points, points[1:], durations)
        if duration > 0
    ]


def approach_duration(pos: ArrayLike, start: ArrayLike, p_vic: NDArray, a3: Mapping) -> float:
    """Seconds of the moves of `approach`."""
    return sum(seg.duration for seg in approach(pos, start, p_vic, a3))


def pre_leg_time(cfg: Mapping) -> float:
    """Seconds from t = 0 to the start of the first leg."""
    return FIRST_HOLD + 2 * (cfg["sync"]["hop_time"] + HOP_GAP)


def build_flight(
    design_rows: Sequence[Mapping], cfg: Mapping, skipped: Collection[int] = ()
) -> tuple[Timeline, Timeline]:
    """Timelines of the victim and the source over one flight.

    A skipped leg is replaced by a hold of the same duration where the source is; the next leg
    then flies back from there. Outside the passes the source never flies over the victim, see
    `approach`.

    A sync hop rises by `hop_height`, unless that would break the ceiling margin; then it dips.

    Args:
        design_rows: The rows of one flight from the design, in any order.
        cfg: The experiment config.
        skipped: Global indices of legs to skip.

    Returns:
        The victim's and the source's timeline, of the same duration.
    """
    rows = sorted(design_rows, key=lambda row: int(row["index_in_flight"]))
    if len({int(row["flight"]) for row in rows}) != 1:
        raise ValueError("The rows of one flight are expected")
    a3, sync = cfg["a3"], cfg["sync"]
    hop_time = sync["hop_time"]
    ceiling = cfg["safety"]["arena_max"][2] - CEILING_MARGIN
    p_vic = victim_hover_point(cfg)

    def hop(pos: NDArray) -> Hop:
        rises = pos[2] + sync["hop_height"] <= ceiling + 1e-9
        return Hop(pos, sync["hop_height"] if rises else -sync["hop_height"], hop_time)

    def leg_points(row: Mapping) -> tuple[NDArray, NDArray]:
        y, z = p_vic[1] + float(row["lateral_offset"]), float(row["z_src"])
        return np.array([float(row["x_start"]), y, z]), np.array([float(row["x_end"]), y, z])

    pos, _ = leg_points(rows[0])  # The takeoff ends at the first leg's start point
    victim = [
        Hold(p_vic, FIRST_HOLD),
        hop(p_vic),
        Hold(p_vic, HOP_GAP),
        Hold(p_vic, hop_time),
        Hold(p_vic, HOP_GAP),
    ]
    source = [
        Hold(pos, FIRST_HOLD),
        Hold(pos, hop_time),
        Hold(pos, HOP_GAP),
        hop(pos),
        Hold(pos, HOP_GAP),
    ]
    for row in rows:
        leg = int(row["leg"])
        start, end = leg_points(row)
        segments = [
            *approach(pos, start, p_vic, a3, leg, float(row["move_time"])),
            Hold(start, a3["hover_time"], "hover", leg),
            Pass(start, end, float(row["speed"]), a3["ramp"], leg),
            Hold(end, a3["pause_time"], "pause", leg),
            Hold(end, a3["wait_time"], "wait", leg),
        ]
        duration = sum(seg.duration for seg in segments)
        victim.append(Hold(p_vic, duration, "hold", leg))
        if leg in skipped:
            source.append(Hold(pos, duration, "hold", leg))
        else:
            source.extend(segments)
            pos = end
    victim += [hop(p_vic), Hold(p_vic, HOP_GAP), Hold(p_vic, hop_time), Hold(p_vic, HOP_GAP)]
    source += [Hold(pos, hop_time), Hold(pos, HOP_GAP), hop(pos), Hold(pos, HOP_GAP)]
    victim_timeline, source_timeline = Timeline(victim), Timeline(source)
    if not math.isclose(victim_timeline.duration, source_timeline.duration, abs_tol=1e-9):
        raise RuntimeError("The victim's and the source's timeline differ in duration")
    return victim_timeline, source_timeline
