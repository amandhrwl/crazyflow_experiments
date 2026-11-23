"""A3 on the stand: the legs of A3 flown over the victim clamped on the load cell.

The source flies A3's legs with their speeds and height separations, centred on the victim's
position on the stand instead of its hover point, so the recordings line up leg by leg with the
free-hover flights. The victim reads the load cell, and in a spin run it runs its motors at hover
through `motorPowerSet` around each pass.

The victim stays on its battery, which A1 saw sag from 4.0 to 3.4 V within about a minute near
hover. So it spins per leg only: motors off for a tare while the source is still far out, a spin-up
`arm_lead_s` before the command, `spin_settle_s` for the rpm to settle (A1 discarded the first
1.5 s of every level), a `baseline_s` spin-alone baseline while the source hovers at the leg's
start, the pass, and `spin_after_s` after it; then motors off until the next leg. A still run
records the same windows with the motors off. Between legs the battery is read at rest; below
`vbat_rest_min` the remaining legs fly without spinning and are listed for a refly with `--legs`.
Under load, A1's `vbat_load_min` stops the victim's motors only, never the source.

This module holds what the deploy script, the conversion and the tests share and needs no radio,
ROS or motion capture: the stand geometry, the hover command from A1, the spin schedule of a
flight, the battery budget and the thread that drives the victim along the schedule.
"""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
import tomllib
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from crazyflow_experiments.downwash import A1_RESULTS_DIR
from crazyflow_experiments.downwash.timeline import CEILING_MARGIN

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from numpy.typing import ArrayLike, NDArray

    from crazyflow_experiments.downwash.radio import RadioLog
    from crazyflow_experiments.downwash.timeline import Timeline

logger = logging.getLogger(__name__)

MODES = ("still", "spin")
A1_CURVE_PATH = A1_RESULTS_DIR / "rpm2thrust.toml"  # cf10's thrust curve, written by a1_fit.py
HOVER_CMD_TOL = 0.01  # Relative difference of the configured hover command from A1's that warns
POLL = 0.05  # s between the checks of the victim thread
VBAT_RECENT = 1.0  # s of status packets whose voltage counts as the present one under load


class SpinVictim(Protocol):
    """What the schedule needs of the victim, `radio.Victim` or a test fake."""

    radio_log: RadioLog

    @property
    def spinning(self) -> bool:
        """Whether the victim is armed and its motors run under `motorPowerSet`."""

    def spin(self, pwm: int):
        """Arm and run all four motors at `pwm`."""

    def stop_spin(self):
        """Stop the motors and disarm, never raises."""


def parse_legs(text: str) -> list[int]:
    """Leg indices of `--legs`: comma separated indices and ranges, e.g. "3,5-7"."""
    legs: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            first, last = (int(x) for x in part.split("-"))
            if last < first:
                raise ValueError(f"Empty leg range {part!r}")
            legs.extend(range(first, last + 1))
        elif part:
            legs.append(int(part))
    if not legs or len(set(legs)) != len(legs):
        raise ValueError(f"--legs {text!r} names no leg or one twice")
    return legs


def select_legs(rows: Sequence[Mapping], legs: Sequence[int] | None) -> list[dict]:
    """The rows of one flight whose global leg index is in `legs`, all of them for None."""
    if legs is None:
        return [dict(row) for row in rows]
    known = {int(row["leg"]) for row in rows}
    if missing := sorted(set(legs) - known):
        raise ValueError(f"Legs {missing} are not in this flight, it has {sorted(known)}")
    return [dict(row) for row in rows if int(row["leg"]) in legs]


def stand_rows(rows: Sequence[Mapping], cfg: Mapping) -> list[dict]:
    """The rows whose height separation is one the stand runs fly, `[stand] height_seps`."""
    keep = [float(dz) for dz in cfg["stand"]["height_seps"]]
    if unknown := [dz for dz in keep if dz not in [float(d) for d in cfg["a3"]["height_seps"]]]:
        raise ValueError(f"[stand] height_seps {unknown} are not separations of the A3 design")
    return [
        dict(row) for row in rows if any(abs(float(row["height_sep"]) - dz) < 1e-6 for dz in keep)
    ]


def stand_config(cfg: Mapping, stand_pos: ArrayLike) -> dict:
    """The config of a run over the victim on the stand, a copy.

    The victim's hover point becomes the stand position, the ceiling of the arena the stand's
    `arena_max_z`, since the passes fly above the stand's height, and the separations of the
    takeoff the stand's: it climbs to the stand's z plus the largest of `a3.height_seps`, which
    were the free flights' 1.0 m, 2.52 m, out of the motion capture's reach (22 Nov).

    Args:
        cfg: The A3 config, with its `[stand]` section.
        stand_pos: Position of the victim on the stand as (3,), its marker frame.
    """
    stand_pos = np.asarray(stand_pos, dtype=float)
    stand_cfg = copy.deepcopy(dict(cfg))
    victim = stand_cfg["drones"]["victim"]
    victim["hover_xy"] = stand_pos[:2].tolist()
    victim["hover_z"] = float(stand_pos[2])
    stand_cfg["a3"]["height_seps"] = [float(dz) for dz in cfg["stand"]["height_seps"]]
    stand_cfg["safety"]["arena_max"] = [
        *stand_cfg["safety"]["arena_max"][:2],
        cfg["stand"]["arena_max_z"],
    ]
    top = float(stand_pos[2]) + max(cfg["stand"]["height_seps"])
    if top > cfg["stand"]["arena_max_z"] - CEILING_MARGIN + 1e-9:
        raise ValueError(
            f"The highest pass flies at {top:.2f} m, within {CEILING_MARGIN} m of the stand's "
            f"arena_max_z = {cfg['stand']['arena_max_z']} m"
        )
    return stand_cfg


def stand_flight(
    cfg: Mapping, rows: Sequence[Mapping], stand_pos: ArrayLike
) -> tuple[dict, list[dict]]:
    """Config and design rows of a flight centred on the victim on the stand.

    The passes are shifted along x by the stand's x, and each pass flies `height_sep` above the
    stand's victim. Only the legs whose separation is in the stand's `height_seps` are kept (at most
    0.5 m, user, 21 Nov). See `stand_config` for the config.

    Args:
        cfg: The A3 config, with its `[stand]` section.
        rows: Design rows of one flight.
        stand_pos: Position of the victim on the stand as (3,), its marker frame.

    Returns:
        The config and the rows, both copies.
    """
    stand_pos = np.asarray(stand_pos, dtype=float)
    stand_cfg = stand_config(cfg, stand_pos)
    shifted = []
    for row in stand_rows(rows, cfg):
        row = dict(row)
        row["x_start"] = float(row["x_start"]) + stand_pos[0]
        row["x_end"] = float(row["x_end"]) + stand_pos[0]
        row["z_src"] = float(stand_pos[2]) + float(row["height_sep"])
        shifted.append(row)
    return stand_cfg, shifted


def check_source_start(ground: ArrayLike, stand_pos: ArrayLike, clearance: float):
    """The source climbs straight up from where it stands: that spot keeps clear of the victim."""
    ground, stand_pos = np.asarray(ground, dtype=float), np.asarray(stand_pos, dtype=float)
    if (distance := float(np.linalg.norm(ground[:2] - stand_pos[:2]))) < clearance:
        raise ValueError(
            f"The source stands {distance:.2f} m from the victim, less than the clearance of "
            f"{clearance} m: its straight climb would pass the stand"
        )


def hover_command(steady: Sequence[Mapping], weight: float) -> tuple[int, dict]:
    """Motor command at which A1 measured the victim's weight in thrust.

    Interpolated linearly between the steady windows of A1's block all that passed and drove all
    four motors alike. On the battery the thrust at a fixed command falls as the voltage sags, so
    the conversion takes the victim's thrust from its rpm, not from this command.

    Args:
        steady: Rows of A1's steady.csv, see `a1_convert.read_steady`.
        weight: The victim's weight in N.

    Returns:
        The command and how it was found.
    """
    rows = [r for r in steady if r["ok"] and r["m1"] == r["m2"] == r["m3"] == r["m4"]]
    commands = np.array([r["m1"] for r in rows], dtype=float)
    forces = np.array([r["F_mean"] for r in rows], dtype=float)
    order = np.argsort(commands)
    commands, forces = commands[order], forces[order]
    if len(rows) < 2 or np.any(np.diff(forces) <= 0):
        raise ValueError("A1's ok windows do not give a thrust that rises with the command")
    if not forces[0] <= weight <= forces[-1]:
        raise ValueError(
            f"The weight {weight:.3f} N lies outside A1's measured thrust "
            f"{forces[0]:.3f} to {forces[-1]:.3f} N"
        )
    command = float(np.interp(weight, forces, commands))
    k = int(np.searchsorted(forces, weight))
    k = min(max(k, 1), len(forces) - 1)
    return round(command), {
        "weight_N": weight,
        "below": {"cmd": int(commands[k - 1]), "F_N": float(forces[k - 1])},
        "above": {"cmd": int(commands[k]), "F_N": float(forces[k])},
        "n_windows": len(rows),
    }


@dataclass(frozen=True)
class LegSpin:
    """The victim's windows of one leg, in the source timeline's time.

    Attributes:
        leg: Global leg index.
        tare_start: Motors off from here, the tare and the battery reading at rest.
        arm_at: The arming request goes out, `arm_lead_s` before the command.
        spin_on: The hover command is applied.
        baseline_start: The spin-alone baseline starts, after `spin_settle_s`.
        pass_start: The source starts its pass; the baseline ends.
        cross: The source is over the victim.
        pass_end: The source ends its pass.
        spin_off: The motors stop, `spin_after_s` after the pass.
        leg_end: The leg ends.
    """

    leg: int
    tare_start: float
    arm_at: float
    spin_on: float
    baseline_start: float
    pass_start: float
    cross: float
    pass_end: float
    spin_off: float
    leg_end: float

    @property
    def spin_time(self) -> float:
        """Seconds the motors run in this leg, the arming included."""
        return self.spin_off - self.arm_at


def spin_schedule(source: Timeline, rows: Sequence[Mapping], stand: Mapping) -> list[LegSpin]:
    """The victim's windows of every leg, checked against the leg timeline.

    The tare, the arming, the settling and the baseline all fall before the pass, the motors stop
    within the leg, and each tare starts after the previous leg's motors stopped, with
    `rest_gap_s` for the battery to recover before it is read at rest.

    Args:
        source: The source's timeline of the flight, see `timeline.build_flight`.
        rows: The design rows of the flight.
        stand: The `[stand]` config.

    Returns:
        The windows of the legs in flight order.
    """
    schedule = []
    previous_off = -math.inf
    lead = stand["arm_lead_s"] + stand["spin_settle_s"] + stand["baseline_s"]
    for row in sorted(rows, key=lambda r: int(r["index_in_flight"])):
        leg = int(row["leg"])
        times = source.leg_times(leg)
        spin_on = times["pass_start"] - stand["spin_settle_s"] - stand["baseline_s"]
        arm_at = spin_on - stand["arm_lead_s"]
        tare_start = arm_at - stand["tare_s"]
        spin_off = times["pass_end"] + stand["spin_after_s"]
        if tare_start < 0.0:
            raise ValueError(f"Leg {leg}: the tare would start {-tare_start:.2f} s before t = 0")
        if tare_start < previous_off + stand["rest_gap_s"]:
            raise ValueError(
                f"Leg {leg}: the tare starts {tare_start - previous_off:.2f} s after the last spin "
                f"stopped, less than rest_gap_s = {stand['rest_gap_s']} s"
            )
        if times["pass_start"] - times["start"] < lead and schedule:
            logger.info(f"Leg {leg}: the spin-up starts while the source still approaches")
        if spin_off > times["end"]:
            raise ValueError(
                f"Leg {leg}: the motors would run {spin_off - times['end']:.2f} s past it"
            )
        schedule.append(
            LegSpin(
                leg=leg,
                tare_start=tare_start,
                arm_at=arm_at,
                spin_on=spin_on,
                baseline_start=spin_on + stand["spin_settle_s"],
                pass_start=times["pass_start"],
                cross=times["cross"],
                pass_end=times["pass_end"],
                spin_off=spin_off,
                leg_end=times["end"],
            )
        )
        previous_off = spin_off
    return schedule


def battery_budget(schedule: Sequence[LegSpin], stand: Mapping) -> dict:
    """Spin seconds per leg and how many legs one battery carries, `battery_spin_s` of hover."""
    per_leg = [leg.spin_time for leg in schedule]
    longest = max(per_leg)
    per_battery = int(stand["battery_spin_s"] // longest)
    return {
        "spin_s_per_leg": {leg.leg: round(leg.spin_time, 2) for leg in schedule},
        "spin_s_total": round(sum(per_leg), 2),
        "battery_spin_s": stand["battery_spin_s"],
        "legs_per_battery": per_battery,
        "batteries": math.ceil(len(schedule) / per_battery) if per_battery else math.inf,
    }


def landing_waypoints(pos: ArrayLike, stand_pos: ArrayLike, cfg: Mapping) -> list[NDArray]:
    """Where the source flies before it lands, so that it never comes down onto the stand.

    Outside the clearance it lands in place. Within it, it first climbs to the transfer height,
    the largest height separation above the victim (capped below the ceiling), and then flies
    sideways into the return lane, outside the wake footprint and the clearance, to land there.

    Args:
        pos: The source's position as (3,).
        stand_pos: The victim's position on the stand as (3,).
        cfg: The stand's config, see `stand_flight`.

    Returns:
        The points to fly through before landing, none when it lands in place.
    """
    pos, stand_pos = np.asarray(pos, dtype=float), np.asarray(stand_pos, dtype=float)
    if np.linalg.norm(pos[:2] - stand_pos[:2]) >= cfg["stand"]["clearance"]:
        return []
    ceiling = cfg["safety"]["arena_max"][2] - CEILING_MARGIN
    z = min(max(pos[2], stand_pos[2] + max(cfg["stand"]["height_seps"])), ceiling)
    lane = np.array([pos[0], stand_pos[1] + cfg["a3"]["return_lane"], z])
    return [np.array([pos[0], pos[1], z]), lane]


@dataclass
class LegOutcome:
    """What happened to the victim in one leg.

    Attributes:
        leg: Global leg index.
        spun: Whether the motors ran through the pass.
        vbat_rest: Battery voltage at rest at the end of the tare, None without a reading.
        vbat_min_load: Lowest voltage under load during the spin.
        reason: Why the leg was not spun or the spin stopped early, None otherwise.
        events: Wall time of every event of the leg, by name.
    """

    leg: int
    spun: bool = False
    vbat_rest: float | None = None
    vbat_min_load: float | None = None
    reason: str | None = None
    events: dict[str, float] = field(default_factory=dict)


class VictimSchedule:
    """Drives the victim along the spin schedule of a flight from a thread of its own.

    The source's control loop must not wait on the victim's radio, whose arming and parameter
    writes take up to seconds. `run_flight` calls `start` with its loop's start time, and the
    thread follows the schedule on the same `time.perf_counter` clock. In a still run it only
    stamps the windows, so both modes are cut alike.
    """

    def __init__(
        self,
        victim: SpinVictim,
        schedule: Sequence[LegSpin],
        stand: Mapping,
        spin: bool,
        clock: Callable[[], float] = time.perf_counter,
        wall: Callable[[], float] = time.time,
    ):
        """Prepare the thread, which starts with `start`.

        Args:
            victim: The victim on the stand.
            schedule: Its windows per leg, see `spin_schedule`.
            stand: The `[stand]` config, with `hover_cmd` and the voltage limits.
            spin: Run the motors, or only stamp the windows.
            clock: Monotonic clock of the source's loop.
            wall: Wall clock of the recorded events.
        """
        self.victim, self.schedule, self.stand, self.spin = victim, list(schedule), stand, spin
        self.clock, self.wall = clock, wall
        self.outcomes = {leg.leg: LegOutcome(leg.leg) for leg in self.schedule}
        self.battery_low = False
        self.error: str | None = None
        self._t0: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, t0: float):
        """Follow the schedule, timeline time zero at `t0` on `clock`."""
        self._t0 = t0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """End the schedule and stop the motors if they run. Never raises."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.victim.spinning:
            self.victim.stop_spin()

    def refly(self) -> list[int]:
        """Legs that were to spin but did not spin through their pass."""
        return [leg for leg, outcome in self.outcomes.items() if self.spin and not outcome.spun]

    def report(self) -> dict:
        """The outcomes for meta.json."""
        return {
            "spin": self.spin,
            "battery_low": self.battery_low,
            "error": self.error,
            "refly_legs": self.refly(),
            "legs": {leg: asdict(outcome) for leg, outcome in self.outcomes.items()},
            "schedule": [asdict(leg) for leg in self.schedule],
        }

    def _now(self) -> float:
        return self.clock() - self._t0

    def _wait_until(self, t: float) -> bool:
        """Wait for timeline time `t`; False when the schedule was stopped first."""
        while (remaining := t - self._now()) > 0:
            if self._stop.wait(min(remaining, POLL)):
                return False
        return not self._stop.is_set()

    def _stamp(self, outcome: LegOutcome, name: str):
        outcome.events[name] = self.wall()

    def _run(self):
        finished = False
        try:
            for leg in self.schedule:
                outcome = self.outcomes[leg.leg]
                if not self._wait_until(leg.tare_start):
                    break
                self._stamp(outcome, "tare_start")
                if not self._wait_until(leg.arm_at):
                    break
                self._stamp(outcome, "tare_end")
                self._leg(leg, outcome)
                if self._stop.is_set():
                    break
            finished = True
        finally:
            if not finished:  # An arming that failed raises in this thread; the source flies on
                self.error = "the victim's schedule ended on an exception, see the log above"
                logger.error(f"Victim schedule: {self.error}; the source flies on")

    def _leg(self, leg: LegSpin, outcome: LegOutcome):
        rest = self.victim.radio_log.recent("pm.vbat", self.stand["tare_s"])
        outcome.vbat_rest = float(np.mean(rest)) if len(rest) else None
        if self.spin and not self.battery_low:
            if outcome.vbat_rest is None or outcome.vbat_rest < self.stand["vbat_rest_min"]:
                self.battery_low = True
                logger.warning(
                    f"Victim battery {outcome.vbat_rest} V at rest before leg {leg.leg}, below "
                    f"{self.stand['vbat_rest_min']} V: the remaining legs fly without spinning"
                )
        if not self.spin or self.battery_low:
            outcome.reason = "battery low at rest" if self.spin else None
            if self._wait_until(leg.spin_on):  # The still run's windows, motors off
                self._stamp(outcome, "spin_on")
            if self._wait_until(leg.spin_off):
                self._stamp(outcome, "spin_off")
            return
        self.victim.spin(self.stand["hover_cmd"])
        self._stamp(outcome, "spin_on")
        lowest = math.inf
        while self._now() < leg.spin_off:
            if self._stop.wait(POLL):
                break
            recent = self.victim.radio_log.recent("pm.vbat", VBAT_RECENT)
            if len(recent):
                lowest = min(lowest, float(np.min(recent)))
            if lowest < self.stand["vbat_load_min"]:
                outcome.reason = f"battery {lowest:.2f} V under load"
                logger.warning(f"Leg {leg.leg}: victim battery {lowest:.2f} V, its motors stop")
                break
        self._stamp(outcome, "spin_off")  # The stop goes out now; stop_spin returns ~1 s later
        self.victim.stop_spin()
        self._stamp(outcome, "motors_stopped")
        outcome.vbat_min_load = None if math.isinf(lowest) else lowest
        outcome.spun = outcome.reason is None and not self._stop.is_set()


def a1_curve(path: Path = A1_CURVE_PATH) -> tuple[NDArray, str]:
    """cf10's rotor thrust curve from A1, the fit its decision chose, and which one that is."""
    result = tomllib.loads(path.read_text())
    fit = result["decision"]["fit"]
    return np.asarray(result[fit]["rpm2thrust"], dtype=float), fit
