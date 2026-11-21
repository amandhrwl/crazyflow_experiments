"""What the scripts of block A1 share: config, crazyflow's thrust curve, the limits and the plan.

The victim sits upright on the load cell and its motors are driven through fixed `motorPowerSet`
commands. A block is planned as a list of timed `Step`s: tares with the motors off, the sync
pulses, and the levels at which the load cell and the rpm are averaged. The thrust a command makes
is not known before it has run, so `ThrustPredictor` starts from the linear thrust map of
`sysid/config_drones.toml` and switches to the measured levels as they come in, and the sweep
skips every level whose predicted thrust exceeds `thrust_cap`.

Nothing here talks to a drone, so the plan, the limits and the dry run work without one.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.control.config import drone_params
from crazyflow_experiments.downwash import A1_CONFIG_PATH

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

N_MOTORS = 4
RPM_INVALID = 65535  # motor.mX_rpm of a motor whose DShot telemetry did not answer
DSHOT_STOP = 1504  # Commands below this become DShot 0, the motor stops (ratio >> 5 < 47)
SUPPORTED_CASES = ("S", "R1")
BLOCKS = ("all", "single", "bench")
# Motor-on steps; tares are motors off, and the runner arms and disarms around them
MOTOR_KINDS = ("idle", "pulse", "ramp", "center", "level")
BENCH_CMD = 10000  # B5, props off
BENCH_HOLD = 3.0  # s
EXTRAPOLATION_MARGIN = 1.1  # Predicted thrust beyond the measured levels is raised by this


@dataclass(frozen=True)
class Step:
    """One timed step of a block.

    Attributes:
        kind: "tare" (motors off), "idle", "pulse", "ramp", "center" or "level".
        duration: Length in s.
        commands: motorPowerSet command of each motor, None while the motors are off.
        level_id: Index of the command in `cmd_levels` (or of the single-motor level), -1 if none.
        motor: The driven motor of the single-motor block, 1 to 4, 0 for all four alike.
        repeat: Which randomized pass through the levels, 0 outside them.
    """

    kind: str
    duration: float
    commands: tuple[int, int, int, int] | None = None
    level_id: int = -1
    motor: int = 0
    repeat: int = 0

    @property
    def motors_on(self) -> bool:
        """Whether the motors run in this step."""
        return self.commands is not None

    def row(self) -> dict:
        """The step as a flat dict for events.csv and meta.json."""
        out = asdict(self)
        commands = out.pop("commands") or (0,) * N_MOTORS
        return out | {f"m{i + 1}": int(c) for i, c in enumerate(commands)}


def load_config(path: Path = A1_CONFIG_PATH) -> dict:
    """Read and check `a1_config.toml`."""
    config = tomllib.loads(Path(path).read_text())
    if config["case"] not in SUPPORTED_CASES:
        raise ValueError(
            f"case {config['case']!r}: only {SUPPORTED_CASES} apply to this stand, where the "
            "victim reads the load cell itself (R2 and P need a second device or a PC amplifier)"
        )
    if config["load_cell"]["force_sign"] not in (1, -1):
        raise ValueError("force_sign must be +1 or -1")
    if config["victim"]["mount"] not in ("upright", "upside_down"):
        raise ValueError(f"Unknown mount {config['victim']['mount']!r}")
    levels = config["sweep"]["cmd_levels"]
    if sorted(levels) != levels or min(levels) < DSHOT_STOP or max(levels) > 65535:
        raise ValueError(f"cmd_levels must rise within [{DSHOT_STOP}, 65535]: {levels}")
    if config["sweep"]["discard_s"] >= config["sweep"]["hold_s"]:
        raise ValueError("discard_s must be shorter than hold_s, or no window is left")
    return config


def crazyflow_rpm2thrust(config: dict) -> NDArray:
    """Crazyflow's rpm2thrust of the victim's model, checked against the config.

    The config holds the coefficients the comparison was planned with; a crazyflow update that
    changes them has to be noticed rather than silently compared against.
    """
    from crazyflow.dynamics import load_params  # JAX, only where the curve is needed

    model = config["victim"]["model"]
    coef = np.asarray(load_params("first_principles", model)["rpm2thrust"], dtype=float)
    coef = coef.reshape(-1)
    if not np.allclose(coef, config["crazyflow"]["rpm2thrust"], rtol=1e-9, atol=0.0):
        raise ValueError(
            f"crazyflow's rpm2thrust of {model} is {coef.tolist()}, the config has "
            f"{config['crazyflow']['rpm2thrust']}; update [crazyflow] knowingly"
        )
    return coef


def rotor_thrust(rpm: ArrayLike, coef: ArrayLike) -> NDArray:
    """Thrust of one rotor in N at `rpm`, f = c0 + c1 rpm + c2 rpm^2."""
    c0, c1, c2 = np.asarray(coef, dtype=float)
    rpm = np.asarray(rpm, dtype=float)
    return c0 + c1 * rpm + c2 * rpm**2


def rotor_rpm(thrust: ArrayLike, coef: ArrayLike) -> NDArray:
    """Rpm at which one rotor makes `thrust` in N, the positive root of the curve."""
    c0, c1, c2 = np.asarray(coef, dtype=float)
    thrust = np.asarray(thrust, dtype=float)
    return (-c1 + np.sqrt(c1**2 - 4 * c2 * (c0 - thrust))) / (2 * c2)


def victim_mass(config: dict, dry_run: bool = False) -> float:
    """Weighed mass of the victim in kg; a dry run falls back to the model's mass."""
    if (mass := config["victim"]["mass_kg"]) > 0:
        return float(mass)
    model_mass = float(drone_params(config["victim"]["model"])["mass"])
    if not dry_run:
        raise ValueError(
            "[victim] mass_kg is not filled in; weigh the victim as it sits on the cell"
        )
    logger.warning(f"mass_kg not filled in, planning with the model's {model_mass * 1e3:.1f} g")
    return model_mass


def check_limits(config: dict):
    """Refuse a clamped run whose load cell rating is not filled in.

    A clamped victim pulls on the cell with up to `max_thrust_factor_hover` times its weight, more
    than the 0-50 g the cell was calibrated over, so the rating has to bound the top levels.
    """
    if config["victim"]["clamped"] and config["load_cell"]["capacity_g"] <= 0:
        raise ValueError(
            "[load_cell] capacity_g is not filled in; with the victim clamped the rating of the "
            "cell bounds the top levels"
        )


def thrust_cap(config: dict, mass: float) -> tuple[float, str]:
    """Largest total thrust in N the sweep may reach, and what sets it.

    A victim that only rests on the load cell stays below `resting_thrust_fraction` of its weight,
    so it never lifts off; a clamped one below `max_thrust_factor_hover` times its weight. The
    load cell's rating caps both once it is filled in.
    """
    limits, g = config["limits"], config["g"]
    if config["victim"]["clamped"]:
        caps = [(limits["max_thrust_factor_hover"] * mass * g, "2 x hover, clamped victim")]
    else:
        caps = [(limits["resting_thrust_fraction"] * mass * g, "victim resting on the cell")]
    if (capacity := config["load_cell"]["capacity_g"]) > 0:
        cap = limits["max_thrust_frac_capacity"] * capacity * 1e-3 * g
        caps.append((cap, "load cell capacity"))
    return min(caps)


class ThrustPredictor:
    """Total thrust of the victim at a set of motor commands, before they are run.

    Until a command has been measured, the linear thrust map of `deploy.crazyflie.Drone`
    (thrust_max at pwm_max) predicts it, as for the spinning load cell test. Within the measured
    commands the measured per-motor thrust is interpolated. Beyond them the largest of three
    predictions counts: the linear map, a parabola through the measured commands, and the highest
    measured thrust scaled with the square of the command. The last bounds the thrust as long as
    the rpm rise at most in proportion to the command, and it already holds after one level; the
    two extrapolations are raised by `EXTRAPOLATION_MARGIN`.
    """

    def __init__(self, model: str):
        """Start from the linear thrust map of the model.

        Args:
            model: Drone model in `sysid/config_drones.toml`.
        """
        params = drone_params(model)
        self.thrust_max, self.pwm_max = float(params["thrust_max"]), float(params["pwm_max"])
        self.measured: dict[int, float] = {}  # Command of all four motors -> thrust per motor

    def add(self, command: int, total_thrust: float):
        """Record the measured total thrust with all four motors at `command`."""
        self.measured[int(command)] = float(total_thrust) / N_MOTORS

    def linear(self, command: ArrayLike) -> NDArray:
        """Per-motor thrust of the linear map in N."""
        return np.asarray(command, dtype=float) / self.pwm_max * self.thrust_max

    def per_motor(self, command: ArrayLike) -> NDArray:
        """Predicted thrust of one motor at each command in N."""
        command = np.atleast_1d(np.asarray(command, dtype=float))
        linear = self.linear(command)
        if not self.measured:
            return linear
        cmds = np.array(sorted(self.measured), dtype=float)
        thrust = np.array([self.measured[int(c)] for c in cmds])
        inside = np.interp(command, cmds, thrust)
        beyond = thrust[-1] * (command / cmds[-1]) ** 2
        if len(cmds) >= 2:
            fit = np.polyval(np.polyfit(cmds, thrust, min(len(cmds) - 1, 2)), command)
            beyond = np.maximum(beyond, fit)
        beyond = np.maximum(EXTRAPOLATION_MARGIN * beyond, linear)
        return np.where(command <= cmds.max(), inside, beyond)

    def total(self, commands: Sequence[int]) -> float:
        """Predicted total thrust of the four motors at their commands in N."""
        return float(self.per_motor(np.asarray(commands)).sum())


def uniform(command: int) -> tuple[int, int, int, int]:
    """The same command for all four motors."""
    return (int(command),) * N_MOTORS


def plan_all(config: dict) -> list[Step]:
    """Steps of block `all`, every motor at the same command.

    Tare, sync pulse, an ascending ramp through the levels (which finds how far the limits allow
    and gives the up-steps), the center level, then `repeats` seeded random orders of the levels
    with the center level after every `center_every` of them and a tare between the repeats, and
    tare, sync pulse, tare at the end. The runner skips any level its limits forbid.
    """
    sweep, pulse = config["sweep"], config["sync_pulse"]
    levels, hold = sweep["cmd_levels"], sweep["hold_s"]
    center = uniform(sweep["center_cmd"] or levels[len(levels) // 2])
    rng = np.random.default_rng(sweep["seed"])
    tare = Step("tare", sweep["tare_s"])

    def pulse_steps() -> list[Step]:
        idle = Step("idle", sweep["idle_s"], uniform(sweep["idle_cmd"]))
        return [idle, Step("pulse", pulse["duration_s"], uniform(pulse["cmd"])), idle]

    steps = [tare, *pulse_steps()]
    steps += [Step("ramp", hold, uniform(c), level_id=i) for i, c in enumerate(levels)]
    steps.append(Step("center", hold, center))
    for repeat in range(1, sweep["repeats"] + 1):
        if repeat > 1:
            steps.append(tare)  # The tare of the middle of the block
        for k, i in enumerate(rng.permutation(len(levels))):
            steps.append(Step("level", hold, uniform(levels[i]), level_id=int(i), repeat=repeat))
            if (k + 1) % sweep["center_every"] == 0:
                steps.append(Step("center", hold, center, repeat=repeat))
    return [*steps, tare, *pulse_steps(), tare]


def single_levels(config: dict, mass: float) -> list[int]:
    """Commands of one motor in the single-motor block, from above idle to `single_max_cmd`.

    Without `single_max_cmd` the end is the command at which the linear thrust map gives
    1.5 x hover thrust per rotor.
    """
    sweep = config["sweep"]
    end = sweep["single_max_cmd"]
    if not end:
        params = drone_params(config["victim"]["model"])
        end = 1.5 * mass * config["g"] / N_MOTORS / params["thrust_max"] * params["pwm_max"]
    end = min(int(end), 65535)
    if end <= sweep["idle_cmd"]:
        raise ValueError(f"single_max_cmd {end} is not above idle_cmd {sweep['idle_cmd']}")
    n = sweep["single_motor_levels"]
    return [int(round(c, -2)) for c in np.linspace(sweep["idle_cmd"], end, n + 1)[1:]]


def plan_single(config: dict, mass: float) -> list[Step]:
    """Steps of the single-motor block: each motor ramps alone while the others idle.

    A tare before each motor and one at the end; the other three motors idle at `idle_cmd`
    rather than stopping, see the config.
    """
    sweep = config["sweep"]
    tare, idle = Step("tare", sweep["tare_s"]), sweep["idle_cmd"]
    steps = []
    for motor in range(1, N_MOTORS + 1):
        steps.append(tare)
        steps.append(Step("idle", sweep["idle_s"], uniform(idle), motor=motor))
        for i, command in enumerate(single_levels(config, mass)):
            commands = tuple(command if m == motor else idle for m in range(1, N_MOTORS + 1))
            steps.append(Step("level", sweep["hold_s"], commands, level_id=i, motor=motor))
    return [*steps, tare]


def plan_bench(config: dict) -> list[Step]:
    """B5, props off: a short tare, all motors at `BENCH_CMD` for `BENCH_HOLD`, a tare."""
    return [Step("tare", 2.0), Step("level", BENCH_HOLD, uniform(BENCH_CMD)), Step("tare", 2.0)]


def plan(config: dict, block: str, mass: float) -> list[Step]:
    """The steps of a block."""
    if block == "all":
        return plan_all(config)
    if block == "single":
        return plan_single(config, mass)
    if block == "bench":
        return plan_bench(config)
    raise ValueError(f"Unknown block {block!r}, expected one of {BLOCKS}")


def schedule_report(
    config: dict, block: str, steps: list[Step], mass: float, coef: NDArray
) -> list[str]:
    """Lines of the dry run: every step with its predicted thrust and rpm, and the totals.

    The prediction is the linear thrust map, as the sweep has measured nothing yet; the rpm is
    where crazyflow's curve makes that thrust per rotor. Steps over the cap are marked, the sweep
    skips them unless the measured levels predict less.
    """
    g = config["g"]
    cap, reason = thrust_cap(config, mass)
    predictor = ThrustPredictor(config["victim"]["model"])
    hover = mass * g
    lines = [
        f"Block {block}, case {config['case']}, victim cf{config['victim']['id']} "
        f"{config['victim']['mount']}, {'clamped' if config['victim']['clamped'] else 'resting'}, "
        f"{mass * 1e3:.1f} g (hover {hover:.3f} N)",
        f"Thrust cap {cap:.3f} N = {cap / hover:.2f} x hover ({reason})",
        f"{'#':>3} {'kind':<7} {'s':>5} {'commands':<27} {'pred N':>7} {'x hover':>7} "
        f"{'rpm cf':>7}",
    ]
    for i, step in enumerate(steps):
        if not step.motors_on:
            lines.append(f"{i:>3} {step.kind:<7} {step.duration:5.1f} motors off")
            continue
        per_motor = predictor.per_motor(np.asarray(step.commands))
        total = float(per_motor.sum())
        rpm = rotor_rpm(per_motor.max(), coef)
        mark = "  over the cap, skipped unless measured lower" if total > cap else ""
        lines.append(
            f"{i:>3} {step.kind:<7} {step.duration:5.1f} {str(step.commands):<27} {total:7.3f} "
            f"{total / hover:7.2f} {rpm:7.0f}{mark}"
        )
    motor_time = sum(s.duration for s in steps if s.motors_on)
    lines.append(
        f"Total {sum(s.duration for s in steps) / 60:.1f} min, motors on {motor_time / 60:.1f} min "
        "(plus about 1 s per arming)"
    )
    if config["victim"]["clamped"] and config["load_cell"]["capacity_g"] <= 0:
        lines.append(
            "capacity_g not filled in: the cap above ignores the load cell's rating, and the "
            "sweep refuses to run until it is filled in"
        )
    if not config["victim"]["clamped"]:
        band_low = 0.6 * hover
        lines.append(
            f"Resting victim: the sweep stops at {cap / hover:.2f} x hover, so the band of the "
            f"decision (0.6-1.6 x hover per rotor) is only measured up to {cap / hover:.2f} x and "
            f"rho at hover is an extrapolation. Clamp the victim to measure it "
            f"(lower end of the band {band_low:.3f} N)."
        )
    return lines
