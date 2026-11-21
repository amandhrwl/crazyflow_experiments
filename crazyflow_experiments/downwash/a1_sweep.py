"""Block A1: drive the motors of the victim on the load cell through fixed commands.

The victim is rebooted and connected as in the load cell tests (`load_cell.radio.Victim`, not the
`Drone` class: no motion capture, no estimator reset, no spin-up at connect), its load cell
calibration and sample rate are set, its card logs everything at 500 Hz (case S) and the radio
streams the load cell and the motors at 100 Hz for the live screen and the limits. Each block is
planned in `a1_stand.py`; the runner arms the victim for every stretch of motor steps and stops
it for every tare.

Safety. `motorPowerSet` overrides the motor ratio even in the firmware's `motorsStop`, so neither a
disarm nor the supervisor's watchdogs stop the motors (motors.c, `motorsSetRatio`); only zero
commands sent over the radio do. The script stops them in `finally`, but a lost link or a killed
script leaves them at their last command: keep the power switch of the victim (bench supply)
within reach. The victim is still armed, for the arming rpm check of all four motors, and fed zero
setpoints so its supervisor neither levels out nor locks. The victim is clamped to the load cell,
so the thrust may reach 2 x its weight, bounded by the cell's rating (`[limits]`).

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/a1_sweep.py --block all [--dry-run]
    pixi run -e sim2real python crazyflow_experiments/downwash/a1_sweep.py --block single \
        --from results/downwash/a1/all_<timestamp>
    pixi run -e sim2real python crazyflow_experiments/downwash/a1_sweep.py --block bench [--unarmed]

`--block bench` is B5 with the props off. Output: `results/downwash/a1/<block>_<timestamp>/` with
`meta.json`, `events.csv` and `radio.npz`; copy the card file in with `a1_convert.py --collect`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crazyflow_experiments.downwash import A1_CONFIG_PATH, A1_RESULTS_DIR, SD_CONFIG_DIR
from crazyflow_experiments.downwash.a1_stand import (
    BLOCKS,
    DSHOT_STOP,
    N_MOTORS,
    RPM_INVALID,
    Step,
    ThrustPredictor,
    check_limits,
    crazyflow_rpm2thrust,
    load_config,
    plan,
    rotor_thrust,
    schedule_report,
    thrust_cap,
    uniform,
    victim_mass,
)
from crazyflow_experiments.load_cell import usd
from crazyflow_experiments.load_cell.radio import Victim
from crazyflow_experiments.utils.paths import REPO_ROOT, new_recording_path

if TYPE_CHECKING:
    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("crazyflow_experiments.downwash.a1_sweep")  # Not __name__, __main__
logging.getLogger("crazyflow_experiments").setLevel(logging.INFO)

SD_CONFIG = SD_CONFIG_DIR / "config_a1.txt"
MOTOR_VARIABLES = ("motor.m1", "motor.m1_rpm", "motor.m2_rpm", "motor.m3_rpm", "motor.m4_rpm")
ARMED, LOCKED, CRASHED = 0x0002, 0x0040, 0x0080  # Bits of supervisor.info
LOOP_PERIOD = 0.05  # s between the checks of the limits
SCREEN_PERIOD = 0.25  # s between updates of the live line
LIMIT_WINDOW = 0.2  # s of load cell readings averaged for the thrust limit
FIRST_PACKETS_TIMEOUT = 5.0  # s for the radio blocks to deliver after the connect
SELFTEST_TIMEOUT = 2.0  # s


def git_state() -> dict[str, str | bool]:
    """Commit and whether the working tree differs from it, for meta.json."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def predictor_from_block(folder: Path, model: str) -> ThrustPredictor:
    """A thrust predictor seeded with the steady windows of a converted block `all`."""
    predictor = ThrustPredictor(model)
    steady = folder / "steady.csv"
    if not steady.is_file():
        raise FileNotFoundError(f"{steady} missing, convert the block first (a1_convert.py)")
    with open(steady) as f:
        for row in csv.DictReader(f):
            if int(row["motor"]) == 0 and row["ok"] == "True":
                predictor.add(int(row["m1"]), float(row["F_mean"]))
    return predictor


class Stand:
    """The victim on the load cell: its motors, the live thrust and the limits."""

    def __init__(self, victim: Victim, config: dict, cap: float, armed: bool = True):
        """Wrap a connected victim.

        Args:
            victim: The connected victim, streaming the load cell, its status and its motors.
            config: The A1 config.
            cap: Largest total thrust in N, see `a1_stand.thrust_cap`.
            armed: Arm the victim for the motor steps. Only the props-off bench check runs
                unarmed, where the motors idle faster than the arming rpm window.
        """
        self.victim, self.config, self.cap, self.armed = victim, config, cap, armed
        self.g = config["g"]
        self.sign = config["load_cell"]["force_sign"]
        self.tare: float | None = None
        self.commands: tuple[int, int, int, int] | None = None
        self.link_lost = threading.Event()
        self._invalid_since: list[float | None] = [None] * N_MOTORS
        victim.cf.connection_lost.add_callback(self._on_link_lost)

    def _on_link_lost(self, link_uri: str, message: str):
        self.link_lost.set()
        logger.critical(
            "Victim link lost. Motors under motorPowerSet keep their last command without the "
            "radio: cut the victim's power now"
        )

    @property
    def on(self) -> bool:
        """Whether the motors were commanded and not stopped."""
        return self.commands is not None

    def set(self, commands: tuple[int, int, int, int]):
        """Run the motors at their commands, arming the victim first if they were off."""
        if self.armed:
            if not self.victim.spinning:
                self.victim.arm()
            uniform_command = len(set(commands)) == 1
            self.victim.set_motors(commands[0] if uniform_command else commands)
        else:
            for motor, command in enumerate(commands, start=1):
                usd.set_param(self.victim.cf, f"motorPowerSet.m{motor}", command)
            usd.set_param(self.victim.cf, "motorPowerSet.enable", 1)
        self.commands = commands
        self._invalid_since = [None] * N_MOTORS

    def stop(self):
        """Stop the motors; never raises, it also runs in the cleanup."""
        if self.armed and self.victim.spinning:
            self.victim.stop_spin()  # Disarm, zero commands, enable 0
        else:  # Zero commands are harmless, so they go out even if nothing seems to run
            for motor in range(1, N_MOTORS + 1):
                self.victim.cf.param.set_value(f"motorPowerSet.m{motor}", 0)
            self.victim.cf.param.set_value("motorPowerSet.enable", 0)
        self.commands = None

    def weight(self, seconds: float) -> float:
        """Mean load cell reading in g over the last `seconds` of the radio."""
        values = self.victim.radio_log.recent("loadcell.weight", seconds)
        if len(values) == 0:
            raise RuntimeError(f"No load cell reading over the radio in the last {seconds} s")
        return float(values.mean())

    def thrust(self, seconds: float = LIMIT_WINDOW) -> float:
        """Live force on the victim in N, up positive, relative to the last tare."""
        if self.tare is None:
            raise RuntimeError("No tare yet, every block starts with one")
        return self.sign * self.g * 1e-3 * (self.weight(seconds) - self.tare)

    def rpm(self) -> NDArray:
        """Latest rpm of the four motors, 65535 where the telemetry did not answer."""
        log = self.victim.radio_log
        return np.array([log.latest(f"motor.m{i}_rpm") for i in range(1, N_MOTORS + 1)])

    def violation(self) -> str | None:
        """Why the block has to stop now, None while all is well."""
        log, limits = self.victim.radio_log, self.config["limits"]
        if self.link_lost.is_set():
            return "radio link lost"
        if (vbat := float(log.latest("pm.vbat"))) < limits["vbat_min_abort"]:
            return f"battery {vbat:.2f} V below {limits['vbat_min_abort']} V"
        if not self.on:
            return None
        now = time.monotonic()
        for i, rpm in enumerate(self.rpm()):
            if self.commands[i] < DSHOT_STOP or rpm != RPM_INVALID:
                self._invalid_since[i] = None
            elif self._invalid_since[i] is None:
                self._invalid_since[i] = now
            elif now - self._invalid_since[i] > limits["rpm_invalid_s"]:
                return f"motor {i + 1} rpm invalid (65535) for {limits['rpm_invalid_s']} s"
        if (thrust := self.thrust()) > self.cap:
            return f"thrust {thrust:.3f} N over the cap of {self.cap:.3f} N"
        if self.armed:
            info = int(log.latest("supervisor.info"))
            if info & LOCKED or info & CRASHED:
                return f"supervisor locked or crashed, supervisor.info {info:#06x}"
            if not info & ARMED:
                return f"the supervisor disarmed the victim, supervisor.info {info:#06x}"
        return None

    def screen(self, index: int, n_steps: int, step: Step, remaining: float, coef: NDArray) -> str:
        """The live line: step, commands, rpm, thrust, crazyflow's thrust from the rpm, battery."""
        log = self.victim.radio_log
        rpm = self.rpm()
        valid = np.where(rpm == RPM_INVALID, np.nan, rpm)
        thrust = self.thrust() if self.tare is not None else float("nan")
        thrust_cf = float(np.nansum(rotor_thrust(valid, coef))) if self.on else 0.0
        commands = "off" if self.commands is None else "/".join(str(c) for c in self.commands)
        return (
            f"\r{index + 1:>3}/{n_steps} {step.kind:<6} {commands:<23} rpm "
            f"{' '.join(f'{r:5.0f}' for r in rpm)} | F {thrust * 1e3:7.1f} mN, cf "
            f"{thrust_cf * 1e3:7.1f} mN | {float(log.latest('pm.vbat')):.2f} V | "
            f"{remaining:4.1f} s "
        )


def hold(
    stand: Stand, step: Step, index: int, n_steps: int, coef: NDArray, stop: threading.Event
) -> str | None:
    """Wait out a step, checking the limits; returns why the block must stop, None if it ran out."""
    t_end = time.monotonic() + step.duration
    t_screen = 0.0
    while (now := time.monotonic()) < t_end:
        if stop.is_set():
            return "Ctrl+C"
        if reason := stand.violation():
            return reason
        if now - t_screen >= SCREEN_PERIOD:
            sys.stdout.write(stand.screen(index, n_steps, step, t_end - now, coef))
            sys.stdout.flush()
            t_screen = now
        time.sleep(LOOP_PERIOD)
    return None


def allowed(predictor: ThrustPredictor, cap: float, wanted: int) -> int | None:
    """The largest command up to `wanted`, for all four motors, with an allowed thrust."""
    for command in range(int(wanted), DSHOT_STOP, -500):
        if predictor.total(uniform(command)) <= cap:
            return command
    return None


def run_block(
    stand: Stand,
    steps: list[Step],
    predictor: ThrustPredictor,
    coef: NDArray,
    stop: threading.Event,
) -> tuple[list[dict], str | None]:
    """Run the steps of a block, skipping every motor step its limits forbid.

    The ramp of block `all` rises until the next level is predicted over the cap, and every level
    it measures teaches the predictor. After the ramp the center level becomes the configured one,
    or the middle of the allowed levels if that is over the cap.

    Returns:
        A row per step for events.csv, and why the block stopped early, None if it did not.
    """
    sweep = stand.config["sweep"]
    rows, abort, ramp_ended, center = [], None, False, None
    for index, step in enumerate(steps):
        if step.kind == "center":
            if center is None:  # The ramp found what is allowed
                center = allowed(predictor, stand.cap, step.commands[0])
                if center is not None and center != step.commands[0]:
                    logger.warning(f"Center level lowered to {center}, the limits forbid more")
            step = step if center is None else replace(step, commands=uniform(center))
        if step.kind == "pulse" and predictor.total(step.commands) > stand.cap:
            if (lower := allowed(predictor, stand.cap, step.commands[0])) is not None:
                logger.warning(f"Sync pulse lowered to {lower}, the limits forbid more")
                step = replace(step, commands=uniform(lower))
        predicted = predictor.total(step.commands) if step.motors_on else 0.0
        row = step.row() | {"step": index, "predicted_N": predicted, "measured_N": np.nan}
        skip = None
        if step.motors_on and (ramp_ended and step.kind == "ramp"):
            skip = "ramp ended below it"
        elif step.motors_on and predicted > stand.cap:
            skip = f"predicted {predicted:.3f} N over the cap"
            ramp_ended |= step.kind == "ramp"
        if skip is not None:
            rows.append(row | {"t_start": np.nan, "t_end": np.nan, "skipped": skip})
            continue
        if step.motors_on:
            stand.set(step.commands)
        elif stand.on:
            stand.stop()
        t_start = time.time()
        abort = hold(stand, step, index, len(steps), coef, stop)
        t_end = time.time()
        settled = min(step.duration - sweep["discard_s"], t_end - t_start - sweep["discard_s"])
        if not step.motors_on and settled > 0 and abort is None:
            stand.tare = stand.weight(settled)
        if step.motors_on and settled > 0 and stand.tare is not None:
            row["measured_N"] = stand.thrust(settled)
            if len(set(step.commands)) == 1 and step.kind != "pulse":
                predictor.add(step.commands[0], row["measured_N"])
        rows.append(row | {"t_start": t_start, "t_end": t_end, "skipped": ""})
        if abort is not None:
            break
    sys.stdout.write("\n")
    return rows, abort


def write_events(path: Path, rows: list[dict]):
    """Write one row per step."""
    columns = ["step", "kind", "level_id", "motor", "repeat", "m1", "m2", "m3", "m4"]
    columns += ["duration", "t_start", "t_end", "predicted_N", "measured_N", "skipped"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_victim(victim: Victim, config: dict, block: str):
    """Self-test, battery at rest and load cell rate before any motor runs."""
    cf = victim.cf
    t_end = time.monotonic() + FIRST_PACKETS_TIMEOUT
    while not all(
        len(victim.radio_log.arrays().get(f"{b}/t_pc", ())) for b in ("motors", "status")
    ):
        if time.monotonic() > t_end:
            raise RuntimeError(
                "The motors or status radio block delivered nothing, is the link up?"
            )
        time.sleep(0.1)
    # A drone whose boot self-test failed (e.g. an SD buffer the heap cannot hold) never arms
    if int(cf.param.get_value("system.selftestPassed")) != 1:
        raise RuntimeError(
            f"{victim.name}: system.selftestPassed is 0, it cannot arm. Check the console and the "
            "card's buffer (at most 8192 B), then power cycle it with the button"
        )
    if config["case"] == "S":
        usd.check_variables(cf, victim.name, usd.read_config(SD_CONFIG).variables)
    vbat = float(victim.radio_log.latest("pm.vbat"))
    if block != "bench" and vbat < config["limits"]["vbat_min_block"]:
        raise RuntimeError(f"Battery {vbat:.2f} V at rest, start a block at >= 3.7 V")
    rate = float(victim.radio_log.latest("loadcell.rate"))
    logger.info(f"{victim.name}: battery {vbat:.2f} V at rest, load cell at {rate:.0f} Hz")


def main(block: str, dry_run: bool, seed_from: Path | None, unarmed: bool):
    """Plan a block, and unless it is a dry run, run and record it.

    Args:
        block: "all", "single" or "bench".
        dry_run: Only print the plan and its limits, connect nothing.
        seed_from: A converted block `all` whose measured levels predict the thrust.
        unarmed: Run the bench block without arming (props off, see `Stand`).
    """
    config = load_config(A1_CONFIG_PATH)
    coef = crazyflow_rpm2thrust(config)
    mass = victim_mass(config, dry_run=dry_run)
    steps = plan(config, block, mass)
    for line in schedule_report(config, block, steps, mass, coef):
        logger.info(line)
    if dry_run:
        return
    check_limits(config)
    if unarmed and block != "bench":
        raise ValueError("--unarmed is for the props-off bench check only")
    model = config["victim"]["model"]
    if seed_from is None:
        predictor = ThrustPredictor(model)
    else:
        predictor = predictor_from_block(seed_from, model)
    cap, cap_reason = thrust_cap(config, mass)
    folder = new_recording_path(A1_RESULTS_DIR, block, suffix="")
    recorded = datetime.strptime(folder.name, f"{block}_%Y-%m-%d_%H-%M-%S")  # Names the card file
    folder.mkdir()
    victim_cfg, load_cell = config["victim"], config["load_cell"]
    period = load_cell["radio_period_ms"]
    victim = Victim(
        victim_cfg["id"],
        victim_cfg["channel"],
        victim_cfg["radio"],
        load_cell,
        extra_blocks=[("motors", MOTOR_VARIABLES, period)],
    )
    stand = Stand(victim, config, cap, armed=not unarmed)
    stop = threading.Event()

    def on_interrupt(signum: int, frame: object):
        stop.set()

    previous_handler = signal.signal(signal.SIGINT, on_interrupt)
    meta = {
        "config": config,
        "block": block,
        "case": config["case"],
        "git": git_state(),
        "mass_kg": mass,
        "thrust_cap": {"N": cap, "reason": cap_reason},
        "crazyflow_rpm2thrust": coef.tolist(),
        "armed": not unarmed,
        "seeded_from": None if seed_from is None else str(seed_from),
        "sd": {},
        "abort": None,
    }
    rows: list[dict] = []
    try:
        check_victim(victim, config, block)
        meta["calibration"] = {
            name: float(victim.cf.param.get_value(f"loadcell.{name}"))
            for name in ("a", "b", "sampleRate")
        }
        if config["case"] == "S":
            meta["sd"]["victim"] = {"start": usd.start_logging(victim.cf, victim.name, recorded)}
        rows, meta["abort"] = run_block(stand, steps, predictor, coef, stop)
    finally:
        stand.stop()
        signal.signal(signal.SIGINT, previous_handler)
        if "victim" in meta["sd"]:
            meta["sd"]["victim"]["stop"] = usd.stop_logging(victim.cf, victim.name)
        meta["sd"]["victim"] = meta["sd"].get("victim", {}) | {
            "files": victim.console.files(),
            "written": victim.console.written(),
        }
        meta["console"] = victim.console.lines
        victim.close()
        write_events(folder / "events.csv", rows)
        radio = {f"victim/{k}": v for k, v in victim.radio_log.arrays().items()}
        np.savez(folder / "radio.npz", **radio)
        with open(folder / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=lambda x: np.asarray(x).tolist())
        logger.info(f"Recording in {folder}")

    if meta["abort"] is not None:
        logger.warning(f"Stopped early: {meta['abort']}")
    skipped = [r for r in rows if r["skipped"]]
    if skipped:
        logger.info(
            f"{len(skipped)} steps skipped by the limits, the first: {skipped[0]['skipped']}"
        )
    for row in rows:
        if row["kind"] in ("ramp", "center", "level") and np.isfinite(row["measured_N"]):
            logger.info(f"{row['kind']:>6} {row['m1']:>5}...: {row['measured_N'] * 1e3:7.1f} mN")
    files = meta["sd"]["victim"]["files"]
    logger.info(
        f"Copy {files} of the victim card in with: a1_convert.py --collect <card> {folder.name}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block A1: motor sweep on the load cell")
    parser.add_argument("--block", choices=BLOCKS, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Plan and check, connect nothing")
    parser.add_argument(
        "--from", dest="seed_from", type=Path, help="Converted block all that predicts the thrust"
    )
    parser.add_argument(
        "--unarmed", action="store_true", help="Bench check props off without arming"
    )
    args = parser.parse_args()
    main(args.block, args.dry_run, args.seed_from, args.unarmed)
