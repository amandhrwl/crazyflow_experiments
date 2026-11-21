"""Block A1 bench checks B1 to B4, props off: who reads the load cell, calibration, rate, card.

    pixi run -e sim2real python crazyflow_experiments/downwash/a1_probe.py              # B1
    pixi run -e sim2real python crazyflow_experiments/downwash/a1_probe.py --calibration-check
    pixi run -e sim2real python crazyflow_experiments/downwash/a1_probe.py --rate-check
    pixi run -e sim2real python crazyflow_experiments/downwash/a1_probe.py --sd-check 60
    pixi run -e sim2real python crazyflow_experiments/downwash/a1_probe.py --decode <card file>

B1 reboots and connects the victim as the sweep does, reads its decks, self-test, supervisor and
log TOC and names the case: S when the victim reads the load cell and its card logs, R1 when the
card does not work with the load cell deck stacked. B2 asks for known weights on the victim and
compares the load cell with them, at the sample rate the sweep uses (the calibration was taken at
10 SPS). B3 measures the conversion rate and the noise floor at 80 and 320 SPS. B4 logs the A1
card config for a while; `--decode` checks the file without a drone: CRC, variables, record
spacing (1978 µs instead of 2000 µs is the 85/84 timer of firmware before bitcraze #1684) and the
card stalls. B5, the motors with the props off, is `a1_sweep.py --block bench`.
Every result is appended to `results/downwash/a1/NOTES.md`.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from crazyflow_experiments.downwash import A1_CONFIG_PATH, A1_RESULTS_DIR, usd
from crazyflow_experiments.downwash.a1_stand import N_MOTORS, load_config
from crazyflow_experiments.downwash.a1_sweep import MOTOR_VARIABLES, SD_CONFIG
from crazyflow_experiments.downwash.radio import Victim
from crazyflow_experiments.downwash.usd import RPM_INVALID

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("crazyflow_experiments.downwash.a1_probe")  # Not __name__, __main__
logging.getLogger("crazyflow_experiments").setLevel(logging.INFO)

CALIBRATION_WEIGHTS = (10.0, 20.0, 50.0)  # g
CALIBRATION_TOL = 0.3  # g
MEASURE_TIME = 3.0  # s averaged per weight
RATE_SETTLE = 2.0  # s after a sample rate change
RATE_MEASURE = 10.0  # s per sample rate
RATES = {3: 80.0, 7: 320.0}  # loadcell.sampleRate -> SPS
RATE_TOL = 0.1  # Relative deviation of the measured rate
SPACING = {2000: "fixed card timer", 1978: "85/84 card timer, firmware before bitcraze #1684"}
CARD_GAP = 0.010  # s
DECK_PARAMS = ("deck.bcLoadcell", "deck.bcUSD", "usd.canLog", "deck.bcRpm", "deck.bcACS37800")


def note(lines: list[str]):
    """Append lines under a dated heading to NOTES.md and log them."""
    for line in lines:
        logger.info(line)
    path = A1_RESULTS_DIR / "NOTES.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(f"\n### Probe {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("".join(f"- {line}\n" for line in lines))


def param(victim: Victim, name: str) -> float | None:
    """A parameter's value, None if the firmware does not have it."""
    if victim.cf.param.toc.get_element_by_complete_name(name) is None:
        return None
    return float(victim.cf.param.get_value(name))


def probe(victim: Victim) -> list[str]:
    """B1: decks, self-test, supervisor and TOC of the victim, and which case applies."""
    values = {name: param(victim, name) for name in (*DECK_PARAMS, "system.selftestPassed")}
    lines = [", ".join(f"{k} {v if v is None else int(v)}" for k, v in values.items())]
    toc = victim.cf.log.toc
    card = usd.read_config(SD_CONFIG).variables
    missing = [v for v in (*card, *MOTOR_VARIABLES) if toc.get_element_by_complete_name(v) is None]
    lines.append(f"log TOC: {'all A1 variables present' if not missing else f'missing {missing}'}")
    lines.append(
        "usd.fileDate: "
        + (
            "present, card files named after the run"
            if param(victim, "usd.fileDate") is not None
            else "missing, the card numbers its files"
        )
    )
    info = int(victim.radio_log.latest("supervisor.info"))
    lines.append(
        f"supervisor.info {info:#06x}, load cell at "
        f"{victim.radio_log.latest('loadcell.rate'):.0f} Hz"
    )
    if values["deck.bcLoadcell"] != 1:
        case = "none: the victim does not read the load cell, stop and ask (spec section 1)"
    elif values["deck.bcUSD"] == 1 and values["usd.canLog"] == 1:
        case = "S: the victim reads the load cell and its card logs"
    else:
        case = (
            "R1: the victim reads the load cell but its card does not log; reseat the SD deck "
            "with the drone off and power it with the button before settling on R1"
        )
    if values["system.selftestPassed"] != 1:
        lines.append("self-test FAILED: the victim cannot arm, check the console and the buffer")
    rpm = [victim.radio_log.latest(f"motor.m{i}_rpm") for i in range(1, N_MOTORS + 1)]
    lines.append(
        f"rpm at rest {rpm} ({'65535: no DShot telemetry' if RPM_INVALID in rpm else 'ok'})"
    )
    lines.append(f"case {case}")
    return lines


def calibration_check(victim: Victim, config: dict) -> list[str]:
    """B2: known weights on the victim against the load cell, at the sweep's sample rate."""
    sign = config["load_cell"]["force_sign"]

    def reading() -> float:
        time.sleep(MEASURE_TIME)
        return float(victim.radio_log.recent("loadcell.weight", MEASURE_TIME).mean())

    input("Nothing on the victim, then press Enter ")
    zero = reading()
    lines = [f"zero {zero:+.3f} g (the calibration's offset holds if near 0)"]
    for weight in CALIBRATION_WEIGHTS:
        input(f"Put {weight:.0f} g on the victim, then press Enter ")
        # A weight pushes the victim down, which the drone-on-top calibration reads as negative
        delta = sign * (reading() - zero)
        error = delta + weight
        verdict = "ok" if abs(error) <= CALIBRATION_TOL else "FAIL, recalibrate with calibscale.py"
        lines.append(f"{weight:.0f} g reads {delta:+.3f} g, off by {error:+.3f} g: {verdict}")
    return lines


def rate_check(victim: Victim, config: dict) -> list[str]:
    """B3: conversion rate and noise floor per sample rate, then back to the config's rate."""
    lines = []
    for setting, sps in RATES.items():
        usd.set_param(victim.cf, "loadcell.sampleRate", setting)
        time.sleep(RATE_SETTLE)
        time.sleep(RATE_MEASURE)
        rate = float(victim.radio_log.recent("loadcell.rate", RATE_MEASURE).mean())
        weight = victim.radio_log.recent("loadcell.weight", RATE_MEASURE)
        per_second = np.array_split(weight, int(RATE_MEASURE))
        noise = float(np.median([w.std() for w in per_second if len(w) > 2]))
        verdict = "ok" if abs(rate / sps - 1) <= RATE_TOL else "FAIL"
        lines.append(
            f"sampleRate {setting}: {rate:.0f} Hz of {sps:.0f} ({verdict}), noise {noise:.4f} g "
            "std over 1 s"
        )
    usd.set_param(victim.cf, "loadcell.sampleRate", config["load_cell"]["sample_rate"])
    lines.append(
        f"back at sampleRate {config['load_cell']['sample_rate']}; switch the config to 7 only if "
        "320 SPS reached its rate without more noise per averaged window"
    )
    return lines


def sd_check(victim: Victim, seconds: float) -> list[str]:
    """B4: log the A1 card config for `seconds`; the file is checked with `--decode`."""
    usd.check_variables(victim.cf, victim.name, usd.read_config(SD_CONFIG).variables)
    usd.start_logging(victim.cf, victim.name, datetime.now())
    logger.info(f"Logging {seconds:.0f} s; press the load cell lightly once meanwhile")
    time.sleep(seconds)
    usd.stop_logging(victim.cf, victim.name)
    return [
        f"card logged {seconds:.0f} s to {victim.console.files()}, "
        f"written {victim.console.written()}",
        "copy the file off the card and check it with --decode",
    ]


def decode_check(path: Path) -> list[str]:
    """B4 without a drone: CRC, variables, record spacing, stalls and the load cell's changes."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sd = usd.decode(path)
    crc = "CRC mismatch" if any("CRC" in str(w.message) for w in caught) else "CRC ok"
    expected = usd.read_config(SD_CONFIG).variables
    variables = [k for k in sd if k != "t_us"]
    spacing = float(np.median(np.diff(sd["t_us"])))
    timer = SPACING.get(round(spacing), f"unexpected spacing {spacing:.0f} µs")
    t = sd["t_us"] / 1e6
    gaps = np.diff(t)[np.diff(t) > CARD_GAP]
    raw = sd.get("loadcell.rawWeight", np.empty(0))
    lines = [
        f"{Path(path).name}: {crc}, {len(t)} records over {t[-1] - t[0]:.1f} s",
        "variables " + (f"as {SD_CONFIG.name}" if variables == list(expected) else f"{variables}"),
        f"record spacing {spacing:.0f} µs: {timer}",
        f"{len(gaps)} card gaps over 10 ms, longest {gaps.max() * 1e3 if len(gaps) else 0:.0f} ms",
    ]
    if len(raw):
        lines.append(f"load cell: {len(np.unique(raw))} distinct raw values, range {np.ptp(raw)}")
    return lines


def main(args: argparse.Namespace):
    """Run the checks the arguments ask for."""
    if args.decode is not None:
        note(decode_check(args.decode))
        return
    config = load_config(A1_CONFIG_PATH)
    victim_cfg = config["victim"]
    victim = Victim(
        victim_cfg["id"],
        victim_cfg["channel"],
        victim_cfg["radio"],
        config["load_cell"],
        extra_blocks=[("motors", MOTOR_VARIABLES, config["load_cell"]["radio_period_ms"])],
    )
    try:
        time.sleep(1.0)  # The first status packet
        lines = probe(victim)
        if args.calibration_check:
            lines += calibration_check(victim, config)
        if args.rate_check:
            lines += rate_check(victim, config)
        if args.sd_check:
            lines += sd_check(victim, args.sd_check)
    finally:
        victim.close()
    note(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block A1 bench checks B1-B4")
    parser.add_argument("--calibration-check", action="store_true", help="B2: known weights")
    parser.add_argument("--rate-check", action="store_true", help="B3: 80 and 320 SPS")
    parser.add_argument("--sd-check", type=float, default=0.0, help="B4: seconds of card logging")
    parser.add_argument("--decode", type=Path, help="B4: check a card file, no drone")
    main(parser.parse_args())
