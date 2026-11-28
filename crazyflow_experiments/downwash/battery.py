"""Battery voltage of the A3 drones at rest, read over the radio without rebooting them.

Connects to the victim and the source of `config.toml`, reads `pm.vbat` once and prints it against
the limits the runs check: the source takes off from `vbat_min_start` up, a spin run of the stand
starts from `[stand] vbat_rest_min` up. `--watch` repeats the reading until Ctrl+C. Read it with the
motors off: under load the voltage sags by 0.1 to 0.3 V. `--load-cell` also sets the victim's load
cell calibration (lost at every reboot) and prints the cell's reading in grams, the mean of
`LOAD_CELL_READINGS`, which is the victim's weight on the cell with the motors off.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/battery.py [--watch] [--load-cell]
"""

from __future__ import annotations

import argparse
import logging
import time
from types import SimpleNamespace

import cflib.crtp
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from crazyflow_experiments.downwash import usd
from crazyflow_experiments.downwash.a1_stand import load_config as load_a1_config
from crazyflow_experiments.downwash.a3_design import load_config
from crazyflow_experiments.downwash.radio import TOC_CACHE, uri

logger = logging.getLogger("crazyflow_experiments.downwash.battery")  # Not __name__, the __main__
logger.setLevel(logging.INFO)  # deploy.crazyflie configures the root logger at WARNING on import
WATCH_PERIOD = 2.0  # s between two readings of --watch
LOAD_CELL_READINGS = 10  # samples averaged for the load cell
LOAD_CELL_SETTLE = 1.0  # s the cell gets after its calibration is set


def limits(cfg: dict) -> dict[str, tuple[float, str]]:
    """The voltage at rest each role needs, and what needs it."""
    return {
        "victim": (cfg["stand"]["vbat_rest_min"], "spin run on the stand"),
        "source": (cfg["safety"]["vbat_min_start"], "takeoff"),
    }


def report(role: str, drone_id: int, vbat: float, limit: tuple[float, str]) -> str:
    """One line on a drone's battery against its limit."""
    status = "ok" if vbat >= limit[0] else f"LOW, below {limit[0]:.2f} V"
    return f"{role:6s} cf{drone_id:<3d} {vbat:.2f} V  ({limit[1]} from {limit[0]:.2f} V: {status})"


def read_load_cell(victim: SimpleNamespace, load_cell: dict) -> tuple[float, float]:
    """Set the load cell calibration on the victim and read it.

    Args:
        victim: The connected victim, with its `cf`.
        load_cell: The `[load_cell]` section of the A1 config.

    Returns:
        Mean and standard deviation of `loadcell.weight`, in g, upward force positive.
    """
    for name, key in (("a", "a"), ("b", "b"), ("sampleRate", "sample_rate")):
        usd.set_param(victim.cf, f"loadcell.{name}", load_cell[key])
    time.sleep(LOAD_CELL_SETTLE)
    values = [
        usd.read_once(victim, ["loadcell.weight"])["loadcell.weight"]
        for _ in range(LOAD_CELL_READINGS)
    ]
    return float(np.mean(values)), float(np.std(values))


def main(watch: bool = False, load_cell: bool = False):
    """Print the battery voltage of the victim and the source, and optionally the load cell.

    Args:
        watch: Repeat the reading every `WATCH_PERIOD` until Ctrl+C.
        load_cell: Also print the load cell's reading, on the first round only.
    """
    cfg = load_config(require_masses=False)
    cflib.crtp.init_drivers()
    needed = limits(cfg)
    a1_cfg = load_a1_config() if load_cell else {}
    links = {}
    for role in ("victim", "source"):
        drone = cfg["drones"][role]
        link_uri = uri(drone["radio"], drone["channel"], drone["id"])
        links[role] = (drone["id"], SyncCrazyflie(link_uri, cf=Crazyflie(rw_cache=str(TOC_CACHE))))
    for _, scf in links.values():
        scf.open_link()
    while True:
        for role, (drone_id, scf) in links.items():
            vbat = float(usd.read_once(SimpleNamespace(cf=scf.cf), ["pm.vbat"])["pm.vbat"])
            logger.info(report(role, drone_id, vbat, needed[role]))
        if load_cell:
            victim = SimpleNamespace(cf=links["victim"][1].cf)
            mean, std = read_load_cell(victim, a1_cfg["load_cell"])
            logger.info(f"load cell {mean:.2f} g (std {std:.2f} g, {mean / 1000 * 9.81:.3f} N)")
            load_cell = False
        if not watch:
            break
        time.sleep(WATCH_PERIOD)
    for _, scf in links.values():
        scf.close_link()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Battery voltage of the A3 drones at rest")
    parser.add_argument("--watch", action="store_true", help="Repeat until Ctrl+C")
    parser.add_argument("--load-cell", action="store_true", help="Also read the victim's load cell")
    args = parser.parse_args()
    main(args.watch, args.load_cell)
