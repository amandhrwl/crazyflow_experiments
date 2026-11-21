"""Battery voltage of the A3 drones at rest, read over the radio without rebooting them.

Connects to the victim and the source of `config.toml`, reads `pm.vbat` once and prints it against
the limits the runs check: the source takes off from `vbat_min_start` up, a spin run of the stand
starts from `[stand] vbat_rest_min` up. `--watch` repeats the reading until Ctrl+C. Read it with the
motors off: under load the voltage sags by 0.1 to 0.3 V.

Run with
    pixi run -e sim2real python crazyflow_experiments/downwash/battery.py [--watch]
"""

from __future__ import annotations

import argparse
import logging
import time
from types import SimpleNamespace

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from crazyflow_experiments.downwash import usd
from crazyflow_experiments.downwash.a3_design import load_config
from crazyflow_experiments.downwash.radio import TOC_CACHE, uri

logger = logging.getLogger("crazyflow_experiments.downwash.battery")  # Not __name__, the __main__
logger.setLevel(logging.INFO)  # deploy.crazyflie configures the root logger at WARNING on import
WATCH_PERIOD = 2.0  # s between two readings of --watch


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


def main(watch: bool = False):
    """Print the battery voltage of the victim and the source.

    Args:
        watch: Repeat the reading every `WATCH_PERIOD` until Ctrl+C.
    """
    cfg = load_config(require_masses=False)
    cflib.crtp.init_drivers()
    needed = limits(cfg)
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
        if not watch:
            break
        time.sleep(WATCH_PERIOD)
    for _, scf in links.values():
        scf.close_link()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Battery voltage of the A3 drones at rest")
    parser.add_argument("--watch", action="store_true", help="Repeat until Ctrl+C")
    main(parser.parse_args().watch)
