"""Downwash of a source Crazyflie on a hovering victim, flown as straight flyover passes.

Block A3 of the real-flight protocol: the victim hovers at a fixed point while the source flies
straight passes over it at three heights and three speeds. Both drones log onboard to their
micro-SD card decks, the PC records the motion capture and the setpoints it sends.

`design.py` writes the leg table, `timeline.py` turns one flight of it into the setpoints of both
drones, `deploy_a3.py` flies the flight through `pair.py` and `usd.py`, and `convert_a3.py` maps the
SD logs onto PC time, aligns them with the motion capture and cuts the legs.
"""

from __future__ import annotations

from pathlib import Path

from crazyflow_experiments.utils.paths import PLOTS_DIR, RESULTS_DIR

CONFIG_PATH = Path(__file__).parent / "config.toml"
SD_CONFIG_DIR = Path(__file__).parent / "sd"  # config.txt of the victim's and the source's card
A3_RESULTS_DIR = RESULTS_DIR / "downwash" / "a3"  # Design, notes, bags and one folder per flight
DOWNWASH_PLOTS_DIR = PLOTS_DIR / "downwash"
