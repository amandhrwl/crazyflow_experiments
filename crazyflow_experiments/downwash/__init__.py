"""Downwash of a source Crazyflie on a victim, in flight and with the victim on the load cell.

Block A3 of the real-flight protocol: the victim hovers at a fixed point while the source flies
straight passes over it at three heights and three speeds. Both drones log onboard to their
micro-SD card decks, the PC records the motion capture and the setpoints it sends.
`a3_design.py` writes the leg table, `timeline.py` turns one flight of it into the setpoints of both
drones, `a3_deploy.py` flies the flight through `pair.py`, and `a3_convert.py` maps the SD logs
onto PC time, aligns them with the motion capture and cuts the legs.

Block A1 checks crazyflow's rpm2thrust with the victim on the load cell: `a1_probe.py` runs the
bench checks, `a1_sweep.py` drives the motors through fixed commands, `a1_convert.py` cuts the
steady windows and `a1_fit.py` fits the thrust curve. `a1_stand.py` holds what they share.

The load cell setup test (`lc_test_*`) hovers the source beside, over and beside the victim on the
load cell: `lc_test_deploy.py` flies it and `lc_test_convert.py` puts its logs onto the PC clock.

The blocks share one layer: `usd.py` for the card configs, the logging and the decoder, `sync.py`
for the onboard clocks, `radio.py` for the radio logs and the victim's link on the load cell, and
`thrust.py` for the rotor thrust from the measured rpm.
"""

from __future__ import annotations

from pathlib import Path

from crazyflow_experiments.utils.paths import PLOTS_DIR, RESULTS_DIR

ROLES = ("victim", "source")
CONFIG_PATH = Path(__file__).parent / "config.toml"
SD_CONFIG_DIR = Path(__file__).parent / "sd"  # config.txt of every card, config_<block>_<role>.txt
A3_RESULTS_DIR = RESULTS_DIR / "downwash" / "a3"  # Design, notes, bags and one folder per flight
DOWNWASH_PLOTS_DIR = PLOTS_DIR / "downwash"
A1_CONFIG_PATH = Path(__file__).parent / "a1_config.toml"
A1_RESULTS_DIR = RESULTS_DIR / "downwash" / "a1"  # NOTES.md, rpm2thrust.toml, one folder per block
A1_PLOTS_DIR = DOWNWASH_PLOTS_DIR / "a1"
LC_CONFIG_PATH = Path(__file__).parent / "lc_test_config.toml"
LC_RESULTS_DIR = RESULTS_DIR / "load_cell"  # One recording per load cell test
LC_PLOTS_DIR = PLOTS_DIR / "load_cell"  # Their quicklooks
