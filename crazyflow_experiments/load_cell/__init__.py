"""Load cell setup test: the source hovers over the victim, which sits on the load cell.

The victim is mounted upright on the load cell and does not fly, its firmware reads the load cell
(the `loadcell` deck driver of the sysid build). The source takes off beside it, holds there, moves
over it, holds, moves back beside it and lands. Both drones log to their micro-SD cards, and the
victim also streams the load cell over the radio at 100 Hz. `deploy.py` flies the test and
`convert.py` puts the SD logs, the radio logs and the motion capture onto the PC clock.
"""

from __future__ import annotations

from pathlib import Path

from crazyflow_experiments.utils.paths import PLOTS_DIR, RESULTS_DIR

CONFIG_PATH = Path(__file__).parent / "config.toml"
SD_CONFIG_DIR = Path(__file__).parent / "sd"
RECORDINGS_DIR = RESULTS_DIR / "load_cell"
QUICKLOOK_DIR = PLOTS_DIR / "load_cell"
ROLES = ("victim", "source")
