"""Export of the A3 stand legs as a versioned dataset in LEGS_FORMAT v1 (`docs/LEGS_FORMAT.md`).

The dataset is the only interface to afterwash, which replays and fits it. The code here reuses the
stand conversion (`a3_stand_convert`), the clocks (`sync`), the card decoder (`usd`) and the
free-flight converter's onboard helpers (`a3_convert`), and adds what the contract needs on top:

- `chain.py`: the stand's measurement chain `C[x] = P[G * x]` and `chain_test.npz`;
- `stand_gain.py`: the stand's gain and delay against the rpm thrust at the victim's motor edges;
- `source_sd.py`: the source's card onto PC time, and its thrust from the motion capture;
- `export.py`: the legs, `manifest.csv`, `dataset.json`, the command line;
- `validate.py`: the checks of LEGS_FORMAT §8;
- `report.py`: the regression against the existing cell table, quicklooks and the notes.

The data lives in the data repository `crazyflow_experiments_data` (`--data`): the runs under
`results/downwash/a3/`, the A1 curve under `results/downwash/a1/`, the dataset under
`legs/<version>/`. Nothing is written outside `legs/<version>/`.
"""

from __future__ import annotations

from pathlib import Path

FORMAT = "LEGS_FORMAT"
FORMAT_VERSION = 1
GRID_FREQ = 500.0  # Hz
G = 9.81  # m/s^2, grams of force to N, as `a3_stand_convert`
MOUNT_SIGN = 1  # The calibrated reading already points along the force (LEGS_FORMAT §6.2)
# The 22 Nov evening session: the current design (Δz 0.25 and 0.5 m, 0.5 to 2.0 m/s, the stand at
# x = 0.47 to 0.54 m). The runs of 21 Nov and of 22 Nov 00:00 to 00:19 flew the earlier design.
EVENING22_RUNS = "stand_flight*_2026-11-22_2*"
# The runs of every dataset version, so a version re-exports from the same runs. legs_v2 re-exports
# legs_v1 with the rpm glitches removed in `a3_stand_convert.load_run`. A version that is not listed
# names its runs with `export.py --runs <glob>`.
VERSION_RUNS = {"legs_v1": EVENING22_RUNS, "legs_v2": EVENING22_RUNS}


def version_runs(version: str, pattern: str | None = None) -> str:
    """Glob of a dataset version's runs: `pattern` if given, else the version's in `VERSION_RUNS`."""
    if pattern is not None:
        return pattern
    if version not in VERSION_RUNS:
        raise ValueError(f"{version} has no runs in VERSION_RUNS: name them with --runs <glob>")
    return VERSION_RUNS[version]


def runs_dir(data: Path) -> Path:
    """Folder of the A3 runs in the data repository."""
    return Path(data) / "results" / "downwash" / "a3"


def a1_curve_path(data: Path) -> Path:
    """cf10's thrust curve of block A1 in the data repository."""
    return Path(data) / "results" / "downwash" / "a1" / "rpm2thrust.toml"


def dataset_dir(data: Path, version: str) -> Path:
    """Folder of a dataset version in the data repository."""
    return Path(data) / "legs" / version
