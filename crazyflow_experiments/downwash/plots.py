"""Shared look of the downwash plots.

Series colors are the categorical slots of the reference palette in fixed order: the victim is
always slot 1, the source slot 2, and the four rotors take slots 1 to 4. Grid and axes are
hairlines one shade off the surface, so the data carries the figure. Figures are built without
pyplot, so no GUI backend is involved when a flight script saves one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.figure import Figure

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
VICTIM, SOURCE = SERIES[0], SERIES[1]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"  # Reference lines: hover height, crossing time, thresholds
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SHADE = "#f0efec"  # Background spans: passes, settled windows
LINE_WIDTH = 1.0  # pt, about 2 px at the saved resolution
DPI = 150


def style(ax: Axes):
    """Recessive chrome: hairline grid below the data, no top and right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.title.set_color(INK)
    ax.title.set_size(10)


def figure(
    rows: int, cols: int = 1, width: float = 10.0, row_height: float = 2.2, sharex: bool = True
) -> tuple[Figure, NDArray]:
    """A styled figure with a grid of axes.

    Args:
        rows: Rows of axes.
        cols: Columns of axes.
        width: Figure width in inches.
        row_height: Height of one row in inches.
        sharex: Share the x axis within each column.

    Returns:
        The figure and its axes as (rows, cols).
    """
    fig = Figure(figsize=(width, row_height * rows + 0.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(rows, cols, sharex="col" if sharex else False, squeeze=False)
    for ax in axes.flat:
        style(ax)
    return fig, axes


def legend(ax: Axes, **kwargs: object):
    """Frameless legend in secondary ink."""
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, **kwargs)


def save(fig: Figure, path: Path) -> Path:
    """Save the figure as an image, creating the folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    return path
