"""Shared matplotlib style for all replication figures.

Colours come from a validated categorical palette, assigned in fixed slot
order and never cycled. The same entity always gets the same colour across
figures:

    slot 1 blue   -> unimanual context
    slot 2 orange -> bimanual context
    slot 3 aqua, slot 4 yellow, ... -> further series, in this order

Marks: 2-px lines, markers of about 8 px, light grid, no top/right spines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are only saved to files
import matplotlib.pyplot as plt

from bimanual.cache import RESULTS_DIR
from bimanual.pngio import save_png_rgb

FIG_DIR = RESULTS_DIR / "figures"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
UNI, BI = SERIES[0], SERIES[1]
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "legend.frameon": False,
})


def save(fig, name: str) -> Path:
    """Save to results/figures/<name>.png (plain RGB) and .svg; returns the PNG path."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    save_png_rgb(fig, png, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    return png
