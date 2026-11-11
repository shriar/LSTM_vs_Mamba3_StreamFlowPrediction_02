"""House style for every paper figure (D4 §4.5, normative).

One import, one call.  The palette, fonts, DPI, and error-band conventions
are identical for result figures, mechanism figures, and the graphical
abstract, so a reviewer never sees a colour change meaning between panels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Model identity colours: identical in every figure and the abstract.
COLOR_LSTM = "#1B6CA8"
COLOR_MAMBA = "#C25B1E"
COLOR_OBS = "#000000"
COLOR_PRCP = "#7FB0C9"
COLOR_BEST = "#6A3D9A"
COLOR_GREY = "#8A8A8A"

MODEL_COLORS = {"lstm": COLOR_LSTM, "mamba3": COLOR_MAMBA}

DPI_PNG = 300
DPI_FIG = 150

RC = {
    "figure.dpi": DPI_FIG,
    "savefig.dpi": DPI_PNG,
    "savefig.bbox": "tight",
    "font.family": "serif",     # matches txfonts; libtexgus STIX renders like Times
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "lines.linewidth": 1.8,
}


def apply_house_style() -> None:
    """Install the house style; call once per process before drawing."""
    plt.rcParams.update(RC)


def model_color(model: str) -> str:
    """Colour for a model id; anything unknown falls back to grey."""
    key = str(model).lower()
    if "mamba" in key:
        return COLOR_MAMBA
    if "lstm" in key:
        return COLOR_LSTM
    return COLOR_GREY


def save_figure(fig: plt.Figure, path, stem: str | None = None):
    """Save PNG (+ PDF) at print resolution."""
    from pathlib import Path

    target = Path(path)
    if target.suffix == "":
        target = target.with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=DPI_PNG, bbox_inches="tight")
    if stem or target.suffix == ".png":
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return target
