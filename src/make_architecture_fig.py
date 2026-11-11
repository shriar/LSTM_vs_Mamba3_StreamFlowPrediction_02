"""Draw the standard-wiring Mamba-3 schematic (replaces the Adapter drawing)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from figure_style import COLOR_LSTM, COLOR_MAMBA, apply_house_style  # noqa: E402
from paths import FIGURES_DIR, PROJECT_ROOT  # noqa: E402

apply_house_style()

INK = "#222222"
FILL_INPUT = "#F6E7D8"
FILL_OP = "#DCE6F2"
FILL_SSM = "#F4D9D0"
FILL_OUT = "#E8E3F2"


def box(ax, x, y, w, h, text, fill, size=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fill, ec=INK, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=0.9, color=INK))


fig = plt.figure(figsize=(13, 6.4))
gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.3], hspace=0.35, wspace=0.25)

# ---- top: data flow -------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.05, 0.30, 1.45, 0.42, "input $(T,B,40)$\n5 forcings + 35 statics", FILL_INPUT)
box(ax, 1.95, 0.30, 1.15, 0.42, "Linear\n$40\\rightarrow H$", FILL_OP)
box(ax, 3.55, 0.30, 1.0, 0.42, "RMSNorm", FILL_OP)
box(ax, 5.0, 0.30, 2.2, 0.42, "Mamba-3 block $\\times L$\npre-norm + residual", FILL_SSM)
box(ax, 7.65, 0.30, 0.9, 0.42, "Linear\n$H\\rightarrow1$", FILL_OP)
box(ax, 8.95, 0.30, 0.95, 0.42, "forecast\n$(T,B,1)$", FILL_OUT)
for x0, x1 in ((1.50, 1.95), (3.10, 3.55), (4.55, 5.00), (7.20, 7.65), (8.55, 8.95)):
    arrow(ax, x0, 0.51, x1, 0.51)
ax.text(5.9, 0.88, "Mamba-3, standard wiring: the same 40-dimensional input as the LSTM at every step",
        ha="center", fontsize=10)

# ---- bottom-left: block detail -------------------------------------------
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(a) Mamba-3 residual block", fontsize=10)
box(ax, 0.10, 0.80, 0.80, 0.10, "$h$ in $(T,B,H)$", FILL_INPUT, 8.5)
box(ax, 0.25, 0.64, 0.50, 0.09, "RMSNorm", FILL_OP, 8.5)
box(ax, 0.25, 0.49, 0.50, 0.09, "Mamba3 (selective SSM)", FILL_SSM, 8.5)
box(ax, 0.25, 0.34, 0.50, 0.09, "Dropout $p=0.06$", FILL_OP, 8.5)
box(ax, 0.25, 0.19, 0.50, 0.09, "$\\times\\ \\alpha$ (learned)", FILL_OP, 8.5)
ax.text(0.50, 0.06, "$+$ residual", ha="center", fontsize=9)
for y0, y1 in ((0.80, 0.73), (0.64, 0.58), (0.49, 0.43), (0.34, 0.28), (0.19, 0.10)):
    arrow(ax, 0.50, y0, 0.50, y1)

# ---- bottom-right: input composition -------------------------------------
ax = fig.add_subplot(gs[1, 1:])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(b) Shared input: concatenation at every step", fontsize=10)
box(ax, 0.03, 0.62, 0.34, 0.16, "5 dynamic forcings\n(dayl, prcp, srad, tmean, vp)", FILL_INPUT, 8.5)
box(ax, 0.03, 0.34, 0.34, 0.16, "35 static attributes\n(area, soils, snow, aridity, ...)", FILL_INPUT, 8.5)
box(ax, 0.46, 0.46, 0.20, 0.20, "concat\n$\\rightarrow 40$", FILL_OP, 9)
box(ax, 0.75, 0.40, 0.22, 0.32, "same tensor\nfor LSTM\nand Mamba-3", FILL_OUT, 9)
arrow(ax, 0.37, 0.70, 0.46, 0.60)
arrow(ax, 0.37, 0.42, 0.46, 0.50)
arrow(ax, 0.66, 0.56, 0.75, 0.56)

out = PROJECT_ROOT.parent / "Writing" / "Submission_WRR" / "05_LaTeX_Source" / "figures" / "fig02b_mamba_architecture.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(FIGURES_DIR / "paper" / "fig02b_mamba_architecture.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
