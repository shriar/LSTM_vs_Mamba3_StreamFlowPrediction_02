"""Build the graphical abstract from the same data as the paper figures."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from figure_style import COLOR_LSTM, COLOR_MAMBA, apply_house_style  # noqa: E402
from paths import FIGURES_DIR, PROJECT_ROOT  # noqa: E402

apply_house_style()
PAPER = FIGURES_DIR / "paper"
WRITING = PROJECT_ROOT.parent / "Writing" / "Jounal_Writting_latex"

fig = plt.figure(figsize=(20, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])


def label(ax, text):
    ax.set_title(text, fontsize=12, loc="left", pad=6)


# (a) parameter scaling ----------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
d = pd.read_csv(PAPER / "fig04_param_scaling_all_layers_data.csv")
styles = {1: ("o", "-"), 2: ("s", "--"), 3: ("^", ":")}
for model, color in (("LSTM", COLOR_LSTM), ("Mamba-3", COLOR_MAMBA)):
    for depth in (1, 2, 3):
        s = d[(d.model == model) & (d.depth == depth)].groupby("tier").agg(
            params=("params", "first"), nse=("median_nse", "mean")).reset_index()
        marker, ls = styles[depth]
        ax.plot(s.params, s.nse, marker=marker, ls=ls, color=color, ms=5,
                label=f"{model} {depth}L")
ax.set_xscale("log"); ax.set_xlabel("parameters"); ax.set_ylabel("median test NSE")
ax.legend(fontsize=7, ncol=2, frameon=False); label(ax, "(a) capacity ladder")

# (b) ECDF NSE -------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
e = pd.read_csv(PAPER / "fig03_ecdf_data.csv")
for model, color in (("LSTM", COLOR_LSTM), ("Mamba", COLOR_MAMBA)):
    s = e[(e.model == model) & (e.metric == "NSE")].sort_values("value")
    ax.plot(s.value, np.arange(1, len(s) + 1) / len(s), color=color, lw=2, label=model)
ax.set_xlim(0, 1); ax.set_xlabel("per-basin NSE"); ax.set_ylabel("cumulative fraction")
ax.legend(frameon=False); label(ax, "(b) per-basin NSE")

# (c) validation curves ----------------------------------------------------
ax = fig.add_subplot(gs[0, 2])
c = pd.read_csv(PAPER / "fig06_learning_curves_data.csv")
for model, color in (("lstm", COLOR_LSTM), ("mamba3", COLOR_MAMBA)):
    s = c[c.model == model].sort_values("epoch")
    ax.fill_between(s.epoch, s.q25_nse, s.q75_nse, color=color, alpha=0.18, lw=0)
    ax.plot(s.epoch, s.median_nse, color=color, lw=2,
            label="LSTM" if model == "lstm" else "Mamba-3")
ax.set_xlabel("epoch"); ax.set_ylabel("median validation NSE")
ax.legend(frameon=False); label(ax, "(c) five-seed validation curves")

# (d) hydrograph -----------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
h = pd.read_csv(PAPER / "fig09_hydrographs_data.csv")
gauge = h.gauge_id.iloc[0]
h = h[h.gauge_id == gauge].copy()
h["date"] = pd.to_datetime(h["date"])
for col, color, lab in (("observed_m3s", "#000000", "observed"),
                        ("lstm_m3s", COLOR_LSTM, "LSTM"),
                        ("mamba_m3s", COLOR_MAMBA, "Mamba-3")):
    ax.plot(h["date"], h[col], color=color, lw=1.1, label=lab)
ax.set_xlabel("date"); ax.set_ylabel("discharge (m3/s)")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"(d) hydrograph — gauge {gauge}", fontsize=12, loc="left", pad=6)

# (e) spatial map ----------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
m = pd.read_csv(PAPER / "fig07_spatial_data.csv")
sc = ax.scatter(m.gauge_lon, m.gauge_lat, c=m.dNSE, cmap="RdBu", vmin=-0.3, vmax=0.3,
                s=10, alpha=0.85, edgecolors="none")
ax.set_xlim(-125, -66); ax.set_ylim(24, 50)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="dNSE (Mamba - LSTM)")
label(ax, "(e) per-basin difference")

# (f) seasonal -------------------------------------------------------------
ax = fig.add_subplot(gs[1, 2])
se = pd.read_csv(PAPER / "fig08_seasonal_heatmaps_data.csv")
se = se[(se.metric == "NSE") & (se.model != "Mamba − LSTM")]
piv = se.pivot(index="season", columns="model", values="value").reindex(
    ["Winter", "Spring", "Summer", "Autumn"])
x = np.arange(len(piv))
ax.bar(x - 0.2, piv["LSTM"], width=0.4, color=COLOR_LSTM, label="LSTM")
ax.bar(x + 0.2, piv["Mamba"], width=0.4, color=COLOR_MAMBA, label="Mamba-3")
ax.set_xticks(x, piv.index, rotation=0); ax.set_ylabel("median NSE")
ax.legend(frameon=False, fontsize=8); label(ax, "(f) seasonal NSE")

fig.suptitle("Gated recurrence versus selective state-space mixing at one-year lookback — 671 CAMELS basins, matched 40-dim wiring, matched tuning",
             fontsize=15)
png = WRITING / "graphical_abstract.png"
svg = WRITING / "graphical_abstract.svg"
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
plt.close(fig)

b64 = base64.b64encode(png.read_bytes()).decode("ascii")
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Graphical abstract</title>
<style>body{{margin:0;background:#fff;font-family:Georgia,serif}}img{{width:100%;height:auto}}</style>
</head><body>
<img alt="Graphical abstract: six panels comparing Mamba-3 and a CuDNN LSTM on 671 CAMELS basins under a shared input wiring and a matched tuning budget. LSTM scores higher at all three tiers." src="data:image/png;base64,{b64}">
</body></html>
"""
(WRITING / "graphical_abstract.html").write_text(html, encoding="utf-8")
print("wrote", png, svg, "and html")
