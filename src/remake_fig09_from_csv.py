"""Rebuild fig09 (hydrographs) from its own plotted-data CSV.

The original figure is produced by ``plot_figures.py`` from prediction bundles,
which are no longer kept.  The plotted window and every value shown are stored
in ``fig09_hydrographs_data.csv``, so the panel can be redrawn exactly from it.
This version also replaces the em-dash separator in the panel titles with a
colon, and drops the r/beta/gamma suffix (the caption only promises NSE and KGE).
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "Writing" / "Submission_WRR" / "05_LaTeX_Source" / "figures" / "results"
CSV = FIG_DIR / "fig09_hydrographs_data.csv"

COLOR_OBS = "#333333"
COLOR_LSTM = "#1B6CA8"
COLOR_MAMBA = "#C25B1E"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)


def main() -> None:
    df = pd.read_csv(CSV, parse_dates=["date"])
    groups = list(df.groupby("gauge_id", sort=False))

    fig, axes = plt.subplots(
        len(groups), 1, figsize=(11, 3.2 * len(groups)), sharex=False,
        constrained_layout=True,
    )
    if len(groups) == 1:
        axes = [axes]

    for i, (ax, (gid, g)) in enumerate(zip(axes, groups)):
        ax.plot(g["date"], g["observed_m3s"], color=COLOR_OBS, lw=1.0,
                alpha=0.85, label="Observed")
        ax.plot(g["date"], g["lstm_m3s"], color=COLOR_LSTM, ls="--", lw=1.1,
                label="LSTM")
        ax.plot(g["date"], g["mamba_m3s"], color=COLOR_MAMBA, ls="-", lw=1.1,
                label="Mamba")

        label = g["label"].iloc[0]
        name = g["gauge_name"].iloc[0]
        nse_l, kge_l = g["nse_lstm"].iloc[0], g["kge_lstm"].iloc[0]
        nse_m, kge_m = g["nse_mamba"].iloc[0], g["kge_mamba"].iloc[0]
        ax.set_title(
            f"({chr(ord('a') + i)}) {label}: {gid}  {name}\n"
            f"Mamba: NSE={nse_m:.2f} | KGE={kge_m:.2f}   \u00b7   "
            f"LSTM: NSE={nse_l:.2f} | KGE={kge_l:.2f}",
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("Q (m\u00b3/s)")
        ax.legend(frameon=False, ncol=3, loc="upper right")

    axes[-1].set_xlabel("Date")

    for ext, dpi in (("png", 300), ("pdf", None)):
        out = FIG_DIR / f"fig09_hydrographs.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print("saved", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
