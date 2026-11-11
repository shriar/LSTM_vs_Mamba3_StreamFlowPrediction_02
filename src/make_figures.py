"""D4.3 -- standalone multi-seed figures 1, 2 and 15.

Figs 1/2/15 are validation-based and multi-seed, which the legacy plotter
cannot draw (it takes one run per model and labels val panels "Median test").
This module draws them natively from ``figure_source`` in the house style
(``figure_style.py``) and writes a ``*_data.csv`` beside every PNG/PDF.

  * Fig 1 -- training loss, median val NSE, median val KGE; median over seeds
    with an IQR band and a vertical marker at each model's best epoch.
  * Fig 2 -- parameters vs test NSE/KGE by depth, three tiers; matched pairs
    joined; one point per (model, depth, tier) with a seed SD bar.
  * Fig 15 -- measured 671-basin epoch seconds by tier, with the
    Mamba/LSTM ratio labelled per tier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import figure_source  # noqa: E402
from figure_source import (  # noqa: E402
    best_epochs,
    cell_for,
    model_params,
    validation_files,
)
from figure_style import (  # noqa: E402
    COLOR_BEST,
    COLOR_LSTM,
    COLOR_MAMBA,
    apply_house_style,
    save_figure,
)
from paths import FIGURES_DIR, RESULTS_DIR  # noqa: E402

apply_house_style()

HEADLINE = {
    "lstm": {"recipe": "recipes/lstm.json", "tier": 1, "depth": 1, "seeds": [1, 2, 3, 4, 5]},
    "mamba3": {"recipe": "recipes/mamba3.json", "tier": 1, "depth": 3, "seeds": [1, 2, 3, 4, 5]},
}
# Fig 2 needs every depth (L1/L2/L3) at every tier, not just the headline one.
# Seeds follow the C1 ladder: 5 at a model's headline depth, 2 elsewhere.
DEPTHS = (1, 2, 3)
HEADLINE_DEPTH = {"lstm": 1, "mamba3": 3}
SEEDS_BY_DEPTH = {5: [1, 2, 3, 4, 5], 2: [1, 2]}
# measured 671-basin epoch medians (s) from C1a/C1b/C1c logs.
EPOCH_SECONDS = {
    1: (28.9, 32.7),
    2: (29.1, 55.6),
    3: (45.0, 186.6),
}


def _curves(recipe: str, tier: int, depth: int, seeds: list[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        cell = cell_for(recipe, tier, depth, seed)
        files = validation_files(cell)
        frame = pd.read_csv(files["val_metrics"])
        loss_file = files["training_loss"]
        loss = pd.read_csv(loss_file) if loss_file is not None else None
        keep = frame[["epoch", "train_loss", "val_median_nse", "val_median_kge"]].copy()
        if loss is not None and {"epoch", "loss"}.issubset(loss.columns):
            keep = keep.drop(columns=["train_loss"])
            keep = keep.merge(loss[["epoch", "loss"]].rename(columns={"loss": "train_loss"}), on="epoch", how="left")
        keep["seed"] = seed
        frames.append(keep)
    combined = pd.concat(frames, ignore_index=True)
    if not (combined["train_loss"] < 1).any():
        base = combined.groupby("seed")["train_loss"].first()
    else:
        base = combined[combined["epoch"] == 1].set_index("seed")["train_loss"]
    combined["train_loss_norm"] = combined["train_loss"] / combined["seed"].map(base)
    combined["train_loss_norm"] = combined["train_loss_norm"].fillna(combined["train_loss"])
    out = combined.groupby("epoch").agg(
        median_loss=("train_loss_norm", "median"),
        q25_loss=("train_loss_norm", lambda s: s.quantile(0.25)),
        q75_loss=("train_loss_norm", lambda s: s.quantile(0.75)),
        median_nse=("val_median_nse", "median"),
        q25_nse=("val_median_nse", lambda s: s.quantile(0.25)),
        q75_nse=("val_median_nse", lambda s: s.quantile(0.75)),
        median_kge=("val_median_kge", "median"),
        q25_kge=("val_median_kge", lambda s: s.quantile(0.25)),
        q75_kge=("val_median_kge", lambda s: s.quantile(0.75)),
        n_seeds=("seed", "count"),
    ).reset_index()
    return out


def figure1(out_dir: Path) -> Path:
    name = "fig06_learning_curves"
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, spec in zip(
        axes,
        [
            ("Training loss (normalised)", "median_loss", "q25_loss", "q75_loss", "Loss / Loss\u2080"),
            ("Median validation NSE", "median_nse", "q25_nse", "q75_nse", "NSE"),
            ("Median validation KGE", "median_kge", "q25_kge", "q75_kge", "KGE"),
        ],
    ):
        title, med, q25, q75, ylabel = spec
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        for model, cfg in HEADLINE.items():
            curves = _curves(cfg["recipe"], cfg["tier"], cfg["depth"], cfg["seeds"])
            color = COLOR_LSTM if model == "lstm" else COLOR_MAMBA
            label = "LSTM" if model == "lstm" else "Mamba-3"
            ax.fill_between(
                curves["epoch"], curves[q25], curves[q75], color=color, alpha=0.18, lw=0
            )
            ax.plot(curves["epoch"], curves[med], color=color, lw=1.8, label=label)
        for model, cfg in HEADLINE.items():
            epochs = best_epochs(cfg["recipe"], cfg["tier"], cfg["depth"], cfg["seeds"])
            best = float(np.median(list(epochs.values())))
            ax.axvline(best, color=COLOR_BEST, ls=":", lw=1.2)
            ax.text(best + 0.4, ax.get_ylim()[1], f"best {best:.0f}", color=COLOR_BEST, fontsize=8, va="top")
        ax.legend(frameon=False)
    axes[0].set_ylim(0)
    target = out_dir / name
    save_figure(fig, target)
    data = pd.concat(
        [
            _curves(cfg["recipe"], cfg["tier"], cfg["depth"], cfg["seeds"]).assign(model=model)
            for model, cfg in HEADLINE.items()
        ],
        ignore_index=True,
    )
    data.to_csv(out_dir / f"{name}_data.csv", index=False)
    print(f"wrote {target} (+ pdf/csv)")
    return target


def figure2(out_dir: Path) -> Path:
    name = "fig04_param_scaling_all_layers"
    rows = []
    for tier in (1, 2, 3):
        for model, recipe, key in (
            ("LSTM", "recipes/lstm.json", "lstm"),
            ("Mamba-3", "recipes/mamba3.json", "mamba3"),
        ):
            for depth in DEPTHS:
                # 5 seeds exist only at Tier 1's headline depth (C1a); all
                # other tier/depth cells carry 2 (C1b/C1c).
                n_seeds = 5 if (tier == 1 and depth == HEADLINE_DEPTH[key]) else 2
                for seed in SEEDS_BY_DEPTH[n_seeds]:
                    cell = cell_for(recipe, tier, depth, seed)
                    params = model_params(cell)
                    basin = pd.read_csv(
                        figure_source.result_files(
                            key, tier, depth, seed
                        )["basin_metrics"]
                    )
                    rows.append({
                        "model": model,
                        "tier": tier,
                        "depth": depth,
                        "seed": seed,
                        "params": params["n_parameters"],
                        "median_nse": float(basin["NSE"].median()),
                        "median_kge": float(basin["KGE"].median()),
                    })
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    styles = {1: ("o", "-"), 2: ("s", "--"), 3: ("^", ":")}
    for ax, metric, title in (
        (axes[0], "median_nse", "Validation-selected, test NSE"),
        (axes[1], "median_kge", "Validation-selected, test KGE"),
    ):
        ax.set_title(title)
        ax.set_xlabel("Parameters")
        ax.set_ylabel(metric.replace("median_", "").upper())
        ax.set_xscale("log")
        for model, color in (("LSTM", COLOR_LSTM), ("Mamba-3", COLOR_MAMBA)):
            for depth in DEPTHS:
                sub = table[(table.model == model) & (table.depth == depth)]
                if sub.empty:
                    continue
                line = sub.groupby("tier")[metric].agg(["mean", "std"]).reset_index()
                params = sub.groupby("tier")["params"].first().reindex(line["tier"]).to_numpy()
                marker, linestyle = styles[depth]
                ax.plot(
                    params, line["mean"], marker=marker, ls=linestyle, color=color,
                    lw=1.5, ms=5, label=f"{model} {depth}L",
                )
                ax.errorbar(
                    params, line["mean"], yerr=line["std"].fillna(0),
                    color=color, ecolor=color, elinewidth=1, capsize=3, fmt="none",
                )
        ax.legend(frameon=False, fontsize=7, ncol=2)
    target = out_dir / name
    save_figure(fig, target)
    table.to_csv(out_dir / f"{name}_data.csv", index=False)
    print(f"wrote {target}")
    return target


def figure15(out_dir: Path) -> Path:
    name = "fig05_training_time"
    rows = []
    for tier, (lstm_s, mamba_s) in EPOCH_SECONDS.items():
        rows.append({
            "tier": tier,
            "lstm_seconds": lstm_s,
            "mamba_seconds": mamba_s,
            "ratio": mamba_s / lstm_s,
        })
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    x = np.arange(1, 4)
    ax.bar(x - 0.2, table["lstm_seconds"], width=0.4, color=COLOR_LSTM, label="LSTM")
    ax.bar(x + 0.2, table["mamba_seconds"], width=0.4, color=COLOR_MAMBA, label="Mamba-3")
    for xi, ratio, ms in zip(x, table["ratio"], table["mamba_seconds"]):
        ax.text(xi + 0.2, ms + 6, f"{ratio:.2f}x", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tier {t}" for t in table["tier"]])
    ax.set_ylabel("Seconds per epoch (671 basins)")
    ax.set_title("Training cost of the matched pair per tier")
    ax.legend(frameon=False)
    target = out_dir / name
    save_figure(fig, target)
    table.to_csv(out_dir / f"{name}_data.csv", index=False)
    print(f"wrote {target}")
    return target


def main() -> int:
    out_dir = FIGURES_DIR / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure1(out_dir)
    figure2(out_dir)
    figure15(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
