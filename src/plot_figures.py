#!/usr/bin/env python3
"""Plot paper figures from completed CAMELS LSTM vs Mamba tier runs.

Implements Figures 1–8 described in ../figures.md. Designed to run *after*
a full `run_all_tiers.py` campaign; missing artifacts are skipped with a clear
warning so partial ladders still produce whatever plots they can.

Usage (WSL + conda py312)::

    cd experiment
    conda activate py312
    python src/plot_figures.py
    python src/plot_figures.py --figures 1 2 3 8 --compare-tier 3
    python src/plot_figures.py --lstm-exp LSTM_1L --mamba-exp Mamba3_3L --epoch 20

Outputs are written under ``experiment/figures/`` (PNG + ``*_data.csv``).
Every figure saves the exact dataframe(s) it plots as ``{stem}_data.csv``
in the same folder, so all numbers behind the plots are reproducible.
"""

from __future__ import annotations

import os
import sys

# =====================================================================
# Windows OpenMP dual-runtime guard (must run BEFORE numpy/torch/hydroDL
# imports, otherwise we abort with:
#     OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll
#                     already initialized.
# conda numpy/scipy ship libiomp5md.dll; hydroDL/torch ship libomp.dll;
# both try to become the active OpenMP runtime. The documented workarounds
# below let the process continue without aborting.
# =====================================================================
if sys.platform.startswith("win") or os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable, Optional

from paths import DATA_DIR, FIGURES_DIR, MPL_CONFIG_DIR, OUTPUT_ROOT

SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from config import (
    ExperimentConfig,
    LSTM_1L_TIER_CONFIG,
    LSTM_2L_TIER_CONFIG,
    LSTM_3L_TIER_CONFIG,
    MAMBA_1L_TIER_CONFIG,
    MAMBA_2L_TIER_CONFIG,
    MAMBA_3L_TIER_CONFIG,
    TIER_NAMES,
)

DEFAULT_FIG_DIR = FIGURES_DIR / "paper"
# Same Zenodo CAMELS source family used by config.py/data_utils.py.
CLIM_URL = "https://zenodo.org/records/15529996/files/camels_clim.txt?download=1"

# =========================================================================
#  ======================  ALL HYPERPARAMETERS HERE  ======================
# =========================================================================
#  Edit this block to control what gets plotted.
#  CLI arguments (--tier-1, --tier-2, --tiers, --compare-tier, --epoch,
#  --lstm-exp, --mamba-exp, --lstm-layers, --mamba-layers, --mamba-version,
#  --basin-ids, --fig-dir, --figures) override the values below when given.
# =========================================================================

# ---- Experiment / run discovery ------------------------------------------
DEFAULT_EPOCH: int | None = None           # Match the completed Ep30 runs
DEFAULT_LSTM_EPOCH: int | None = None    # Default epoch for LSTM (None = use DEFAULT_EPOCH or highest completed)
DEFAULT_MAMBA_EPOCH: int | None = None   # Default epoch for Mamba (None = use DEFAULT_EPOCH or highest completed)
# Keep these defaults in sync with ExperimentConfig and run_all_tiers.py.
# The runner can create all controlled depths; explicit layer/experiment CLI
# arguments remain available when a matched-depth comparison is desired.
DEFAULT_LSTM_EXP: str | None = None       # derive from DEFAULT_LSTM_LAYERS
DEFAULT_MAMBA_EXP: str | None = None      # derive from version/layers
DEFAULT_LSTM_LAYERS: int | None = 1
DEFAULT_MAMBA_LAYERS: int | None = 3
DEFAULT_MAMBA_VERSION: int | None = 3
DEFAULT_MAMBA_VARIANT: str = "standard"
TIER_CHOICES = tuple(sorted(TIER_NAMES))

# ---- Per-figure tier selection ------------------------------------------
# Edit DEFAULT_TIER to select the tier used by all single-tier figures.
# Figs 2 and 5 intentionally remain tier ladders.
DEFAULT_TIER: int = 1
TIER_FIG1_LEARNING_CURVES: int = DEFAULT_TIER
TIER_FIG3_ECDF: int = DEFAULT_TIER
TIER_FIG4_SIGNATURES: int = DEFAULT_TIER
TIER_FIG6_SPATIAL: int = DEFAULT_TIER
TIER_FIG7_HYDROGRAPHS: int = DEFAULT_TIER
TIER_FIG8_FDC: int = DEFAULT_TIER
TIER_FIG9_SEASONAL: int = DEFAULT_TIER
TIER_FIG10_RESIDUALS: int = DEFAULT_TIER
TIER_FIG11_EVENTS: int = DEFAULT_TIER
TIER_FIG12_UNCERTAINTY: int = DEFAULT_TIER
TIERS_FIG2_PARAM_SCALING: list[int] = sorted(TIER_NAMES)
TIERS_FIG5_TIER_SCALING: list[int] = sorted(TIER_NAMES)
DEFAULT_COMPARE_TIER: int = DEFAULT_TIER

# ---- Which figures to render by default ----------------------------------
DEFAULT_FIGURES: list[int] = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15]

# ---- Colors --------------------------------------------------------------
# Mamba-3 blue, LSTM red. The hue pair is blue-vs-red, not the Okabe-Ito
# vermillion+blue, so this is no longer a literal Okabe-Ito palette. The
# accessibility intent survives the change: blue and red stay separable under
# deuteranopia, protanopia and tritanopia, as blue-vs-orange did. Line style
# carries the same distinction independently of hue (LSTM dashed, Mamba solid),
# so the traces remain separable in greyscale too.
# D4 (2026-11-10): house style.  The old palette (Mamba #0072B2 blue, LSTM red)
# is replaced by the D4 convention in figure_style.py, where LSTM is blue and
# Mamba-3 is orange, matching every rebuilt figure.
COLOR_MAMBA = "#C25B1E"
COLOR_LSTM = "#1B6CA8"
COLOR_OBS = "#333333"

# ---- Style / DPI ---------------------------------------------------------
FIG_DPI: int = 120
SAVE_DPI: int = 300
FONT_SIZE: int = 10
AX_TITLE_SIZE: int = 11
AX_LABEL_SIZE: int = 10
LEGEND_SIZE: int = 9
GRID_ALPHA: float = 0.25
GRID_LS: str = "--"

# ---- Figure 1 (learning curves) ------------------------------------------
FIG1_FIGSIZE: tuple[float, float] = (12.5, 3.6)
FIG1_MARKER_MS: int = 4
FIG1_LINE_LW: float = 1.8

# ---- Figure 2 (param scaling) --------------------------------------------
FIG2_FIGSIZE: tuple[float, float] = (10, 4)
FIG2_MARKER_MS: int = 6
FIG2_LINE_LW: float = 1.8

# ---- Figure 3 (ECDF) -----------------------------------------------------
FIG3_FIGSIZE: tuple[float, float] = (12, 8)
FIG3_LINE_LW: float = 1.8

# ---- Figure 4 (signatures / violin) --------------------------------------
FIG4_COL_W: float = 2.2
FIG4_H: float = 2.5
FIG4_VIOLIN_ALPHA: float = 0.55

# ---- Figure 5 (tier scaling with CI) -------------------------------------
FIG5_FIGSIZE: tuple[float, float] = (9, 4.5)
FIG5_NSE_LINE_LW: float = 1.8

# ---- Figure 15 (training time vs tier) ------------------------------------
FIG15_FIGSIZE: tuple[float, float] = (8, 5)
FIG15_MARKER_MS: int = 6
FIG15_LINE_LW: float = 1.8
FIG5_NSE_MS: int = 6
FIG5_KGE_LINE_LW: float = 1.5
FIG5_KGE_MS: int = 5
FIG5_CI_ALPHA_NSE: float = 0.15
FIG5_CI_ALPHA_KGE: float = 0.08
FIG5_BOOT_N: int = 2000
FIG5_BOOT_ALPHA: float = 0.05
FIG5_BOOT_SEED: int = 111111

# ---- Figure 6 (1:1 scatter + spatial) ------------------------------------
FIG6_FIGSIZE: tuple[float, float] = (12, 8)
FIG6_SCATTER_S: int = 12
FIG6_SCATTER_ALPHA: float = 0.75
FIG6_MAP_S: int = 18
FIG6_MAP_ALPHA: float = 0.85
FIG6_CMAP: str = "RdBu"
FIG6_VMIN: float = -0.3
FIG6_VMAX: float = 0.3
FIG6_LON_LIM: tuple[float, float] = (-125, -66)
FIG6_LAT_LIM: tuple[float, float] = (24, 50)

# ---- Figure 7 (hydrographs) ----------------------------------------------
FIG7_FIGW: float = 11
FIG7_ROW_H: float = 3.2
FIG7_MAX_DAYS: int = 730
FIG7_T0: str = "1995-10-01"        # Validation window start
FIG7_BASIN_IDS: list[str] | None = None   # None = auto-pick 3 case basins
FIG7_OBS_LW: float = 1.0
FIG7_LSTM_LW: float = 1.1
FIG7_MAMBA_LW: float = 1.1
FIG7_OBS_ALPHA: float = 0.85

# ---- Figure 8 (flow-duration curves) --------------------------------------
FIG8_FIGSIZE: tuple[float, float] = (11, 4.8)
FIG8_N_POINTS: int = 180
FIG8_BAND_LO: float = 0.10
FIG8_BAND_HI: float = 0.90
FIG8_EPSILON: float = 1e-6
FIG8_YLIM: tuple[float, float] = (1e-3, 1e3)
FIG8_LINE_LW: float = 2.0
FIG8_BAND_ALPHA: float = 0.18

# ---- Figure 9 (seasonal performance heatmaps) -----------------------------
FIG9_FIGSIZE: tuple[float, float] = (13, 10)
FIG9_CMAP_SCORE: str = "viridis"
FIG9_CMAP_BIAS: str = "RdBu_r"
FIG9_ANNOTATION_FORMAT: str = ".3f"

# ---- Figure 10 (residual diagnostics) -------------------------------------
FIG10_FIGSIZE: tuple[float, float] = (13, 5.2)
FIG10_MAX_SCATTER: int = 100_000
FIG10_SCATTER_ALPHA: float = 0.035
FIG10_N_FLOW_BINS: int = 40
FIG10_CMAP: str = "RdBu_r"
FIG10_ANNOTATION_FORMAT: str = ".3f"

# ---- Figure 11 (peak-event diagnostics) -----------------------------------
FIG11_FIGSIZE: tuple[float, float] = (14, 10)
FIG11_TOP_N_EVENTS: int = 5
FIG11_EVENT_THRESHOLD: float = 0.90
FIG11_MIN_PEAK_SEPARATION: int = 7
FIG11_EVENT_HALF_WINDOW: int = 3
FIG11_VIOLIN_ALPHA: float = 0.55
FIG11_HEATMAP_CMAP: str = "viridis"
FIG11_THRESHOLDS: tuple[float, ...] = (0.90, 0.95, 0.99)

# ---- Figure 12 (uncertainty and reliability) ------------------------------
FIG12_FIGSIZE: tuple[float, float] = (14, 10)
FIG12_MAX_DAYS: int = 730
FIG12_MAX_EVAL_BASINS: int = 50
FIG12_MAX_EVAL_DAYS: int = 1000
FIG12_LEVELS: tuple[float, ...] = (0.50, 0.80, 0.95)
FIG12_PIT_BINS: int = 10
FIG12_BAND_ALPHA: float = 0.16
FIG12_SAMPLE_FILES: tuple[str, ...] = (
    "pred_ensemble_daily_m3s.npy",
    "pred_mc_dropout_daily_m3s.npy",
    "pred_samples_daily_m3s.npy",
)

# ---- Figure 13 (NSE, PR, and KGE vs layers at fixed tier) ---------------
FIG13_FIGSIZE: tuple[float, float] = (9, 4.2)
FIG13_MARKER_MS: int = 7
FIG13_LINE_LW: float = 1.8
TIER_FIG13_DEPTH: int = 1               # Tier for depth scaling (Fig 13)

# ---- Basin auto-pick thresholds (Fig 7 case basins) ----------------------
BASIN_SNOW_FRAC_MIN: float = 0.4
BASIN_RAIN_SNOW_MAX: float = 0.15
BASIN_RAIN_ARIDITY_MAX: float = 1.0
BASIN_ARID_ARIDITY_MIN: float = 1.5

# ---- Misc -----------------------------------------------------------------
# Combined discovery regex — matches BOTH the legacy LSTM folder format AND
# the new condensed Mamba folder format defined in config.save_path().
# Named groups not present in a given branch will be None (callers already
# guard on that via `m.group("layers")` / etc. returning None).
TIER_DIR_RE = re.compile(
    r"^(?:"
    # ---------------------------------------------------------------------
    # Legacy / LSTM folder pattern:
    #   Tier{N}_{Name}_Ep{epoch}_Bs{bs}_Rho{rho}_H{hidden}[_L{layers}][_Seed{n}]
    # e.g. Tier1_Micro_Ep30_Bs100_Rho365_H108_L1_Seed1
    #      Tier1_Micro_Ep30_Bs100_Rho365_H108_L1        (pre-B1 legacy)
    # ---------------------------------------------------------------------
    r"Tier(?P<tier>\d+)_(?P<name>[^_]+)_Ep(?P<epoch>\d+)"
    r"_Bs(?P<bs>\d+)_Rho(?P<rho>\d+)_H(?P<hidden>\d+)"
    r"(?:_L(?P<layers>\d+))?"
    r"(?:_Seed(?P<seed>\d+))?"
    r"|"
    # ---------------------------------------------------------------------
    # New condensed Mamba folder pattern (config.ExperimentConfig.save_path):
    #   Tier{N}_Ep{epoch}_Variant-{variant}_V{version}_head{headdim}
    #   _H{hidden}_L{layers}_dConv{d_conv}[_Rho{rho}][_Seed{n}]
    # e.g. Tier1_Ep2_Variant-standard_V3_head64_H128_L3_dConv7_Seed1
    # ---------------------------------------------------------------------
    r"Tier(?P<tier_mamba>\d+)_Ep(?P<epoch_mamba>\d+)"
    r"_Variant-(?P<variant>[^_]+)_V(?P<version>\d+)"
    r"_head(?P<headdim>\d+)_H(?P<hidden_mamba>\d+)"
    r"_L(?P<layers_mamba>\d+)_dConv(?P<d_conv>\d+)"
    r"(?:_Rho(?P<rho_mamba>\d+))?"
    r"(?:_Seed(?P<seed_mamba>\d+))?"
    r")$"
)
# Normalized group access helpers (one value per logical field regardless of
# which branch matched).
def _re_field(match: re.Match, name: str) -> Optional[str]:
    """Return the value of a logical regex field regardless of the matched branch.

    The combined regex uses alternate named groups for some fields (``tier`` vs
    ``tier_mamba``, ``epoch`` vs ``epoch_mamba``, …) because REs forbid the
    same group name twice. This helper bridges the two namespaces.
    """
    alt = {
        "tier":   match.group("tier")   or match.group("tier_mamba"),
        "epoch":  match.group("epoch")  or match.group("epoch_mamba"),
        "hidden": match.group("hidden") or match.group("hidden_mamba"),
        "layers": match.group("layers") or match.group("layers_mamba"),
        "rho":    match.group("rho")   or match.group("rho_mamba"),
        "seed":   match.group("seed")  or match.group("seed_mamba"),
    }
    if name in alt:
        return alt[name]
    return match.group(name)  # "name" (tier tag e.g. Micro), bs, rho, version, variant, headdim, d_conv
#  ======================  END CONFIGURATION  =============================
# =========================================================================


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _read_params(run_dir: Path) -> Optional[int]:
    params_file = run_dir / "total_params.txt"
    if not params_file.is_file():
        return None
    text = params_file.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "total_params" in line.lower():
            token = line.split(":")[-1].strip().replace(",", "")
            if token.isdigit():
                return int(token)
    return None


def _parse_run_csv_loss(run_dir: Path) -> pd.DataFrame:
    """Parse hydroDL ``run.csv`` lines: ``Epoch N Loss X time Y``."""
    run_csv = run_dir / "run.csv"
    if not run_csv.is_file():
        return pd.DataFrame(columns=["epoch", "training_loss"])
    rows = []
    for line in run_csv.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"Epoch\s+(\d+)\s+Loss\s+([0-9.eE+-]+)", line)
        if m:
            rows.append({"epoch": int(m.group(1)), "training_loss": float(m.group(2))})
    return pd.DataFrame(rows)


def _val_metrics_frame(run_dir: Path) -> Optional[pd.DataFrame]:
    """Adapt the revision's val_metrics.csv to the legacy checkpoint schema."""
    path = run_dir / "val_metrics.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    frame = frame.rename(columns={
        "train_loss": "training_loss",
        "val_median_nse": "median_nse",
        "val_median_kge": "median_kge",
    })
    keep = [c for c in ("epoch", "training_loss", "median_nse", "median_kge")
            if c in frame.columns]
    if "epoch" not in keep or len(keep) < 2:
        return None
    return frame[keep].sort_values("epoch").reset_index(drop=True)


def load_checkpoint_metrics(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load checkpoint_metrics.csv and fill training_loss from fallbacks.

    New runs do not write checkpoint_metrics.csv; for those, adapt
    val_metrics.csv so the legacy plotter can still draw them.
    """
    cp = run_dir / "checkpoint_metrics.csv"
    if not cp.is_file():
        return _val_metrics_frame(run_dir)
    df = pd.read_csv(cp)
    if "training_loss" not in df.columns or df["training_loss"].isna().all():
        loss_file = run_dir / "training_loss.csv"
        if loss_file.is_file():
            loss = pd.read_csv(loss_file)
            if {"epoch", "loss"}.issubset(loss.columns):
                df = df.drop(columns=["training_loss"], errors="ignore")
                df = df.merge(
                    loss.rename(columns={"loss": "training_loss"})[["epoch", "training_loss"]],
                    on="epoch",
                    how="left",
                )
        if "training_loss" not in df.columns or df["training_loss"].isna().all():
            parsed = _parse_run_csv_loss(run_dir)
            if not parsed.empty:
                df = df.drop(columns=["training_loss"], errors="ignore")
                df = df.merge(parsed, on="epoch", how="left")
    return df.sort_values("epoch").reset_index(drop=True)


def load_basin_bundle(run_dir: Path) -> Optional[dict]:
    """Load basin_metrics (+ optional signature tables) for one completed run."""
    metrics_path = run_dir / "basin_metrics.csv"
    if not metrics_path.is_file():
        return None
    metrics = pd.read_csv(metrics_path)
    if len(metrics) != 671:
        return None
    if "NSE" not in metrics.columns or "KGE" not in metrics.columns:
        return None
    if "gauge_id" in metrics.columns:
        metrics["gauge_id"] = metrics["gauge_id"].astype(str).str.zfill(8)

    kge = None
    kge_path = run_dir / "kge_components.csv"
    if kge_path.is_file():
        kge = pd.read_csv(kge_path)
        if "gauge_id" in kge.columns:
            kge["gauge_id"] = kge["gauge_id"].astype(str).str.zfill(8)

    # Prefer dedicated flow_signatures.csv; else columns already in basin_metrics
    flow = None
    flow_path = run_dir / "flow_signatures.csv"
    if flow_path.is_file():
        flow = pd.read_csv(flow_path)
        if "gauge_id" in flow.columns:
            flow["gauge_id"] = flow["gauge_id"].astype(str).str.zfill(8)
    elif {"FHV", "FLV"}.issubset(metrics.columns):
        cols = ["gauge_id", "FHV", "FLV"] if "gauge_id" in metrics.columns else ["FHV", "FLV"]
        flow = metrics[cols].copy()

    return {
        "metrics": metrics,
        "kge": kge,
        "flow": flow,
        "params": _read_params(run_dir),
        "checkpoint": load_checkpoint_metrics(run_dir),
        "pred": run_dir / "pred_daily_m3s.npy",
        "obs": run_dir / "obs_daily_m3s.npy",
        "run_dir": run_dir,
    }


def discover_exp_runs(
    exp_name: str,
    *,
    epoch: Optional[int] = None,
    layers: Optional[int] = None,
    mamba_variant: Optional[str] = None,
    output_root: Optional[Path] = None,
) -> dict[int, Path]:
    """Map tier → best matching ``.../All`` directory under an experiment family."""
    root = output_root or OUTPUT_ROOT
    exp_root = root / exp_name
    if not exp_root.is_dir():
        return {}

    candidates: dict[int, list[tuple[int, Path]]] = {}
    for tier_dir in sorted(exp_root.iterdir()):
        if not tier_dir.is_dir():
            continue
        m = TIER_DIR_RE.match(tier_dir.name)
        if not m:
            continue
        tier = int(_re_field(m, "tier"))
        ep = int(_re_field(m, "epoch"))
        lyr = _re_field(m, "layers")
        lyr_i = int(lyr) if lyr is not None else None
        if mamba_variant is not None and m.group("variant") is not None:
            if m.group("variant") != mamba_variant:
                continue
        if epoch is not None and ep != epoch:
            continue
        if layers is not None and lyr_i is not None and lyr_i != layers:
            continue
        all_dir = tier_dir / "All"
        if not (all_dir / "basin_metrics.csv").is_file():
            continue
        # Prefer longer (higher-epoch) runs first; exact layer-suffix match
        # only breaks ties so a 2-epoch smoke test never beats a full run.
        if layers is not None:
            if lyr_i == layers:
                layer_bonus = 5
            elif lyr_i is None:
                layer_bonus = 2  # legacy folders without _L suffix
            else:
                layer_bonus = 0  # unreachable (filtered above)
        else:
            layer_bonus = 0
        score = ep * 10 + layer_bonus
        candidates.setdefault(tier, []).append((score, all_dir))

    chosen: dict[int, Path] = {}
    for tier, opts in candidates.items():
        opts.sort(key=lambda x: x[0], reverse=True)
        chosen[tier] = opts[0][1]
    return chosen


def resolve_exp_names(args) -> tuple[str, str]:
    """Pick LSTM / Mamba experiment folder names from CLI or config defaults."""
    cfg_lstm = ExperimentConfig(
        model_type="lstm", param_tier=3,
        lstm_layers=args.lstm_layers or DEFAULT_LSTM_LAYERS,
    )
    cfg_mamba = ExperimentConfig(
        model_type="mamba", param_tier=3,
        mamba_version=args.mamba_version or DEFAULT_MAMBA_VERSION,
        mamba_layers=args.mamba_layers or DEFAULT_MAMBA_LAYERS,
        mamba_variant=DEFAULT_MAMBA_VARIANT,
    )
    if args.lstm_layers is not None:
        cfg_lstm = ExperimentConfig(
            model_type="lstm", param_tier=3, lstm_layers=args.lstm_layers
        )
    if args.mamba_version is not None or args.mamba_layers is not None:
        cfg_mamba = ExperimentConfig(
            model_type="mamba",
            param_tier=3,
            mamba_version=args.mamba_version or cfg_mamba.mamba_version,
            mamba_layers=args.mamba_layers or cfg_mamba.mamba_layers,
        )

    lstm_exp = args.lstm_exp or cfg_lstm.exp_name()
    mamba_exp = args.mamba_exp or cfg_mamba.exp_name()
    return lstm_exp, mamba_exp


def collect_ladder(
    lstm_exp: str,
    mamba_exp: str,
    *,
    epoch: Optional[int],
    lstm_epoch: Optional[int] = None,
    mamba_epoch: Optional[int] = None,
    lstm_layers: Optional[int],
    mamba_layers: Optional[int],
    mamba_variant: Optional[str] = None,
    output_root: Optional[Path] = None,
) -> pd.DataFrame:
    """One row per (model, tier) with median metrics and param counts."""
    rows = []
    lstm_tier_ladders = {
        1: LSTM_1L_TIER_CONFIG,
        2: LSTM_2L_TIER_CONFIG,
        3: LSTM_3L_TIER_CONFIG,
    }
    mamba_tier_ladders = {
        1: MAMBA_1L_TIER_CONFIG,
        2: MAMBA_2L_TIER_CONFIG,
        3: MAMBA_3L_TIER_CONFIG,
    }
    for model, exp, layers, model_epoch in [
        ("LSTM", lstm_exp, lstm_layers, lstm_epoch if lstm_epoch is not None else epoch),
        ("Mamba", mamba_exp, mamba_layers, mamba_epoch if mamba_epoch is not None else epoch),
    ]:
        tier_ladders = lstm_tier_ladders if model == "LSTM" else mamba_tier_ladders
        tier_cfg = tier_ladders.get(layers or 1)
        if tier_cfg is None:
            raise ValueError(f"Unsupported {model} depth={layers}; expected 1, 2, or 3.")
        runs = discover_exp_runs(
            exp,
            epoch=model_epoch,
            layers=layers,
            mamba_variant=mamba_variant if model == "Mamba" else None,
            output_root=output_root,
        )
        for tier, run_dir in sorted(runs.items()):
            bundle = load_basin_bundle(run_dir)
            if bundle is None:
                continue
            metrics = bundle["metrics"]
            # Parameter count must come from the completed run itself.
            # Static estimates are unsafe because Mamba counts depend on version,
            # d_state, expand, headdim, and the installed mamba_ssm release.
            params = bundle["params"]
            rows.append(
                {
                    "model": model,
                    "exp": exp,
                    "tier": tier,
                    "tier_name": TIER_NAMES.get(tier, f"Tier{tier}"),
                    "params": params,
                    "median_nse": float(np.nanmedian(metrics["NSE"])),
                    "median_kge": float(np.nanmedian(metrics["KGE"])),
                    "n_basins": len(metrics),
                    "run_dir": str(run_dir),
                }
            )
    return pd.DataFrame(rows)


def collect_training_time(
    lstm_exp: str,
    mamba_exp: str,
    *,
    lstm_layers: Optional[int],
    mamba_layers: Optional[int],
    mamba_variant: Optional[str] = None,
    output_root: Optional[Path] = None,
) -> pd.DataFrame:
    """One row per (model, tier) with total training time for 30 epochs."""
    rows = []
    lstm_tier_ladders = {
        1: LSTM_1L_TIER_CONFIG,
        2: LSTM_2L_TIER_CONFIG,
        3: LSTM_3L_TIER_CONFIG,
    }
    mamba_tier_ladders = {
        1: MAMBA_1L_TIER_CONFIG,
        2: MAMBA_2L_TIER_CONFIG,
        3: MAMBA_3L_TIER_CONFIG,
    }
    for model, exp, layers in [
        ("LSTM", lstm_exp, lstm_layers),
        ("Mamba", mamba_exp, mamba_layers),
    ]:
        tier_ladders = lstm_tier_ladders if model == "LSTM" else mamba_tier_ladders
        tier_cfg = tier_ladders.get(layers or 1)
        if tier_cfg is None:
            raise ValueError(f"Unsupported {model} depth={layers}; expected 1, 2, or 3.")
        runs = discover_exp_runs(
            exp,
            epoch=None,  # Get highest completed epoch
            layers=layers,
            mamba_variant=mamba_variant if model == "Mamba" else None,
            output_root=output_root,
        )
        for tier, run_dir in sorted(runs.items()):
            run_csv = run_dir / "run.csv"
            if not run_csv.is_file():
                continue
            try:
                run_df = pd.read_csv(run_csv, sep=r"\s+", header=None, 
                                    names=["epoch", "loss", "time"])
                # Calculate total time for 30 epochs (or all available if fewer)
                # If more than 30 epochs, only use first 30 for fair comparison
                if len(run_df) > 30:
                    run_df = run_df.head(30)
                # Time is in seconds, convert to minutes
                total_time_seconds = run_df["time"].sum()
                total_time_minutes = total_time_seconds / 60.0
                n_epochs = len(run_df)
                rows.append(
                    {
                        "model": model,
                        "exp": exp,
                        "tier": tier,
                        "tier_name": TIER_NAMES.get(tier, f"Tier{tier}"),
                        "total_time_minutes": total_time_minutes,
                        "n_epochs": n_epochs,
                        "run_dir": str(run_dir),
                    }
                )
            except Exception as e:
                print(f"[collect_training_time] Failed to parse {run_csv}: {e}")
                continue
    return pd.DataFrame(rows)


def collect_all_layers_training_time(
    lstm_exps: Optional[list[str]] = None,
    mamba_exps: Optional[list[str]] = None,
    *,
    mamba_version: Optional[int] = None,
    mamba_variant: Optional[str] = None,
    output_root: Optional[Path] = None,
) -> pd.DataFrame:
    """One row per (model, tier, layers) with total training time for 30 epochs.
    
    Collects training time data from all available layer depths for both LSTM and Mamba models.
    Useful for plotting training time efficiency across different depths.
    """
    root = output_root or OUTPUT_ROOT
    if lstm_exps is None:
        lstm_exps = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and p.name.startswith("LSTM")
        ) if root.is_dir() else []
    if mamba_exps is None:
        mamba_exps = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and p.name.startswith("Mamba")
        ) if root.is_dir() else []
    
    rows = []
    for exp in list(lstm_exps) + list(mamba_exps):
        model = "LSTM" if exp.startswith("LSTM") else "Mamba"
        if exp.endswith("_RhoScaling"):
            continue
        if model == "Mamba":
            version_match = re.match(r"^Mamba(?P<version>\d+)_", exp)
            if version_match is None:
                continue
            # If mamba_version is specified, check the experiment name for version
            if mamba_version is not None:
                exp_version = int(version_match.group("version"))
                if exp_version != mamba_version:
                    continue
            # If mamba_variant is specified, check the experiment name for variant
            if mamba_variant is not None:
                variant_match = re.search(r"_Variant-(\w+)_", exp)
                if variant_match and variant_match.group(1) != mamba_variant:
                    continue
        
        exp_root = root / exp
        if not exp_root.is_dir():
            continue
        
        # Extract layer depth from experiment name
        m_exp = re.search(r"_(\d+)L$", exp)
        exp_layers = int(m_exp.group(1)) if m_exp else None
        
        for tier_dir in sorted(exp_root.iterdir()):
            if not tier_dir.is_dir():
                continue
            m = TIER_DIR_RE.match(tier_dir.name)
            if not m:
                continue
            if model == "Mamba" and mamba_variant is not None:
                if m.group("variant") != mamba_variant:
                    continue
            try:
                t = int(_re_field(m, "tier"))
            except (TypeError, ValueError):
                continue
            
            all_dir = tier_dir / "All"
            if not all_dir.is_dir():
                continue
            
            run_csv = all_dir / "run.csv"
            if not run_csv.is_file():
                continue
            
            try:
                run_df = pd.read_csv(run_csv, sep=r"\s+", header=None, 
                                    names=["epoch", "loss", "time"])
                # Calculate total time for 30 epochs (or all available if fewer)
                # If more than 30 epochs, only use first 30 for fair comparison
                if len(run_df) > 30:
                    run_df = run_df.head(30)
                # Skip runs with fewer than 30 epochs for fair comparison
                if len(run_df) < 30:
                    continue
                # Time is in seconds, convert to minutes
                total_time_seconds = run_df["time"].sum()
                total_time_minutes = total_time_seconds / 60.0
                n_epochs = len(run_df)
                rows.append(
                    {
                        "model": model,
                        "exp": exp,
                        "tier": t,
                        "tier_name": TIER_NAMES.get(t, f"Tier{t}"),
                        "layers": exp_layers,
                        "total_time_minutes": total_time_minutes,
                        "n_epochs": n_epochs,
                        "run_dir": str(all_dir),
                    }
                )
            except Exception as e:
                print(f"[collect_all_layers_training_time] Failed to parse {run_csv}: {e}")
                continue
    
    return pd.DataFrame(rows)


def collect_all_layers_ladder(
    lstm_exps: Optional[list[str]] = None,
    mamba_exps: Optional[list[str]] = None,
    *,
    epoch: Optional[int],
    lstm_epoch: Optional[int] = None,
    mamba_epoch: Optional[int] = None,
    mamba_version: Optional[int] = None,
    mamba_variant: Optional[str] = None,
    output_root: Optional[Path] = None,
) -> pd.DataFrame:
    """One row per (model, tier, layers) with median metrics and param counts.
    
    Collects data from all available layer depths for both LSTM and Mamba models.
    Useful for plotting parameter efficiency across different depths.
    """
    root = output_root or OUTPUT_ROOT
    if lstm_exps is None:
        lstm_exps = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and p.name.startswith("LSTM")
        ) if root.is_dir() else []
    if mamba_exps is None:
        mamba_exps = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and p.name.startswith("Mamba")
        ) if root.is_dir() else []
    
    rows = []
    for exp in list(lstm_exps) + list(mamba_exps):
        model = "LSTM" if exp.startswith("LSTM") else "Mamba"
        if exp.endswith("_RhoScaling"):
            continue
        if model == "Mamba":
            version_match = re.match(r"^Mamba(?P<version>\d+)_", exp)
            if version_match is None:
                continue
            # If mamba_version is specified, check the experiment name for version
            if mamba_version is not None:
                exp_version = int(version_match.group("version"))
                if exp_version != mamba_version:
                    continue
            # If mamba_variant is specified, check the experiment name for variant
            if mamba_variant is not None:
                variant_match = re.search(r"_Variant-(\w+)_", exp)
                if variant_match and variant_match.group(1) != mamba_variant:
                    continue
        
        exp_root = root / exp
        if not exp_root.is_dir():
            continue
        
        # Extract layer depth from experiment name
        m_exp = re.search(r"_(\d+)L$", exp)
        exp_layers = int(m_exp.group(1)) if m_exp else None
        
        for tier_dir in sorted(exp_root.iterdir()):
            if not tier_dir.is_dir():
                continue
            m = TIER_DIR_RE.match(tier_dir.name)
            if not m:
                continue
            if model == "Mamba" and mamba_variant is not None:
                if m.group("variant") != mamba_variant:
                    continue
            try:
                t = int(_re_field(m, "tier"))
            except (TypeError, ValueError):
                continue
            ep_raw = _re_field(m, "epoch")
            ep = int(ep_raw) if ep_raw is not None else -1
            # Check model-specific epoch
            model_epoch = lstm_epoch if model == "LSTM" else mamba_epoch
            if model_epoch is not None and ep != model_epoch:
                continue
            # Fall back to general epoch if model-specific not set
            if model_epoch is None and epoch is not None and ep != epoch:
                continue
            lyr_raw = _re_field(m, "layers")
            lyr = int(lyr_raw) if lyr_raw is not None else exp_layers
            if lyr is None:
                continue
            all_dir = tier_dir / "All"
            if not (all_dir / "basin_metrics.csv").is_file():
                continue
            bundle = load_basin_bundle(all_dir)
            if bundle is None:
                continue
            metrics = bundle["metrics"]
            params = bundle["params"]
            rows.append(
                {
                    "model": model,
                    "exp": exp,
                    "tier": t,
                    "tier_name": TIER_NAMES.get(t, f"Tier{t}"),
                    "layers": lyr,
                    "params": params,
                    "median_nse": float(np.nanmedian(metrics["NSE"])),
                    "median_kge": float(np.nanmedian(metrics["KGE"])),
                    "n_basins": len(metrics),
                    "run_dir": str(all_dir),
                }
            )
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["model", "layers", "tier"]).reset_index(drop=True)
    return df


def bootstrap_median_ci(values: np.ndarray, n_boot: int = FIG5_BOOT_N, alpha: float = FIG5_BOOT_ALPHA,
                        seed: int = FIG5_BOOT_SEED) -> tuple[float, float, float]:
    """Return (median, lo, hi) of the median via bootstrap."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    med = float(np.median(v))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(v, size=len(v), replace=True)
        boots[i] = np.median(sample)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return med, float(lo), float(hi)


def load_gauge_meta() -> pd.DataFrame:
    path = DATA_DIR / "gauge_ids.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(8)
    return df


def load_camels_clim(cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load CAMELS climate attributes (frac_snow, aridity, …), downloading if needed."""
    cache_dir = cache_dir or (DATA_DIR / "camels_attributes_v2.0")
    cache_dir.mkdir(parents=True, exist_ok=True)
    clim_path = cache_dir / "camels_clim.txt"
    if not clim_path.is_file():
        try:
            import urllib.request

            print(f"[download] {CLIM_URL}")
            urllib.request.urlretrieve(CLIM_URL, clim_path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not download camels_clim.txt ({exc}). "
                f"Place it at {clim_path} manually."
            ) from exc
    clim = pd.read_csv(clim_path, sep=";", dtype={"gauge_id": str})
    clim["gauge_id"] = clim["gauge_id"].astype(str).str.zfill(8)
    return clim


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------


def apply_style():
    # D4 (2026-11-10): serif + 300 dpi so the generated figures match the
    # house style (figure_style.py).  Times-like serif matches txfonts.
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": FONT_SIZE,
            "axes.titlesize": AX_TITLE_SIZE,
            "axes.labelsize": AX_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": GRID_LS,
        }
    )


def save_fig(fig: plt.Figure, fig_dir: Path, stem: str):
    # D4 §4.5.1: every figure ships as PNG (300 dpi) + vector PDF.
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{stem}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  saved {out}")
    plt.close(fig)


def save_data(df: pd.DataFrame, fig_dir: Path, stem: str):
    """Save the dataframe behind a figure next to its PNG."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{stem}_data.csv"
    df.to_csv(out, index=False)
    print(f"  saved {out}")


def _model_style(model: str) -> dict:
    if model.lower().startswith("mamba"):
        return {"color": COLOR_MAMBA, "ls": "-", "label": "Mamba", "marker": "o"}
    return {"color": COLOR_LSTM, "ls": "--", "label": "LSTM", "marker": "s"}


def _layer_text_from_bundles(*bundles: dict) -> str:
    """Return a compact model/depth label from loaded run bundles."""
    labels = []
    for model, bundle in zip(("LSTM", "Mamba"), bundles):
        run_dir = bundle.get("run_dir")
        layer = "?L"
        if run_dir is not None:
            for part in (Path(run_dir).parent.name, Path(run_dir).parent.parent.name):
                match = re.search(r"(?:_L(\d+)|_(\d+)L)(?:_|$)", part)
                if match:
                    layer = f"{next(group for group in match.groups() if group)}L"
                    break
        labels.append(f"{model} {layer}")
    return ", ".join(labels)


def _layer_text_from_frame(frame: pd.DataFrame) -> str:
    """Return model/depth labels from a scaling dataframe."""
    labels = []
    if not frame.empty and {"model", "layers"}.issubset(frame.columns):
        for model in ("LSTM", "Mamba"):
            values = frame.loc[frame["model"] == model, "layers"].dropna().unique()
            if len(values):
                labels.append(f"{model} {', '.join(f'{int(v)}L' for v in sorted(values))}")
    return ", ".join(labels)


def _layer_text_from_experiments(frame: pd.DataFrame) -> str:
    """Return model/depth labels by parsing experiment folder names."""
    labels = []
    if not frame.empty and {"model", "exp"}.issubset(frame.columns):
        for model in ("LSTM", "Mamba"):
            values = set()
            for exp in frame.loc[frame["model"] == model, "exp"].dropna():
                match = re.search(r"_(\d+)L(?:_RhoScaling)?$", str(exp))
                if match:
                    values.add(int(match.group(1)))
            if values:
                labels.append(f"{model} {', '.join(f'{v}L' for v in sorted(values))}")
    return ", ".join(labels)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure1_learning_curves(
    lstm_run: Optional[Path],
    mamba_run: Optional[Path],
    fig_dir: Path,
    compare_tier: int,
):
    """Fig 1: 3-panel Loss / NSE / KGE vs epoch."""
    panels = []
    if lstm_run is not None:
        cp = load_checkpoint_metrics(lstm_run)
        if cp is None:
            print(f"[Fig1] skip: missing checkpoint_metrics.csv in {lstm_run}")
            return
        panels.append(("LSTM", cp))
    if mamba_run is not None:
        cp = load_checkpoint_metrics(mamba_run)
        if cp is None:
            print(f"[Fig1] skip: missing checkpoint_metrics.csv in {mamba_run}")
            return
        panels.append(("Mamba", cp))
    
    if not panels:
        print(f"[Fig1] skip: no valid model runs provided")
        return

    fig, axes = plt.subplots(1, 3, figsize=FIG1_FIGSIZE, constrained_layout=True)
    # D4 (2026-11-10): these curves come from val_metrics.csv, so they are
    # validation scores, NOT test scores.  Relabelled to keep the figure honest.
    titles = [
        "(a) Training loss (normalised)",
        "(b) Median validation NSE",
        "(c) Median validation KGE",
    ]
    ycols = ["training_loss", "median_nse", "median_kge"]

    for model, cp in panels:
        sty = _model_style(model)
        loss = cp["training_loss"].astype(float) if "training_loss" in cp else None
        if loss is not None and loss.notna().any():
            # Normalize each curve by its own first finite value for panel A
            base = loss[loss.notna()].iloc[0]
            cp = cp.copy()
            cp["training_loss_norm"] = loss / base if base and np.isfinite(base) else loss
        else:
            cp = cp.copy()
            cp["training_loss_norm"] = np.nan
            print(f"[Fig1] note: no training_loss for {model}; Panel A skips it")

        series = {
            "training_loss": cp["training_loss_norm"],
            "median_nse": cp["median_nse"],
            "median_kge": cp["median_kge"],
        }
        for ax, title, ycol in zip(axes, titles, ycols):
            y = series[ycol]
            if y.isna().all():
                continue
            ax.plot(
                cp["epoch"],
                y,
                color=sty["color"],
                ls=sty["ls"],
                marker=sty["marker"],
                ms=FIG1_MARKER_MS,
                label=sty["label"],
                lw=FIG1_LINE_LW,
            )
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[0].set_ylabel("Loss / Loss\u2080")
    axes[1].set_ylabel("Median validation NSE")
    axes[2].set_ylabel("Median validation KGE")
    axes[0].legend(frameon=False)
    def _layer_label(run: Path) -> str:
        # Runs are passed as .../<experiment>/<tier-run>/All.
        for part in (run.parent.name, run.parent.parent.name):
            match = re.search(r"(?:_L(\d+)|_(\d+)L)(?:_|$)", part)
            if match:
                return f"{next(group for group in match.groups() if group)}L"
        return "?L"

    tier_label = TIER_NAMES.get(compare_tier, "")
    # fig.suptitle(
    #     f"Convergence (Tier {compare_tier}: {tier_label}; "
    #     f"LSTM {_layer_label(lstm_run)}, Mamba {_layer_label(mamba_run)})",
    #     y=1.04,
    # )
    save_fig(fig, fig_dir, "fig06_learning_curves")
    # Persist the exact plotted curves (one row per model × epoch)
    frames = []
    for model, cp in panels:
        f = cp.copy()
        if "training_loss_norm" not in f.columns:
            loss = f["training_loss"].astype(float) if "training_loss" in f else None
            if loss is not None and loss.notna().any():
                base = loss[loss.notna()].iloc[0]
                f["training_loss_norm"] = loss / base if base and np.isfinite(base) else loss
            else:
                f["training_loss_norm"] = np.nan
        f["model"] = model
        cols = ["model", "epoch", "training_loss", "training_loss_norm",
                "median_nse", "median_kge"]
        frames.append(f[[c for c in cols if c in f.columns]])
    save_data(pd.concat(frames, ignore_index=True), fig_dir, "fig06_learning_curves")


def figure2_param_scaling(
    ladder: pd.DataFrame,
    fig_dir: Path,
    tiers: Optional[list[int]] = None,
    *,
    lstm_layers: Optional[int] = None,
    mamba_layers: Optional[int] = None,
    all_layers: bool = False,
):
    """Fig 2: Parameters (log) vs median NSE / KGE.

    Args:
        ladder: Full per-(model,tier) metrics frame.
        fig_dir: Output directory.
        tiers: Optional allow-list of tier numbers to plot (e.g. [4, 6, 7]).
        all_layers: If True, plot all available layer depths instead of single depth.
    """
    if ladder.empty:
        print("[Fig2] skip: no completed tiers found")
        return
    plot_ladder = ladder.copy()
    if tiers is not None:
        plot_ladder = plot_ladder[plot_ladder["tier"].isin(tiers)].copy()
        if plot_ladder.empty:
            print(f"[Fig2] skip: no tiers match filter {tiers}")
            return
    
    fig, axes = plt.subplots(1, 2, figsize=FIG2_FIGSIZE, constrained_layout=False)
    
    custom_lines = []
    custom_labels = []
    
    if all_layers and "layers" in plot_ladder.columns:
        # Plot all layers with different markers/styles
        layer_markers = {1: 'o', 2: 's', 3: '^'}
        layer_styles = {1: '-', 2: '--', 3: ':'}
        
        # Use tier names as x-axis categories
        tier_order = sorted(plot_ladder["tier"].unique())
        tier_labels = [plot_ladder[plot_ladder["tier"] == t]["tier_name"].values[0] for t in tier_order]
        
        for model, g in plot_ladder.groupby("model"):
            sty = _model_style(model)
            g = g.dropna(subset=["params"]).sort_values(["layers", "tier"])
            if g.empty:
                continue
            
            for layer, layer_group in g.groupby("layers"):
                marker = layer_markers.get(layer, 'o')
                ls = layer_styles.get(layer, '-')
                
                # Plot using tier numbers as x positions
                tier_nums = layer_group["tier"].tolist()
                tier_params = layer_group["params"].tolist()
                
                # Sort by tier number to maintain order
                sorted_data = sorted(zip(tier_nums, tier_params, layer_group["median_nse"].tolist()))
                tier_nums_sorted, params_sorted, nse_sorted = zip(*sorted_data) if sorted_data else ([], [], [])
                
                axes[0].plot(
                    tier_nums_sorted, nse_sorted,
                    color=sty["color"], ls="none", marker=marker,
                    ms=FIG2_MARKER_MS, lw=FIG2_LINE_LW,
                )
                
                # KGE - use same tier order
                kge_sorted = [layer_group.set_index("tier").loc[t, "median_kge"] for t in tier_nums_sorted]
                
                axes[1].plot(
                    tier_nums_sorted, kge_sorted,
                    color=sty["color"], ls="none", marker=marker,
                    ms=FIG2_MARKER_MS, lw=FIG2_LINE_LW,
                )
        
        # Set x-axis to show tier names
        axes[0].set_xticks(tier_order)
        axes[0].set_xticklabels(tier_labels)
        axes[1].set_xticks(tier_order)
        axes[1].set_xticklabels(tier_labels)
        
        # Create custom legend: show layers with markers, then colors for models
        # Layer entries with markers
        for layer in sorted(layer_markers.keys()):
            marker = layer_markers[layer]
            line = Line2D([0], [0], color='gray', ls="none", marker=marker, 
                          ms=FIG2_MARKER_MS, lw=FIG2_LINE_LW)
            custom_lines.append(line)
            custom_labels.append(f"{layer}L")
        
        # Model entries with colors
        for model in ["LSTM", "Mamba"]:
            sty = _model_style(model)
            line = Line2D([0], [0], color=sty["color"], ls="none", marker='o',
                          ms=FIG2_MARKER_MS, lw=FIG2_LINE_LW)
            custom_lines.append(line)
            custom_labels.append(sty["label"])
        
        # Set x-axis to show tier names
        axes[0].set_xticks(tier_order)
        axes[0].set_xticklabels(tier_labels)
        axes[1].set_xticks(tier_order)
        axes[1].set_xticklabels(tier_labels)
        axes[0].set_xlim(min(tier_order) - 0.5, max(tier_order) + 0.5)
        axes[1].set_xlim(min(tier_order) - 0.5, max(tier_order) + 0.5)
    else:
        # Original single-layer plotting with tier names on x-axis
        tier_order = sorted(plot_ladder["tier"].unique())
        tier_labels = [plot_ladder[plot_ladder["tier"] == t]["tier_name"].values[0] for t in tier_order]
        
        for model, g in plot_ladder.groupby("model"):
            sty = _model_style(model)
            g = g.dropna(subset=["params"]).sort_values("tier")
            if g.empty:
                continue
            
            tier_nums = g["tier"].tolist()
            axes[0].plot(
                tier_nums, g["median_nse"],
                color=sty["color"], ls="none", marker=sty["marker"],
                ms=FIG2_MARKER_MS, lw=FIG2_LINE_LW, label=sty["label"],
            )
            axes[1].plot(
                tier_nums, g["median_kge"],
                color=sty["color"], ls="none", marker=sty["marker"],
                ms=FIG2_MARKER_MS, lw=FIG2_LINE_LW, label=sty["label"],
            )
        
        # Set x-axis to show tier names
        axes[0].set_xticks(tier_order)
        axes[0].set_xticklabels(tier_labels)
        axes[1].set_xticks(tier_order)
        axes[1].set_xticklabels(tier_labels)
        axes[0].set_xlim(min(tier_order) - 0.5, max(tier_order) + 0.5)
        axes[1].set_xlim(min(tier_order) - 0.5, max(tier_order) + 0.5)
    
    for ax, ylab, title in zip(
        axes,
        ["Median test NSE", "Median test KGE"],
        ["(a) Parameter efficiency (NSE)", "(b) Parameter efficiency (KGE)"],
    ):
        if all_layers and "layers" in plot_ladder.columns:
            ax.set_xlabel("Parameter tier")
        else:
            ax.set_xscale("log")
            ax.set_xlabel("Trainable parameters")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    
    # Place a single legend on the right side outside the plot
    if all_layers and "layers" in plot_ladder.columns and custom_lines:
        fig.legend(custom_lines, custom_labels, frameon=False, loc='center left', bbox_to_anchor=(0.85, 0.5))
        plt.subplots_adjust(right=0.88)
    else:
        lines1, labels1 = axes[0].get_legend_handles_labels()
        fig.legend(lines1, labels1, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.subplots_adjust(right=0.88)
    title_parts = []
    if tiers is not None:
        title_parts.append(f"tiers {sorted(tiers)}")
    if all_layers:
        title_parts.append("all layers")
    else:
        layer_parts = []
        if lstm_layers is not None:
            layer_parts.append(f"LSTM {lstm_layers}L")
        if mamba_layers is not None:
            layer_parts.append(f"Mamba {mamba_layers}L")
        if layer_parts:
            title_parts.append("layers: " + ", ".join(layer_parts))
    title_extra = f" ({'; '.join(title_parts)})" if title_parts else ""
    # fig.suptitle(f"Capacity & parameter efficiency{title_extra}", y=1.03)
    stem = "fig04_param_scaling_all_layers" if all_layers else "fig04_param_scaling"
    save_fig(fig, fig_dir, stem)
    data_cols = ["model", "exp", "tier", "tier_name", "params",
                 "median_nse", "median_kge", "n_basins"]
    if all_layers and "layers" in plot_ladder.columns:
        data_cols.insert(3, "layers")
    save_data(
        plot_ladder.sort_values(["model", "params"])[data_cols],
        fig_dir, stem,
    )


def figure3_ecdf(
    lstm_bundle: dict,
    mamba_bundle: dict,
    fig_dir: Path,
    compare_tier: int,
):
    """Fig 3: ECDF of basin NSE, KGE, FHV, and FLV at matched tier.
    
    NSE and KGE are from bundle["metrics"], while FHV and FLV are from bundle["flow"].
    """
    fig, axes = plt.subplots(2, 2, figsize=FIG3_FIGSIZE, constrained_layout=True)
    axes = axes.flatten()
    ecdf_rows: list[dict] = []
    
    # Metrics to plot: NSE, KGE from bundle["metrics"], FHV, FLV from bundle["flow"]
    metrics_config = [
        ("NSE", "metrics", "(a) ECDF of basin NSE", 0.0, 1.0),
        ("KGE", "metrics", "(b) ECDF of basin KGE", 0.0, 1.0),
        ("FHV", "flow", "(c) ECDF of FHV", None, None),
        ("FLV", "flow", "(d) ECDF of FLV", None, None),
    ]
    
    for ax, (metric, source, title, xlim_min, xlim_max) in zip(axes, metrics_config):
        for model, bundle in [("LSTM", lstm_bundle), ("Mamba", mamba_bundle)]:
            sty = _model_style(model)
            
            # Get data from appropriate source
            if source == "metrics":
                data = bundle["metrics"]
            else:  # flow
                data = bundle.get("flow")
                if data is None:
                    # Fallback to metrics if flow not available
                    data = bundle["metrics"]
            
            vals = np.sort(data[metric].dropna().to_numpy())
            if len(vals) == 0:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, y, color=sty["color"], ls=sty["ls"], lw=FIG3_LINE_LW, label=sty["label"])
            for v, q in zip(vals, y):
                ecdf_rows.append({"model": model, "metric": metric,
                                  "value": float(v), "ecdf": float(q)})
        
        ax.set_xlabel(metric)
        ax.set_ylabel("Cumulative fraction of basins")
        if xlim_min is not None and xlim_max is not None:
            ax.set_xlim(xlim_min, xlim_max)
        elif metric == "FLV":
            # D4 art review (2026-11-10): FLV is a signed, heavy-tailed ratio
            # (1st-99th percentile spans ~8 orders of magnitude because the
            # denominator is the low-flow volume).  A linear ECDF collapses to
            # a wall at 0.  Draw the same ECDF on a symlog axis, keeping every
            # basin; the axis label states the transform.
            ax.set_xscale("symlog", linthresh=1.0, linscale=0.6)
            ax.set_xlabel("FLV (symlog)")
            parts = []
            for bundle in (lstm_bundle, mamba_bundle):
                frame = bundle.get("flow") if bundle.get("flow") is not None else bundle["metrics"]
                if frame is not None and metric in frame:
                    parts.append(frame[metric].dropna().to_numpy())
            if parts:
                combined = np.concatenate(parts)
                lo, hi = np.nanpercentile(combined, [0.5, 99.5])
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    ax.set_xlim(lo, hi)
                ax.text(
                    0.02, 0.04,
                    f"n = {combined.size // 2} basins per model\n(symlog; tails clipped at 0.5/99.5 %)",
                    transform=ax.transAxes, fontsize=7, color="#555555",
                    va="bottom", ha="left",
                )
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.legend(frameon=False, loc="lower right")
    
    # fig.suptitle(
    #     f"Basin ECDFs (Tier {compare_tier}; "
    #             f"layers: {_layer_text_from_bundles(lstm_bundle, mamba_bundle)})",
    #     y=1.03,
    # )
    save_fig(fig, fig_dir, "fig03_ecdf")
    if ecdf_rows:
        save_data(pd.DataFrame(ecdf_rows), fig_dir, "fig03_ecdf")


def figure4_signatures(
    lstm_bundle: dict,
    mamba_bundle: dict,
    fig_dir: Path,
    compare_tier: int,
):
    """Fig 4: Violin / box plots of hydrologic signatures."""
    records = []
    for model, bundle in [("LSTM", lstm_bundle), ("Mamba", mamba_bundle)]:
        m = bundle["metrics"]
        for metric in ["NSE", "KGE"]:
            for v in m[metric].dropna():
                records.append({"model": model, "metric": metric, "value": float(v)})

        kge = bundle.get("kge")
        if kge is not None:
            for col in ["r", "beta", "gamma"]:
                if col in kge.columns:
                    for v in kge[col].dropna():
                        records.append({"model": model, "metric": col, "value": float(v)})

        flow = bundle.get("flow")
        if flow is not None:
            for col in ["FHV", "FLV"]:
                if col in flow.columns:
                    for v in flow[col].dropna():
                        records.append({"model": model, "metric": col, "value": float(v)})
        else:
            for col in ["FHV", "FLV"]:
                if col in m.columns:
                    for v in m[col].dropna():
                        records.append({"model": model, "metric": col, "value": float(v)})

    df = pd.DataFrame(records)
    if df.empty:
        print("[Fig4] skip: no signature columns found")
        return

    order = [c for c in ["NSE", "KGE", "r", "beta", "FHV", "FLV"] if c in set(df["metric"])]
    fig, axes = plt.subplots(2, 3, figsize=(FIG4_COL_W * 3, FIG4_H * 2), constrained_layout=True)
    axes = axes.flatten()

    for idx, (ax, metric) in enumerate(zip(axes, order)):
        parts = []
        labels = []
        colors = []
        for model in ["LSTM", "Mamba"]:
            vals = df.loc[(df["model"] == model) & (df["metric"] == metric), "value"].to_numpy()
            if len(vals) == 0:
                continue
            parts.append(vals)
            labels.append(model)
            colors.append(_model_style(model)["color"])
        if not parts:
            continue
        # D4 art review (2026-11-10): display-clip the heavy-tailed metrics
        # for the violin body only.  The CSV keeps the raw values; the panel
        # states how many points fall outside the displayed range.
        display = [np.asarray(p, dtype=float) for p in parts]
        n_clipped = 0
        if metric in {"FLV", "FHV"}:
            combined = np.concatenate(display)
            lo, hi = np.nanpercentile(combined, [5, 95])
            clipped = [np.clip(p, lo, hi) for p in display]
        elif metric == "beta":
            lo, hi = 0.0, 3.0
            clipped = [np.clip(p, lo, hi) for p in display]
        else:
            clipped = display
        if metric in {"FLV", "FHV", "beta"}:
            n_clipped = int(sum(np.sum((p < lo) | (p > hi)) for p in display))
        vp = ax.violinplot(clipped, positions=list(range(len(clipped))), showmeans=False,
                           showmedians=True, showextrema=False)
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(FIG4_VIOLIN_ALPHA)
            body.set_edgecolor(color)
        if "cmedians" in vp:
            vp["cmedians"].set_color("black")
        ax.set_xticks(list(range(len(labels))))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(f"({chr(97 + idx)}) {metric}")
        # D4 art review (2026-11-10): per-metric view rules.  NSE/KGE/r are
        # bounded but a few Mamba blow-ups stretch the violins; FLV/FHV are
        # heavy-tailed by construction; beta is a ratio around 1.
        values = np.concatenate(parts) if parts else np.array([])
        if metric in {"NSE", "KGE"}:
            lo, hi = -1.5, 1.0
            ax.set_ylim(lo, hi)
            n_out = int(np.sum((values < lo) | (values > hi)))
            if n_out:
                ax.text(0.03, 0.03, f"{n_out} values outside axis", transform=ax.transAxes,
                        fontsize=6.5, color="#7a1f1f", va="bottom")
        elif metric == "r":
            ax.set_ylim(-0.2, 1.0)
        elif metric == "beta":
            ax.set_ylim(0.0, 3.0)
            if n_clipped:
                ax.text(0.03, 0.03, f"display clipped [0,3]\n{n_clipped} outside",
                        transform=ax.transAxes, fontsize=6.5, color="#7a1f1f", va="bottom")
        elif metric in {"FLV", "FHV"}:
            ax.set_ylim(lo, hi)
            if n_clipped:
                ax.text(0.03, 0.03, f"display clipped 5-95 %\n{n_clipped} outside",
                        transform=ax.transAxes, fontsize=6.5, color="#7a1f1f", va="bottom")
        ax.set_ylabel("Value" if idx in [0, 3] else "")
    # fig.suptitle(
    #         f"Hydrologic signatures (Tier {compare_tier}; "
    #         f"layers: {_layer_text_from_bundles(lstm_bundle, mamba_bundle)})",
    #         y=1.03,
    #     )
    save_fig(fig, fig_dir, "figA1_signatures")
    save_data(df, fig_dir, "figA1_signatures")


def figure5_tier_scaling(ladder: pd.DataFrame, fig_dir: Path,
                         tiers: Optional[list[int]] = None):
    """Fig 5: Dual-axis NSE/KGE vs tier with bootstrap CI shading.

    Args:
        ladder: Full per-(model,tier) metrics frame.
        fig_dir: Output directory.
        tiers: Optional allow-list of tier numbers to plot (e.g. [4, 6, 7]).
    """
    if ladder.empty:
        print("[Fig5] skip: no completed tiers found")
        return
    plot_ladder = ladder.copy()
    if tiers is not None:
        plot_ladder = plot_ladder[plot_ladder["tier"].isin(tiers)].copy()
        if plot_ladder.empty:
            print(f"[Fig5] skip: no tiers match filter {tiers}")
            return

    fig, ax_nse = plt.subplots(figsize=FIG5_FIGSIZE, constrained_layout=True)
    ax_kge = ax_nse.twinx()
    ax_kge.spines["right"].set_visible(True)

    scaling_rows: list[dict] = []
    for model, g in plot_ladder.groupby("model"):
        sty = _model_style(model)
        g = g.sort_values("tier")
        tiers = g["tier"].to_numpy()
        nse = g["median_nse"].to_numpy()
        kge = g["median_kge"].to_numpy()

        # Bootstrap CI from basin metrics when available
        nse_lo, nse_hi, kge_lo, kge_hi = [], [], [], []
        for _, row in g.iterrows():
            bundle = load_basin_bundle(Path(row["run_dir"]))
            if bundle is None:
                nse_lo.append(np.nan); nse_hi.append(np.nan)
                kge_lo.append(np.nan); kge_hi.append(np.nan)
                continue
            _, lo, hi = bootstrap_median_ci(bundle["metrics"]["NSE"].to_numpy())
            nse_lo.append(lo); nse_hi.append(hi)
            _, lo, hi = bootstrap_median_ci(bundle["metrics"]["KGE"].to_numpy())
            kge_lo.append(lo); kge_hi.append(hi)
        for row, lo, hi, klo, khi in zip(
            g.itertuples(), nse_lo, nse_hi, kge_lo, kge_hi
        ):
            scaling_rows.append({
                "model": model, "tier": int(row.tier),
                "tier_name": row.tier_name,
                "params": row.params,
                "median_nse": float(row.median_nse),
                "nse_ci_lo": float(lo), "nse_ci_hi": float(hi),
                "median_kge": float(row.median_kge),
                "kge_ci_lo": float(klo), "kge_ci_hi": float(khi),
            })

        ax_nse.plot(tiers, nse, color=sty["color"], ls=sty["ls"], marker=sty["marker"],
                    lw=FIG5_NSE_LINE_LW, ms=FIG5_NSE_MS, label=f"{sty['label']} NSE")
        ax_nse.fill_between(tiers, nse_lo, nse_hi, color=sty["color"], alpha=FIG5_CI_ALPHA_NSE)
        ax_kge.plot(tiers, kge, color=sty["color"], ls=":", marker=sty["marker"],
                    lw=FIG5_KGE_LINE_LW, ms=FIG5_KGE_MS, label=f"{sty['label']} KGE")
        ax_kge.fill_between(tiers, kge_lo, kge_hi, color=sty["color"], alpha=FIG5_CI_ALPHA_KGE)

    # Secondary x tick labels with param counts (prefer Mamba params, else LSTM)
    tick_labels = []
    for t in sorted(plot_ladder["tier"].unique()):
        name = TIER_NAMES.get(int(t), str(t))
        sub = plot_ladder[plot_ladder["tier"] == t]
        bits = []
        for model in ["Mamba", "LSTM"]:
            hit = sub.loc[sub["model"] == model, "params"]
            if len(hit) and pd.notna(hit.iloc[0]):
                p = int(hit.iloc[0])
                bits.append(f"{model[0]}:{_fmt_params(p)}")
        tick_labels.append(f"{t}\n{name}\n" + " / ".join(bits) if bits else f"{t}\n{name}")

    ax_nse.set_xticks(sorted(plot_ladder["tier"].unique()))
    ax_nse.set_xticklabels(tick_labels)
    ax_nse.set_xlabel("Parameter tier")
    ax_nse.set_ylabel("Median test NSE", color=COLOR_LSTM)
    ax_kge.set_ylabel("Median test KGE", color=COLOR_MAMBA)
    title_parts = []
    if tiers is not None:
        title_parts.append(f"tiers {sorted(tiers)}")
    layer_text = _layer_text_from_experiments(plot_ladder)
    if layer_text:
        title_parts.append(f"layers: {layer_text}")
    title_extra = f" ({'; '.join(title_parts)})" if title_parts else ""
    ax_nse.set_title(f"Scaling across tiers (median ± 95% bootstrap CI){title_extra}")
    # Combined legend
    lines1, labels1 = ax_nse.get_legend_handles_labels()
    lines2, labels2 = ax_kge.get_legend_handles_labels()
    ax_nse.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower right", ncol=2)
    save_fig(fig, fig_dir, "fig05_tier_scaling")
    if scaling_rows:
        save_data(pd.DataFrame(scaling_rows).sort_values(["model", "tier"]),
                  fig_dir, "fig05_tier_scaling")


def _fmt_params(p: int) -> str:
    if p >= 1_000_000:
        return f"{p / 1_000_000:.2f}M"
    if p >= 1_000:
        return f"{p / 1_000:.0f}k"
    return str(p)


def figure6_spatial(
    lstm_bundle: dict,
    mamba_bundle: dict,
    fig_dir: Path,
    compare_tier: int,
):
    """Fig 6: 1:1 scatter + CONUS ΔNSE / ΔKGE maps."""
    gauges = load_gauge_meta()
    a = lstm_bundle["metrics"][["gauge_id", "NSE", "KGE"]].rename(
        columns={"NSE": "NSE_LSTM", "KGE": "KGE_LSTM"}
    )
    b = mamba_bundle["metrics"][["gauge_id", "NSE", "KGE"]].rename(
        columns={"NSE": "NSE_Mamba", "KGE": "KGE_Mamba"}
    )
    merged = a.merge(b, on="gauge_id").merge(
        gauges[["gauge_id", "gauge_lat", "gauge_lon"]], on="gauge_id", how="left"
    )
    merged["dNSE"] = merged["NSE_Mamba"] - merged["NSE_LSTM"]
    merged["dKGE"] = merged["KGE_Mamba"] - merged["KGE_LSTM"]

    fig = plt.figure(figsize=FIG6_FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # Scatter panels
    for i, (metric, dcol) in enumerate([("NSE", "dNSE"), ("KGE", "dKGE")]):
        ax = fig.add_subplot(gs[0, i])
        x = merged[f"{metric}_LSTM"]
        y = merged[f"{metric}_Mamba"]
        # D4 art review (2026-11-10): a handful of Mamba blow-ups (NSE -12)
        # crushed 95 % of the points into the corner.  Clip the 1:1 axes and
        # state how many basins fall outside, so the blow-ups stay visible as
        # a count rather than as an axis stretch.
        lo, hi = -2.0, 1.0
        inside = x.between(lo, hi) & y.between(lo, hi)
        n_out = int((~inside).sum())
        ax.scatter(x[inside], y[inside], c=merged.loc[inside, dcol], cmap=FIG6_CMAP,
                   vmin=FIG6_VMIN, vmax=FIG6_VMAX,
                   s=FIG6_SCATTER_S, alpha=FIG6_SCATTER_ALPHA, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"LSTM {metric}")
        ax.set_ylabel(f"Mamba {metric}")
        ax.set_title(f"{'(a)' if i == 0 else '(b)'}  1:1 basin {metric}")
        ax.set_aspect("equal", adjustable="box")
        if n_out:
            ax.text(0.03, 0.93, f"{n_out} basins outside axis\n(Mamba-3 blow-ups)",
                    transform=ax.transAxes, fontsize=7, va="top", color="#7a1f1f")

    # CONUS maps
    for i, (dcol, title) in enumerate([
        ("dNSE", "(c) ΔNSE (Mamba − LSTM)"),
        ("dKGE", "(d) ΔKGE (Mamba − LSTM)"),
    ]):
        ax = fig.add_subplot(gs[1, i])
        sc = ax.scatter(
            merged["gauge_lon"], merged["gauge_lat"],
            c=merged[dcol], cmap=FIG6_CMAP, vmin=FIG6_VMIN, vmax=FIG6_VMAX,
            s=FIG6_MAP_S, alpha=FIG6_MAP_ALPHA, edgecolors="k", linewidths=0.15,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        ax.set_xlim(*FIG6_LON_LIM)
        ax.set_ylim(*FIG6_LAT_LIM)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(dcol.replace("d", "Δ"))

    # fig.suptitle(
    #         f"Spatial comparison (Tier {compare_tier}; "
    #         f"layers: {_layer_text_from_bundles(lstm_bundle, mamba_bundle)})",
    #         y=1.01,
    #     )
    save_fig(fig, fig_dir, "fig07_spatial")
    save_data(merged, fig_dir, "fig07_spatial")


def _pick_case_basins(merged_attrs: pd.DataFrame) -> dict[str, str]:
    """Pick snow / flashy / arid representative basins."""
    df = merged_attrs.dropna(subset=["frac_snow", "aridity", "NSE_Mamba", "NSE_LSTM"]).copy()
    picks = {}
    # Snow-dominated: high frac_snow, decent skill
    snow = df[df["frac_snow"] >= BASIN_SNOW_FRAC_MIN].sort_values("frac_snow", ascending=False)
    if len(snow):
        picks["Snow-dominated"] = snow.iloc[0]["gauge_id"]
    # Flashy / rain-fed: low snow, low aridity, high precip seasonality if present
    rain = df[(df["frac_snow"] < BASIN_RAIN_SNOW_MAX) & (df["aridity"] < BASIN_RAIN_ARIDITY_MAX)]
    if "p_seasonality" in rain.columns:
        rain = rain.sort_values("p_seasonality", ascending=False)
    if len(rain):
        gid = rain.iloc[0]["gauge_id"]
        if gid not in picks.values():
            picks["Flashy / rain-fed"] = gid
    # Arid / baseflow: high aridity
    arid = df[df["aridity"] >= BASIN_ARID_ARIDITY_MIN].sort_values("aridity", ascending=False)
    for _, row in arid.iterrows():
        if row["gauge_id"] not in picks.values():
            picks["Arid / baseflow"] = row["gauge_id"]
            break
    # Fallbacks if filters empty
    if len(picks) < 3:
        for label, row in zip(
            ["Snow-dominated", "Flashy / rain-fed", "Arid / baseflow"],
            df.sort_values("NSE_Mamba", ascending=False).itertuples(),
        ):
            if label not in picks and row.gauge_id not in picks.values():
                picks[label] = row.gauge_id
            if len(picks) == 3:
                break
    return picks


def figure7_hydrographs(
    lstm_bundle: dict,
    mamba_bundle: dict,
    fig_dir: Path,
    compare_tier: int,
    basin_ids: Optional[list[str]] = None,
    max_days: int = FIG7_MAX_DAYS,
):
    """Fig 7: Observed vs LSTM vs Mamba hydrographs for 3 basins."""
    if not lstm_bundle["pred"].is_file() or not mamba_bundle["pred"].is_file():
        print("[Fig7] skip: pred/obs .npy missing")
        return
    if not lstm_bundle["obs"].is_file():
        print("[Fig7] skip: obs_daily_m3s.npy missing")
        return

    gauges = load_gauge_meta()
    lstm_m = lstm_bundle["metrics"].set_index("gauge_id")
    mamba_m = mamba_bundle["metrics"].set_index("gauge_id")
    # Align basin order with gauge_ids / metrics
    gauge_order = lstm_bundle["metrics"]["gauge_id"].astype(str).str.zfill(8).tolist()

    obs = np.load(lstm_bundle["obs"])
    pred_lstm = np.load(lstm_bundle["pred"])
    pred_mamba = np.load(mamba_bundle["pred"])
    if obs.shape[0] != len(gauge_order):
        print(f"[Fig7] skip: obs rows ({obs.shape[0]}) != gauges ({len(gauge_order)})")
        return

    # Optional climate attributes for automatic basin selection
    labels_and_ids: list[tuple[str, str]] = []
    if basin_ids:
        for i, gid in enumerate(basin_ids[:3]):
            labels_and_ids.append((f"Basin {gid}", str(gid).zfill(8)))
    else:
        try:
            clim = load_camels_clim()
            merged = (
                lstm_bundle["metrics"][["gauge_id", "NSE", "KGE"]]
                .rename(columns={"NSE": "NSE_LSTM", "KGE": "KGE_LSTM"})
                .merge(
                    mamba_bundle["metrics"][["gauge_id", "NSE", "KGE"]].rename(
                        columns={"NSE": "NSE_Mamba", "KGE": "KGE_Mamba"}
                    ),
                    on="gauge_id",
                )
                .merge(clim, on="gauge_id", how="left")
            )
            picks = _pick_case_basins(merged)
            labels_and_ids = list(picks.items())
        except Exception as exc:
            warnings.warn(f"[Fig7] clim attributes unavailable ({exc}); using top-NSE basins")
            top = (
                mamba_bundle["metrics"]
                .sort_values("NSE", ascending=False)["gauge_id"]
                .head(3)
                .tolist()
            )
            labels_and_ids = [(f"Basin {g}", g) for g in top]

    if not labels_and_ids:
        print("[Fig7] skip: no basins selected")
        return

    # KGE components for annotation
    def _kge_ann(bundle: dict, gid: str) -> str:
        parts = []
        kge = bundle.get("kge")
        if kge is not None and "gauge_id" in kge.columns:
            row = kge.loc[kge["gauge_id"] == gid]
            if len(row):
                r, b, g = row.iloc[0].get("r"), row.iloc[0].get("beta"), row.iloc[0].get("gamma")
                kge_parts = []
                if r is not None:
                    kge_parts.append(f"r={r:.2f}")
                if b is not None:
                    kge_parts.append(f"β={b:.2f}")
                if g is not None:
                    kge_parts.append(f"γ={g:.2f}")
                if kge_parts:
                    parts.append(", ".join(kge_parts))
        return " (" + parts[0] + ")" if parts else ""

    n = len(labels_and_ids)
    fig, axes = plt.subplots(n, 1, figsize=(FIG7_FIGW, FIG7_ROW_H * n), sharex=False, constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Validation window starts per config
    t0 = np.datetime64(FIG7_T0)
    hydro_rows: list[dict] = []
    for i, (ax, (label, gid)) in enumerate(zip(axes, labels_and_ids)):
        if gid not in gauge_order:
            ax.set_visible(False)
            continue
        idx = gauge_order.index(gid)
        n_t = min(max_days, obs.shape[1], pred_lstm.shape[1], pred_mamba.shape[1])
        t = t0 + np.arange(n_t)
        ax.plot(t, obs[idx, :n_t], color=COLOR_OBS, lw=FIG7_OBS_LW, label="Observed", alpha=FIG7_OBS_ALPHA)
        ax.plot(t, pred_lstm[idx, :n_t], color=COLOR_LSTM, ls="--", lw=FIG7_LSTM_LW, label="LSTM")
        ax.plot(t, pred_mamba[idx, :n_t], color=COLOR_MAMBA, ls="-", lw=FIG7_MAMBA_LW, label="Mamba")

        meta = gauges.loc[gauges["gauge_id"] == gid]
        name = meta.iloc[0]["gauge_name"] if len(meta) else ""
        nse_l = float(lstm_m.loc[gid, "NSE"]) if gid in lstm_m.index else np.nan
        kge_l = float(lstm_m.loc[gid, "KGE"]) if gid in lstm_m.index else np.nan
        nse_m = float(mamba_m.loc[gid, "NSE"]) if gid in mamba_m.index else np.nan
        kge_m = float(mamba_m.loc[gid, "KGE"]) if gid in mamba_m.index else np.nan
        ann_l = _kge_ann(lstm_bundle, gid)
        ann_m = _kge_ann(mamba_bundle, gid)
        subfig_label = chr(ord('a') + i)
        ax.set_title(
            f"({subfig_label}) {label}: {gid}  {name}\n"
            f"Mamba: NSE={nse_m:.2f} | KGE={kge_m:.2f}{ann_m}   ·   "
            f"LSTM: NSE={nse_l:.2f} | KGE={kge_l:.2f}{ann_l}",
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("Q (m³/s)")
        ax.legend(frameon=False, ncol=3, loc="upper right")
        # Persist the exact plotted window for this basin
        for d, o, pl, pm in zip(t.astype(str),
                                np.asarray(obs[idx, :n_t], dtype=float),
                                np.asarray(pred_lstm[idx, :n_t], dtype=float),
                                np.asarray(pred_mamba[idx, :n_t], dtype=float)):
            hydro_rows.append({
                "date": str(d), "gauge_id": gid, "label": label,
                "gauge_name": name, "observed_m3s": float(o),
                "lstm_m3s": float(pl), "mamba_m3s": float(pm),
                "nse_lstm": nse_l, "kge_lstm": kge_l,
                "nse_mamba": nse_m, "kge_mamba": kge_m,
            })

    axes[-1].set_xlabel("Date")
    # fig.suptitle(
    #         f"Hydrograph case studies (Tier {compare_tier}; "
    #         f"layers: {_layer_text_from_bundles(lstm_bundle, mamba_bundle)})",
    #         y=1.01,
    #     )
    save_fig(fig, fig_dir, "fig09_hydrographs")
    if hydro_rows:
        save_data(pd.DataFrame(hydro_rows), fig_dir, "fig09_hydrographs")


def figure8_fdc(
    lstm_bundle: dict,
    mamba_bundle: dict,
    fig_dir: Path,
    compare_tier: int,
):
    """Fig 8: basin-aggregated flow-duration curves and relative FDC error.

    Each basin contributes one FDC computed from its observed and predicted
    daily discharge. Curves show the median across basins with 10--90% bands.
    Exceedance probability is plotted from high flows (small p) to low flows
    (large p); zero/negative values are excluded because both axes use a
    logarithmic discharge scale.
    """
    bundles = [("Observed", lstm_bundle, "obs", COLOR_OBS),
               ("LSTM", lstm_bundle, "pred_lstm", COLOR_LSTM),
               ("Mamba", mamba_bundle, "pred_mamba", COLOR_MAMBA)]
    if not lstm_bundle["obs"].is_file():
        print("[Fig8] skip: obs_daily_m3s.npy missing")
        return
    if not lstm_bundle["pred"].is_file() or not mamba_bundle["pred"].is_file():
        print("[Fig8] skip: pred_daily_m3s.npy missing")
        return

    obs = np.asarray(np.load(lstm_bundle["obs"]), dtype=float)
    pred_lstm = np.asarray(np.load(lstm_bundle["pred"]), dtype=float)
    pred_mamba = np.asarray(np.load(mamba_bundle["pred"]), dtype=float)
    n_basins = min(obs.shape[0], pred_lstm.shape[0], pred_mamba.shape[0],
                   len(lstm_bundle["metrics"]), len(mamba_bundle["metrics"]))
    if n_basins == 0 or obs.ndim != 2 or pred_lstm.ndim != 2 or pred_mamba.ndim != 2:
        print("[Fig8] skip: expected 2-D observed and predicted arrays")
        return

    # Avoid p=0 because it cannot be displayed on a logarithmic x-axis.
    exceedance = np.geomspace(1.0e-3, 1.0, FIG8_N_POINTS)
    fdc_values: dict[str, np.ndarray] = {}
    for label, _, key, _ in bundles:
        if key == "obs":
            arr = obs[:n_basins]
        elif key == "pred_lstm":
            arr = pred_lstm[:n_basins]
        else:
            arr = pred_mamba[:n_basins]
        curves = np.full((n_basins, len(exceedance)), np.nan)
        for i in range(n_basins):
            flow = arr[i]
            flow = flow[np.isfinite(flow) & (flow > 0)]
            if len(flow):
                # Q(p) is the (1-p) quantile: p=0.001 is a high flow.
                curves[i] = np.quantile(flow, 1.0 - exceedance)
        fdc_values[label] = curves

    obs_fdc = fdc_values["Observed"]
    rows: list[dict] = []
    fig, axes = plt.subplots(1, 2, figsize=FIG8_FIGSIZE, constrained_layout=True)

    # Panel A: median FDC and central basin envelope.
    ax = axes[0]
    for label, _, _, color in bundles:
        curves = fdc_values[label]
        median = np.nanmedian(curves, axis=0)
        lo = np.nanquantile(curves, FIG8_BAND_LO, axis=0)
        hi = np.nanquantile(curves, FIG8_BAND_HI, axis=0)
        ax.plot(exceedance, np.maximum(median, FIG8_EPSILON), color=color,
                lw=FIG8_LINE_LW, label=label)
        if label != "Observed":
            ax.fill_between(exceedance, np.maximum(lo, FIG8_EPSILON),
                            np.maximum(hi, FIG8_EPSILON), color=color,
                            alpha=FIG8_BAND_ALPHA, linewidth=0)
        for p, med, band_lo, band_hi in zip(exceedance, median, lo, hi):
            rows.append({"panel": "fdc", "model": label,
                         "exceedance_probability": float(p),
                         "median_discharge_m3s": float(med),
                         "q10_discharge_m3s": float(band_lo),
                         "q90_discharge_m3s": float(band_hi)})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*FIG8_YLIM)
    ax.set_xlabel("Exceedance probability")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title("(a) Flow-duration curves")
    ax.legend(frameon=False)

    # Panel B: calculate relative error basin by basin before aggregating.
    ax = axes[1]
    for label, color in [("LSTM", COLOR_LSTM), ("Mamba", COLOR_MAMBA)]:
        model_fdc = fdc_values[label]
        valid = np.isfinite(obs_fdc) & np.isfinite(model_fdc) & (obs_fdc > 0)
        errors = np.where(valid, (model_fdc - obs_fdc) / obs_fdc, np.nan)
        median = np.nanmedian(errors, axis=0)
        lo = np.nanquantile(errors, FIG8_BAND_LO, axis=0)
        hi = np.nanquantile(errors, FIG8_BAND_HI, axis=0)
        ax.plot(exceedance, median, color=color, lw=FIG8_LINE_LW, label=label)
        ax.fill_between(exceedance, lo, hi, color=color,
                        alpha=FIG8_BAND_ALPHA, linewidth=0)
        for p, med, band_lo, band_hi in zip(exceedance, median, lo, hi):
            rows.append({"panel": "relative_error", "model": label,
                         "exceedance_probability": float(p),
                         "median_relative_error": float(med),
                         "q10_relative_error": float(band_lo),
                         "q90_relative_error": float(band_hi)})
    ax.axhline(0.0, color="black", lw=0.9, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Exceedance probability")
    ax.set_ylabel("Relative FDC error")
    ax.set_title("(b) Relative error: (model − observed) / observed")
    ax.legend(frameon=False)

    # Keep the hydrologically meaningful regions visible without obscuring data.
    for ax in axes:
        ax.axvspan(1e-3, 0.30, color="#999999", alpha=0.06, zorder=0)
        ax.axvspan(0.70, 1.0, color="#999999", alpha=0.06, zorder=0)
        ax.text(0.012, 0.03, "high flow", transform=ax.transAxes, fontsize=8)
        ax.text(0.79, 0.03, "low flow", transform=ax.transAxes, fontsize=8)

    # fig.suptitle(
    #     f"Flow-duration curves and relative error (Tier {compare_tier}; "
    #             f"layers: {_layer_text_from_bundles(lstm_bundle, mamba_bundle)})",
    #     y=1.03,
    # )
    save_fig(fig, fig_dir, "figA2_flow_duration")
    save_data(pd.DataFrame(rows), fig_dir, "figA2_flow_duration")


def seasonal_basin_scores(
    o: np.ndarray,
    p: np.ndarray,
    season_mask: np.ndarray,
    high_threshold: float,
    low_threshold: float,
    metric_names: list[str] | tuple[str, ...] = ("NSE", "KGE", "High-flow bias", "Low-flow bias"),
) -> dict:
    """Per-basin seasonal scores for Fig 9.

    ``o``/``p`` are one basin's full daily series; ``season_mask`` selects the
    days of the current season. KGE is the Gupta (2009) form -- the same
    formula as the headline metric in ``hydroDL.post.stat.statError`` and the
    one the paper declares in §3.4. The 2012 form (a ratio of coefficients of
    variation) returns different values and must not be used here; audit_kge.py
    checks this equivalence stays true.
    """
    valid = season_mask & np.isfinite(o) & np.isfinite(p)
    if valid.sum() < 2:
        return {metric: np.nan for metric in metric_names}
    ov, pv = o[valid], p[valid]
    denominator = np.sum((ov - np.mean(ov)) ** 2)
    nse = np.nan if denominator <= 0 else 1.0 - np.sum((ov - pv) ** 2) / denominator

    mean_o, mean_p = np.mean(ov), np.mean(pv)
    std_o, std_p = np.std(ov), np.std(pv)
    if mean_o <= 0 or mean_p <= 0 or std_o <= 0 or std_p <= 0:
        kge = np.nan
    else:
        correlation = float(np.corrcoef(ov, pv)[0, 1])
        beta = mean_p / mean_o
        # Variability term must use the 2009 definition, alpha = std_p /
        # std_o, so that the seasonal scores are the same metric as the
        # headline KGE in evaluation.py and the one the paper declares.
        # The 2012 form, a ratio of coefficients of variation
        # ((std_p/mean_p)/(std_o/mean_o)), returns different values and
        # must not be used here.
        alpha = std_p / std_o
        kge = 1.0 - np.sqrt(
            (correlation - 1.0) ** 2
            + (beta - 1.0) ** 2
            + (alpha - 1.0) ** 2
        )

    high = valid & (o >= high_threshold)
    low = valid & (o <= low_threshold)
    high_bias = np.nan
    low_bias = np.nan
    if high.sum() > 0 and np.mean(o[high]) > 0:
        high_bias = np.mean(p[high] - o[high]) / np.mean(o[high])
    if low.sum() > 0 and np.mean(o[low]) > 0:
        low_bias = np.mean(p[low] - o[low]) / np.mean(o[low])
    return {"NSE": nse, "KGE": kge,
            "High-flow bias": high_bias, "Low-flow bias": low_bias}


def figure9_seasonal_heatmaps(
    lstm_bundle: dict,
    mamba_bundle: dict,
    fig_dir: Path,
    compare_tier: int,
):
    """Fig 9: seasonal NSE, KGE, high-flow bias, and low-flow bias.

    Scores are calculated separately for each basin and season, then the
    basin-level scores are summarized by their median. This prevents large
    basins with more discharge from dominating the continental result.
    """
    if not lstm_bundle["obs"].is_file():
        print("[Fig9] skip: obs_daily_m3s.npy missing")
        return
    if not lstm_bundle["pred"].is_file() or not mamba_bundle["pred"].is_file():
        print("[Fig9] skip: pred_daily_m3s.npy missing")
        return

    obs = np.asarray(np.load(lstm_bundle["obs"]), dtype=float)
    pred_lstm = np.asarray(np.load(lstm_bundle["pred"]), dtype=float)
    pred_mamba = np.asarray(np.load(mamba_bundle["pred"]), dtype=float)
    if obs.ndim != 2 or pred_lstm.ndim != 2 or pred_mamba.ndim != 2:
        print("[Fig9] skip: expected 2-D observed and predicted arrays")
        return
    n_basins = min(obs.shape[0], pred_lstm.shape[0], pred_mamba.shape[0],
                   len(lstm_bundle["metrics"]), len(mamba_bundle["metrics"]))
    n_days = min(obs.shape[1], pred_lstm.shape[1], pred_mamba.shape[1])
    if n_basins == 0 or n_days == 0:
        print("[Fig9] skip: empty observed or predicted arrays")
        return

    dates = pd.date_range(FIG7_T0, periods=n_days, freq="D")
    month = dates.month.to_numpy()
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    season_months = {
        "Winter": {12, 1, 2},
        "Spring": {3, 4, 5},
        "Summer": {6, 7, 8},
        "Autumn": {9, 10, 11},
    }
    model_arrays = {"LSTM": pred_lstm[:n_basins, :n_days],
                    "Mamba": pred_mamba[:n_basins, :n_days]}
    observed = obs[:n_basins, :n_days]
    metric_names = ["NSE", "KGE", "High-flow bias", "Low-flow bias"]
    score_rows: list[dict] = []

    for season in seasons:
        season_mask = np.isin(month, list(season_months[season]))
        for model, prediction in model_arrays.items():
            basin_scores = {metric: [] for metric in metric_names}
            for i in range(n_basins):
                all_valid_obs = observed[i][np.isfinite(observed[i]) & (observed[i] > 0)]
                if len(all_valid_obs) == 0:
                    thresholds = (np.nan, np.nan)
                else:
                    thresholds = (float(np.quantile(all_valid_obs, 0.95)),
                                  float(np.quantile(all_valid_obs, 0.30)))
                scores = seasonal_basin_scores(
                    observed[i], prediction[i], season_mask,
                    thresholds[0], thresholds[1], metric_names,
                )
                for metric in metric_names:
                    basin_scores[metric].append(scores[metric])
            for metric in metric_names:
                value = float(np.nanmedian(basin_scores[metric]))
                score_rows.append({"season": season, "model": model,
                                   "metric": metric, "value": value,
                                   "n_basins": int(np.isfinite(basin_scores[metric]).sum())})

    scores = pd.DataFrame(score_rows)
    if scores.empty:
        print("[Fig9] skip: no seasonal scores available")
        return

    # Add paired model differences after aggregation, not before aggregation.
    difference_rows = []
    for season in seasons:
        for metric in metric_names:
            pivot = scores[(scores["season"] == season) &
                           (scores["metric"] == metric)].set_index("model")["value"]
            mamba_value = float(pivot["Mamba"]) if "Mamba" in pivot.index else np.nan
            lstm_value = float(pivot["LSTM"]) if "LSTM" in pivot.index else np.nan
            difference_rows.append({
                "season": season, "model": "Mamba − LSTM",
                "metric": metric,
                "value": mamba_value - lstm_value,
                "n_basins": np.nan,
            })
    scores = pd.concat([scores, pd.DataFrame(difference_rows)], ignore_index=True)

    columns = ["LSTM", "Mamba", "Mamba − LSTM"]
    fig, axes = plt.subplots(2, 2, figsize=FIG9_FIGSIZE, constrained_layout=True)
    for i, (ax, metric) in enumerate(zip(axes.flat, metric_names)):
        subfig_label = chr(ord('a') + i)
        matrix = scores[scores["metric"] == metric].pivot(
            index="season", columns="model", values="value"
        ).reindex(index=seasons, columns=columns).to_numpy(dtype=float)
        finite = matrix[np.isfinite(matrix)]
        if len(finite) == 0:
            continue
        
        if "bias" in metric.lower():
            max_abs = max(abs(float(np.min(finite))), abs(float(np.max(finite))), 1e-6)
            vmin, vmax, cmap = -max_abs, max_abs, FIG9_CMAP_BIAS
            image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            vmin, vmax, cmap = float(np.min(finite)), float(np.max(finite)), FIG9_CMAP_SCORE
            if vmin == vmax:
                vmin, vmax = vmin - 1e-6, vmax + 1e-6
            # D4 art review (2026-11-10): the delta column is ~0.05 on a 0.6-0.7
            # scale, so a shared sequential colourmap made real gaps look like
            # zero.  Draw columns 1-2 on the score scale and the delta column on
            # its own diverging scale, centred at zero.
            model_only = matrix.copy()
            model_only[:, 2] = np.nan
            image = ax.imshow(model_only, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            delta_col = np.full_like(matrix, np.nan)
            delta_col[:, 2] = matrix[:, 2]
            dmax = max(abs(float(np.nanmin(delta_col))), abs(float(np.nanmax(delta_col))), 1e-6)
            ax.imshow(delta_col, aspect="auto", cmap=FIG9_CMAP_BIAS, vmin=-dmax, vmax=dmax)
            ax.text(2, -0.62, f"$\\Delta$ scale $\\pm${dmax:.3f}", ha="center",
                    va="bottom", fontsize=7, color="#444444")
        ax.set_xticks(np.arange(len(columns)), labels=columns)
        ax.set_yticks(np.arange(len(seasons)), labels=seasons)
        ax.set_title(f"({subfig_label}) {metric}")
        ax.set_ylabel("Season")
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                value = matrix[r, c]
                if np.isfinite(value):
                    ax.text(c, r, format(value, FIG9_ANNOTATION_FORMAT),
                            ha="center", va="center", fontsize=9,
                            color="white" if abs(value - (vmin + vmax) / 2) > (vmax - vmin) * 0.25 else "black")
        cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Median score" if metric in {"NSE", "KGE"} else "Relative bias")

    # fig.suptitle(
    #     f"Seasonal performance (Tier {compare_tier}; "
    #             f"layers: {_layer_text_from_bundles(lstm_bundle, mamba_bundle)})",
    #     y=1.02,
    # )
    save_fig(fig, fig_dir, "fig08_seasonal_heatmaps")
    save_data(scores, fig_dir, "fig08_seasonal_heatmaps")


def figure10_residual_diagnostics(
    lstm_bundle: dict,
    mamba_bundle: dict,
    fig_dir: Path,
    compare_tier: int,
):
    """Fig 10: residual-versus-flow and flow-percentile diagnostics.

    Residuals are defined as simulated minus observed discharge. The scatter
    panel is deterministically subsampled for readable files, while all valid
    paired values contribute to the percentile summaries.
    """
    if not lstm_bundle["obs"].is_file():
        print("[Fig10] skip: obs_daily_m3s.npy missing")
        return
    if not lstm_bundle["pred"].is_file() or not mamba_bundle["pred"].is_file():
        print("[Fig10] skip: pred_daily_m3s.npy missing")
        return

    obs = np.asarray(np.load(lstm_bundle["obs"]), dtype=float)
    pred_lstm = np.asarray(np.load(lstm_bundle["pred"]), dtype=float)
    pred_mamba = np.asarray(np.load(mamba_bundle["pred"]), dtype=float)
    if obs.ndim != 2 or pred_lstm.ndim != 2 or pred_mamba.ndim != 2:
        print("[Fig10] skip: expected 2-D observed and predicted arrays")
        return
    n_basins = min(obs.shape[0], pred_lstm.shape[0], pred_mamba.shape[0])
    n_days = min(obs.shape[1], pred_lstm.shape[1], pred_mamba.shape[1])
    if n_basins == 0 or n_days == 0:
        print("[Fig10] skip: empty observed or predicted arrays")
        return

    observed = obs[:n_basins, :n_days].ravel()
    predictions = {
        "LSTM": pred_lstm[:n_basins, :n_days].ravel(),
        "Mamba": pred_mamba[:n_basins, :n_days].ravel(),
    }
    flattened: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_flow_parts = []
    for model, prediction in predictions.items():
        valid = np.isfinite(observed) & np.isfinite(prediction) & (observed > 0)
        flow = observed[valid]
        residual = prediction[valid] - flow
        flattened[model] = (flow, residual)
        all_flow_parts.append(flow)
    if not all_flow_parts:
        print("[Fig10] skip: no valid positive observed flows")
        return

    pooled_flow = np.concatenate(all_flow_parts)
    diagnostic_rows: list[dict] = []

    fig, ax = plt.subplots(1, 1, figsize=(FIG10_FIGSIZE[0] / 2, FIG10_FIGSIZE[1]), constrained_layout=True)

    # Panel A: residual versus observed flow, with a median trend in log-flow bins.
    # Use observed-flow quantiles so the annotations describe the x-axis itself.
    low_flow_limit, high_flow_limit = np.nanquantile(pooled_flow, [0.30, 0.70])
    flow_min = float(np.nanmin(pooled_flow))
    flow_max = float(np.nanmax(pooled_flow))
    ax.axvspan(flow_min, low_flow_limit, color="#999999", alpha=0.06,
               zorder=0)
    ax.axvspan(high_flow_limit, flow_max, color="#999999", alpha=0.06,
               zorder=0)
    ax.text(0.02, 0.96, "low observed flow", transform=ax.transAxes,
            fontsize=8, va="top")
    ax.text(0.98, 0.96, "high observed flow", transform=ax.transAxes,
            fontsize=8, va="top", ha="right")
    all_residuals = []
    for model, (flow, residual) in flattened.items():
        all_residuals.append(residual)
        if len(flow) > FIG10_MAX_SCATTER:
            sample_idx = np.linspace(0, len(flow) - 1, FIG10_MAX_SCATTER, dtype=int)
        else:
            sample_idx = np.arange(len(flow))
        color = _model_style(model)["color"]
        ax.scatter(flow[sample_idx], residual[sample_idx], s=3,
                   color=color, alpha=FIG10_SCATTER_ALPHA,
                   edgecolors="none", rasterized=True)

        xmin = max(float(np.nanmin(flow)), FIG8_EPSILON)
        xmax = float(np.nanmax(flow))
        if xmax > xmin:
            edges = np.geomspace(xmin, xmax, FIG10_N_FLOW_BINS + 1)
            centers, medians = [], []
            for left, right in zip(edges[:-1], edges[1:]):
                in_bin = (flow >= left) & (flow < right)
                if in_bin.any():
                    centers.append(np.sqrt(left * right))
                    medians.append(float(np.nanmedian(residual[in_bin])))
                    diagnostic_rows.append({
                        "panel": "residual_trend", "model": model,
                        "flow_range": "logarithmic bin",
                        "median_signed_residual_m3s": float(np.nanmedian(residual[in_bin])),
                        "flow_center_m3s": float(np.sqrt(left * right)),
                    })
            ax.plot(centers, medians, color=color, lw=2.2)

    residual_values = np.concatenate(all_residuals)
    y_limits = np.nanquantile(residual_values, [0.005, 0.995])
    if y_limits[0] == y_limits[1]:
        y_limits = (y_limits[0] - 1.0, y_limits[1] + 1.0)
    ax.axhline(0.0, color="black", lw=0.9, alpha=0.8)
    ax.set_xscale("log")
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Observed discharge (m³/s)")
    ax.set_ylabel("Residual: simulated − observed (m³/s)")
    # ax.set_title("Residual versus observed flow")
    
    # Create custom legend with model colors
    legend_lines = []
    legend_labels = []
    for model in ["LSTM", "Mamba"]:
        sty = _model_style(model)
        line = Line2D([0], [0], marker='o', color=sty["color"], 
                      linestyle='None', markersize=8, label=model)
        legend_lines.append(line)
        legend_labels.append(model)
    ax.legend(legend_lines, legend_labels, frameon=False, fontsize=8)

    # fig.suptitle(
    #     f"Residual diagnostics (Tier {compare_tier}; "
    #             f"layers: {_layer_text_from_bundles(lstm_bundle, mamba_bundle)})",
    #     y=1.02,
    # )
    save_fig(fig, fig_dir, "figA3_residual_diagnostics")
    save_data(pd.DataFrame(diagnostic_rows), fig_dir,
              "figA3_residual_diagnostics")


def figure11_peak_events(
    lstm_bundle: dict,
    mamba_bundle: dict,
    fig_dir: Path,
    compare_tier: int,
):
    """Fig 11: observed flood-event errors and threshold skill.

    The top observed local peaks above the basin 90th percentile are selected
    with a minimum separation. Model peaks are searched in the same observed
    event window, making timing, magnitude, and volume comparisons paired.
    """
    if not lstm_bundle["obs"].is_file():
        print("[Fig11] skip: obs_daily_m3s.npy missing")
        return
    if not lstm_bundle["pred"].is_file() or not mamba_bundle["pred"].is_file():
        print("[Fig11] skip: pred_daily_m3s.npy missing")
        return

    obs = np.asarray(np.load(lstm_bundle["obs"]), dtype=float)
    pred_lstm = np.asarray(np.load(lstm_bundle["pred"]), dtype=float)
    pred_mamba = np.asarray(np.load(mamba_bundle["pred"]), dtype=float)
    if obs.ndim != 2 or pred_lstm.ndim != 2 or pred_mamba.ndim != 2:
        print("[Fig11] skip: expected 2-D observed and predicted arrays")
        return
    n_basins = min(obs.shape[0], pred_lstm.shape[0], pred_mamba.shape[0],
                   len(lstm_bundle["metrics"]), len(mamba_bundle["metrics"]))
    n_days = min(obs.shape[1], pred_lstm.shape[1], pred_mamba.shape[1])
    if n_basins == 0 or n_days == 0:
        print("[Fig11] skip: empty observed or predicted arrays")
        return

    def _event_peaks(series: np.ndarray) -> list[int]:
        valid = np.isfinite(series) & (series > 0)
        if valid.sum() < 3:
            return []
        threshold = float(np.quantile(series[valid], FIG11_EVENT_THRESHOLD))
        candidates = [
            i for i in range(1, len(series) - 1)
            if np.isfinite(series[i - 1]) and np.isfinite(series[i])
            and np.isfinite(series[i + 1]) and series[i] >= threshold
            and series[i] >= series[i - 1] and series[i] >= series[i + 1]
        ]
        candidates.sort(key=lambda i: series[i], reverse=True)
        selected: list[int] = []
        for candidate in candidates:
            if all(abs(candidate - previous) >= FIG11_MIN_PEAK_SEPARATION
                   for previous in selected):
                selected.append(candidate)
                if len(selected) == FIG11_TOP_N_EVENTS:
                    break
        return sorted(selected)

    def _safe_corr(x: list[float], y: list[float]) -> float:
        if len(x) < 2 or len(y) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    model_arrays = {
        "LSTM": pred_lstm[:n_basins, :n_days],
        "Mamba": pred_mamba[:n_basins, :n_days],
    }
    gauge_ids = lstm_bundle["metrics"]["gauge_id"].astype(str).str.zfill(8).tolist()
    event_rows: list[dict] = []
    threshold_rows: list[dict] = []
    basin_peak_pairs: dict[tuple[str, int], tuple[list[float], list[float]]] = {}

    for basin_idx in range(n_basins):
        observed = obs[basin_idx, :n_days]
        peaks = _event_peaks(observed)
        if not peaks:
            continue
        gid = gauge_ids[basin_idx] if basin_idx < len(gauge_ids) else str(basin_idx)
        observed_valid = observed[np.isfinite(observed) & (observed > 0)]
        if len(observed_valid) == 0:
            continue
        event_threshold = float(np.quantile(observed_valid, FIG11_EVENT_THRESHOLD))
        for model, prediction_array in model_arrays.items():
            peak_observed_values: list[float] = []
            peak_predicted_values: list[float] = []
            pair_key = (model, basin_idx)
            basin_peak_pairs[pair_key] = (peak_observed_values, peak_predicted_values)
            for event_number, peak_idx in enumerate(peaks, start=1):
                start = max(0, peak_idx - FIG11_EVENT_HALF_WINDOW)
                end = min(n_days, peak_idx + FIG11_EVENT_HALF_WINDOW + 1)
                observed_window = observed[start:end]
                predicted_window = prediction_array[basin_idx, start:end]
                if not np.isfinite(predicted_window).any():
                    continue
                predicted_peak_idx = int(np.nanargmax(predicted_window))
                predicted_peak = float(predicted_window[predicted_peak_idx])
                observed_peak = float(observed[peak_idx])
                if not np.isfinite(observed_peak) or observed_peak <= 0:
                    continue
                observed_volume = float(np.nansum(np.maximum(observed_window, 0)))
                predicted_volume = float(np.nansum(np.maximum(predicted_window, 0)))
                if observed_volume <= 0:
                    continue
                observed_rise = observed_peak - float(observed_window[0])
                predicted_rise = predicted_peak - float(predicted_window[0])
                observed_fall = observed_peak - float(observed_window[-1])
                predicted_fall = predicted_peak - float(predicted_window[-1])
                rise_scale = max(abs(observed_rise), FIG8_EPSILON)
                fall_scale = max(abs(observed_fall), FIG8_EPSILON)
                peak_observed_values.append(observed_peak)
                peak_predicted_values.append(predicted_peak)
                event_rows.append({
                    "record_type": "event",
                    "model": model,
                    "gauge_id": gid,
                    "basin_index": basin_idx,
                    "event_number": event_number,
                    "observed_peak_m3s": observed_peak,
                    "predicted_peak_m3s": predicted_peak,
                    "peak_timing_error_days": float(predicted_peak_idx + start - peak_idx),
                    "peak_magnitude_error": (predicted_peak - observed_peak) / observed_peak,
                    "event_volume_error": (predicted_volume - observed_volume) / observed_volume,
                    "rising_limb_error": (predicted_rise - observed_rise) / rise_scale,
                    "falling_limb_error": (predicted_fall - observed_fall) / fall_scale,
                    "event_start_index": start,
                    "event_end_index": end - 1,
                    "observed_event_threshold_m3s": event_threshold,
                })

    # Peak-flow correlation is calculated per basin from the selected events.
    for (model, basin_idx), (observed_peaks, predicted_peaks) in basin_peak_pairs.items():
        if not observed_peaks:
            continue
        gid = gauge_ids[basin_idx] if basin_idx < len(gauge_ids) else str(basin_idx)
        event_rows.append({
            "record_type": "basin_summary",
            "model": model,
            "gauge_id": gid,
            "basin_index": basin_idx,
            "peak_flow_correlation": _safe_corr(observed_peaks, predicted_peaks),
            "n_events": len(observed_peaks),
        })

    # Daily threshold classification: a hit is a predicted exceedance on a day
    # when the observed discharge also exceeds the same basin-specific threshold.
    for threshold_probability in FIG11_THRESHOLDS:
        for model, prediction_array in model_arrays.items():
            observed_labels: list[np.ndarray] = []
            predicted_labels: list[np.ndarray] = []
            for basin_idx in range(n_basins):
                observed = obs[basin_idx, :n_days]
                prediction = prediction_array[basin_idx]
                valid = np.isfinite(observed) & np.isfinite(prediction) & (observed > 0)
                if not valid.any():
                    continue
                threshold = float(np.quantile(observed[valid], threshold_probability))
                observed_labels.append(observed[valid] >= threshold)
                predicted_labels.append(prediction[valid] >= threshold)
            if not observed_labels:
                continue
            actual = np.concatenate(observed_labels)
            predicted = np.concatenate(predicted_labels)
            true_positive = int(np.sum(actual & predicted))
            false_positive = int(np.sum(~actual & predicted))
            false_negative = int(np.sum(actual & ~predicted))
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else np.nan
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else np.nan
            f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and precision + recall else np.nan
            threshold_rows.append({
                "record_type": "threshold_summary",
                "model": model,
                "threshold_probability": threshold_probability,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
            })

    events = pd.DataFrame(event_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    if events.empty or not (events["record_type"] == "event").any():
        print("[Fig11] skip: no observed flood events found")
        return

    fig, axes = plt.subplots(2, 2, figsize=FIG11_FIGSIZE, constrained_layout=True)
    metric_specs = [
        ("peak_timing_error_days", "(a) Peak timing error", "Timing error (days)"),
        ("peak_magnitude_error", "(b) Peak magnitude error", "Relative peak error"),
        ("event_volume_error", "(c) Event volume bias", "Relative event-volume error"),
    ]
    event_only = events[events["record_type"] == "event"]
    for ax, (column, title, ylabel) in zip(axes.flat[:3], metric_specs):
        data, labels, colors = [], [], []
        for model in ["LSTM", "Mamba"]:
            values = event_only.loc[event_only["model"] == model, column].dropna().to_numpy()
            if len(values):
                data.append(values)
                labels.append(model)
                colors.append(_model_style(model)["color"])
        if data:
            violin = ax.violinplot(data, positions=np.arange(1, len(data) + 1),
                                   showmeans=False, showmedians=True, showextrema=False)
            for body, color in zip(violin["bodies"], colors):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(FIG11_VIOLIN_ALPHA)
            if "cmedians" in violin:
                violin["cmedians"].set_color("black")
            ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        if column != "peak_timing_error_days":
            ax.axhline(0.0, color="black", lw=0.9, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")

    # Panel D: rows are thresholds; columns are precision, recall, F1 per model.
    ax = axes[1, 1]
    ax.set_title("(d) Threshold exceedance skill")
    if not threshold_df.empty:
        columns = ["LSTM precision", "LSTM recall", "LSTM F1",
                   "Mamba precision", "Mamba recall", "Mamba F1"]
        matrix = np.full((len(FIG11_THRESHOLDS), len(columns)), np.nan)
        for row_idx, probability in enumerate(FIG11_THRESHOLDS):
            for model_idx, model in enumerate(["LSTM", "Mamba"]):
                hit = threshold_df[(threshold_df["threshold_probability"] == probability) &
                                   (threshold_df["model"] == model)]
                if hit.empty:
                    continue
                values = hit.iloc[0]
                matrix[row_idx, model_idx * 3:model_idx * 3 + 3] = [
                    values["precision"], values["recall"], values["f1"]
                ]
        image = ax.imshow(matrix, aspect="auto", cmap=FIG11_HEATMAP_CMAP, vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(columns)), labels=columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(FIG11_THRESHOLDS)),
                      labels=[f"{int(p * 100)}th percentile" for p in FIG11_THRESHOLDS])
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                if np.isfinite(matrix[r, c]):
                    ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", fontsize=8)
        cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Score")
    else:
        ax.text(0.5, 0.5, "No threshold scores", ha="center", va="center")
        ax.set_axis_off()

    # fig.suptitle(
    #         f"Peak-event diagnostics (Tier {compare_tier}; "
    #         f"layers: {_layer_text_from_bundles(lstm_bundle, mamba_bundle)})",
    #         y=1.02,
    #     )
    save_fig(fig, fig_dir, "figA4_peak_events")
    save_data(pd.concat([events, threshold_df], ignore_index=True, sort=False),
              fig_dir, "figA4_peak_events")


def figure12_uncertainty(
    lstm_bundle: dict,
    mamba_bundle: dict,
    fig_dir: Path,
    compare_tier: int,
):
    """Fig 12: prediction intervals, coverage, reliability, and PIT.

    This figure requires ensemble or Monte Carlo prediction arrays. It never
    constructs uncertainty intervals from a single deterministic prediction.
    Supported files are listed in ``FIG12_SAMPLE_FILES`` and should contain a
    three-dimensional array with members, basins, and days in any recognizable
    order.
    """
    if not lstm_bundle["obs"].is_file():
        print("[Fig12] skip: obs_daily_m3s.npy missing")
        return
    obs = np.asarray(np.load(lstm_bundle["obs"]), dtype=float)
    if obs.ndim != 2:
        print("[Fig12] skip: observed array is not 2-D")
        return
    n_metric_basins = min(len(lstm_bundle["metrics"]), len(mamba_bundle["metrics"]))

    def _load_samples(bundle: dict) -> tuple[Optional[np.ndarray], Optional[str]]:
        for filename in FIG12_SAMPLE_FILES:
            path = bundle["run_dir"] / filename
            if not path.is_file():
                continue
            raw = np.asarray(np.load(path), dtype=float)
            if raw.ndim != 3:
                warnings.warn(f"[Fig12] ignoring {path.name}: expected 3-D array")
                continue
            # Normalize to (members, basins, days).
            if raw.shape[1] == n_metric_basins:
                normalized = raw
            elif raw.shape[0] == n_metric_basins:
                normalized = np.transpose(raw, (2, 0, 1))  # basin, day, member
            elif raw.shape[2] == n_metric_basins:
                normalized = np.transpose(raw, (0, 2, 1))  # member, day, basin
            else:
                warnings.warn(f"[Fig12] ignoring {path.name}: no axis matches basin count")
                continue
            if normalized.shape[0] < 2:
                warnings.warn(f"[Fig12] ignoring {path.name}: requires >=2 members")
                continue
            return normalized, path.name
        return None, None

    samples = {}
    sample_names = {}
    for model, bundle in [("LSTM", lstm_bundle), ("Mamba", mamba_bundle)]:
        samples[model], sample_names[model] = _load_samples(bundle)
        if samples[model] is None:
            print(
                f"[Fig12] skip: no ensemble samples for {model}; expected one of "
                f"{', '.join(FIG12_SAMPLE_FILES)}"
            )
            return

    assert samples["LSTM"] is not None and samples["Mamba"] is not None
    n_basins = min(obs.shape[0], n_metric_basins,
                   samples["LSTM"].shape[1], samples["Mamba"].shape[1])
    n_days = min(obs.shape[1], samples["LSTM"].shape[2], samples["Mamba"].shape[2])
    if n_basins == 0 or n_days == 0:
        print("[Fig12] skip: empty uncertainty arrays")
        return
    obs = obs[:n_basins, :n_days]
    samples = {model: values[:, :n_basins, :n_days]
               for model, values in samples.items()}
    dates = pd.date_range(FIG7_T0, periods=n_days, freq="D")
    rows: list[dict] = []

    # Use the full time series for one representative basin, but bound the
    # expensive all-basin diagnostics to a reproducible subset.
    eval_basin_idx = np.linspace(
        0, n_basins - 1, min(n_basins, FIG12_MAX_EVAL_BASINS), dtype=int
    )
    eval_day_idx = np.linspace(
        0, n_days - 1, min(n_days, FIG12_MAX_EVAL_DAYS), dtype=int
    )
    eval_obs = obs[np.ix_(eval_basin_idx, eval_day_idx)]
    eval_samples = {
        model: values[:, eval_basin_idx, :][:, :, eval_day_idx]
        for model, values in samples.items()
    }

    def _intervals(values: np.ndarray) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        result = {}
        for level in FIG12_LEVELS:
            lower = (1.0 - level) * 50.0
            upper = 100.0 - lower
            result[level] = (
                np.nanpercentile(values, lower, axis=0),
                np.nanpercentile(values, upper, axis=0),
            )
        return result

    # Pick the basin whose mean observed flow is closest to the continental median.
    observed_means = np.nanmean(np.where(obs > 0, obs, np.nan), axis=1)
    if np.isfinite(observed_means).any():
        target_mean = np.nanmedian(observed_means)
        representative = int(np.nanargmin(np.abs(observed_means - target_mean)))
    else:
        representative = 0
    plot_days = min(FIG12_MAX_DAYS, n_days)
    representative_intervals = {
        model: _intervals(values[:, representative, :plot_days])
        for model, values in samples.items()
    }
    representative_medians = {
        model: np.nanmedian(values[:, representative, :plot_days], axis=0)
        for model, values in samples.items()
    }
    diagnostic_intervals = {
        model: _intervals(values)
        for model, values in eval_samples.items()
    }
    print(
        f"[Fig12] diagnostics use {len(eval_basin_idx)} basins × "
        f"{len(eval_day_idx)} days; representative panel uses {plot_days} days"
    )

    fig, axes = plt.subplots(2, 2, figsize=FIG12_FIGSIZE, constrained_layout=True)

    # Panel A: one representative basin with nested prediction intervals.
    ax = axes[0, 0]
    ax.plot(dates[:plot_days], obs[representative, :plot_days],
            color=COLOR_OBS, lw=1.0, label="Observed")
    for model in ["LSTM", "Mamba"]:
        color = _model_style(model)["color"]
        for level in sorted(FIG12_LEVELS, reverse=True):
            lower, upper = representative_intervals[model][level]
            ax.fill_between(dates[:plot_days], lower, upper, color=color,
                            alpha=FIG12_BAND_ALPHA * (0.8 if level == 0.95 else 1.0),
                            linewidth=0, label=f"{model} {int(level * 100)}% PI")
        ax.plot(dates[:plot_days], representative_medians[model],
                color=color, lw=1.3, label=f"{model} median")
    ax.set_title("(a) Prediction intervals")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    # Panels B/C: empirical coverage and interval width for all basin-days.
    ax_cov = axes[0, 1]
    ax_rel = axes[1, 0]
    for model in ["LSTM", "Mamba"]:
        color = _model_style(model)["color"]
        coverage_values, width_values = [], []
        for level in FIG12_LEVELS:
            lower, upper = diagnostic_intervals[model][level]
            valid = np.isfinite(eval_obs) & np.isfinite(lower) & np.isfinite(upper)
            coverage = float(np.mean((eval_obs[valid] >= lower[valid]) & (eval_obs[valid] <= upper[valid]))) if valid.any() else np.nan
            width = float(np.nanmean((upper - lower)[valid])) if valid.any() else np.nan
            coverage_values.append(coverage)
            width_values.append(width)
            rows.append({
                "record_type": "coverage_summary", "model": model,
                "nominal_level": level, "empirical_coverage": coverage,
                "mean_interval_width_m3s": width,
                "sample_file": sample_names[model],
            })
        ax_cov.plot(width_values, coverage_values, marker="o", color=color, label=model)
        for width, coverage, level in zip(width_values, coverage_values, FIG12_LEVELS):
            if np.isfinite(width) and np.isfinite(coverage):
                ax_cov.annotate(f"{int(level * 100)}%", (width, coverage),
                                xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax_rel.plot(np.asarray(FIG12_LEVELS) * 100.0,
                    np.asarray(coverage_values) * 100.0,
                    marker="o", color=color, label=model)
    ax_cov.set_title("(b) Coverage versus interval width")
    ax_cov.set_xlabel("Mean interval width (m³/s)")
    ax_cov.set_ylabel("Empirical coverage")
    ax_cov.set_ylim(0, 1.02)
    ax_cov.legend(frameon=False)
    ax_rel.plot([0, 100], [0, 100], "k--", lw=1.0, label="Perfect reliability")
    ax_rel.set_title("(c) Reliability diagram")
    ax_rel.set_xlabel("Nominal coverage (%)")
    ax_rel.set_ylabel("Observed coverage (%)")
    ax_rel.set_xlim(45, 100)
    ax_rel.set_ylim(45, 100)
    ax_rel.legend(frameon=False, fontsize=8)

    # Panel D: PIT based on the empirical ensemble CDF.
    ax = axes[1, 1]
    for model in ["LSTM", "Mamba"]:
        values = eval_samples[model]
        valid_obs = np.isfinite(eval_obs)
        valid_obs &= np.any(np.isfinite(values), axis=0)
        if not valid_obs.any():
            continue
        observed_values = eval_obs[valid_obs]
        member_values = values[:, valid_obs]
        ranks = np.sum(member_values <= observed_values[None, :], axis=0)
        pit = (ranks + 0.5) / (values.shape[0] + 1.0)
        counts, edges = np.histogram(pit, bins=FIG12_PIT_BINS, range=(0, 1), density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax.step(centers, counts, where="mid", color=_model_style(model)["color"],
                lw=1.8, label=model)
        for center, count in zip(centers, counts):
            rows.append({"record_type": "pit_histogram", "model": model,
                         "pit_bin_center": float(center), "pit_density": float(count),
                         "sample_file": sample_names[model]})
    ax.axhline(1.0, color="black", ls="--", lw=0.9, label="Uniform PIT")
    ax.set_title("(d) PIT histogram")
    ax.set_xlabel("Probability integral transform")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)

    # fig.suptitle(
    #         f"Uncertainty and reliability (Tier {compare_tier}; "
    #         f"layers: {_layer_text_from_bundles(lstm_bundle, mamba_bundle)})",
    #         y=1.02,
    #     )
    save_fig(fig, fig_dir, "fig12_uncertainty_reliability")
    save_data(pd.DataFrame(rows), fig_dir, "fig12_uncertainty_reliability")


def collect_depth_ladder(
    tier: int,
    *,
    epoch: Optional[int] = None,
    lstm_exps: Optional[list[str]] = None,
    mamba_exps: Optional[list[str]] = None,
    mamba_version: Optional[int] = None,
    mamba_variant: Optional[str] = DEFAULT_MAMBA_VARIANT,
    output_root: Optional[Path] = None,
) -> pd.DataFrame:
    """One row per (model, layers) at a fixed tier: median NSE/KGE vs depth.

    Scans ``output_root/<exp>/Tier{tier}_*`` folders. Layer depth is taken
    from the run-folder ``_L{n}`` suffix when present, else from the exp-name
    ``_<n>L`` suffix (e.g. ``LSTM_3L``, ``Mamba2_5L``). Best (highest-epoch)
    completed run wins per (model, layers).
    """
    root = output_root or OUTPUT_ROOT
    if lstm_exps is None:
        lstm_exps = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and p.name.startswith("LSTM")
        ) if root.is_dir() else []
    if mamba_exps is None:
        mamba_exps = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and p.name.startswith("Mamba")
        ) if root.is_dir() else []
    rows: list[dict] = []
    for exp in list(lstm_exps) + list(mamba_exps):
        model = "LSTM" if exp.startswith("LSTM") else "Mamba"
        if exp.endswith("_RhoScaling"):
            continue
        if model == "Mamba":
            version_match = re.match(r"^Mamba(?P<version>\d+)_", exp)
            if mamba_version is not None and (
                version_match is None or int(version_match.group("version")) != mamba_version
            ):
                continue
        exp_root = root / exp
        if not exp_root.is_dir():
            continue
        m_exp = re.search(r"_(\d+)L$", exp)
        exp_layers = int(m_exp.group(1)) if m_exp else None
        best: tuple[int, Path, Optional[int]] | None = None  # (ep, dir, layers)
        for tier_dir in sorted(exp_root.iterdir()):
            if not tier_dir.is_dir():
                continue
            m = TIER_DIR_RE.match(tier_dir.name)
            if not m:
                continue
            if model == "Mamba" and mamba_variant is not None:
                if m.group("variant") != mamba_variant:
                    continue
            try:
                t = int(_re_field(m, "tier"))
            except (TypeError, ValueError):
                continue
            if t != tier:
                continue
            ep_raw = _re_field(m, "epoch")
            ep = int(ep_raw) if ep_raw is not None else -1
            if epoch is not None and ep != epoch:
                continue
            lyr_raw = _re_field(m, "layers")
            lyr = int(lyr_raw) if lyr_raw is not None else exp_layers
            if lyr is None:
                continue
            all_dir = tier_dir / "All"
            if not (all_dir / "basin_metrics.csv").is_file():
                continue
            if best is None or ep > best[0]:
                best = (ep, all_dir, lyr)
        if best is None:
            continue
        _, run_dir, lyr = best
        bundle = load_basin_bundle(run_dir)
        if bundle is None:
            continue
        metrics = bundle["metrics"]
        # Pearson r is emitted by some hydroDL versions directly in
        # basin_metrics.csv and by this project's evaluation fallback in
        # kge_components.csv. Prefer the former when available.
        if "r" in metrics.columns:
            pearson = metrics["r"]
        elif "correlation" in metrics.columns:
            pearson = metrics["correlation"]
        else:
            kge = bundle.get("kge")
            pearson = kge["r"] if kge is not None and "r" in kge.columns else None
        median_pr = float(np.nanmedian(pearson)) if pearson is not None else np.nan
        rows.append({
            "model": model,
            "exp": exp,
            "layers": int(lyr),  # type: ignore[arg-type]
            "tier": tier,
            "params": bundle["params"],
            "median_nse": float(np.nanmedian(metrics["NSE"])),
            "median_pr": median_pr,
            "median_kge": float(np.nanmedian(metrics["KGE"])),
            "n_basins": len(metrics),
            "run_dir": str(run_dir),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["model", "layers"]).reset_index(drop=True)
    return df


def collect_rho_ladder(
    tier: int,
    *,
    epoch: Optional[int] = None,
    lstm_layers: int = 1,
    mamba_layers: int = DEFAULT_MAMBA_LAYERS,
    mamba_version: int = DEFAULT_MAMBA_VERSION,
    mamba_variant: Optional[str] = DEFAULT_MAMBA_VARIANT,
    lstm_exp: Optional[str] = None,
    mamba_exp: Optional[str] = None,
    output_root: Optional[Path] = None,
) -> pd.DataFrame:
    """One row per (model, rho) at a fixed tier and layer depth.

    The scan targets the dedicated ``*_RhoScaling`` experiment folders.  For
    Mamba, the omitted ``_Rho`` folder suffix is the default ``rho=365``.
    When multiple completed runs exist for a rho, the highest-epoch run wins.
    """
    root = output_root or OUTPUT_ROOT
    experiments = [
        ("LSTM", lstm_exp or f"LSTM_{lstm_layers}L_RhoScaling", lstm_layers),
        (
            "Mamba",
            mamba_exp or f"Mamba{mamba_version}_{mamba_layers}L_RhoScaling",
            mamba_layers,
        ),
    ]
    rows: list[dict] = []
    for model, exp, exp_layers in experiments:
        exp_root = root / exp
        if not exp_root.is_dir():
            continue
        best: dict[int, tuple[int, Path]] = {}
        for tier_dir in sorted(exp_root.iterdir()):
            if not tier_dir.is_dir():
                continue
            match = TIER_DIR_RE.match(tier_dir.name)
            if not match:
                continue
            if model == "Mamba" and mamba_variant is not None:
                if match.group("variant") != mamba_variant:
                    continue
            try:
                if int(_re_field(match, "tier")) != tier:
                    continue
                run_epoch = int(_re_field(match, "epoch"))
            except (TypeError, ValueError):
                continue
            if epoch is not None and run_epoch != epoch:
                continue
            run_layers = _re_field(match, "layers")
            if run_layers is not None and int(run_layers) != exp_layers:
                continue
            rho_raw = _re_field(match, "rho")
            rho = int(rho_raw) if rho_raw is not None else 365
            all_dir = tier_dir / "All"
            if not (all_dir / "basin_metrics.csv").is_file():
                continue
            if rho not in best or run_epoch > best[rho][0]:
                best[rho] = (run_epoch, all_dir)

        for rho, (run_epoch, run_dir) in sorted(best.items()):
            bundle = load_basin_bundle(run_dir)
            if bundle is None:
                continue
            metrics = bundle["metrics"]
            rows.append({
                "model": model,
                "exp": exp,
                "layers": exp_layers,
                "tier": tier,
                "rho": rho,
                "params": bundle["params"],
                "median_nse": float(np.nanmedian(metrics["NSE"])),
                "median_kge": float(np.nanmedian(metrics["KGE"])),
                "n_basins": len(metrics),
                "epoch": run_epoch,
                "run_dir": str(run_dir),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["model", "rho"]).reset_index(drop=True)
    return df


def figure14_rho_scaling(
    rho_data: pd.DataFrame,
    fig_dir: Path,
    tier: int,
    metric: str = "both",
):
    """Plot median NSE and/or KGE against sequence length ``rho``."""
    if rho_data.empty:
        print(f"[Fig14] skip: no completed rho-scaling runs at tier {tier}")
        return
    metrics = ["nse", "kge"] if metric == "both" else [metric]
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6.2 * len(metrics), 4.2),
        squeeze=False, constrained_layout=True,
    )
    axes = axes[0]
    for model, group in rho_data.groupby("model"):
        sty = _model_style(model)
        group = group.sort_values("rho")
        for ax, name in zip(axes, metrics):
            ax.plot(
                group["rho"], group[f"median_{name}"],
                color=sty["color"], ls=sty["ls"], marker=sty["marker"],
                ms=6, lw=1.8, label=sty["label"],
            )
    for ax, name in zip(axes, metrics):
        ax.set_xlabel("Sequence length (rho)")
        ax.set_ylabel(f"Median test {name.upper()}")
        ax.set_title(f"{name.upper()} vs rho")
        ax.legend(frameon=False)
    layer_text = _layer_text_from_frame(rho_data)
    layer_extra = f"; layers: {layer_text}" if layer_text else ""
    # fig.suptitle(f"Rho scaling at Tier {tier}{layer_extra}", y=1.03)
    stem = "fig14_rho_scaling" if metric == "both" else f"fig14_rho_scaling_{metric}"
    save_fig(fig, fig_dir, stem)
    save_data(rho_data, fig_dir, stem)


def figure15_training_time(
    time_data: pd.DataFrame,
    fig_dir: Path,
    *,
    lstm_layers: Optional[int] = None,
    mamba_layers: Optional[int] = None,
    all_layers: bool = False,
):
    """Fig 15: Training time (minutes) vs tier for LSTM and Mamba.
    
    Plots total time to complete 30 epochs across parameter tiers.
    Similar to Figure 2 but with training time on y-axis instead of NSE/KGE.
    
    Args:
        time_data: DataFrame with training time data
        fig_dir: Output directory
        lstm_layers: Optional filter for LSTM layer depth
        mamba_layers: Optional filter for Mamba layer depth
        all_layers: If True, plot all available layer depths with different markers
    """
    if time_data.empty:
        print("[Fig15] skip: no training time data found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=FIG15_FIGSIZE, constrained_layout=True)
    
    custom_lines = []
    custom_labels = []
    
    if all_layers and "layers" in time_data.columns:
        # Plot all layers with different markers/styles
        layer_markers = {1: 'o', 2: 's', 3: '^'}
        layer_styles = {1: '-', 2: '--', 3: ':'}
        
        # Use tier names as x-axis categories
        tier_order = sorted(time_data["tier"].unique())
        tier_labels = [time_data[time_data["tier"] == t]["tier_name"].values[0] for t in tier_order]
        
        for model, g in time_data.groupby("model"):
            sty = _model_style(model)
            g = g.sort_values(["layers", "tier"])
            if g.empty:
                continue
            
            for layer, layer_group in g.groupby("layers"):
                marker = layer_markers.get(layer, 'o')
                ls = layer_styles.get(layer, '-')
                
                # Plot using tier numbers as x positions
                tier_nums = layer_group["tier"].tolist()
                tier_times = layer_group["total_time_minutes"].tolist()
                
                # Sort by tier number to maintain order
                sorted_data = sorted(zip(tier_nums, tier_times))
                tier_nums_sorted, times_sorted = zip(*sorted_data) if sorted_data else ([], [])
                
                ax.plot(
                    tier_nums_sorted, times_sorted,
                    color=sty["color"], ls="none", marker=marker,
                    ms=FIG15_MARKER_MS, lw=FIG15_LINE_LW,
                )
        
        # Set x-axis to show tier names
        ax.set_xticks(tier_order)
        ax.set_xticklabels(tier_labels)
        
        # Create custom legend: show layers with markers, then colors for models
        # Layer entries with markers
        for layer in sorted(layer_markers.keys()):
            marker = layer_markers[layer]
            ls = layer_styles[layer]
            line = Line2D([0], [0], color='gray', ls="none", marker=marker, 
                          ms=FIG15_MARKER_MS, lw=FIG15_LINE_LW)
            custom_lines.append(line)
            custom_labels.append(f"{layer}L")
        
        # Model entries with colors
        for model in ["LSTM", "Mamba"]:
            sty = _model_style(model)
            line = Line2D([0], [0], color=sty["color"], ls="none", marker='o',
                          ms=FIG15_MARKER_MS, lw=FIG15_LINE_LW)
            custom_lines.append(line)
            custom_labels.append(sty["label"])
        
        ax.set_xlim(min(tier_order) - 0.5, max(tier_order) + 0.5)
        ax.legend(custom_lines, custom_labels, frameon=False, loc="upper left")
        
        layer_text = " (all layers)"
    else:
        # Original single-layer plotting with tier names on x-axis
        tier_order = sorted(time_data["tier"].unique())
        tier_labels = [time_data[time_data["tier"] == t]["tier_name"].values[0] for t in tier_order]
        
        for model, g in time_data.groupby("model"):
            sty = _model_style(model)
            g = g.sort_values("tier")
            if g.empty:
                continue
            
            tier_nums = g["tier"].tolist()
            ax.plot(
                tier_nums, g["total_time_minutes"],
                color=sty["color"], ls="none", marker=sty["marker"],
                ms=FIG15_MARKER_MS, lw=FIG15_LINE_LW, label=sty["label"],
            )
        
        ax.set_xticks(tier_order)
        ax.set_xticklabels(tier_labels)
        ax.set_xlim(min(tier_order) - 0.5, max(tier_order) + 0.5)
        ax.legend(frameon=False, loc="upper left")
        
        layer_text = f" (LSTM {lstm_layers or 1}L, Mamba {mamba_layers or 3}L)" if lstm_layers or mamba_layers else ""
    
    ax.set_xlabel("Parameter Tier")
    ax.set_ylabel("Total Training Time (minutes)")
    # ax.set_title("Training Time to Complete 30 Epochs")
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, ls="--")
    
    # fig.suptitle(f"Training Time vs Parameter Tier{layer_text}", y=1.03)
    
    save_fig(fig, fig_dir, "fig05_training_time")
    save_data(time_data, fig_dir, "fig05_training_time")


def figure13_depth_scaling(depth: pd.DataFrame, fig_dir: Path, tier: int):
    """Fig 13: median NSE, Pearson r (PR), and KGE vs layers at a fixed tier."""
    if depth.empty:
        print(f"[Fig13] skip: no completed runs at tier {tier}")
        return
    metric_specs = [
        ("median_nse", "Median test NSE", "(a) Depth scaling (NSE)"),
        ("median_pr", "Median test PR", "(b) Depth scaling (PR)"),
        ("median_kge", "Median test KGE", "(c) Depth scaling (KGE)"),
    ]
    fig, axes = plt.subplots(
        1, len(metric_specs), figsize=(FIG13_FIGSIZE[0] * 1.35, FIG13_FIGSIZE[1]),
        constrained_layout=True,
    )
    for model, g in depth.groupby("model"):
        sty = _model_style(model)
        g = g.sort_values("layers")
        for ax, (column, _, _) in zip(axes, metric_specs):
            ax.plot(
                g["layers"], g[column],
                color=sty["color"], ls=sty["ls"], marker=sty["marker"],
                ms=FIG13_MARKER_MS, lw=FIG13_LINE_LW, label=sty["label"],
            )
    for ax, (_, ylab, title) in zip(axes, metric_specs):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Layers")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(frameon=False)
    layer_text = _layer_text_from_frame(depth)
    layer_extra = f"; layers: {layer_text}" if layer_text else ""
    # fig.suptitle(
    #     f"NSE, PR, and KGE vs layers (Tier {tier}{layer_extra})",
    #     y=1.03,
    # )
    save_fig(fig, fig_dir, "fig13_depth_scaling")
    save_data(
        depth.sort_values(["model", "layers"])[
            ["model", "exp", "layers", "tier", "params",
             "median_nse", "median_pr", "median_kge", "n_basins"]
        ],
        fig_dir, "fig13_depth_scaling",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Plot figures.md Figs 1–15 from experiment outputs")
    p.add_argument("--figures", "--figure", type=int, nargs="+", default=DEFAULT_FIGURES,
                   help="Which figure numbers to plot (default: %(default)s)")
    p.add_argument("--tiers", type=int, nargs="+", choices=TIER_CHOICES, default=None,
                   help="Global tier allow-list for ladder figs 2 & 5 (e.g. 1 2 3). "
                        "Also sets all single-tier figs to the FIRST tier given here "
                        "unless --compare-tier is also given.")
    p.add_argument("--compare-tier", type=int, choices=TIER_CHOICES, default=None,
                   help=f"Matched-parameter tier for Figs 1,3,4,6,7,8,9,10,11,12 "
                        f"(default: top of CONFIG block {DEFAULT_COMPARE_TIER}).")
    p.add_argument("--tier-fig1", type=int, choices=TIER_CHOICES, default=None,
                   help=f"Override tier for Fig 1 only (default: config TIER_FIG1={TIER_FIG1_LEARNING_CURVES})")
    p.add_argument("--tier-fig3", type=int, choices=TIER_CHOICES, default=None,
                   help=f"Override tier for Fig 3 only (default: config TIER_FIG3={TIER_FIG3_ECDF})")
    p.add_argument("--tier-fig4", type=int, choices=TIER_CHOICES, default=None,
                   help=f"Override tier for Fig 4 only (default: config TIER_FIG4={TIER_FIG4_SIGNATURES})")
    p.add_argument("--tier-fig6", type=int, choices=TIER_CHOICES, default=None,
                   help=f"Override tier for Fig 6 only (default: config TIER_FIG6={TIER_FIG6_SPATIAL})")
    p.add_argument("--tier-fig7", type=int, choices=TIER_CHOICES, default=None,
                   help=f"Override tier for Fig 7 only (default: config TIER_FIG7={TIER_FIG7_HYDROGRAPHS})")
    p.add_argument("--tier-fig8", type=int, choices=TIER_CHOICES, default=None,
                   help=f"Override tier for Fig 8 only (default: config TIER_FIG8={TIER_FIG8_FDC})")
    p.add_argument("--tier-fig9", type=int, choices=TIER_CHOICES, default=None,
                   help=f"Override tier for Fig 9 only (default: config TIER_FIG9={TIER_FIG9_SEASONAL})")
    p.add_argument("--tier-fig10", type=int, choices=TIER_CHOICES, default=None,
                   help=f"Override tier for Fig 10 only (default: config TIER_FIG10={TIER_FIG10_RESIDUALS})")
    p.add_argument("--tier-fig11", type=int, choices=TIER_CHOICES, default=None,
                   help=f"Override tier for Fig 11 only (default: config TIER_FIG11={TIER_FIG11_EVENTS})")
    p.add_argument("--tier-fig12", type=int, choices=TIER_CHOICES, default=None,
                   help=f"Override tier for Fig 12 only (default: config TIER_FIG12={TIER_FIG12_UNCERTAINTY})")
    p.add_argument("--tiers-fig2", type=int, nargs="+", choices=TIER_CHOICES, default=None,
                   help=f"Override tier list for Fig 2 (param scaling) (default: config TIERS_FIG2={TIERS_FIG2_PARAM_SCALING})")
    p.add_argument("--tiers-fig5", type=int, nargs="+", choices=TIER_CHOICES, default=None,
                    help=f"Override tier list for Fig 5 (tier scaling) (default: config TIERS_FIG5={TIERS_FIG5_TIER_SCALING})")
    p.add_argument("--tier-fig13", type=int, choices=TIER_CHOICES, default=None,
                    help=f"Tier for Fig 13 depth scaling (default: config TIER_FIG13_DEPTH={TIER_FIG13_DEPTH})")
    p.add_argument("--depth-lstm-exps", type=str, nargs="+", default=None,
                    help="LSTM exp folders to scan for Fig 13 (default: all LSTM_* under output/)")
    p.add_argument("--depth-mamba-exps", type=str, nargs="+", default=None,
                    help="Mamba exp folders to scan for Fig 13 (default: all Mamba* under output/)")
    p.add_argument("--tier-fig14", type=int, choices=TIER_CHOICES, default=None,
                   help=f"Tier for Fig 14 rho scaling (default: --tiers first value or {DEFAULT_COMPARE_TIER})")
    p.add_argument("--rho-metric", choices=["nse", "kge", "both"], default="both",
                   help="Metric(s) for Fig 14 (default: both)")
    p.add_argument("--rho-lstm-exp", type=str, default=None,
                   help="Rho-scaling LSTM experiment folder (default: LSTM_<layers>L_RhoScaling)")
    p.add_argument("--rho-mamba-exp", type=str, default=None,
                   help="Rho-scaling Mamba experiment folder (default: Mamba<version>_<layers>L_RhoScaling)")
    p.add_argument("--output-root", type=str, default=None,
                   help="Custom output directory root (default: experiment/output)")

    p.add_argument("--epoch", type=int, default=DEFAULT_EPOCH,
                   help="Require this exact completed epoch for both models; if omitted, use the highest completed epoch")
    p.add_argument("--lstm-epoch", type=int, default=DEFAULT_LSTM_EPOCH,
                   help="Require this exact completed epoch for LSTM only (overrides --epoch for LSTM)")
    p.add_argument("--mamba-epoch", type=int, default=DEFAULT_MAMBA_EPOCH,
                   help="Require this exact completed epoch for Mamba only (overrides --epoch for Mamba)")
    p.add_argument("--lstm-exp", type=str, default=DEFAULT_LSTM_EXP,
                   help=f"LSTM experiment folder under output/ (default: from config)")
    p.add_argument("--mamba-exp", type=str, default=DEFAULT_MAMBA_EXP,
                   help=f"Mamba experiment folder under output/ (default: from config)")
    p.add_argument("--lstm-layers", type=int, choices=(1, 2, 3),
                   default=DEFAULT_LSTM_LAYERS,
                   help="LSTM depth used for the selected LSTM experiment")
    p.add_argument("--mamba-layers", type=int, choices=(1, 2, 3),
                   default=DEFAULT_MAMBA_LAYERS,
                   help="Mamba depth used for the selected Mamba experiment")
    p.add_argument("--mamba-version", type=int, choices=(1, 2, 3),
                   default=DEFAULT_MAMBA_VERSION,
                   help="Mamba implementation version used for discovery")
    p.add_argument("--mamba-variant", type=str,
                   choices=["standard", "film", "hybrid", "prefix", "residual", "adapter"],
                   default=DEFAULT_MAMBA_VARIANT,
                   help="Mamba variant to use for depth/rho scaling discovery "
                        f"(default: {DEFAULT_MAMBA_VARIANT})")
    p.add_argument("--no-fig2-all-layers", action="store_true", default=False,
                   help="Disable all-layers mode for Figure 2 (default: all-layers enabled)")
    p.add_argument("--no-fig15-all-layers", action="store_true", default=False,
                   help="Disable all-layers mode for Figure 15 (default: all-layers enabled)")
    p.add_argument("--basin-ids", type=str, nargs="+", default=FIG7_BASIN_IDS,
                   help="Optional USGS gauge IDs for Fig 7 (overrides auto-pick)")
    p.add_argument("--fig-dir", type=str, default=str(DEFAULT_FIG_DIR))
    p.add_argument("--list-runs", action="store_true",
                   help="Only list discovered completed runs and exit")
    return p.parse_args()


def main():
    args = parse_args()
    apply_style()
    fig_dir = Path(args.fig_dir)

    # ----- Resolve per-figure tiers (CLI > global --tiers > CONFIG block) -----
    def _single_tier(cli_value: int | None, cfg_default: int) -> int:
        if cli_value is not None:
            return cli_value
        if args.tiers is not None and len(args.tiers) > 0:
            return args.tiers[0]
        return cfg_default

    def _tier_list(cli_value: list[int] | None, cfg_default: list[int]) -> list[int]:
        if cli_value is not None:
            return cli_value
        if args.tiers is not None:
            return list(args.tiers)
        return list(cfg_default)

    tier_fig1 = _single_tier(args.tier_fig1, TIER_FIG1_LEARNING_CURVES)
    tier_fig3 = _single_tier(args.tier_fig3, TIER_FIG3_ECDF)
    tier_fig4 = _single_tier(args.tier_fig4, TIER_FIG4_SIGNATURES)
    tier_fig6 = _single_tier(args.tier_fig6, TIER_FIG6_SPATIAL)
    tier_fig7 = _single_tier(args.tier_fig7, TIER_FIG7_HYDROGRAPHS)
    tier_fig8 = _single_tier(args.tier_fig8, TIER_FIG8_FDC)
    tier_fig9 = _single_tier(args.tier_fig9, TIER_FIG9_SEASONAL)
    tier_fig10 = _single_tier(args.tier_fig10, TIER_FIG10_RESIDUALS)
    tier_fig11 = _single_tier(args.tier_fig11, TIER_FIG11_EVENTS)
    tier_fig12 = _single_tier(args.tier_fig12, TIER_FIG12_UNCERTAINTY)
    tiers_fig2 = _tier_list(args.tiers_fig2, TIERS_FIG2_PARAM_SCALING)
    tiers_fig5 = _tier_list(args.tiers_fig5, TIERS_FIG5_TIER_SCALING)
    tier_fig13 = args.tier_fig13
    if tier_fig13 is None:
        tier_fig13 = args.tiers[0] if args.tiers is not None and len(args.tiers) > 0 else TIER_FIG13_DEPTH
    tier_fig14 = args.tier_fig14
    if tier_fig14 is None:
        tier_fig14 = args.tiers[0] if args.tiers is not None and len(args.tiers) > 0 else DEFAULT_COMPARE_TIER
    # compare_tier alias — used only for legacy display and fallback
    compare_tier = args.compare_tier if args.compare_tier is not None else (
        args.tiers[0] if args.tiers is not None else DEFAULT_COMPARE_TIER
    )

    # None means discover the highest matching completed epoch. Do not turn
    # the config default into an implicit filter: older and newer campaigns
    # can coexist under the same experiment family.
    epoch = args.epoch
    lstm_epoch = args.lstm_epoch if args.lstm_epoch is not None else epoch
    mamba_epoch = args.mamba_epoch if args.mamba_epoch is not None else epoch

    # Override the per-figure single-tier values if --compare-tier is given
    # (this lets --compare-tier affect every single-tier figure at once, like the old API)
    if args.compare_tier is not None:
        tier_fig1 = args.compare_tier
        tier_fig3 = args.compare_tier
        tier_fig4 = args.compare_tier
        tier_fig6 = args.compare_tier
        tier_fig7 = args.compare_tier
        tier_fig8 = args.compare_tier
        tier_fig9 = args.compare_tier
        tier_fig10 = args.compare_tier
        tier_fig11 = args.compare_tier
        tier_fig12 = args.compare_tier
        tier_fig14 = args.compare_tier

    # ----- Resolve experiment names -----
    # Temporarily swap defaults so resolve_exp_names respects the CONFIG block
    import types
    fake_args = types.SimpleNamespace(
        lstm_exp=args.lstm_exp,
        mamba_exp=args.mamba_exp,
        lstm_layers=args.lstm_layers,
        mamba_version=args.mamba_version,
        mamba_layers=args.mamba_layers,
    )
    lstm_exp, mamba_exp = resolve_exp_names(fake_args)

    # Layer hints for discovery. If an explicit experiment folder is given,
    # infer its depth when the CLI value is still the default; this prevents
    # --lstm-exp LSTM_1L (or equivalent) from being filtered as 3-layer.
    def _exp_layers(
        exp_name: str | None,
        current: int | None,
        default: int | None,
    ) -> int | None:
        if exp_name is not None and current == default:
            match = re.search(r"_(\d+)L(?:_RhoScaling)?$", exp_name)
            if match:
                return int(match.group(1))
        return current

    lstm_layers = _exp_layers(
        args.lstm_exp, args.lstm_layers, DEFAULT_LSTM_LAYERS
    )
    mamba_layers = _exp_layers(
        args.mamba_exp, args.mamba_layers, DEFAULT_MAMBA_LAYERS
    )

    # Resolve custom output root if provided
    output_root = Path(args.output_root) if args.output_root else OUTPUT_ROOT

    print("=" * 72)
    print("CAMELS figure plotter".center(72))
    print("=" * 72)
    print(f"LSTM exp   : {lstm_exp}  (layers filter={lstm_layers})")
    print(f"Mamba exp  : {mamba_exp}  (layers filter={mamba_layers}, version={args.mamba_version}, variant={args.mamba_variant})")
    print(f"Epoch filter: {epoch if epoch is not None else 'highest completed'}")
    print(f"Figure dir : {fig_dir}")
    print(f"Figures    : {args.figures}")
    print(f"  Fig 1 tier: {tier_fig1}  |  Fig 3 tier: {tier_fig3}  |  Fig 4 tier: {tier_fig4}")
    print(f"  Fig 6 tier: {tier_fig6}  |  Fig 7 tier: {tier_fig7}  |  Fig 8 tier: {tier_fig8}  |  Fig 9 tier: {tier_fig9}  |  Fig 10 tier: {tier_fig10}  |  Fig 11 tier: {tier_fig11}  |  Fig 12 tier: {tier_fig12}")
    print(f"  Fig 2 tiers: {tiers_fig2}")
    print(f"  Fig 5 tiers: {tiers_fig5}")
    print(f"  Fig 14 tier: {tier_fig14}  | rho metric: {args.rho_metric}")
    print(f"(--compare-tier legacy alias: {compare_tier})")

    ladder = collect_ladder(
        lstm_exp, mamba_exp,
        epoch=epoch,
        lstm_epoch=lstm_epoch,
        mamba_epoch=mamba_epoch,
        lstm_layers=lstm_layers,
        mamba_layers=mamba_layers,
        mamba_variant=args.mamba_variant,
        output_root=output_root,
    )

    # Per-model fallback: if the preferred epoch finds nothing for one model
    # (e.g. LSTM ran at Ep30 while Mamba ran at Ep20), rediscover that model
    # with any completed epoch instead of dropping it entirely.
    if args.epoch is None and args.lstm_epoch is None and args.mamba_epoch is None:  # only auto-fallback when user didn't pin any epoch
        have = set(ladder["model"].unique()) if not ladder.empty else set()
        if have != {"LSTM", "Mamba"}:
            any_ladder = collect_ladder(
                lstm_exp, mamba_exp,
                epoch=None,
                lstm_epoch=None,
                mamba_epoch=None,
                lstm_layers=lstm_layers,
                mamba_layers=mamba_layers,
                mamba_variant=args.mamba_variant,
            )
            for _model in ["LSTM", "Mamba"]:
                if _model in have or _model not in set(any_ladder["model"].unique()):
                    continue
                _any = any_ladder[any_ladder["model"] == _model]
                model_specific_epoch = lstm_epoch if _model == "LSTM" else mamba_epoch
                epoch_str = f"epoch={model_specific_epoch}" if model_specific_epoch else f"epoch={epoch}"
                print(
                    f"[warn] No {_model} runs at {epoch_str}; "
                    f"using any completed epoch instead "
                    f"(e.g. {_any['run_dir'].iloc[0]})"
                )
                ladder = pd.concat([ladder, _any], ignore_index=True)

    if ladder.empty and args.epoch is None and args.lstm_epoch is None and args.mamba_epoch is None:
        print(f"[warn] No completed runs found; discovering any completed tiers…")
        ladder = collect_ladder(
            lstm_exp, mamba_exp,
            epoch=None,
            lstm_epoch=None,
            mamba_epoch=None,
            lstm_layers=lstm_layers,
            mamba_layers=mamba_layers,
            mamba_variant=args.mamba_variant,
        )

    if not ladder.empty:
        print("\nDiscovered completed runs:")
        show = ladder[["model", "tier", "tier_name", "params", "median_nse", "median_kge", "run_dir"]]
        print(show.to_string(index=False))
    else:
        print("\n[warn] No completed runs found under the selected experiment folders.")
        print("       Finish `python src/run_all_tiers.py` first, or pass --lstm-exp / --mamba-exp.")

    if args.list_runs:
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    if not ladder.empty:
        ladder.to_csv(fig_dir / "ladder_summary.csv", index=False)
        print(f"\nWrote {fig_dir / 'ladder_summary.csv'}")

    # Resolve per-figure (tier) run dirs + bundles
    def _tier_run(model_name: str, tier_n: int) -> Optional[Path]:
        hit = ladder[(ladder["model"] == model_name) & (ladder["tier"] == tier_n)]
        if hit.empty:
            return None
        return Path(hit.iloc[0]["run_dir"])

    wanted = set(args.figures)

    # Pre-resolve bundles that share tiers (each tier resolved once)
    tier_to_bundle = {}
    for fig, tier in [("fig1", tier_fig1), ("fig3", tier_fig3),
                      ("fig4", tier_fig4), ("fig6", tier_fig6),
                      ("fig7", tier_fig7), ("fig8", tier_fig8),
                      ("fig9", tier_fig9), ("fig10", tier_fig10),
                      ("fig11", tier_fig11), ("fig12", tier_fig12) ]:
        if tier in tier_to_bundle:
            continue
        tier_to_bundle[tier] = {
            "LSTM": None, "Mamba": None,
        }
        for m in ["LSTM", "Mamba"]:
            rd = _tier_run(m, tier)
            if rd:
                tier_to_bundle[tier][m] = load_basin_bundle(rd)

    def _bundle_for(tier_n: int, model: str) -> Optional[dict]:
        return tier_to_bundle.get(tier_n, {}).get(model)

    def _run_for(tier_n: int, model: str) -> Optional[Path]:
        return _tier_run(model, tier_n)

    if 1 in wanted:
        print("\n[Fig1] Learning curves…")
        r1, r2 = _run_for(tier_fig1, "LSTM"), _run_for(tier_fig1, "Mamba")
        if r1 or r2:
            figure1_learning_curves(r1, r2, fig_dir, tier_fig1)
        else:
            print(f"  skip: need at least one model at tier {tier_fig1}")

    if 2 in wanted:
        print("\n[Fig2] Parameter scaling…")
        if not args.no_fig2_all_layers:
            print("  Using all layers mode...")
            all_layers_ladder = collect_all_layers_ladder(
                lstm_exps=args.depth_lstm_exps,
                mamba_exps=args.depth_mamba_exps,
                epoch=epoch if args.epoch is not None else None,
                lstm_epoch=lstm_epoch if args.lstm_epoch is not None else None,
                mamba_epoch=mamba_epoch if args.mamba_epoch is not None else None,
                mamba_version=args.mamba_version,
                mamba_variant=args.mamba_variant,
                output_root=output_root,
            )
            if not all_layers_ladder.empty:
                print(f"  Found {len(all_layers_ladder)} layer configurations")
                print(all_layers_ladder[["model", "exp", "layers", "tier", "tier_name", "params", 
                                          "median_nse", "median_kge"]].to_string(index=False))
            figure2_param_scaling(
                all_layers_ladder,
                fig_dir,
                tiers=tiers_fig2,
                all_layers=True,
            )
        else:
            figure2_param_scaling(
                ladder,
                fig_dir,
                tiers=tiers_fig2,
                lstm_layers=lstm_layers,
                mamba_layers=mamba_layers,
            )

    if 3 in wanted:
        print("\n[Fig3] ECDFs…")
        b1, b2 = _bundle_for(tier_fig3, "LSTM"), _bundle_for(tier_fig3, "Mamba")
        if b1 and b2:
            figure3_ecdf(b1, b2, fig_dir, tier_fig3)
        else:
            print(f"  skip: need both models at tier {tier_fig3}")

    if 4 in wanted:
        print("\n[Fig4] Hydrologic signatures…")
        b1, b2 = _bundle_for(tier_fig4, "LSTM"), _bundle_for(tier_fig4, "Mamba")
        if b1 and b2:
            figure4_signatures(b1, b2, fig_dir, tier_fig4)
        else:
            print(f"  skip: need both models at tier {tier_fig4}")

    if 5 in wanted:
        print("\n[Fig5] Tier scaling…")
        figure5_tier_scaling(ladder, fig_dir, tiers=tiers_fig5)

    if 6 in wanted:
        print("\n[Fig6] Spatial maps…")
        b1, b2 = _bundle_for(tier_fig6, "LSTM"), _bundle_for(tier_fig6, "Mamba")
        if b1 and b2:
            figure6_spatial(b1, b2, fig_dir, tier_fig6)
        else:
            print(f"  skip: need both models at tier {tier_fig6}")

    if 7 in wanted:
        print("\n[Fig7] Hydrographs…")
        b1, b2 = _bundle_for(tier_fig7, "LSTM"), _bundle_for(tier_fig7, "Mamba")
        if b1 and b2:
            figure7_hydrographs(
                b1, b2, fig_dir, tier_fig7,
                basin_ids=args.basin_ids,
            )
        else:
            print(f"  skip: need both models at tier {tier_fig7}")

    if 8 in wanted:
        print("\n[Fig8] Flow-duration curves…")
        b1, b2 = _bundle_for(tier_fig8, "LSTM"), _bundle_for(tier_fig8, "Mamba")
        if b1 and b2:
            figure8_fdc(b1, b2, fig_dir, tier_fig8)
        else:
            print(f"  skip: need both models at tier {tier_fig8}")

    if 9 in wanted:
        print("\n[Fig9] Seasonal performance heatmaps…")
        b1, b2 = _bundle_for(tier_fig9, "LSTM"), _bundle_for(tier_fig9, "Mamba")
        if b1 and b2:
            figure9_seasonal_heatmaps(b1, b2, fig_dir, tier_fig9)
        else:
            print(f"  skip: need both models at tier {tier_fig9}")

    if 10 in wanted:
        print("\n[Fig10] Residual diagnostics…")
        b1, b2 = _bundle_for(tier_fig10, "LSTM"), _bundle_for(tier_fig10, "Mamba")
        if b1 and b2:
            figure10_residual_diagnostics(b1, b2, fig_dir, tier_fig10)
        else:
            print(f"  skip: need both models at tier {tier_fig10}")

    if 11 in wanted:
        print("\n[Fig11] Peak-event diagnostics…")
        b1, b2 = _bundle_for(tier_fig11, "LSTM"), _bundle_for(tier_fig11, "Mamba")
        if b1 and b2:
            figure11_peak_events(b1, b2, fig_dir, tier_fig11)
        else:
            print(f"  skip: need both models at tier {tier_fig11}")

    if 12 in wanted:
        print("\n[Fig12] Uncertainty and reliability…")
        b1, b2 = _bundle_for(tier_fig12, "LSTM"), _bundle_for(tier_fig12, "Mamba")
        if b1 and b2:
            figure12_uncertainty(b1, b2, fig_dir, tier_fig12)
        else:
            print(f"  skip: need both models at tier {tier_fig12}")

    if 13 in wanted:
        print(f"\n[Fig13] Depth scaling (NSE vs layers at tier {tier_fig13})…")
        depth = collect_depth_ladder(
            tier_fig13,
            epoch=epoch if args.epoch is not None else None,
            lstm_exps=args.depth_lstm_exps,
            mamba_exps=args.depth_mamba_exps,
            mamba_version=args.mamba_version,
            mamba_variant=args.mamba_variant,
            output_root=output_root,
        )
        if not depth.empty:
            print(depth[["model", "exp", "layers", "params",
                          "median_nse", "median_pr", "median_kge"]].to_string(index=False))
            depth.to_csv(fig_dir / "depth_summary.csv", index=False)
        figure13_depth_scaling(depth, fig_dir, tier_fig13)

    if 14 in wanted:
        print(f"\n[Fig14] Rho scaling ({args.rho_metric}) at tier {tier_fig14}…")
        rho_data = collect_rho_ladder(
            tier_fig14,
            epoch=epoch if args.epoch is not None else None,
            lstm_layers=lstm_layers or DEFAULT_LSTM_LAYERS,
            mamba_layers=mamba_layers or DEFAULT_MAMBA_LAYERS,
            mamba_version=args.mamba_version,
            mamba_variant=args.mamba_variant,
            lstm_exp=args.rho_lstm_exp,
            mamba_exp=args.rho_mamba_exp,
            output_root=output_root,
        )
        if not rho_data.empty:
            print(rho_data[["model", "rho", "median_nse", "median_kge",
                            "run_dir"]].to_string(index=False))
        figure14_rho_scaling(rho_data, fig_dir, tier_fig14, args.rho_metric)

    if 15 in wanted:
        print("\n[Fig15] Training time vs tier…")
        if not args.no_fig15_all_layers:
            time_data = collect_all_layers_training_time(
                mamba_version=args.mamba_version,
                mamba_variant=args.mamba_variant,
                output_root=output_root,
            )
            if not time_data.empty:
                print(time_data[["model", "tier", "tier_name", "layers", "total_time_minutes", 
                                "n_epochs"]].to_string(index=False))
            figure15_training_time(
                time_data, fig_dir,
                all_layers=True,
            )
        else:
            time_data = collect_training_time(
                lstm_exp,
                mamba_exp,
                lstm_layers=lstm_layers,
                mamba_layers=mamba_layers,
                mamba_variant=args.mamba_variant,
                output_root=output_root,
            )
            if not time_data.empty:
                print(time_data[["model", "tier", "tier_name", "total_time_minutes", 
                                "n_epochs"]].to_string(index=False))
            figure15_training_time(
                time_data, fig_dir,
                lstm_layers=lstm_layers,
                mamba_layers=mamba_layers,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
