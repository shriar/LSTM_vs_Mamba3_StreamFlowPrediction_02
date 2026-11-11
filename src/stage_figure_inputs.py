"""D4.2 -- stage legacy-shaped run trees from the revision's artefacts.

``src/plot_figures.py`` expects the legacy layout (``basin_metrics.csv``,
``checkpoint_metrics.csv``, ``obs.npy`` / ``pred.npy`` inside a run dir, and
a tier directory that matches ``TIER_DIR_RE``).  The revision stores test
metrics under ``results/test/<cell>/`` and predictions in ``pred.npz``.

This script copies nothing: it writes a *staging* tree under
``experiment/.figure_staging/`` (git-ignored, delete after use) that the
plotter can read unchanged, and records where each staged file came from
in ``STAGING_MANIFEST.csv``.

Hard rules, mirroring D4_PLAN §0:
* per-epoch data is copied from ``val_metrics.csv`` / ``training_loss.csv``
  only (columns renamed toward the legacy names, never interpolated);
* test data is copied from ``results/test/<cell>/`` only;
* only C1 cells that passed the split-hash check are staged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from figure_source import (  # noqa: E402
    check_split,
    cell_for,
    load_basin_metrics,
    load_predictions,
    model_params,
    validation_files,
)
from paths import RESULTS_DIR  # noqa: E402

STAGING_ROOT = RESULTS_DIR.parent / ".figure_staging"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kge_components(obs: np.ndarray, sim: np.ndarray, gauge_ids: np.ndarray) -> pd.DataFrame:
    """Per-basin KGE components (Gupta 2009): r, beta, gamma.

    Computed from the audited test predictions, with the same definitions the
    manuscript uses: beta = mean(sim)/mean(obs), gamma = CV_sim/CV_obs.
    """
    rows = []
    for i, gauge in enumerate(gauge_ids):
        o = np.asarray(obs[i], dtype=float)
        s = np.asarray(sim[i], dtype=float)
        mask = np.isfinite(o) & np.isfinite(s)
        o, s = o[mask], s[mask]
        if o.size < 2 or o.std() == 0 or s.std() == 0:
            rows.append({"gauge_id": str(gauge), "r": np.nan, "beta": np.nan, "gamma": np.nan})
            continue
        r = float(np.corrcoef(o, s)[0, 1])
        beta = float(s.mean() / o.mean()) if o.mean() != 0 else np.nan
        gamma = float((s.std() / s.mean()) / (o.std() / o.mean())) if o.mean() and s.mean() else np.nan
        rows.append({"gauge_id": str(gauge), "r": r, "beta": beta, "gamma": gamma})
    return pd.DataFrame(rows)


def _legacy_exp_and_dir(cell: dict, params: dict) -> tuple[str, Path]:
    """Legacy (experiment, tier_dir) for one manifest cell."""
    if cell.get("model") == "mamba3" or "mamba3" in str(cell.get("recipe", "")):
        # Condensed Mamba pattern.
        exp = "Mamba3_standard"
        tier_dir = (
            f"Tier{cell['tier']}_Ep{params['epochs_requested']}"
            f"_Variant-standard_V3_head{params['headdim']}_H{params['hidden_size']}"
            f"_L{params['depth']}_dConv{params['d_conv']}"
            f"_Rho{params['rho']}_Seed{cell['seed']}"
        )
    else:
        exp = "LSTM"
        # Match the real output names: Tier1_200K, Tier2_500K, Tier3_1M.
        tier_name = {1: "200K", 2: "500K", 3: "1M"}.get(int(cell["tier"]), "1M")
        tier_dir = (
            f"Tier{cell['tier']}_{tier_name}_Ep{params['epochs_requested']}"
            f"_Bs{params['batch_size']}_Rho{params['rho']}_H{params['hidden_size']}"
            f"_L{params['depth']}_Seed{cell['seed']}"
        )
    return exp, Path(tier_dir)


def _infer_model(cell: dict) -> str:
    recipe = str(cell.get("recipe", ""))
    if "mamba3" in recipe:
        return "mamba3"
    if "lstm" in recipe:
        return "lstm"
    raise ValueError(f"cannot infer model from {cell}")


def stage_cell(
    recipe: str,
    tier: int,
    depth: int,
    seed: int,
    root: Path = STAGING_ROOT,
) -> dict:
    """Write one staged run tree and return its provenance row."""
    cell = cell_for(recipe, tier, depth, seed)
    check_split(cell)  # refuse a run trained on another split
    files = validation_files(cell)
    params = model_params(cell)

    exp, tier_dir = _legacy_exp_and_dir(cell, params)
    staged = root / exp / tier_dir / "All"
    staged.mkdir(parents=True, exist_ok=True)

    # 1. basin_metrics.csv from the audited test artefacts (never from
    #    the run dir).
    basin = load_basin_metrics(_infer_model(cell), tier, depth, seed)
    basin_path = staged / "basin_metrics.csv"
    basin.to_csv(basin_path, index=False)
    basin_sha = _sha256(basin_path)

    # 2. checkpoint_metrics.csv from val_metrics.csv + training_loss.csv
    val = pd.read_csv(files["val_metrics"])
    cp = val[["epoch", "train_loss", "val_median_nse", "val_median_kge"]].rename(
        columns={
            "train_loss": "training_loss",
            "val_median_nse": "median_nse",
            "val_median_kge": "median_kge",
        }
    )
    if files["training_loss"] is not None:
        loss = pd.read_csv(files["training_loss"])
        if {"epoch", "loss"}.issubset(loss.columns):
            cp = cp.drop(columns=["training_loss"], errors="ignore")
            cp = cp.merge(
                loss.rename(columns={"loss": "training_loss"})[["epoch", "training_loss"]],
                on="epoch",
                how="left",
            )
    cp_path = staged / "checkpoint_metrics.csv"
    cp.to_csv(cp_path, index=False)

    # 3. obs.npy / pred.npy from pred.npz (test only).
    pred = load_predictions(_infer_model(cell), tier, depth, seed)
    np.save(staged / "obs_daily_m3s.npy", pred["obs_m3s"].astype(np.float32))
    np.save(staged / "pred_daily_m3s.npy", pred["pred_m3s"].astype(np.float32))

    # 3b. kge_components.csv (r, beta, gamma) — the legacy figure 4 needs
    #     these three signatures and the revision never wrote them out.
    kge = _kge_components(pred["obs_m3s"], pred["pred_m3s"], pred["gauge_ids"])
    kge.to_csv(staged / "kge_components.csv", index=False)

    # 4. run.csv in the legacy 'Epoch N Loss X time Y' spelling.
    run_csv = staged / "run.csv"
    if "epoch_seconds" in val.columns:
        with run_csv.open("w", encoding="utf-8") as handle:
            for _, row in val.iterrows():
                handle.write(
                    f"Epoch {int(row['epoch'])} Loss {float(row['train_loss']):.6f} "
                    f"time {float(row['epoch_seconds']):.3f}\n"
                )

    return {
        "cell_id": cell["id"],
        "model": _infer_model(cell),
        "exp": exp,
        "tier_dir": str(tier_dir),
        "tier": tier,
        "depth": depth,
        "seed": seed,
        "params": params["n_parameters"],
        "hidden_size": params["hidden_size"],
        "run_dir": str(files["run_dir"]),
        "basin_metrics_sha256": basin_sha,
        "epochs": int(len(val)),
        "best_epoch": params["best_epoch"],
    }


HEADLINE = [
    # (recipe, tier, depth, seeds)
    ("recipes/lstm.json", 1, 1, [1, 2, 3, 4, 5]),
    ("recipes/mamba3.json", 1, 3, [1, 2, 3, 4, 5]),
    ("recipes/lstm.json", 2, 1, [1, 2]),
    ("recipes/lstm.json", 3, 1, [1, 2]),
    ("recipes/mamba3.json", 2, 3, [1, 2]),
    ("recipes/mamba3.json", 3, 3, [1, 2]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=str(STAGING_ROOT),
        help="staging tree root (git-ignored, recreated each run)",
    )
    parser.add_argument("--clear", action="store_true",
                        help="delete the staging tree before writing")
    args = parser.parse_args()
    root = Path(args.root)
    if args.clear and root.is_dir():
        import shutil

        shutil.rmtree(root)

    rows = []
    for recipe, tier, depth, seeds in HEADLINE:
        for seed in seeds:
            row = stage_cell(recipe, tier, depth, seed, root=root)
            rows.append(row)
            print(f"  staged {row['cell_id']} -> {row['exp']}/{row['tier_dir']}/All "
                  f"({row['params']:,} params, {row['epochs']} epochs)")

    manifest = pd.DataFrame(rows)
    out = root / "STAGING_MANIFEST.csv"
    manifest.to_csv(out, index=False)
    print(f"staged {len(rows)} cells -> {root} (manifest: {out})")
    print("run the plotter with --output-root pointing at the staging tree, "
          "e.g. --output-root experiment/.figure_staging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
