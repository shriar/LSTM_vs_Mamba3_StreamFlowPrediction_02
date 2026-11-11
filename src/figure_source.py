"""D4.1 -- resolve (recipe, tier, depth, seed) to its on-disk artefacts.

The figure pipeline must never guess a run directory.  This module answers
one question per cell: where do the validation curves, the test per-basin
metrics, and the test predictions live?  Two hard rules live here:

* per-epoch data comes from ``val_metrics.csv`` / ``training_loss.csv`` only;
* test data comes from ``results/test/<cell>/`` only.

Anything else is a ``KeyError`` raised by :func:`result_files` /
:func:`validation_files`, never a fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval_test import _read_json, _resolve  # noqa: E402
from paths import OUTPUT_ROOT, PROJECT_ROOT, RESULTS_DIR  # noqa: E402

MANIFEST = PROJECT_ROOT / "recipes" / "eval_manifest.json"
RESULTS_TEST = RESULTS_DIR / "test"
EXPECTED_SPLIT_SHA = "9efbe60e790965f203b037d7e630828f7fe36285d4c86ee5c3ddbd93ec13cc58"


def _manifest_cells() -> list[dict]:
    payload = _read_json(MANIFEST, "evaluation manifest")
    return list(payload.get("cells", []))


def cell_for(
    recipe: str,
    tier: int,
    depth: int,
    seed: int,
    cells: list[dict] | None = None,
) -> dict:
    """Find the manifest cell for a recipe/tier/depth/seed, or raise."""
    recipe_path = _resolve(recipe)
    for cell in cells if cells is not None else _manifest_cells():
        if (
            _resolve(str(cell["recipe"])) == recipe_path
            and int(cell["tier"]) == int(tier)
            and int(cell["depth"]) == int(depth)
            and int(cell["seed"]) == int(seed)
        ):
            return cell
    raise KeyError(
        f"no manifest cell for recipe={recipe} tier={tier} depth={depth} seed={seed}"
    )


def run_dir_for(cell: dict) -> Path:
    return _resolve(str(cell["run_dir"]))


def result_files(recipe_id: str, tier: int, depth: int, seed: int) -> dict:
    """Test artefacts for one cell.  Raises if the cell was never evaluated."""
    out = RESULTS_TEST / f"{recipe_id}_T{tier}_L{depth}_Seed{seed}"
    basin = out / "basin_test_metrics.csv"
    pred = out / "pred.npz"
    if not basin.is_file():
        raise FileNotFoundError(
            f"{out} has no basin_test_metrics.csv; the cell was not evaluated"
        )
    return {"basin_metrics": basin, "predictions": pred if pred.is_file() else None}


def validation_files(cell: dict) -> dict:
    """Per-epoch artefacts for one run.  Raises if the run is incomplete."""
    run = run_dir_for(cell)
    master_file = run / "master.json"
    if not master_file.is_file():
        raise FileNotFoundError(f"{run} has no master.json; the run is incomplete")
    master = _read_json(master_file, "run master")
    val = run / "val_metrics.csv"
    loss = run / "training_loss.csv"
    if not val.is_file():
        raise FileNotFoundError(f"{run} has no val_metrics.csv")
    return {
        "run_dir": run,
        "master": master,
        "val_metrics": val,
        "training_loss": loss if loss.is_file() else None,
    }


def check_split(cell: dict) -> str:
    """Return the split hash, refusing a run trained on another split."""
    master = _read_json(run_dir_for(cell) / "master.json", "run master")
    data = master.get("data") or {}
    sha = str(data.get("split_sha256", ""))
    if sha != EXPECTED_SPLIT_SHA:
        raise RuntimeError(
            f"run {cell['id']} was trained on split {sha!r}, expected "
            f"{EXPECTED_SPLIT_SHA}"
        )
    return sha


def load_basin_metrics(recipe_id: str, tier: int, depth: int, seed: int) -> pd.DataFrame:
    """One cell's per-basin test metrics, joined on ``gauge_id``."""
    path = result_files(recipe_id, tier, depth, seed)["basin_metrics"]
    return pd.read_csv(path)


def seed_averaged_basin_scores(
    recipe_id: str,
    tier: int,
    depth: int,
    seeds: list[int],
) -> pd.DataFrame:
    """Mean of per-basin metrics across seeds, joined on ``gauge_id``.

    Returns one row per basin: ``gauge_id`` plus every numeric metric
    averaged over seeds.  The join is by id, never by row position; a cell
    missing a basin drops that basin from the mean (NaN-aware).
    """
    per_seed: list[pd.DataFrame] = []
    for seed in seeds:
        frame = load_basin_metrics(recipe_id, tier, depth, seed)
        per_seed.append(
            frame.set_index("gauge_id").select_dtypes(include=[np.number])
        )
    # Stack and average per metric column, keeping the id index.
    names = per_seed[0].columns
    frames = [frame.reindex(columns=names) for frame in per_seed]
    combined = pd.concat(frames, axis=0)
    averaged = combined.groupby(level=0).mean(skipna=True)
    return averaged.reset_index().rename(columns={"index": "gauge_id"})


def load_predictions(recipe_id: str, tier: int, depth: int, seed: int) -> dict:
    """One cell's audited test predictions (pred/obs m3/s, ids, dates)."""
    path = result_files(recipe_id, tier, depth, seed)["predictions"]
    if path is None:
        raise FileNotFoundError(
            f"no pred.npz for {recipe_id} T{tier} L{depth} S{seed}"
        )
    with np.load(path) as data:
        return {
            "pred_m3s": np.asarray(data["pred_m3s"], dtype=float),
            "obs_m3s": np.asarray(data["obs_m3s"], dtype=float),
            "gauge_ids": np.asarray(data["gauge_ids"]).astype(str),
            "dates": np.asarray(data["dates"]).astype(str),
        }


def learning_curves(
    recipe: str, tier: int, depth: int, seeds: list[int]
) -> pd.DataFrame:
    """Median + IQR of per-epoch validation scores across seeds.

    Columns: ``epoch, median_nse, q25_nse, q75_nse, median_kge, q25_kge, q75_kge``.
    """
    frames = []
    for seed in seeds:
        cell = cell_for(recipe, tier, depth, seed)
        check_split(cell)
        files = validation_files(cell)
        frame = pd.read_csv(files["val_metrics"])
        frame = frame[["epoch", "val_median_nse", "val_median_kge"]].copy()
        frame["seed"] = seed
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    out = combined.groupby("epoch").agg(
        median_nse=("val_median_nse", "median"),
        q25_nse=("val_median_nse", lambda s: s.quantile(0.25)),
        q75_nse=("val_median_nse", lambda s: s.quantile(0.75)),
        median_kge=("val_median_kge", "median"),
        q25_kge=("val_median_kge", lambda s: s.quantile(0.25)),
        q75_kge=("val_median_kge", lambda s: s.quantile(0.75)),
        n_seeds=("seed", "count"),
    ).reset_index()
    return out


def best_epochs(recipe: str, tier: int, depth: int, seeds: list[int]) -> dict:
    """Validation-selected best epoch per seed, from ``master.json``."""
    out = {}
    for seed in seeds:
        cell = cell_for(recipe, tier, depth, seed)
        files = validation_files(cell)
        selection = (files["master"] or {}).get("selection") or {}
        out[seed] = int(selection.get("best_epoch", 0))
    return out


def model_params(cell: dict) -> dict:
    """Widths, parameters, state size and training settings for one run."""
    master = _read_json(run_dir_for(cell) / "master.json", "run master")
    model = master.get("model") or {}
    training = master.get("training") or {}
    return {
        "hidden_size": int(model.get("hidden_size", 0)),
        "param_tier": int(model.get("param_tier", 0)),
        "depth": int(model.get("depth", 0)),
        "n_parameters": int(model.get("total_params", 0)),
        "epochs_requested": int(training.get("epochs_requested", 0)),
        "batch_size": int(training.get("batch_size", 0)),
        "rho": int(training.get("rho", 0)),
        "train_buff": int(training.get("train_buff", 0)),
        "mamba_variant": model.get("mamba_variant"),
        "mamba_version": model.get("mamba_version"),
        "d_state": model.get("d_state"),
        "d_conv": model.get("d_conv"),
        "expand": model.get("expand"),
        "headdim": model.get("headdim"),
        "best_epoch": int((master.get("selection") or {}).get("best_epoch", 0)),
        "n_val_selections": int(
            (master.get("selection") or {}).get("n_val_selections", 0)
        ),
    }
