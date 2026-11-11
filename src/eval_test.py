"""The only entry point allowed to open the test slice (see B2d).

A test evaluation is allowed only when all four locks agree:

1. the ``(recipe, tier, depth, seed)`` cell is listed in
   ``recipes/eval_manifest.json``;
2. the recipe says ``locked: true``;
3. the recipe, manifest, run and ``data/split.json`` all carry the same
   ``split_sha256``;
4. the cell has not already been evaluated, unless ``--force`` is given.

The run directory is read-only here.  Test predictions and metrics are written
under ``results/test/`` only; this script never writes under ``output/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# When run as ``python src/eval_test.py`` sys.path[0] is already src/; keep the
# import path explicit for ``python -m`` and for the unit tests.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from constants import ATTR_LST, forcing_variables  # noqa: E402
from data_utils import (  # noqa: E402
    GAUGE_IDS_PATH,
    SPLIT_PATH,
    _split_sha256,
    load_split_with_context,
)
from inference import (  # noqa: E402
    CFS_TO_M3S,
    combined_stats,
    inverse_runoff,
    load_scaler,
    predict_array,
    scale_attrs,
    scale_series,
)
from paths import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "recipes" / "eval_manifest.json"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "test"
DEFAULT_AUDIT = DEFAULT_OUTPUT_DIR / "audit.jsonl"
CELL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
REQUIRED_CELL_KEYS = ("id", "recipe", "tier", "depth", "seed", "run_dir")


class Refusal(RuntimeError):
    """A protocol lock rejected the request; nothing was evaluated."""

    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = code


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    """Resolve a CLI path against cwd, the experiment root, then its parent."""
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), PROJECT_ROOT, PROJECT_ROOT.parent):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise Refusal(f"{label} not found: {path}", code=5)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Refusal(f"{label} is not valid JSON: {path}: {exc}", code=5) from exc
    if not isinstance(payload, dict):
        raise Refusal(f"{label} must contain a JSON object: {path}", code=5)
    return payload


def _read_split() -> dict:
    """Re-derive the split hash so a hand-edited split cannot be tested."""
    payload = _read_json(SPLIT_PATH, "split spec")
    spec = {
        key: payload[key]
        for key in ("protocol", "dataset", "preprocessed", "arrays",
                    "slices", "context", "basins")
    }
    if _split_sha256(spec) != payload.get("split_sha256"):
        raise Refusal(
            f"{SPLIT_PATH} does not match its recorded split_sha256; "
            "refusing to evaluate a hand-edited split",
            code=3,
        )
    return payload


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_manifest(path: Path) -> dict:
    manifest = _read_json(path, "evaluation manifest")
    if manifest.get("schema") != "eval_manifest/v1":
        raise Refusal(
            f"manifest schema must be 'eval_manifest/v1', got "
            f"{manifest.get('schema')!r}",
            code=5,
        )
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise Refusal("manifest 'cells' must be a list", code=5)
    seen: set[tuple[str, int, int, int]] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise Refusal(f"manifest cell {index} is not an object", code=5)
        missing = [key for key in REQUIRED_CELL_KEYS if key not in cell]
        if missing:
            raise Refusal(
                f"manifest cell {index} is missing {missing}", code=5
            )
        cell_id = str(cell["id"])
        if not CELL_ID_RE.match(cell_id):
            raise Refusal(f"manifest cell id is unsafe: {cell_id!r}", code=5)
        try:
            key = (
                _resolve(str(cell["recipe"])).as_posix(),
                int(cell["tier"]),
                int(cell["depth"]),
                int(cell["seed"]),
            )
        except (TypeError, ValueError) as exc:
            raise Refusal(
                f"manifest cell {cell_id} has non-integer tier/depth/seed",
                code=5,
            ) from exc
        if key in seen:
            raise Refusal(f"duplicate manifest cell {key}", code=5)
        seen.add(key)
    return manifest


def _validate_recipe(recipe: dict, path: Path) -> None:
    if recipe.get("schema") != "eval_recipe/v1":
        raise Refusal(
            f"recipe schema must be 'eval_recipe/v1', got "
            f"{recipe.get('schema')!r}",
            code=5,
        )
    recipe_id = str(recipe.get("id") or "")
    if not CELL_ID_RE.match(recipe_id):
        raise Refusal(f"recipe id is unsafe or missing: {recipe_id!r}", code=5)
    model = recipe.get("model")
    if model not in {"lstm", "mamba3"}:
        raise Refusal(f"recipe model must be 'lstm' or 'mamba3', got {model!r}", code=5)
    for key in ("tier", "depth", "seed", "epochs", "batch_size", "rho",
                "train_buff", "ctx_days"):
        if key not in recipe:
            raise Refusal(f"recipe {path} is missing {key!r}", code=5)
    tier, depth = int(recipe["tier"]), int(recipe["depth"])
    seed, epochs = int(recipe["seed"]), int(recipe["epochs"])
    batch, rho = int(recipe["batch_size"]), int(recipe["rho"])
    buff, ctx = int(recipe["train_buff"]), int(recipe["ctx_days"])
    if tier not in (1, 2, 3):
        raise Refusal(f"recipe tier must be 1..3, got {tier}", code=5)
    if depth not in (1, 2, 3):
        raise Refusal(f"recipe depth must be 1..3, got {depth}", code=5)
    if seed < 0 or epochs < 1 or batch < 1 or rho < 1 or buff < 0:
        raise Refusal("recipe has a non-positive epoch/batch/rho or bad seed", code=5)
    if ctx < rho + buff:
        raise Refusal(
            f"recipe ctx_days={ctx} does not cover rho+train_buff={rho + buff}",
            code=5,
        )
    if not isinstance(recipe.get("hyperparameters"), dict):
        raise Refusal("recipe 'hyperparameters' must be an object", code=5)
    if model == "mamba3":
        if recipe["hyperparameters"].get("mamba_variant") != "standard":
            raise Refusal("headline Mamba recipe must be variant='standard'", code=5)
        if int(recipe["hyperparameters"].get("mamba_version", 0)) != 3:
            raise Refusal("headline Mamba recipe must be version=3", code=5)
    if not recipe.get("run_dir"):
        raise Refusal("recipe is missing run_dir", code=5)
    if not recipe.get("checkpoint"):
        raise Refusal("recipe is missing checkpoint", code=5)
    if not recipe.get("split_sha256"):
        raise Refusal("recipe is missing split_sha256", code=5)


def _cell_matches(
    cell: dict, recipe_path: Path, tier: int, depth: int, seed: int
) -> bool:
    return (
        _resolve(str(cell["recipe"])) == recipe_path
        and int(cell["tier"]) == tier
        and int(cell["depth"]) == depth
        and int(cell["seed"]) == seed
    )


def _expected_training(recipe: dict) -> dict:
    hyper = recipe["hyperparameters"]
    model = recipe["model"]
    if model == "lstm":
        keys = {
            "lr": "lr",
            "min_lr": "lstm_min_lr",
            "weight_decay": "lstm_weight_decay",
            "grad_clip": "lstm_grad_clip",
            "scheduler": "lstm_scheduler",
            "optimizer": "lstm_optimizer",
        }
        betas = ("lstm_beta1", "lstm_beta2")
        warmup = None
    else:
        keys = {
            "lr": "mamba_lr",
            "min_lr": "mamba_min_lr",
            "weight_decay": "mamba_weight_decay",
            "grad_clip": "mamba_grad_clip",
            "scheduler": "mamba_scheduler",
            "optimizer": "mamba_optimizer",
        }
        betas = ("mamba_beta1", "mamba_beta2")
        warmup = "mamba_warmup_steps"
    missing = [name for name in (*keys.values(), *betas) if name not in hyper]
    if warmup and warmup not in hyper:
        missing.append(warmup)
    if "use_amp" not in hyper:
        missing.append("use_amp")
    if missing:
        raise Refusal(
            f"locked recipe is missing training keys {missing}; cannot prove "
            "it produced this run",
            code=5,
        )
    expected = {
        "epochs_requested": int(recipe["epochs"]),
        "batch_size": int(recipe["batch_size"]),
        "rho": int(recipe["rho"]),
        "train_buff": int(recipe["train_buff"]),
        "optimizer": str(hyper[keys["optimizer"]]).lower(),
        "scheduler": str(hyper[keys["scheduler"]]).lower(),
        "lr": float(hyper[keys["lr"]]),
        "min_lr": float(hyper[keys["min_lr"]]),
        "weight_decay": float(hyper[keys["weight_decay"]]),
        "betas": [float(hyper[betas[0]]), float(hyper[betas[1]])],
        "grad_clip": float(hyper[keys["grad_clip"]]),
        "use_amp": bool(hyper["use_amp"]),
    }
    if warmup:
        expected["warmup_steps"] = int(hyper[warmup])
    return expected


def _same_value(actual, expected) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(
                float(actual), expected, rel_tol=1e-9, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(_same_value(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, bool):
        return bool(actual) is expected
    return actual == expected


def _verify_master(
    master: dict,
    recipe: dict,
    *,
    tier: int,
    depth: int,
    seed: int,
    split_sha: str,
) -> None:
    if master.get("protocol") != "train_val/v1":
        raise Refusal(
            f"run master protocol must be 'train_val/v1', got "
            f"{master.get('protocol')!r}",
            code=5,
        )
    if master.get("test_split_opened") is not False:
        raise Refusal(
            "run master does not prove test_split_opened=false; refusing to "
            "trust a run that has already seen test data",
            code=5,
        )
    data = master.get("data") or {}
    if data.get("split_sha256") != split_sha:
        raise Refusal(
            "run was trained on a different split_sha256; "
            f"{data.get('split_sha256')} != {split_sha}",
            code=3,
        )
    if int(data.get("context_days", -1)) != int(recipe["ctx_days"]):
        raise Refusal(
            f"run context_days={data.get('context_days')} != recipe "
            f"{recipe['ctx_days']}",
            code=5,
        )
    determinism = master.get("determinism") or {}
    if int(determinism.get("seed", -1)) != int(seed):
        raise Refusal(
            f"run seed={determinism.get('seed')} != cell seed {seed}", code=5
        )
    model = master.get("model") or {}
    expected_type = "mamba3" if recipe["model"] == "mamba3" else "lstm"
    if model.get("type") != expected_type:
        raise Refusal(
            f"run model type={model.get('type')!r} != recipe {expected_type!r}",
            code=5,
        )
    if int(model.get("param_tier", -1)) != tier:
        raise Refusal(f"run tier != cell tier {tier}", code=5)
    if int(model.get("depth", -1)) != depth:
        raise Refusal(f"run depth != cell depth {depth}", code=5)
    if expected_type == "mamba3":
        if model.get("mamba_variant") != "standard":
            raise Refusal("run is not Mamba-3 'standard'", code=5)
        if int(model.get("mamba_version", -1)) != 3:
            raise Refusal("run is not Mamba-3 (version 3)", code=5)

    training = master.get("training") or {}
    for key, expected in _expected_training(recipe).items():
        if not _same_value(training.get(key), expected):
            raise Refusal(
                f"run training.{key}={training.get(key)!r} does not match "
                f"locked recipe value {expected!r}",
                code=5,
            )
    selection = master.get("selection") or {}
    if int(selection.get("n_val_selections", 0)) < 1:
        raise Refusal("run has no validation selections", code=5)
    if int(selection.get("best_epoch", 0)) < 1:
        raise Refusal("run has no best-on-validation checkpoint", code=5)


def _verify_checkpoint(
    model: torch.nn.Module,
    recipe: dict,
    master: dict,
    *,
    depth: int,
) -> None:
    expected_class = (master.get("model") or {}).get("class")
    actual_class = getattr(model, "name", type(model).__name__)
    if expected_class and actual_class != expected_class:
        raise Refusal(
            f"checkpoint class {actual_class!r} != run master class "
            f"{expected_class!r}",
            code=5,
        )
    expected_hidden = (master.get("model") or {}).get("hidden_size")
    if expected_hidden and int(getattr(model, "hiddenSize", -1)) != int(expected_hidden):
        raise Refusal(
            f"checkpoint hiddenSize={getattr(model, 'hiddenSize', None)} != "
            f"run master {expected_hidden}",
            code=5,
        )
    if recipe["model"] == "lstm":
        if int(getattr(model, "nLayer", -1)) != depth:
            raise Refusal("checkpoint LSTM depth does not match the cell", code=5)
    else:
        if int(getattr(model, "numLayers", -1)) != depth:
            raise Refusal("checkpoint Mamba depth does not match the cell", code=5)
        if int(getattr(model, "mambaVersion", -1)) != 3:
            raise Refusal("checkpoint is not Mamba-3", code=5)
    if int(getattr(model, "ny", -1)) != 1:
        raise Refusal("checkpoint output width is not 1", code=5)


def _verify_best_checkpoint(run_dir: Path, checkpoint: Path, master: dict) -> None:
    best_epoch = int((master.get("selection") or {}).get("best_epoch", 0))
    epoch_path = run_dir / f"model_Ep{best_epoch}.pt"
    if not epoch_path.is_file():
        raise Refusal(
            f"selected epoch checkpoint {epoch_path.name} is missing; cannot "
            "prove model_best.pt came from validation",
            code=5,
        )
    if _sha256(epoch_path) != _sha256(checkpoint):
        raise Refusal(
            f"{checkpoint.name} is not a byte copy of selected epoch "
            f"{epoch_path.name}; refusing a swapped checkpoint",
            code=5,
        )


def _summary(metrics: dict, n_days: int) -> dict:
    def stat(name: str, op: str) -> float:
        values = metrics.get(name)
        if values is None:
            return float("nan")
        array = np.asarray(values, dtype=float)
        with np.errstate(invalid="ignore"):
            return float(getattr(np, f"nan{op}")(array))

    def percentile(name: str, q: float) -> float:
        values = metrics.get(name)
        if values is None:
            return float("nan")
        with np.errstate(invalid="ignore"):
            return float(np.nanpercentile(np.asarray(values, float), q))

    return {
        "n_basins": int(len(next(iter(metrics.values())))),
        "n_days": int(n_days),
        "n_basins_scored": int(np.sum(~np.isnan(np.asarray(metrics["NSE"], float)))),
        "median_nse": stat("NSE", "median"),
        "mean_nse": stat("NSE", "mean"),
        "p10_nse": percentile("NSE", 10),
        "p90_nse": percentile("NSE", 90),
        "median_kge": stat("KGE", "median"),
        "mean_kge": stat("KGE", "mean"),
        "p10_kge": percentile("KGE", 10),
        "p90_kge": percentile("KGE", 90),
        "median_fhv": stat("FHV", "median"),
        "median_flv": stat("FLV", "median"),
        "median_pbias": stat("PBias", "median"),
        "median_rmse": stat("RMSE", "median"),
        "median_corr": stat("Corr", "median"),
    }


def _write_outputs(
    out_dir: Path,
    *,
    recipe: dict,
    recipe_path: Path,
    cell: dict,
    checkpoint: Path,
    split: dict,
    metrics: dict,
    pred_m3s: np.ndarray,
    obs_m3s: np.ndarray,
    gauge_ids: np.ndarray,
    dates: np.ndarray,
    recipe_sha: str,
    forced: bool,
) -> dict:
    if len(gauge_ids) != pred_m3s.shape[0]:
        raise Refusal(
            f"gauge id count {len(gauge_ids)} != basin count {pred_m3s.shape[0]}",
            code=5,
        )
    if len(dates) != pred_m3s.shape[1]:
        raise Refusal(
            f"date count {len(dates)} != test day count {pred_m3s.shape[1]}",
            code=5,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(metrics, pred_m3s.shape[1])
    row = {
        "cell_id": cell["id"],
        "recipe_id": recipe["id"],
        "recipe": _rel(recipe_path),
        "model": recipe["model"],
        # D5 audit fix (2026-11-10): take the identity from the manifest cell,
        # not the shared recipe.  A recipe serves every depth/seed of its
        # architecture, so recipe["depth"]/["seed"] mislabels all non-headline
        # cells (e.g. C1b-lstm-T2-L2-S1 was recorded as tier 1 depth 1 seed 1).
        "tier": int(cell["tier"]),
        "depth": int(cell["depth"]),
        "seed": int(cell["seed"]),
        "epochs": int(recipe["epochs"]),
        "checkpoint": _rel(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "recipe_sha256": recipe_sha,
        "split_sha256": split["split_sha256"],
        "test_start": split["slices"]["test"]["start"],
        "test_end": split["slices"]["test"]["end"],
        **summary,
        "evaluated_utc": _utc_now(),
        "git_sha": _git_sha(),
        "forced": bool(forced),
    }
    pd.DataFrame([row]).to_csv(out_dir / "test_metrics.csv", index=False)

    basin = {"gauge_id": gauge_ids}
    for name in sorted(metrics):
        basin[name] = metrics[name]
    pd.DataFrame(basin).to_csv(out_dir / "basin_test_metrics.csv", index=False)

    np.savez_compressed(
        out_dir / "pred.npz",
        pred_m3s=pred_m3s.astype(np.float32).squeeze(-1),
        obs_m3s=obs_m3s.astype(np.float32).squeeze(-1),
        gauge_ids=gauge_ids.astype(str),
        dates=dates.astype("datetime64[D]").astype(str),
    )
    return row


def _append_audit(
    audit_path: Path, row: dict, metrics_path: Path, prior_lines: int
) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "utc": row["evaluated_utc"],
        "git_sha": row["git_sha"],
        "cell_id": row["cell_id"],
        "recipe_id": row["recipe_id"],
        "recipe": row["recipe"],
        "recipe_sha256": row["recipe_sha256"],
        "tier": row["tier"],
        "depth": row["depth"],
        "seed": row["seed"],
        "split_sha256": row["split_sha256"],
        "checkpoint": row["checkpoint"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "test_metrics": _rel(metrics_path),
        "forced": row["forced"],
        "audit_line_number": int(prior_lines) + 1,
        "status": "ok",
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one pre-registered manifest cell on the test slice."
    )
    parser.add_argument("--recipe", required=True, help="locked eval recipe JSON")
    parser.add_argument("--tier", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument("--depth", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--manifest", default=str(MANIFEST_PATH), help="evaluation manifest JSON"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="results directory (never under output/)",
    )
    parser.add_argument(
        "--audit",
        default=str(DEFAULT_AUDIT),
        help="append-only audit JSONL (never under output/)",
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-evaluate a cell that already has test_metrics.csv",
    )
    return parser.parse_args(argv)


def _guard_results_path(path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(PROJECT_ROOT / "output")
    except ValueError:
        return
    raise Refusal(
        f"{label} may not be under output/: {path}; test artefacts belong in "
        "results/test/",
        code=5,
    )


def evaluate(args: argparse.Namespace) -> int:
    manifest_path = _resolve(args.manifest)
    manifest = _load_manifest(manifest_path)
    recipe_path = _resolve(args.recipe)
    recipe = _read_json(recipe_path, "recipe")
    _validate_recipe(recipe, recipe_path)

    tier = int(args.tier if args.tier is not None else recipe["tier"])
    depth = int(args.depth if args.depth is not None else recipe["depth"])
    seed = int(args.seed if args.seed is not None else recipe["seed"])
    matches = [
        cell
        for cell in manifest["cells"]
        if _cell_matches(cell, recipe_path, tier, depth, seed)
    ]
    if not matches:
        raise Refusal(
            f"cell recipe={_rel(recipe_path)} tier={tier} depth={depth} "
            f"seed={seed} is not listed in {_rel(manifest_path)}"
        )
    cell = matches[0]

    if recipe.get("locked") is not True:
        raise Refusal(
            f"recipe {_rel(recipe_path)} is not locked (locked="
            f"{recipe.get('locked')!r}); G2 must freeze it first"
        )

    split = _read_split()
    split_sha = split["split_sha256"]
    if manifest.get("split_sha256") not in (None, split_sha):
        raise Refusal(
            "manifest split_sha256 does not match data/split.json", code=3
        )
    if recipe["split_sha256"] != split_sha:
        raise Refusal(
            f"recipe split_sha256={recipe['split_sha256']} != data/split.json "
            f"{split_sha}",
            code=3,
        )

    # The manifest is the single source of truth for where a cell's run
    # lives.  A shared recipe -- one locked hyperparameter set serving many
    # seeds and depths (the C1a ladder) -- cannot carry nine run_dirs, so the
    # recipe's own run_dir is binding only for the cell the recipe was
    # written for.  Every other cell is proven to match the recipe by
    # _verify_master below: seed, tier, depth, model type/variant, split
    # sha256, test_split_opened=false and every training hyperparameter.
    run_dir = _resolve(str(cell["run_dir"]))
    recipe_run_dir = _resolve(str(recipe["run_dir"]))
    headline_cell = (
        tier == int(recipe["tier"])
        and depth == int(recipe["depth"])
        and seed == int(recipe["seed"])
    )
    if headline_cell and run_dir != recipe_run_dir:
        raise Refusal(
            f"recipe run_dir {_rel(recipe_run_dir)} != manifest cell run_dir "
            f"{_rel(run_dir)} for the recipe's own cell",
            code=5,
        )
    checkpoint = (run_dir / str(recipe["checkpoint"])).resolve()
    if checkpoint.parent != run_dir:
        raise Refusal("recipe checkpoint escapes its run_dir", code=5)
    if not checkpoint.is_file():
        raise Refusal(f"checkpoint not found: {_rel(checkpoint)}", code=5)
    master_path = run_dir / "master.json"
    master = _read_json(master_path, "run master")
    _verify_master(
        master, recipe, tier=tier, depth=depth, seed=seed, split_sha=split_sha
    )
    _verify_best_checkpoint(run_dir, checkpoint, master)

    out_dir = _resolve(args.output_dir)
    audit_path = _resolve(args.audit)
    _guard_results_path(out_dir, "output directory")
    _guard_results_path(audit_path, "audit file")
    out_dir = out_dir / f"{recipe['id']}_T{tier}_L{depth}_Seed{seed}"
    _guard_results_path(out_dir, "cell output directory")
    metrics_path = out_dir / "test_metrics.csv"
    if metrics_path.is_file() and not args.force:
        raise Refusal(
            f"cell has already been evaluated: {_rel(metrics_path)}; "
            "pass --force only for a documented re-run",
            code=4,
        )
    prior = 0
    if audit_path.is_file():
        prior = sum(1 for line in audit_path.read_text(encoding="utf-8").splitlines() if line)

    print(
        f"Cell {cell['id']} | {recipe['model']} T{tier} L{depth} seed {seed} | "
        f"{_rel(checkpoint)}"
    )
    print(f"Manifest: {_rel(manifest_path)} (audit lines before: {prior})")

    scaler = load_scaler()
    stats = combined_stats(scaler)
    forcing = forcing_variables("daymet")
    ctx_days = int(recipe["ctx_days"])
    test_x, test_y, test_c = load_split_with_context(
        "test", ctx_days=ctx_days, allow_test=True
    )
    if test_x.shape[1] != ctx_days + test_y.shape[1]:
        raise Refusal(
            f"test context misaligned: {test_x.shape[1]} != {ctx_days} + "
            f"{test_y.shape[1]}",
            code=5,
        )
    expected_days = int(split["slices"]["test"]["days"])
    if test_y.shape[1] != expected_days:
        raise Refusal(
            f"test slice has {test_y.shape[1]} days, split.json says "
            f"{expected_days}",
            code=5,
        )
    if test_x.shape[2] != len(forcing):
        raise Refusal(
            f"test pickle has {test_x.shape[2]} forcing channels, expected "
            f"{len(forcing)}",
            code=5,
        )

    area = test_c[:, ATTR_LST.index("area_gages2")][:, None]
    p_mean = test_c[:, ATTR_LST.index("p_mean")][:, None]
    test_x_n = scale_series(test_x, forcing, stats)
    test_c_n = scale_attrs(test_c, stats)
    obs_m3s = test_y * CFS_TO_M3S

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _verify_checkpoint(model, recipe, master, depth=depth)
    expected_nx = int(test_x.shape[2] + test_c.shape[1])
    if int(getattr(model, "nx", -1)) != expected_nx:
        raise Refusal(
            f"checkpoint nx={getattr(model, 'nx', None)} != "
            f"forcing+attributes {expected_nx}",
            code=5,
        )
    model.to(device)
    model.eval()
    print(f"Loaded {type(model).__name__} on {device}; opening test slice...")

    pred_norm = predict_array(
        model, test_x_n, test_c_n, max(1, int(args.batch_size))
    )[:, ctx_days:, :]
    pred_m3s = inverse_runoff(pred_norm, area, p_mean, stats)
    if pred_m3s.shape[1] != test_y.shape[1]:
        raise Refusal(
            f"prediction length {pred_m3s.shape[1]} != test length "
            f"{test_y.shape[1]}",
            code=5,
        )

    from hydroDL.post import stat  # local: the only metric implementation used

    metrics = stat.statError(
        pred_m3s.squeeze(-1), obs_m3s.squeeze(-1)
    )
    row = _write_outputs(
        out_dir,
        recipe=recipe,
        recipe_path=recipe_path,
        cell=cell,
        checkpoint=checkpoint,
        split=split,
        metrics=metrics,
        pred_m3s=pred_m3s,
        obs_m3s=obs_m3s,
        gauge_ids=np.asarray(
            pd.read_csv(GAUGE_IDS_PATH).sort_values("basin_index")["gauge_id"].astype(str)
        ),
        dates=np.asarray(
            pd.date_range(
                split["slices"]["test"]["start"],
                split["slices"]["test"]["end"],
                freq="D",
            )
        ),
        recipe_sha=_sha256(recipe_path),
        forced=bool(args.force),
    )
    _append_audit(audit_path, row, metrics_path, prior)
    print(
        f"Test median NSE {row['median_nse']:.4f} | KGE "
        f"{row['median_kge']:.4f} | {row['n_basins_scored']} basins | "
        f"{row['n_days']} days"
    )
    print(f"Wrote {_rel(out_dir)} and appended {_rel(audit_path)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return evaluate(args)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
