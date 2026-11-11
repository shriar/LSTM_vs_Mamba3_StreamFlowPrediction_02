"""B2c - the only training path: fit on train, score validation every epoch.

One run per invocation.  This module never imports ``eval_test.py`` and never
opens ``data/pickles/test.pkl``; ``data_utils.load_split("test")`` refuses to
return that slice unless the caller passes ``allow_test=True``, which only the
evaluation entry point may do.

Usage
-----
    python src/train_val.py --model lstm   --tier 1 --depth 1 --epochs 30 --seed 1
    python src/train_val.py --model mamba3 --tier 1 --depth 3 --epochs 30 --seed 1
    python src/train_val.py --model mamba3 --tier 1 --depth 3 --epochs 30 --seed 1 \
        --config recipes/custom.json --basins 150
    python src/train_val.py --report output/Mamba3_3L/Tier1_Ep30_..._Seed1

Selection rule (fixed before the first run): maximise ``val_median_nse``,
patience 10 epochs, ``min_delta = 0``.  The best-on-validation checkpoint is
copied to ``model_best.pt``; every epoch's checkpoint is kept as
``model_Ep{k}.pt``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from config import ExperimentConfig, RECORD_END, RECORD_START, SPLIT_WINDOWS, set_seed
from constants import ATTR_LST, forcing_variables
from data_utils import (
    GAUGE_IDS_PATH,
    MANIFEST_PATH,
    SPLIT_PATH,
    _split_sha256,
    load_gauge_ids,
    load_split,
    load_split_with_context,
)
from hydroDL.model.crit import NSELossBatch
from inference import (
    CFS_TO_M3S,
    combined_stats as _combined_stats,
    load_scaler as _load_scaler,
    scale_attrs as _scale_attrs,
    scale_series as _scale_series,
    score_slice,
    target_normalised as _target_normalised,
)
from training import train_model

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
PATIENCE = 10
MIN_DELTA = 0.0

# argv of the current programmatic invocation, when ``main`` was called with one
# instead of the process having been launched as a script.
_RAW_ARGV: "list[str] | None" = None

# Top-level recipe keys that are metadata, not ExperimentConfig fields.
RECIPE_META = {
    "schema",
    "id",
    "name",
    "notes",
    "source",
    "locked",
    "lock_date",
    "created_utc",
    "updated_utc",
    "run_dir",
    "checkpoint",
    "split_sha256",
    "model",
    "tier",
    "depth",
    "seed",
    "epochs",
    "batch_size",
    "rho",
    "train_buff",
    "ctx_days",
    "hyperparameters",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _config_field_names() -> set[str]:
    return {f.name for f in dataclass_fields(ExperimentConfig) if f.init}


def build_config(argv: list[str] | None = None) -> tuple[ExperimentConfig, dict]:
    """Resolve defaults < recipe < CLI into an ``ExperimentConfig``.

    Returns the config and a small dict describing the basin subset request.
    """
    return config_from_args(parse_args(argv))


def config_from_args(args: argparse.Namespace) -> tuple[ExperimentConfig, dict]:
    recipe = _read_recipe(args.config) if args.config else {}

    hyper = {k: v for k, v in recipe.items() if k in _config_field_names()}
    hyper.update(recipe.get("hyperparameters") or {})

    known = _config_field_names()
    unknown = sorted(set(hyper) - known)
    if unknown:
        raise SystemExit(
            f"recipe {args.config}: unknown hyperparameter(s) {unknown}. "
            f"Known fields: {sorted(known)}"
        )
    unknown_top = sorted(set(recipe) - known - RECIPE_META)
    if unknown_top:
        raise SystemExit(
            f"recipe {args.config}: unknown key(s) {unknown_top}. "
            "Put hyperparameters under 'hyperparameters' (or use a "
            "ExperimentConfig field name)."
        )

    kwargs = dict(hyper)

    # Recipe-level identity (checked against, not silently overriding, the CLI).
    recipe_model = recipe.get("model")
    model = args.model or recipe_model or "lstm"
    if recipe_model and args.model and recipe_model != args.model:
        raise SystemExit(
            f"--model {args.model} contradicts recipe model {recipe_model}"
        )
    if model not in {"lstm", "mamba3"}:
        raise SystemExit(f"model must be 'lstm' or 'mamba3', got {model!r}")

    expected_type = "mamba" if model == "mamba3" else "lstm"
    if "model_type" in kwargs and kwargs["model_type"] != expected_type:
        raise SystemExit(
            f"recipe model_type={kwargs['model_type']!r} contradicts "
            f"--model {model}"
        )

    tier = args.tier if args.tier is not None else recipe.get("tier", 1)
    depth = args.depth if args.depth is not None else recipe.get("depth")
    epochs = args.epochs if args.epochs is not None else recipe.get("epochs", 30)
    seed = args.seed if args.seed is not None else recipe.get("seed", 111111)

    if model == "lstm":
        kwargs["model_type"] = "lstm"
        kwargs["lstm_layers"] = int(depth if depth is not None else 1)
    else:
        kwargs["model_type"] = "mamba"
        kwargs["mamba_layers"] = int(depth if depth is not None else 3)
        # The required benchmark wiring is fixed; a recipe cannot change it.
        wanted_variant = kwargs.get("mamba_variant", "standard")
        wanted_version = kwargs.get("mamba_version", 3)
        if wanted_variant != "standard" or wanted_version != 3:
            raise SystemExit(
                "the required wiring is Mamba-3 'standard' "
                f"(got variant={wanted_variant!r}, version={wanted_version}). "
                "Optional wirings are out of scope for B2c."
            )
        kwargs["mamba_version"] = 3
        kwargs["mamba_variant"] = "standard"

    kwargs["param_tier"] = int(tier)
    kwargs["epoch"] = int(epochs)
    kwargs["seed"] = int(seed)

    if args.batch_size is not None:
        kwargs["batch_size"] = int(args.batch_size)
    if args.rho is not None:
        kwargs["rho"] = int(args.rho)
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir
    elif args.basins is not None and args.basins < 671:
        # A subset must never overwrite the full-basin run of the same recipe.
        kwargs["output_dir"] = "output/subsets"

    cfg = ExperimentConfig(**kwargs)
    if args.hidden_size is not None:
        cfg.hidden_size = int(args.hidden_size)
    validate_wiring(cfg)
    return cfg, {"n_basins": args.basins, "config_path": args.config}


def validate_wiring(cfg: ExperimentConfig) -> None:
    """Refuse to train anything that is not the required headline wiring."""
    if cfg.model_type == "mamba":
        if cfg.mamba_version != 3:
            raise SystemExit(
                "B2c trains Mamba-3 only (mamba_version=3); "
                f"got {cfg.mamba_version}"
            )
        if cfg.mamba_variant != "standard":
            raise SystemExit(
                "B2c trains the 'standard' wiring only (40-dim concatenation); "
                f"got {cfg.mamba_variant!r}. The optional wiring track (O2) "
                "needs its own recipe and is out of scope here."
            )
    if cfg.rho < 1:
        raise SystemExit(f"rho must be >= 1, got {cfg.rho}")
    if cfg.batch_size < 1:
        raise SystemExit(f"batch_size must be >= 1, got {cfg.batch_size}")
    if cfg.ctx_days < cfg.rho + cfg.train_buff:
        # Context must cover one lookback window plus the loss warm-up.
        cfg.ctx_days = cfg.rho + cfg.train_buff


def _read_recipe(path: str | os.PathLike) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"recipe not found: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"recipe {p} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"recipe {p} must contain a JSON object")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one LSTM/Mamba-3 run on train, score validation."
    )
    parser.add_argument("--model", choices=("lstm", "mamba3"), default=None)
    parser.add_argument("--tier", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        metavar="H",
        help="override the tier width ladder (side experiments only)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default=None, help="JSON recipe of overrides")
    parser.add_argument(
        "--basins",
        type=int,
        default=None,
        metavar="N",
        help="deterministic aridity-stratified subset (default: all 671)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rho", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing run in the same folder",
    )
    parser.add_argument(
        "--report",
        metavar="RUN_DIR",
        default=None,
        help="print a finished run's val_metrics.csv; performs no writes",
    )
    args = parser.parse_args(argv)

    if args.report is not None:
        return args
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.depth is not None and args.depth < 1:
        parser.error("--depth must be >= 1")
    if args.hidden_size is not None and args.hidden_size < 1:
        parser.error("--hidden-size must be >= 1")
    if args.basins is not None and not 2 <= args.basins <= 671:
        parser.error("--basins must be between 2 and 671")
    return args


# ---------------------------------------------------------------------------
# Data: train + validation only, scaled with the frozen train-only scaler
# ---------------------------------------------------------------------------


def _stratified_rows(c: np.ndarray, n: int) -> np.ndarray:
    """Deterministic aridity-stratified basin subset (median of n rank bins)."""
    values = c[:, ATTR_LST.index("aridity")].astype(float)
    order = np.argsort(values, kind="stable")
    chunks = np.array_split(order, n)
    rows = sorted(int(chunk[len(chunk) // 2]) for chunk in chunks if len(chunk))
    if len(rows) != n:
        raise RuntimeError(f"stratified subset produced {len(rows)} rows, wanted {n}")
    return np.asarray(rows, dtype=int)


def load_datasets(cfg: ExperimentConfig, n_basins: int | None = None) -> dict:
    """Load the train and validation slices, scaled by the frozen scaler.

    The test pickle is never requested: ``load_split`` is called with "train"
    and ``load_split_with_context`` with "val", whose context is the train tail.
    """
    scaler = _load_scaler()
    stats = _combined_stats(scaler)
    forcing = forcing_variables(cfg.for_type)
    ctx_days = max(cfg.ctx_days, cfg.rho + cfg.train_buff)

    train_x, train_y, train_c = load_split("train")
    val_x, val_y, val_c = load_split_with_context("val", ctx_days=ctx_days)

    if train_x.shape[2] != len(forcing):
        raise SystemExit(
            f"train pickle has {train_x.shape[2]} channels, expected {len(forcing)}"
        )
    if val_x.shape[1] != ctx_days + val_y.shape[1]:
        raise SystemExit(
            "validation context does not line up with the scored slice: "
            f"{val_x.shape[1]} != {ctx_days} + {val_y.shape[1]}"
        )

    rows = np.arange(train_x.shape[0])
    subset = {"n": int(train_x.shape[0]), "method": "all"}
    gauge_ids = load_gauge_ids(GAUGE_IDS_PATH, n_basins=int(train_x.shape[0]))
    if n_basins is not None and n_basins < train_x.shape[0]:
        rows = _stratified_rows(train_c, n_basins)
        subset = {
            "n": int(len(rows)),
            "method": "aridity-stratified: sort by aridity, n equal rank bins, median of each",
            "rows": rows.tolist(),
        }
        train_x, train_y, train_c = train_x[rows], train_y[rows], train_c[rows]
        val_x, val_y, val_c = val_x[rows], val_y[rows], val_c[rows]
        gauge_ids = gauge_ids[rows]

    area = train_c[:, ATTR_LST.index("area_gages2")][:, None]
    p_mean = train_c[:, ATTR_LST.index("p_mean")][:, None]

    train_x_n = _scale_series(train_x, forcing, stats)
    val_x_n = _scale_series(val_x, forcing, stats)
    train_c_n = _scale_attrs(train_c, stats)
    val_c_n = _scale_attrs(val_c, stats)
    train_y_n = _target_normalised(train_y, area, p_mean, stats)
    val_y_n = _target_normalised(val_y, area, p_mean, stats)

    return {
        "train_x": train_x_n,
        "train_y": train_y_n,
        "train_c": train_c_n,
        "val_x": val_x_n,
        "val_y": val_y_n,
        "val_c": val_c_n,
        "train_obs_m3s": train_y * CFS_TO_M3S,
        "val_obs_m3s": val_y * CFS_TO_M3S,
        "area": area,
        "p_mean": p_mean,
        "std_array": np.nanstd(train_y_n, axis=1),
        "stats": stats,
        "scaler": scaler,
        "ctx_days": ctx_days,
        "subset": subset,
        "gauge_ids": gauge_ids,
        "shapes": {
            "train_x": list(train_x_n.shape),
            "train_y": list(train_y_n.shape),
            "val_x_with_context": list(val_x_n.shape),
            "val_y": list(val_y_n.shape),
        },
    }


# ---------------------------------------------------------------------------
# Model factory - headline wiring only
# ---------------------------------------------------------------------------


def build_model(cfg: ExperimentConfig, nx: int, ny: int) -> torch.nn.Module:
    if cfg.model_type == "lstm":
        if torch.cuda.is_available():
            from CudnnLstmModel import CudnnLstmModel

            model = CudnnLstmModel(
                nx=nx,
                ny=ny,
                hiddenSize=cfg.hidden_size,
                nLayer=cfg.lstm_layers,
                dr=cfg.lstm_dropout,
            )
        else:
            from hydroDL.model.rnn import CpuLstmModel

            if cfg.lstm_layers > 1:
                raise SystemExit(
                    "multi-layer LSTM experiments require CUDA; "
                    "HydroDL's CPU LSTM path is single-layer only"
                )
            model = CpuLstmModel(
                nx=nx, ny=ny, hiddenSize=cfg.hidden_size
            )
    else:
        from mamba_model import MambaStreamflowModel

        n_dynamic = len(cfg.varF)
        n_static = len(cfg.attrLst)
        if n_dynamic + n_static != nx:
            raise SystemExit(
                f"standard wiring expects {n_dynamic}+{n_static} inputs, got {nx}"
            )
        model = MambaStreamflowModel(
            nx=nx,
            ny=ny,
            hiddenSize=cfg.hidden_size,
            numLayers=cfg.mamba_layers,
            dState=cfg.mamba_d_state,
            dConv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            headdim=cfg.mamba_headdim,
            mambaVersion=cfg.mamba_version,
            dropout=cfg.mamba_dropout,
        )
    return model.to(cfg.device)


# ---------------------------------------------------------------------------
# Inference and scoring (in memory - no prediction CSVs)
# ---------------------------------------------------------------------------


def score(model: torch.nn.Module, x: np.ndarray, c: np.ndarray,
          y_norm: np.ndarray, obs_m3s: np.ndarray, drop: int,
          cfg: ExperimentConfig, data: dict, label_drop: int = 0,
          keep_metrics: bool = False) -> dict:
    """Forward + inverse-transform + metrics on one slice.

    ``drop`` removes the warm-up predictions.  ``label_drop`` does the same for
    the labels: the train slice carries its own warm-up (365 d), while a
    validation slice is stored already scored-length (no labels to drop).
    """
    result = score_slice(
        model,
        x,
        c,
        y_norm,
        obs_m3s,
        drop=drop,
        label_drop=label_drop,
        area=data["area"],
        p_mean=data["p_mean"],
        stats=data["stats"],
        std_array=data["std_array"],
        batch_size=cfg.valid_batch,
    )
    # Training only needs the scalars; do not pin validation predictions in
    # memory for the whole epoch loop.
    result.pop("pred_m3s")
    result.pop("obs_m3s")
    if not keep_metrics:
        result.pop("metrics")
    return result


def _iqr(values) -> float:
    """Inter-quartile range of the finite entries, or NaN if too few."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    q1, q3 = np.percentile(arr, [25.0, 75.0])
    return float(q3 - q1)


def _replace_with_retry(source: Path, destination: Path,
                        attempts: int = 5, pause: float = 0.15) -> None:
    """``os.replace`` with a short retry.

    On a Windows-backed filesystem (WSL ``/mnt/c``, ``/mnt/d``) an indexer or
    antivirus scan can hold the destination open for a few tens of
    milliseconds, which surfaces as ``PermissionError`` from an otherwise
    correct atomic replace.  Retrying is enough; failing every attempt is still
    an error.
    """
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(pause)


def _write_val_basin_metrics(
    out: Path, metrics: dict, gauge_ids: np.ndarray
) -> None:
    """Write the per-basin validation scores of the selected epoch.

    Called once, after training, so the file always describes the checkpoint
    that ``model_best.pt`` points at.  Written to a temporary name and renamed,
    so a crash cannot leave a half file behind.
    """
    nse = np.asarray(metrics.get("NSE"), dtype=float).ravel()
    kge = np.asarray(metrics.get("KGE"), dtype=float).ravel()
    if nse.size != len(gauge_ids) or kge.size != len(gauge_ids):
        raise RuntimeError(
            f"basin metric length {nse.size}/{kge.size} != "
            f"gauge id count {len(gauge_ids)}"
        )
    tmp = out / "val_basin_metrics.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gauge_id", "NSE", "KGE"])
        for gid, a, b in zip(gauge_ids, nse, kge):
            writer.writerow([gid, f"{a:.6f}", f"{b:.6f}"])
    _replace_with_retry(tmp, out / "val_basin_metrics.csv")


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------


class BestOnValidation:
    """Best ``val_median_nse`` with patience and ``min_delta`` fixed in code."""

    def __init__(self, patience: int = PATIENCE, min_delta: float = MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_nse = -math.inf
        self.best_epoch = 0
        self.best_kge = float("nan")
        self.best_loss = float("nan")
        self.last_nse = float("nan")
        self.n_selections = 0
        self._waiting = 0

    def update(self, epoch: int, nse: float, kge: float, loss: float) -> bool:
        """Record one epoch; return True if it improved the best score."""
        self.n_selections += 1
        self.last_nse = nse
        finite = not math.isnan(nse)
        improved = finite and nse > self.best_nse + self.min_delta
        if improved:
            self.best_nse = nse
            self.best_epoch = epoch
            self.best_kge = kge
            self.best_loss = loss
            self._waiting = 0
        else:
            self._waiting += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self._waiting >= self.patience

    @property
    def best_minus_last(self) -> float:
        if math.isnan(self.last_nse) or self.best_epoch == 0:
            return float("nan")
        return self.best_nse - self.last_nse


# ---------------------------------------------------------------------------
# Training recipe
# ---------------------------------------------------------------------------


def _optimizer_kwargs(cfg: ExperimentConfig) -> dict:
    if cfg.model_type == "mamba":
        return {
            "lr": cfg.mamba_lr,
            "max_lr": cfg.mamba_lr,
            "min_lr": cfg.mamba_min_lr,
            "scheduler_type": cfg.mamba_scheduler,
            "weight_decay": cfg.mamba_weight_decay,
            "optimizer_type": cfg.mamba_optimizer,
            "betas": (cfg.mamba_beta1, cfg.mamba_beta2),
            "warmup_steps": cfg.mamba_warmup_steps,
            "warmup_epochs": 0,
            "grad_clip": cfg.mamba_grad_clip,
            "no_decay_params": cfg.mamba_no_decay_params,
        }
    cosine = cfg.lstm_scheduler == "cosine"
    return {
        "lr": cfg.lr,
        "max_lr": cfg.lr,
        "min_lr": cfg.lstm_min_lr if cosine else None,
        "scheduler_type": cfg.lstm_scheduler,
        "weight_decay": cfg.lstm_weight_decay,
        "optimizer_type": cfg.lstm_optimizer,
        "betas": (cfg.lstm_beta1, cfg.lstm_beta2),
        "warmup_steps": 0,
        "warmup_epochs": 0,
        "grad_clip": cfg.lstm_grad_clip,
        "no_decay_params": None,
    }


# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_info() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _repo_root() -> Path:
    try:
        return Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except Exception:
        return PROJECT_ROOT


def _parse_status(text: str) -> list[str]:
    """Repository-relative paths from ``git status --porcelain`` output.

    The command is always run at the repository root, so every entry is already
    root-relative; this only normalises separators, renames and quoting.
    """
    paths: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        entry = line[3:].strip()
        if " -> " in entry:  # rename: keep the destination
            entry = entry.split(" -> ", 1)[1]
        if len(entry) >= 2 and entry[0] == entry[-1] == '"':
            entry = entry[1:-1]
        paths.append(Path(entry).as_posix())
    return sorted(paths)


def _dirty_paths() -> list[str]:
    """Repository-relative names of the files that make the tree dirty.

    Recorded so a ``dirty: true`` run can be shown to be dirty only outside the
    training path, instead of leaving a reviewer to guess what changed.
    """
    root = _repo_root()
    try:
        status = subprocess.check_output(
            ["git", "-c", "core.quotepath=off", "status", "--porcelain"],
            cwd=root, text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return _parse_status(status)


def _relative_command() -> str:
    """The invocation with paths made repository-relative.

    The interpreter is written as ``python`` and nothing outside the project is
    quoted, so a published ``master.json`` does not advertise a local home
    directory.  ``main`` records the argv it was given, so a programmatic call
    is described faithfully instead of quoting the test runner's arguments.
    """
    if _RAW_ARGV is not None:
        arguments = [str(Path(__file__).resolve()), *_RAW_ARGV]
    else:
        arguments = list(sys.argv)
    tokens = ["python"]
    for token in arguments:
        candidate = Path(token)
        if not candidate.is_absolute():
            tokens.append(token)
            continue
        try:
            tokens.append(candidate.resolve().relative_to(PROJECT_ROOT).as_posix())
        except Exception:
            tokens.append(f"<external>/{candidate.name}")
    return " ".join(tokens)


def _resolved_config(cfg: ExperimentConfig, n_basins: int) -> dict:
    """Every resolved tunable, in a form that JSON can round-trip."""
    payload: dict = {}
    for field in dataclass_fields(cfg):
        if not field.init:
            continue
        value = getattr(cfg, field.name)
        if isinstance(value, (bool, int, float, str)) or value is None:
            payload[field.name] = value
        elif isinstance(value, (list, tuple)):
            payload[field.name] = list(value)
        else:
            payload[field.name] = str(value)
    payload["basins"] = int(n_basins)
    return payload


def _sha256_json(payload) -> str:
    """Hash of the canonical JSON form, so the record can be re-verified."""
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2
        )
    for module in ("pandas", "mamba_ssm", "causal_conv1d"):
        try:
            info[module] = __import__(module).__version__
        except Exception:
            info[module] = None
    return info


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_split() -> dict:
    if not SPLIT_PATH.is_file():
        raise SystemExit(
            f"{SPLIT_PATH} is missing; run first: python src/data_utils.py --split"
        )
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    spec = {
        key: payload[key]
        for key in ("protocol", "dataset", "preprocessed", "arrays",
                    "slices", "context", "basins")
    }
    if _split_sha256(spec) != payload.get("split_sha256"):
        raise SystemExit(
            f"{SPLIT_PATH} does not match its recorded split_sha256; "
            "refusing to train on a hand-edited split"
        )
    return payload


def _clear_run_dir(out: Path) -> None:
    for name in ("master.json", "model_best.pt", "training_loss.csv",
                 "run.csv", "val_metrics.csv"):
        (out / name).unlink(missing_ok=True)
    for old in out.glob("model_Ep*.pt"):
        old.unlink(missing_ok=True)


def _model_identity(cfg: ExperimentConfig, model: torch.nn.Module,
                    total_params: int) -> dict:
    payload = {
        "type": "mamba3" if cfg.model_type == "mamba" else "lstm",
        "class": getattr(model, "name", type(model).__name__),
        "param_tier": cfg.param_tier,
        "tier_name": cfg.tier_name,
        "hidden_size": cfg.hidden_size,
        "depth": cfg.mamba_layers if cfg.model_type == "mamba" else cfg.lstm_layers,
        "total_params": int(total_params),
    }
    if cfg.model_type == "mamba":
        payload.update(
            mamba_version=cfg.mamba_version,
            mamba_variant=cfg.mamba_variant,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            headdim=cfg.mamba_headdim,
            dropout=cfg.mamba_dropout,
        )
    else:
        payload.update(dropout=cfg.lstm_dropout)
    return payload


def _training_identity(cfg: ExperimentConfig, optim: dict) -> dict:
    return {
        "epochs_requested": cfg.epoch,
        "batch_size": cfg.batch_size,
        "rho": cfg.rho,
        "train_buff": cfg.train_buff,
        "optimizer": optim["optimizer_type"],
        "scheduler": optim["scheduler_type"],
        "lr": optim["lr"],
        "max_lr": optim["max_lr"],
        "min_lr": optim["min_lr"],
        "weight_decay": optim["weight_decay"],
        "betas": list(optim["betas"]),
        "warmup_steps": optim["warmup_steps"],
        "grad_clip": optim["grad_clip"],
        "loss": "NSELossBatch",
        "use_amp": bool(cfg.use_amp),
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Path:
    cfg, request = build_config_from_args(args)
    set_seed(cfg)

    split = _read_split()
    split_sha = split["split_sha256"]

    out = Path(cfg.output_path())
    existing = out / "master.json"
    if existing.is_file() and not args.force:
        raise SystemExit(
            f"run already exists at {out}\n"
            "pass --force to overwrite it (this deletes its checkpoints)"
        )
    removed = cfg.remove_other_epoch_runs()
    if removed:
        print("Removed previous epoch output(s): "
              + ", ".join(p.name for p in removed))
    out.mkdir(parents=True, exist_ok=True)
    if args.force:
        _clear_run_dir(out)

    print(
        f"\n[{cfg.model_type.upper()}] Tier {cfg.param_tier} ({cfg.tier_name}) "
        f"| H={cfg.hidden_size} | depth="
        f"{cfg.mamba_layers if cfg.model_type == 'mamba' else cfg.lstm_layers} "
        f"| epochs={cfg.epoch} | seed={cfg.seed} | rho={cfg.rho}"
    )
    print(f"Run directory: {out}")

    data = load_datasets(cfg, request["n_basins"])
    print(
        f"Data: train {data['shapes']['train_x']} + {data['shapes']['train_y'][2]} target "
        f"| val {data['shapes']['val_x_with_context']} -> "
        f"{data['shapes']['val_y'][1]} scored days "
        f"({data['subset']['n']} basins, context {data['ctx_days']} d)"
    )

    nx = data["train_x"].shape[-1] + data["train_c"].shape[-1]
    ny = data["train_y"].shape[-1]
    model = build_model(cfg, nx, ny)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    loss_fun = NSELossBatch(data["std_array"], device=cfg.device)
    optim = _optimizer_kwargs(cfg)

    selector = BestOnValidation()
    history: list[dict] = []
    best_iqr = float("nan")
    best_basin: dict = {}
    csv_path = out / "val_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "train_median_nse",
            "val_median_nse", "val_median_kge", "epoch_seconds",
        ])
        handle.flush()

        def on_epoch_end(*, epoch: int, mean_loss: float,
                         epoch_time: float, model: torch.nn.Module):
            nonlocal best_basin
            train_metrics = score(
                model, data["train_x"], data["train_c"], data["train_y"],
                data["train_obs_m3s"], cfg.train_buff, cfg, data,
                label_drop=cfg.train_buff,
            )
            val_metrics = score(
                model, data["val_x"], data["val_c"], data["val_y"],
                data["val_obs_m3s"], data["ctx_days"], cfg, data,
                keep_metrics=True,
            )
            improved = selector.update(
                epoch, val_metrics["median_nse"], val_metrics["median_kge"],
                val_metrics["loss"],
            )
            if improved:
                # Keep the arrays in memory; the file is written once after
                # training, so a Windows file lock cannot kill a run mid-epoch.
                best_basin = {
                    "NSE": np.array(val_metrics["metrics"].get("NSE"), dtype=float),
                    "KGE": np.array(val_metrics["metrics"].get("KGE"), dtype=float),
                }
            val_metrics.pop("metrics", None)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_median_nse": train_metrics["median_nse"],
                "val_median_nse": val_metrics["median_nse"],
                "val_median_kge": val_metrics["median_kge"],
                "epoch_seconds": epoch_time,
            }
            history.append(row)
            writer.writerow([
                epoch, f"{row['train_loss']:.8f}", f"{row['val_loss']:.8f}",
                f"{row['train_median_nse']:.6f}",
                f"{row['val_median_nse']:.6f}",
                f"{row['val_median_kge']:.6f}", f"{epoch_time:.2f}",
            ])
            handle.flush()
            print(
                f"  epoch {epoch:3d} | loss {mean_loss:.5f} | "
                f"train NSE {row['train_median_nse']:.4f} | "
                f"val NSE {row['val_median_nse']:.4f} | "
                f"val KGE {row['val_median_kge']:.4f} | {epoch_time:.0f}s"
            )
            if selector.should_stop:
                print(
                    f"  early stop: no val improvement for "
                    f"{selector.patience} epochs (best epoch "
                    f"{selector.best_epoch})"
                )
                return "stop"
            return None

        train_model(
            model,
            data["train_x"], data["train_y"], data["train_c"],
            loss_fun,
            n_epoch=cfg.epoch,
            mini_batch=[cfg.batch_size, cfg.rho],
            save_epoch=1,
            save_folder=str(out),
            bufftime=cfg.train_buff,
            lr=optim["lr"],
            max_lr=optim["max_lr"],
            scheduler_type=optim["scheduler_type"],
            min_lr=optim["min_lr"],
            weight_decay=optim["weight_decay"],
            optimizer_type=optim["optimizer_type"],
            betas=optim["betas"],
            warmup_epochs=optim["warmup_epochs"],
            warmup_steps=optim["warmup_steps"],
            grad_clip=optim["grad_clip"],
            no_decay_params=optim["no_decay_params"],
            use_amp=cfg.use_amp,
            epoch_callback=on_epoch_end,
        )

    best_path = out / f"model_Ep{selector.best_epoch}.pt"
    if selector.best_epoch and best_path.is_file():
        shutil.copyfile(best_path, out / "model_best.pt")
        print(f"Best-on-validation checkpoint: {best_path.name} -> model_best.pt")
        if best_basin:
            best_iqr = _iqr(best_basin["NSE"])
            _write_val_basin_metrics(out, best_basin, data["gauge_ids"])
    else:
        print("[WARNING] no finite validation score; model_best.pt not written")

    git = _git_info()
    config_spec = _resolved_config(cfg, data["subset"]["n"])
    recipe_path, recipe_sha = None, None
    if request.get("config_path"):
        candidate = Path(request["config_path"])
        try:
            recipe_path = candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
        except Exception:
            recipe_path = candidate.name
        if candidate.is_file():
            recipe_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    manifest_sha = (
        hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
        if MANIFEST_PATH.is_file() else None
    )
    provenance = OrderedDict(
        git_sha=git["commit"],
        git_branch=git["branch"],
        git_dirty=git["dirty"],
        git_dirty_paths=_dirty_paths() if git["dirty"] else [],
        seed=cfg.seed,
        split_dates={key: list(value) for key, value in SPLIT_WINDOWS.items()},
        split_sha256=split_sha,
        data_manifest_sha256=manifest_sha,
        recipe={"path": recipe_path, "sha256": recipe_sha},
        config=config_spec,
        config_sha256=_sha256_json(config_spec),
    )

    payload = OrderedDict(
        protocol="train_val/v1",
        created_utc=_utc_now(),
        command=_relative_command(),
        git=git,
        model=_model_identity(cfg, model, total_params),
        training=_training_identity(cfg, optim),
        data={
            "split_sha256": split_sha,
            "record": {"start": RECORD_START, "end": RECORD_END},
            "train": dict(split["slices"]["train"], dates=SPLIT_WINDOWS["train"]),
            "val": dict(split["slices"]["val"], dates=SPLIT_WINDOWS["val"]),
            "context_days": data["ctx_days"],
            "basin_subset": data["subset"],
            "scaler_fitted_on": data["scaler"].get("fitted_on"),
            "shapes": data["shapes"],
        },
        selection={
            "rule": "max val_median_nse",
            "patience": selector.patience,
            "min_delta": selector.min_delta,
            "n_val_selections": selector.n_selections,
            "epochs_completed": len(history),
            "early_stopped": len(history) < cfg.epoch,
            "best_epoch": selector.best_epoch,
            "best_val_median_nse": selector.best_nse if selector.best_epoch else None,
            "best_val_median_kge": selector.best_kge if selector.best_epoch else None,
            "best_val_loss": selector.best_loss if selector.best_epoch else None,
            "val_median_nse_last": selector.last_nse,
            "best_minus_last": selector.best_minus_last,
            "best_val_nse_iqr": best_iqr,
        },
        determinism={
            "seed": cfg.seed,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "bitwise_reproducible": cfg.model_type == "lstm",
            "mamba_bitwise_reproducible": False,
            "note": (
                "CuDNN LSTM is bitwise reproducible on one machine with this seed; "
                "Mamba-3 is not (Triton atomics), so multi-seed CIs are required."
            ),
        },
        environment=_environment_info(),
        provenance=provenance,
        artifacts={
            "checkpoints": [f"model_Ep{row['epoch']}.pt" for row in history],
            "best_checkpoint": "model_best.pt" if selector.best_epoch else None,
            "training_loss_csv": "training_loss.csv",
            "val_metrics_csv": "val_metrics.csv",
            "val_basin_metrics_csv": (
                "val_basin_metrics.csv" if selector.best_epoch else None
            ),
        },
        test_split_opened=False,
    )
    _write_json(out / "master.json", payload)

    print(
        f"\nFinished {len(history)}/{cfg.epoch} epochs. "
        f"Best val NSE {selector.best_nse:.4f} at epoch {selector.best_epoch} "
        f"(last {selector.last_nse:.4f})."
    )
    return out


def build_config_from_args(args: argparse.Namespace) -> tuple[ExperimentConfig, dict]:
    """Config resolution for an already-parsed namespace (see ``main``)."""
    return config_from_args(args)


# ---------------------------------------------------------------------------
# Read-only report
# ---------------------------------------------------------------------------


def report(run_dir: str | os.PathLike) -> None:
    """Print one finished run's ``val_metrics.csv``.  Performs no writes."""
    path = Path(run_dir)
    if (path / "All").is_dir():
        path = path / "All"
    csv_path = path / "val_metrics.csv"
    if not csv_path.is_file():
        raise SystemExit(f"no val_metrics.csv under {path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"{path}")
    print(
        f"{'ep':>4} {'train_loss':>11} {'val_loss':>10} "
        f"{'train_nse':>9} {'val_nse':>8} {'val_kge':>8}"
    )
    for row in rows:
        print(
            f"{int(row['epoch']):>4} {float(row['train_loss']):>11.6f} "
            f"{float(row['val_loss']):>10.6f} "
            f"{float(row['train_median_nse']):>9.4f} "
            f"{float(row['val_median_nse']):>8.4f} "
            f"{float(row['val_median_kge']):>8.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    global _RAW_ARGV
    _RAW_ARGV = list(argv) if argv is not None else None
    args = parse_args(argv)
    if args.report is not None:
        report(args.report)
        return 0
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
