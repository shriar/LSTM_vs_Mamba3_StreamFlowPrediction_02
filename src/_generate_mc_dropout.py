#!/usr/bin/env python3
"""Generate MC-dropout prediction samples from an existing CAMELS checkpoint.

This script does not train or modify a model. It loads a completed ``All`` run,
reuses the project's validation preprocessing, performs stochastic forward
passes with dropout enabled, and saves:

    pred_mc_dropout_daily_m3s.npy

The saved array has shape ``(samples, basins, days)``.

Example (from experiment):

    python generate_mc_dropout.py \
        --run-dir output/Mamba2_3L/Tier1_Ep20_Variant-standard_V2_head16_H64_L3_dConv4/All \
        --model mamba --epoch 20 --samples 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Match the OpenMP guard used by the other experiment scripts.
if sys.platform.startswith("win") or os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import torch

from paths import PROJECT_ROOT

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(SCRIPT_DIR))
_local_hydrodl = SCRIPT_DIR / "hydroDLpack"
if _local_hydrodl.is_dir():
    sys.path.insert(0, str(_local_hydrodl))

from config import ExperimentConfig
from data_utils import load_and_scale_data
from evaluation import _specify_inputs, inverse_transform_predictions
from hydroDL.master.master import loadModel
from hydroDL.model.settings import make_test_settings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create pred_mc_dropout_daily_m3s.npy from a trained checkpoint."
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Completed run's All directory containing model_Ep{epoch}.pt.",
    )
    parser.add_argument(
        "--model", choices=["lstm", "mamba"], required=True,
        help="Model family used by the run; used to construct the data config.",
    )
    parser.add_argument("--epoch", type=int, required=True,
                        help="Checkpoint epoch to load, e.g. 20.")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of stochastic passes (default: 10).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Validation batch size; default comes from config.")
    parser.add_argument("--param-tier", type=int, default=3,
                        help="Config tier used for loading the preprocessed data.")
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--mamba-version", type=int, default=2)
    parser.add_argument("--mamba-layers", type=int, default=3)
    parser.add_argument("--output-name", default="pred_mc_dropout_daily_m3s.npy")
    return parser.parse_args()


def _progress_bar(current, total, *, prefix="", start_time=None, width=28):
    """Print a dependency-free progress bar with elapsed time and ETA."""
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    fraction = current / total
    filled = int(width * fraction)
    bar = "=" * filled + ">" + " " * max(width - filled - 1, 0)
    elapsed = time.perf_counter() - start_time if start_time is not None else 0.0
    eta = elapsed * (total - current) / current if current else 0.0
    print(
        f"\\r{prefix} [{bar}] {current}/{total} "
        f"{fraction * 100:5.1f}% | elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s",
        end="",
        flush=True,
    )
    if current >= total:
        print()


def _stochastic_prediction(model, val_x_buffered, val_c, batch_size, output_csv,
                           *, sample_index, total_samples):
    """Run one dropout-enabled validation pass through the project's pipeline."""
    if output_csv.exists():
        output_csv.unlink()

    handles = [output_csv.open("a", encoding="utf-8")]
    inputs, settings = make_test_settings(model, val_x_buffered, val_c, batch_size, False, None)

    # Keep dropout active. Gradients remain disabled below.
    model.train(mode=True)
    total_batches = len(settings["iS"])
    sample_start = time.perf_counter()
    for i in range(total_batches):
        data = _specify_inputs(inputs, settings, i)
        with torch.inference_mode():
            prediction = model(
                data["xTest"],
                doDropMC=True,
                dropoutFalse=False,
            )
        y_out = prediction.detach().cpu().numpy().swapaxes(0, 1)
        # Keep the existing CSV layout expected by inverse_transform_predictions,
        # but avoid pandas overhead for every validation batch.
        np.savetxt(handles[0], y_out[:, :, 0], delimiter=",")
        _progress_bar(
            i + 1,
            total_batches,
            prefix=f"Sample {sample_index}/{total_samples}",
            start_time=sample_start,
        )

    handles[0].close()


def main():
    args = parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least 2 for uncertainty estimation")

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoint = run_dir / f"model_Ep{args.epoch}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # This config is used for the same CAMELS data/scaler conventions as the
    # training code. The serialized checkpoint supplies the trained architecture.
    cfg = ExperimentConfig(
        model_type=args.model,
        param_tier=args.param_tier,
        epoch=args.epoch,
        lstm_layers=args.lstm_layers,
        mamba_version=args.mamba_version,
        mamba_layers=args.mamba_layers,
    )
    data = load_and_scale_data(cfg)
    train_x = data["train_x"]
    val_x = data["val_x"]
    val_y = data["val_y"]
    val_c = data["val_c"]
    scaler = data["scaler"]
    basinarea = data["basinarea"]
    meanprep = data["meanprep"]

    # The evaluation pipeline prepends the training tail for sequence warmup.
    test_buff = train_x.shape[1]
    val_x_buffered = np.concatenate([train_x, val_x], axis=1)
    batch_size = args.batch_size or cfg.valid_batch

    print(f"Loading checkpoint: {checkpoint}")
    model = loadModel(str(run_dir), epoch=args.epoch)

    samples = []
    temporary_csv = run_dir / ".mc_dropout_prediction.csv"
    for sample_idx in range(args.samples):
        print(f"MC-dropout sample {sample_idx + 1}/{args.samples}")
        _stochastic_prediction(
            model,
            val_x_buffered,
            val_c,
            batch_size,
            temporary_csv,
            sample_index=sample_idx + 1,
            total_samples=args.samples,
        )
        prediction = inverse_transform_predictions(
            temporary_csv,
            scaler,
            basinarea,
            meanprep,
            cfg.log_norm_cols,
            test_buff,
        )
        samples.append(np.asarray(prediction.squeeze(), dtype=np.float32))

    if temporary_csv.exists():
        temporary_csv.unlink()

    samples_array = np.stack(samples, axis=0)
    output_path = run_dir / args.output_name
    np.save(output_path, samples_array)

    print(f"Saved: {output_path}")
    print(f"Shape: {samples_array.shape}")
    print(f"Mean ensemble spread: {float(np.nanstd(samples_array, axis=0).mean()):.6g} m³/s")


if __name__ == "__main__":
    main()
