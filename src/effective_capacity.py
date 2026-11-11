"""B9 -- effective-capacity profiler (review item R5#6).

Builds the two headline models at their *final* widths -- LSTM ``standard``
1 layer, width 162, and Mamba-3 ``standard`` 3 layers, width 96, both on the
same 40-dim input (5 Daymet forcings + 35 static attributes) -- and measures
the five numbers the capacity claim (A8) needs:

    parameters, recurrent / SSM state size, FLOPs per 365-day pass,
    peak allocated VRAM for one training step, throughput (basin-days/s).

Why this exists
---------------
The headline pair is parameter-matched to 0.75 % (218,053 vs 219,685), but
equal parameter counts are not equal capacity: Mamba's recurrent state is
``depth x H x d_state`` (3 x 96 x 64 = 18,432) against the LSTM's ``H`` (162).
The paper must quote both numbers, so this script measures them rather than
arguing about them.  It does not claim matched capacity; it tabulates the
asymmetry.

What it does *not* do
---------------------
- No data is loaded and no checkpoint is touched: every measurement runs on
  random tensors of the shape the training loop uses.
- It never imports ``eval_test`` and cannot open the test split.
- It does not train; the timing loop is a synthetic optimizer step.
- The timing loop excludes the data pipeline (window sampling, copies,
  validation scoring), so ``basin_days_per_second`` compares the kernels only.
  It is not an epoch time; real epochs are ~8x slower per basin-day and are
  measured by the training runs themselves.

FLOPs convention (documented because no single profiler sees both kernels)
---------------------------------------------------------------------------
Both models call fused CUDA kernels (cuDNN LSTM, mamba_ssm Triton Mamba-3),
so ``torch.utils.flop_counter`` counts neither.  ``flops`` below is the
dominant-arithmetic model, 2 FLOPs per multiply-accumulate:

    1. every ``nn.Linear`` / ``nn.Conv1d`` in the module tree (dense GEMM and
       depthwise conv arithmetic), found by walking the *constructed* model so
       version-specific projections are included automatically;
    2. one selective-scan term of ``2 * d_inner * d_state`` MACs per state
       element per day: the state update (``h <- a*h + b*u``) and the readout
       (``y += C*h``).  The LSTM has no such term (its recurrence is already in
       the 4-gate dense count).

Excluded: elementwise nonlinearities, RMSNorm, dropout, and Mamba-3's extra
trapezoidal correction terms.  Treat the number as a cost model with the same
convention on both sides, not as an exact kernel counter; the measured
throughput column is the ground truth for wall-clock cost.

Usage
-----
    python src/effective_capacity.py
    python src/effective_capacity.py --batch 128 --iters 15

Writes ``results/table_effective_capacity.csv`` (one row per model).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

from config import ExperimentConfig, set_seed
from paths import PROJECT_ROOT, RESULTS_DIR

SEQ_DAYS = 365          # one rho-length pass
DEFAULT_BATCH = 128     # the training batch size (config.py)
NX = 40                 # 5 Daymet forcings + 35 static attributes
NY = 1

OUT_CSV = RESULTS_DIR / "table_effective_capacity.csv"


# ---------------------------------------------------------------------------
# Arithmetic model (pure functions, unit-tested without CUDA)
# ---------------------------------------------------------------------------


def lstm_macs_per_basin_day(nx: int, hidden: int, ny: int, layers: int) -> int:
    """MACs for one basin-day of the headline LSTM wiring.

    ``linearIn`` (nx -> H), then ``layers`` cuDNN LSTM cells whose input is the
    162-wide hidden state (input-to-hidden + hidden-to-hidden, 4 gates each),
    then ``linearOut`` (H -> ny).
    """
    input_proj = nx * hidden
    per_cell = 4 * hidden * (hidden + hidden)
    output_proj = hidden * ny
    return input_proj + layers * per_cell + output_proj


def module_macs(module: nn.Module) -> int:
    """Sum of in*out over every Linear / spatial Conv1d in the module tree."""
    total = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            total += m.in_features * m.out_features
        elif isinstance(m, nn.Conv1d):
            total += ((m.in_channels // m.groups) * m.out_channels
                      * m.kernel_size[0])
    return total


def scan_macs_per_basin_day(d_inner: int, d_state: int, layers: int) -> int:
    """Selective-scan state update + readout: 2 MACs per (d_inner, d_state)."""
    return 2 * layers * d_inner * d_state


def state_sizes(model_type: str, hidden: int, layers: int, d_state: int,
                d_conv: int = 4, expand: int = 2) -> dict:
    """Recurrent state, in the plan's sense, plus every auxiliary buffer.

    LSTM: hidden state ``h`` is ``H`` per layer; the cell state ``c`` doubles it.
    Mamba: the SSM state matrix is ``H * d_state`` per layer; the causal conv
    keeps ``d_conv - 1`` samples per inner channel.
    """
    if model_type == "lstm":
        ssm_per_layer = hidden
        aux = layers * hidden                      # cell state c
    else:
        ssm_per_layer = hidden * d_state
        aux = layers * (expand * hidden) * (d_conv - 1)
    ssm_total = layers * ssm_per_layer
    return {
        "state_ssm_per_layer": ssm_per_layer,
        "state_ssm_total": ssm_total,
        "state_aux_total": aux,
        "state_total_incl_aux": ssm_total + aux,
    }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def build_headline_models(device: torch.device):
    """Instantiate both models through the real training factory."""
    from train_val import build_model  # local: keeps the arithmetic tests cheap

    lstm_cfg = ExperimentConfig(
        model_type="lstm", param_tier=1, lstm_layers=1, epoch=60,
    )
    mamba_cfg = ExperimentConfig(
        model_type="mamba", param_tier=1, mamba_layers=3,
        mamba_version=3, mamba_variant="standard", epoch=60,
    )
    lstm = build_model(lstm_cfg, NX, NY)
    mamba = build_model(mamba_cfg, NX, NY)

    # The Mamba-3 kernel pads d_inner up to a multiple of headdim; at H=96,
    # headdim=32 no padding may occur or the width would silently differ from
    # the ladder the paper reports.
    assert mamba.linearIn.out_features == mamba_cfg.hidden_size, (
        f"Mamba width padded: ladder {mamba_cfg.hidden_size} -> "
        f"{mamba.linearIn.out_features}"
    )
    return lstm_cfg, mamba_cfg, lstm, mamba


def measure_step(model: nn.Module, device: torch.device, batch: int,
                 warmup: int, iters: int) -> dict:
    """Peak VRAM and steady-state wall clock for one training step."""
    model.train()
    x = torch.randn(SEQ_DAYS, batch, NX, device=device)
    y = torch.randn(SEQ_DAYS, batch, NY, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def step():
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):        # Triton autotune + cuDNN plan selection
        step()
    torch.cuda.synchronize(device)

    times = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    return {
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30,
        "step_seconds_median": statistics.median(times),
        "step_seconds_min": min(times),
        "basin_days_per_second": batch * SEQ_DAYS / statistics.median(times),
    }


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def torch_and_mamba_versions() -> tuple[str, str]:
    try:
        import mamba_ssm
        mamba_version = getattr(mamba_ssm, "__version__", "unknown")
    except Exception:                                   # pragma: no cover
        mamba_version = "not-installed"
    return torch.__version__, mamba_version


def profile(name: str, cfg: ExperimentConfig, model: nn.Module,
            timing: dict, torch_version: str, mamba_version: str) -> dict:
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if cfg.model_type == "lstm":
        dense = lstm_macs_per_basin_day(NX, cfg.hidden_size, NY, cfg.lstm_layers)
        scan = 0
        dims = {"depth": cfg.lstm_layers, "width": cfg.hidden_size,
                "d_state": "", "headdim": ""}
        variant = "standard"
    else:
        dense = module_macs(model)
        d_inner = 2 * cfg.hidden_size
        scan = scan_macs_per_basin_day(d_inner, cfg.mamba_d_state, cfg.mamba_layers)
        dims = {"depth": cfg.mamba_layers, "width": cfg.hidden_size,
                "d_state": cfg.mamba_d_state, "headdim": cfg.mamba_headdim}
        variant = cfg.mamba_variant

    macs_day = dense + scan
    flops_day = 2 * macs_day
    state = state_sizes(cfg.model_type, cfg.hidden_size, dims["depth"],
                        cfg.mamba_d_state, cfg.mamba_d_conv, cfg.mamba_expand)

    batch = timing["batch"]
    row = {
        "model": name,
        "variant": variant,
        **dims,
        "nx": NX,
        "ny": NY,
        "params_total": params_total,
        "params_trainable": params_trainable,
        **state,
        "dense_macs_per_basin_day": dense,
        "scan_macs_per_basin_day": scan,
        "flops_per_basin_day": flops_day,
        "flops_per_basin_pass_365d": flops_day * SEQ_DAYS,
        "flops_per_training_step_tflops": flops_day * SEQ_DAYS * batch / 1e12,
        **{k: v for k, v in timing.items() if k != "batch"},
        "batch": batch,
        "seq_days": SEQ_DAYS,
        "device": str(timing["device"]),
        "torch": torch_version,
        "mamba_ssm": mamba_version,
        "git_commit": git_commit(),
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    print(
        f"[{name}] params {params_total:,} | state {state['state_ssm_total']:,} "
        f"(incl. aux {state['state_total_incl_aux']:,}) | "
        f"{flops_day / 1e6:.1f} MFLOP/basin-day | "
        f"peak {timing['peak_alloc_gb']:.2f} GiB | "
        f"{timing['basin_days_per_second']:,.0f} basin-days/s "
        f"[{timing['basin_days_per_second_min']:,.0f}-"
        f"{timing['basin_days_per_second_max']:,.0f}] | "
        f"step {timing['step_seconds_median'] * 1000:.0f} ms"
    )
    return row


def aggregate(rounds: list[dict], batch: int, device: torch.device) -> dict:
    """Median across interleaved rounds; min/max expose the clock drift."""
    bps = [r["basin_days_per_second"] for r in rounds]
    return {
        "peak_alloc_gb": max(r["peak_alloc_gb"] for r in rounds),
        "peak_reserved_gb": max(r["peak_reserved_gb"] for r in rounds),
        "step_seconds_median": statistics.median(
            r["step_seconds_median"] for r in rounds),
        "step_seconds_min": min(r["step_seconds_min"] for r in rounds),
        "basin_days_per_second": statistics.median(bps),
        "basin_days_per_second_min": min(bps),
        "basin_days_per_second_max": max(bps),
        "rounds": len(rounds),
        "batch": batch,
        "device": device,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help="basins per synthetic step (default: the training batch, 128)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="untimed steps before measuring (Triton/cuDNN warm-up)")
    parser.add_argument("--iters", type=int, default=30, help="timed steps per round")
    parser.add_argument("--rounds", type=int, default=5,
                        help="interleaved rounds; the reported value is the median "
                             "(the min/max columns expose GPU clock drift)")
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the memory/throughput columns")
    if args.device == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(args.device)

    torch_version, mamba_version = torch_and_mamba_versions()
    print(f"device={device} torch={torch_version} mamba_ssm={mamba_version} "
          f"batch={args.batch} seq={SEQ_DAYS} iters={args.iters}")

    lstm_cfg, mamba_cfg, lstm, mamba = build_headline_models(device)
    set_seed(lstm_cfg)                  # same determinism flags as training
    models = [("lstm", lstm_cfg, lstm), ("mamba3", mamba_cfg, mamba)]

    rounds: dict[str, list[dict]] = {name: [] for name, _, _ in models}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for round_index in range(args.rounds):
            for name, _cfg, model in models:      # interleave to cancel drift
                single = measure_step(model, device, args.batch,
                                      args.warmup, args.iters)
                rounds[name].append(single)
                print(f"  round {round_index + 1}/{args.rounds} {name}: "
                      f"{single['basin_days_per_second']:,.0f} basin-days/s")

    rows = [
        profile(name, cfg, model, aggregate(rounds[name], args.batch, device),
                torch_version, mamba_version)
        for name, cfg, model in models
    ]

    write_csv(rows, args.out)
    print(f"\nwrote {args.out}")

    # Compact view: the pair of numbers the capacity claim quotes.
    lstm_row, mamba_row = rows
    ratio = mamba_row["state_ssm_total"] / lstm_row["state_ssm_total"]
    print(f"\nheadline parity: {lstm_row['params_total']:,} vs "
          f"{mamba_row['params_total']:,} params "
          f"({100 * abs(mamba_row['params_total'] - lstm_row['params_total']) / lstm_row['params_total']:.2f} %)")
    print(f"state asymmetry: {lstm_row['state_ssm_total']:,} vs "
          f"{mamba_row['state_ssm_total']:,} ({ratio:.1f}x), "
          f"{lstm_row['state_total_incl_aux']:,} vs "
          f"{mamba_row['state_total_incl_aux']:,} counting every buffer")
    print(f"throughput: {lstm_row['basin_days_per_second']:,.0f} "
          f"[{lstm_row['basin_days_per_second_min']:,.0f}-"
          f"{lstm_row['basin_days_per_second_max']:,.0f}] vs "
          f"{mamba_row['basin_days_per_second']:,.0f} "
          f"[{mamba_row['basin_days_per_second_min']:,.0f}-"
          f"{mamba_row['basin_days_per_second_max']:,.0f}] basin-days/s "
          f"(LSTM/Mamba = {lstm_row['basin_days_per_second'] / mamba_row['basin_days_per_second']:.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
