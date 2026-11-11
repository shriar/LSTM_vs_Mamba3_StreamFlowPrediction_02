"""Count the actual trainable parameters in the project LSTM and Mamba models.

Examples
--------
# Use the project's default dimensions and architecture settings:
python count_model_parameters.py

# Use explicit dimensions and architecture settings:
python count_model_parameters.py --nx 40 --ny 1 --lstm-layers 3 --mamba-layers 3 \
    --mamba-version 2 --d-state 32 --expand 2

# On a machine without CUDA, count the portable Mamba fallback explicitly:
python count_model_parameters.py --allow-cpu-fallback

The default behavior requires CUDA for a real mamba_ssm model. The model in
mamba_model.py intentionally uses a different portable sequence mixer when
CUDA is unavailable, so its parameter count is not the parameter count of the
actual mamba_ssm implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


# ============================================================================
# EDIT MODEL HYPERPARAMETERS HERE
# Command-line arguments override these values when supplied.
# ============================================================================

# Experiment dimensions
# The experiment builds x as concatenate([train_x, train_c], axis=2):
#   train_x: 5 Daymet forcing variables
#   train_c: 35 CAMELS static attributes from config.ATTR_LST
# Therefore nx = 5 + 35 = 40. train_y is runoff with one output channel.
N_FORCING_VARIABLES = 5
N_STATIC_ATTRIBUTES = 35
NX = N_FORCING_VARIABLES + N_STATIC_ATTRIBUTES  # 40
NY = 1                                           # runoff/streamflow

# LSTM hyperparameters
LSTM_HIDDEN_SIZE = 128     # LSTM hidden size
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.5

# Mamba hyperparameters
MAMBA_HIDDEN_SIZE = 140    # Mamba d_model
MAMBA_LAYERS = 1
MAMBA_VERSION = 3          # Adapter variant requires Mamba 3
MAMBA_D_STATE = 64
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2
# Same rule as ExperimentConfig.__post_init__:
MAMBA_HEADDIM = 32 if MAMBA_VERSION == 3 else (MAMBA_HIDDEN_SIZE * MAMBA_EXPAND) // 8
MAMBA_DROPOUT = 0.11       # Match the ExperimentConfig adapter setting
MAMBA_ADAPTER_BOTTLENECK = 64

# Headline wiring is `standard` (B4 amendment, 2026-11-04), so the evidence
# helper defaults to it; pass --mamba-variant adapter|film|hybrid to reproduce
# the older wiring audit rows.
MAMBA_VARIANT = "standard"

# Runtime settings
DEVICE = "cuda"            # "auto", "cpu", or "cuda"
ALLOW_CPU_FALLBACK = False # True counts the non-Mamba CPU fallback
RUN_DEPTH_LADDER = False   # True counts all 1L/2L/3L ladder candidates

def hidden_size_ladder():
    """Return the capacity ladder as (tier, LSTM-1L, Mamba-1L, LSTM-2L, ...).

    Tiers 1 (~200K), 2 (~500K), and 3 (~1M) are calibrated for the Mamba-3
    headdim=32 constraint against the LSTM. The numbers are read from
    config.py, the single source of truth the trainer builds models from, so
    this evidence helper cannot drift after a ladder change.
    """
    from config import (
        LSTM_1L_TIER_CONFIG, LSTM_2L_TIER_CONFIG, LSTM_3L_TIER_CONFIG,
        MAMBA_1L_TIER_CONFIG, MAMBA_2L_TIER_CONFIG, MAMBA_3L_TIER_CONFIG,
    )

    return tuple(
        (
            tier,
            LSTM_1L_TIER_CONFIG[tier], MAMBA_1L_TIER_CONFIG[tier],
            LSTM_2L_TIER_CONFIG[tier], MAMBA_2L_TIER_CONFIG[tier],
            LSTM_3L_TIER_CONFIG[tier], MAMBA_3L_TIER_CONFIG[tier],
        )
        for tier in (1, 2, 3)
    )


# Allow this file to be run from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from CudnnLstmModel import CudnnLstmModel
from hydroDL.model.rnn import CpuLstmModel
from mamba_model import (
    MambaStreamflowModel,
    MambaStreamflowModelAdapter,
    MambaStreamflowModelFiLM,
    MambaStreamflowModelFiLMHybrid,
    MambaStreamflowModelPrefixTokens,
    MambaStreamflowModelResidualBlocks,
)


class ParameterCountingLSTM(torch.nn.Module):
    """CPU-safe parameter-equivalent of the HydroDL projected LSTM.

    HydroDL's CpuLstmModel supports one layer in this project. PyTorch's LSTM
    uses the same four-gate weight and bias layout, so this class lets us count
    multi-layer configurations without requiring an NVIDIA driver. It is used
    only for parameter counting; training still uses the project's HydroDL
    model.
    """

    def __init__(self, nx: int, ny: int, hidden_size: int, layers: int):
        super().__init__()
        self.linearIn = torch.nn.Linear(nx, hidden_size)
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=layers,
        )
        self.linearOut = torch.nn.Linear(hidden_size, ny)


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in an instantiated model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_breakdown(model: torch.nn.Module) -> list[tuple[str, int]]:
    """Return trainable parameter counts grouped by top-level module."""
    breakdown = []
    for name, module in model.named_children():
        count = count_parameters(module)
        if count:
            breakdown.append((name, count))
    return breakdown


def format_count(value: int) -> str:
    return f"{value:,} ({value / 1_000_000:.6f}M)"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


MAMBA_MODEL_CLASS_TO_VARIANT = {
    "MambaStreamflowModel": "standard",
    "MambaStreamflowModelFiLMHybrid": "hybrid",
    "MambaStreamflowModelFiLM": "film",
    "MambaStreamflowModelAdapter": "adapter",
    "MambaStreamflowModelResidualBlocks": "residual",
    "MambaStreamflowModelPrefixTokens": "prefix",
}
MAMBA_VARIANT_TO_MODEL_CLASS = {
    variant: model_class
    for model_class, variant in MAMBA_MODEL_CLASS_TO_VARIANT.items()
}
MAMBA_MODEL_CLASSES = {
    "MambaStreamflowModel": MambaStreamflowModel,
    "MambaStreamflowModelFiLMHybrid": MambaStreamflowModelFiLMHybrid,
    "MambaStreamflowModelFiLM": MambaStreamflowModelFiLM,
    "MambaStreamflowModelAdapter": MambaStreamflowModelAdapter,
    "MambaStreamflowModelResidualBlocks": MambaStreamflowModelResidualBlocks,
    "MambaStreamflowModelPrefixTokens": MambaStreamflowModelPrefixTokens,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count actual trainable parameters in the LSTM and Mamba models."
    )
    parser.add_argument("--nx", type=positive_int, default=NX, help="Number of input features.")
    parser.add_argument("--ny", type=positive_int, default=NY, help="Number of output features.")
    parser.add_argument("--lstm-hidden-size", type=positive_int, default=LSTM_HIDDEN_SIZE)
    parser.add_argument("--mamba-hidden-size", type=positive_int, default=MAMBA_HIDDEN_SIZE)
    mamba_model_group = parser.add_mutually_exclusive_group()
    mamba_model_group.add_argument(
        "--mamba-model",
        choices=tuple(MAMBA_MODEL_CLASSES),
        default=None,
        help="Mamba model class to instantiate.",
    )
    mamba_model_group.add_argument(
        "--mamba-variant",
        choices=tuple(MAMBA_VARIANT_TO_MODEL_CLASS),
        default=MAMBA_VARIANT,
        help="Mamba variant, matching run_all_tiers.py (default: standard).",
    )
    parser.add_argument("--lstm-layers", type=positive_int, default=LSTM_LAYERS)
    parser.add_argument("--lstm-dropout", type=float, default=LSTM_DROPOUT)
    parser.add_argument("--mamba-layers", type=positive_int, default=MAMBA_LAYERS)
    parser.add_argument("--mamba-version", type=int, choices=(1, 2, 3), default=MAMBA_VERSION)
    parser.add_argument("--d-state", type=positive_int, default=MAMBA_D_STATE)
    parser.add_argument("--d-conv", type=positive_int, default=MAMBA_D_CONV)
    parser.add_argument("--expand", type=positive_int, default=MAMBA_EXPAND)
    parser.add_argument(
        "--adapter-bottleneck", type=positive_int,
        default=MAMBA_ADAPTER_BOTTLENECK,
        help="Adapter bottleneck dimension for the adapter variant.",
    )
    parser.add_argument(
        "--headdim",
        type=positive_int,
        default=None,
        help="Mamba-2/3 head dimension; default is (mamba_hidden_size * expand) // 8.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    parser.add_argument(
        "--ladder",
        action="store_true",
        default=RUN_DEPTH_LADDER,
        help="Count all 1-, 2-, and 3-layer candidates from different_param_case.md.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        default=ALLOW_CPU_FALLBACK,
        help="Allow counting the project's portable CPU fallback instead of real mamba_ssm.",
    )
    return parser


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_models(args: argparse.Namespace, device: torch.device):
    """Instantiate both models and return them with implementation metadata."""
    using_real_mamba = torch.cuda.is_available()
    if not using_real_mamba and not args.allow_cpu_fallback:
        raise RuntimeError(
            "CUDA is unavailable. mamba_model.py would instantiate its CPU fallback, "
            "not real mamba_ssm. Run this script on a CUDA machine, or pass "
            "--allow-cpu-fallback to count the fallback explicitly."
        )

    if using_real_mamba:
        lstm = CudnnLstmModel(
            nx=args.nx, ny=args.ny, hiddenSize=args.lstm_hidden_size,
            nLayer=args.lstm_layers, dr=args.lstm_dropout,
        ).to(device)
        lstm_implementation = "HydroDL CudnnLstmModel"
    elif args.lstm_layers == 1:
        lstm = CpuLstmModel(
            nx=args.nx, ny=args.ny, hiddenSize=args.lstm_hidden_size,
        ).to(device)
        lstm_implementation = "HydroDL CpuLstmModel"
    else:
        lstm = ParameterCountingLSTM(
            nx=args.nx, ny=args.ny, hidden_size=args.lstm_hidden_size,
            layers=args.lstm_layers,
        ).to(device)
        lstm_implementation = "PyTorch-equivalent parameter counter"

    mamba_inner_size = args.mamba_hidden_size * args.expand
    mamba_headdim = args.headdim
    if mamba_headdim is None:
        if args.mamba_version == 3:
            mamba_headdim = 32
        else:
            if mamba_inner_size % 8 != 0:
                raise RuntimeError(
                    "Invalid Mamba dimensions: hidden_size * expand must be divisible "
                    f"by 8, but got {args.mamba_hidden_size} * {args.expand} = "
                    f"{mamba_inner_size}. For expand=2, use a hidden size divisible "
                    "by 4 (for example 148 or 152 instead of 150)."
                )
            mamba_headdim = mamba_inner_size // 8

    if args.mamba_version != 3 and mamba_inner_size % mamba_headdim != 0:
        raise RuntimeError(
            "Invalid Mamba headdim: d_inner must be divisible by headdim, but got "
            f"d_inner={mamba_inner_size}, headdim={mamba_headdim}."
        )

    mamba_model_name = args.mamba_model or MAMBA_VARIANT_TO_MODEL_CLASS[args.mamba_variant]
    mamba_model_class = MAMBA_MODEL_CLASSES[mamba_model_name]
    mamba = mamba_model_class(
        nx=args.nx, ny=args.ny, hiddenSize=args.mamba_hidden_size,
        numLayers=args.mamba_layers, dState=args.d_state,
        dConv=args.d_conv, expand=args.expand, headdim=mamba_headdim,
        mambaVersion=args.mamba_version, dropout=MAMBA_DROPOUT,
        bottleneck=args.adapter_bottleneck,
    ).to(device)
    return (
        lstm, mamba, lstm_implementation, mamba_headdim,
        using_real_mamba, mamba_model_name,
    )


def run_depth_ladder(args: argparse.Namespace, device: torch.device) -> int:
    """Count the candidate hidden sizes listed in different_param_case.md."""
    print("tier,depth,lstm_hidden,mamba_hidden,lstm_params,mamba_params,relative_diff_percent")
    for tier, lstm_1, mamba_1, lstm_2, mamba_2, lstm_3, mamba_3 in hidden_size_ladder():
        for depth, lstm_hidden, mamba_hidden in (
            (1, lstm_1, mamba_1),
            (2, lstm_2, mamba_2),
            (3, lstm_3, mamba_3),
        ):
            run_args = argparse.Namespace(**vars(args))
            run_args.lstm_hidden_size = lstm_hidden
            run_args.mamba_hidden_size = mamba_hidden
            run_args.lstm_layers = depth
            run_args.mamba_layers = depth
            lstm, mamba, _, _, _, _ = build_models(run_args, device)
            lstm_count = count_parameters(lstm)
            mamba_count = count_parameters(mamba)
            relative_diff = 100 * abs(mamba_count - lstm_count) / lstm_count
            print(f"{tier},{depth},{lstm_hidden},{mamba_hidden},{lstm_count},{mamba_count},{relative_diff:.3f}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    device = select_device(args.device)
    if args.ladder:
        return run_depth_ladder(args, device)

    (
        lstm, mamba, lstm_implementation, mamba_headdim,
        using_real_mamba, mamba_model_name,
    ) = build_models(args, device)
    lstm_count = count_parameters(lstm)
    mamba_count = count_parameters(mamba)
    difference = mamba_count - lstm_count
    percentage = 100.0 * abs(difference) / lstm_count if lstm_count else 0.0

    print(f"Device: {device}")
    print(f"LSTM implementation: {lstm_implementation}")
    print(
        f"Mamba model class: {mamba_model_name} "
        f"({'real mamba_ssm' if using_real_mamba else 'CPU fallback'})"
    )
    print(f"Input/output dimensions: nx={args.nx}, ny={args.ny}")
    print(f"LSTM: hidden_size={args.lstm_hidden_size}, layers={args.lstm_layers}")
    print(
        "Mamba: "
        f"version={args.mamba_version}, d_model={args.mamba_hidden_size}, "
        f"layers={args.mamba_layers}, d_state={args.d_state}, "
        f"d_conv={args.d_conv}, expand={args.expand}, headdim={mamba_headdim}, "
        f"dropout={MAMBA_DROPOUT}, adapter_bottleneck={args.adapter_bottleneck}"
    )
    print()
    print(f"LSTM total trainable parameters:   {format_count(lstm_count)}")
    print(f"Mamba total trainable parameters:  {format_count(mamba_count)}")
    print(f"Mamba - LSTM difference:            {difference:+,} parameters")
    print(f"Absolute relative difference:       {percentage:.3f}%")
    print()
    print("LSTM breakdown:")
    for name, count in parameter_breakdown(lstm):
        print(f"  {name:20s} {format_count(count)}")
    print("Mamba breakdown:")
    for name, count in parameter_breakdown(mamba):
        print(f"  {name:20s} {format_count(count)}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
