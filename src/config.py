"""Shared experiment configuration for LSTM and Mamba streamflow runs.

Every tunable knob lives here so that the main scripts stay short and
only specify what differs (model_type + architecture-specific overrides).
"""

import os
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from paths import PROJECT_ROOT
from constants import ATTR_LST, forcing_variables

# The split is defined once, in the torch-free data layer (B2b). Re-exported
# here so every training/evaluation entry point can quote the same dates
# without importing pandas-heavy code it does not need.
from data_utils import CONTEXT_DAYS, SPLIT_WINDOWS  # noqa: E402

RECORD_START = SPLIT_WINDOWS["train"][0]   # 1980-10-01
RECORD_END = SPLIT_WINDOWS["test"][1]      # 2014-09-30


# cuBLAS reads this once, when it creates its workspace, so it has to be set
# before the first CUDA operation. Without it, cuBLAS results can differ
# between runs even with deterministic algorithms enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

# ---------------------------------------------------------------------------
# Tier definitions (shared across architectures)
# ---------------------------------------------------------------------------

TIER_NAMES = {
    1: "200K",
    2: "500K",
    3: "1M",
}

# Hidden-size ladders for the controlled 1-, 2-, and 3-layer experiments.
#
# The LSTM ladder is the published reference and is never changed.
#
# The Mamba ladder was re-derived on 2026-11-04 (B4 follow-up): the original
# values matched the LSTM parameter count only under the `adapter` wiring they
# were calibrated on. Under the mandatory `standard` wiring (the paper's
# headline) the same widths were 6-24% short, and the headline pair
# (LSTM 1L H162 = 218,053 vs Mamba-3 3L H80 = 159,375) was 27% apart.
#
# The Mamba width cannot be tuned continuously: mamba_model.py forces
# headdim=32 for Mamba-3 and pads d_inner up to a multiple of it, so widths
# move in ~16-wide steps (160 -> 176 -> 192 -> ...). These ladders hold the
# width that minimises |N_mamba - N_lstm| at each (tier, depth) cell under
# `standard`; every value is a multiple of 16, so d_inner = 2H is a multiple
# of 32 and no padding occurs. Resulting parity, per tier:
#   T1 3.8 / 4.7 / 5.0 %,  T2 2.2 / 2.3 / 2.5 %,  T3 1.9 / 1.2 / 1.2 %
# and the headline pair (LSTM 1L 218,053 vs Mamba-3 3L H96 219,685) is 0.75%.
# Equal parameter counts are not equal capacity: Mamba's recurrent state is
# depth * H * d_state against the LSTM's H (see B9 / A8).
LSTM_1L_TIER_CONFIG = {
    1: 162, 2: 247, 3: 345,
}

MAMBA_1L_TIER_CONFIG = {
    1: 176, 2: 272, 3: 384,
}

LSTM_2L_TIER_CONFIG = {
    1: 106, 2: 177, 3: 248,
}

MAMBA_2L_TIER_CONFIG = {
    1: 112, 2: 192, 3: 272,
}

LSTM_3L_TIER_CONFIG = {
    1: 92, 2: 149, 3: 206,
}

# With expand=2, these values keep d_inner divisible by headdim=32.
MAMBA_3L_TIER_CONFIG = {
    1: 96, 2: 160, 3: 224,
}

# Backward-compatible aliases for code that imports the original names.
LSTM_TIER_CONFIG = LSTM_1L_TIER_CONFIG
MAMBA_TIER_CONFIG = MAMBA_1L_TIER_CONFIG

# ---------------------------------------------------------------------------
# Forcing variables and static attributes live in constants.py so the data
# pipeline can import them without pulling in torch.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All hyperparameters for a single CAMELS experiment."""

    # --- Identity ---
    model_type: str = "lstm"          # "lstm" or "mamba"
    param_tier: int = 3               # 1–3

    # --- Seed ---
    seed: int = 111111
    deterministic: bool = True          # deterministic kernels where available
    deterministic_strict: bool = False  # True: raise on nondeterministic ops (after the warn-only sweep)

    # --- Dataset ---
    data_opt: int = 2                 # 1: NCAR download, 2: presaved, 3: local script
    for_type: str = "daymet"
    flow_regime: int = 1              # 0: log-norm + RMSE, 1: standard norm + NSE
    t_train: list = field(default_factory=lambda: [19801001, 19951001])
    t_valid: list = field(default_factory=lambda: [19951001, 19991001])
    subset_train: str = "All"
    ctx_days: int = CONTEXT_DAYS  # warm-up inputs prepended to val/test at load time
    train_file: str = "data/pickles/train.pkl"
    validation_file: str = "data/pickles/val.pkl"
    train_file_gdrive_id: str = "1z9UBkn86dBIKo1wu1v7bYSTHG87RzGWX"
    validation_file_gdrive_id: str = "12f-ZrWLxtWRwzacmxBZ23-76_buGhlNa"
    gauge_id_file: str = "data/gauge_ids.csv"  # USGS IDs, one per basin row
    gauge_metadata_url: str = (
        "https://zenodo.org/records/15529996/files/camels_name.txt?download=1"
    )

    # --- Training ---
    epoch: int = 30
    batch_size: int = 128
    rho: int = 365
    train_buff: int = 365  # Warmup days excluded from loss (hydroDL / CAMELS default)
    save_epoch: int = 5
    use_amp: bool = False  # BF16 autocast; OFF for reproducible, full-precision runs

    # --- LSTM-specific training parameters ---
    lr: float = 2e-3  # LSTM cosine peak/starting LR --- best = 2e-3
    lstm_min_lr: float = 1e-4  # Keep useful updates through a 20-epoch run --- best = 1e-4
    lstm_optimizer: str = "adamw"  # "adam" or "adamw"
    lstm_weight_decay: float = 1e-5  # Light regularization for LSTM weights --- best = 1e-5
    lstm_beta1: float = 0.9
    lstm_beta2: float = 0.999
    lstm_grad_clip: float = 1.0  # Critical for LSTM stability (exploding gradients)
    lstm_scheduler: str = "cosine"  # Decay from lr to lstm_min_lr

    # --- Mamba-specific training parameters ---
    mamba_lr: float = 3e-3  # Cosine peak/starting LR -- best = 5e-4
    mamba_min_lr: float = 3e-4  # Cosine final LR -- best = 2e-4
    mamba_weight_decay: float = 1e-3  # AdamW decay for eligible matrix weights -- best = 9e-3
    mamba_beta1: float = 0.9
    mamba_beta2: float = 0.95  # Mamba/GPT-NeoX convention
    mamba_warmup_steps: int = 5  # Short warmup before cosine decay
    mamba_scheduler: str = "cosine"  # Decay from mamba_lr to mamba_min_lr
    mamba_optimizer: str = "adamw"  # "adam" or "adamw"
    mamba_grad_clip: float = 0.5  # Essential for training stability -- best = 1
    mamba_no_decay_params: list = field(default_factory=lambda: ["A_log", "D", "conv1d.bias", "norm"])  # Params to exclude from weight decay

    # --- RHO-scaling toggle ---
    rho_scaling_experiment: bool = False

    # --- LSTM-specific (ignored for Mamba) ---
    lstm_layers: int = 1
    lstm_dropout: float = 0.2  # Lower recurrent dropout for 20-epoch convergence

    # --- Mamba-specific (ignored for LSTM) ---
    mamba_version: int = 3  # Version selector: 1, 2, or 3
    mamba_variant: str = "standard"  # Variant: "standard", "film", "hybrid", "prefix", "adapter", or "residual"
    mamba_layers: int = 3  # Common to all versions
    mamba_d_state: int = 64  # Common to all versions (v1 caps at 32, else defaults to 16)
    mamba_d_conv: int = 4  # Mamba-1 & 2 only (ignored in v3)
    mamba_expand: int = 2  # Common to all versions
    mamba_headdim = None  # Mamba-2 & 3 only (ignored in v1); derived in __post_init__
    mamba_dropout: float = 0.11  # Mamba-3 only (ignored in v1 & v2) - best = 0.11
    mamba_n_prefix_tokens: int = 2  # Number of prefix tokens for prefix variant
    mamba_residual_scale_init: float = 0.1  # Initial residual scale for residual/film/hybrid/adapter variants
    mamba_adapter_bottleneck: int = 64  # Bottleneck dimension for adapter variant

    # --- Validation ---
    valid_batch: int = 25

    # --- Output directory ---
    output_dir: str = "output"  # Custom output directory name (default: "output")

    # --- Derived (set by post_init) ---
    script_dir: Path = field(init=False)
    hidden_size: int = field(init=False)
    tier_name: str = field(init=False)

    def __post_init__(self):
        # Historical name kept: every path helper builds from it. It is the
        # project root (parent of src/), not the directory holding the code.
        self.script_dir = PROJECT_ROOT
        self.lstm_optimizer = self.lstm_optimizer.lower()
        self.lstm_scheduler = self.lstm_scheduler.lower()
        self.mamba_optimizer = self.mamba_optimizer.lower()
        self.mamba_scheduler = self.mamba_scheduler.lower()
        if self.lstm_scheduler not in {"onecycle", "cosine"}:
            raise ValueError("lstm_scheduler must be 'onecycle' or 'cosine'")
        if self.mamba_scheduler not in {"onecycle", "cosine"}:
            raise ValueError("mamba_scheduler must be 'onecycle' or 'cosine'")
        if self.lstm_min_lr < 0 or self.lstm_min_lr > self.lr:
            raise ValueError("lstm_min_lr must satisfy 0 <= lstm_min_lr <= lr")
        if self.mamba_min_lr < 0 or self.mamba_min_lr > self.mamba_lr:
            raise ValueError("mamba_min_lr must satisfy 0 <= mamba_min_lr <= mamba_lr")
        if self.lstm_optimizer not in {"adam", "adamw"}:
            raise ValueError("lstm_optimizer must be 'adam' or 'adamw'")
        if self.mamba_optimizer not in {"adam", "adamw"}:
            raise ValueError("mamba_optimizer must be 'adam' or 'adamw'")
        if self.model_type == "lstm":
            if self.lstm_layers < 1:
                print(f"[WARNING] lstm_layers={self.lstm_layers} invalid. Using 1 layer.")
                self.lstm_layers = 1
            lstm_ladders = {
                1: LSTM_1L_TIER_CONFIG,
                2: LSTM_2L_TIER_CONFIG,
                3: LSTM_3L_TIER_CONFIG,
            }
            # Use tier config if available, otherwise use nearest neighbor
            if self.lstm_layers in lstm_ladders:
                tier_cfg = lstm_ladders[self.lstm_layers]
            else:
                # Find nearest layer count for hidden size estimation
                nearest_layer = min(lstm_ladders.keys(), key=lambda x: abs(x - self.lstm_layers))
                print(f"[INFO] lstm_layers={self.lstm_layers} not in predefined ladders. Using hidden size from {nearest_layer}L config.")
                tier_cfg = lstm_ladders[nearest_layer]
        else:
            if self.mamba_layers < 1:
                print(f"[WARNING] mamba_layers={self.mamba_layers} invalid. Using 1 layer.")
                self.mamba_layers = 1
            mamba_ladders = {
                1: MAMBA_1L_TIER_CONFIG,
                2: MAMBA_2L_TIER_CONFIG,
                3: MAMBA_3L_TIER_CONFIG,
            }
            # Use tier config if available, otherwise use nearest neighbor
            if self.mamba_layers in mamba_ladders:
                tier_cfg = mamba_ladders[self.mamba_layers]
            else:
                # Find nearest layer count for hidden size estimation
                nearest_layer = min(mamba_ladders.keys(), key=lambda x: abs(x - self.mamba_layers))
                print(f"[INFO] mamba_layers={self.mamba_layers} not in predefined ladders. Using hidden size from {nearest_layer}L config.")
                tier_cfg = mamba_ladders[nearest_layer]
        self.hidden_size = tier_cfg[self.param_tier]
        self.tier_name = TIER_NAMES[self.param_tier]
        if self.model_type == "mamba":
            d_inner = self.hidden_size * self.mamba_expand
            if self.mamba_version == 3:
                # Force headdim=32 for all Mamba-3 adapter models
                self.mamba_headdim = 32
            else:
                # Preserve the existing eight-head layout for Mamba-1/2.
                self.mamba_headdim = d_inner // 8

    # --- Helpers ---

    @property
    def varF(self):
        return forcing_variables(self.for_type)

    @property
    def attrLst(self):
        return ATTR_LST

    @property
    def log_norm_cols(self):
        if self.flow_regime == 0:
            return ["prcp", "usgsFlow", "Precip", "runoff", "Runoff", "Runofferror"]
        return []

    @property
    def device(self):
        return torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

    def exp_name(self) -> str:
        """Top-level experiment folder name."""
        suffix = "_RhoScaling" if self.rho_scaling_experiment else ""
        if self.model_type == "lstm":
            return f"LSTM_{self.lstm_layers}L{suffix}"
        return f"Mamba{self.mamba_version}_{self.mamba_layers}L{suffix}"

    def save_path(self) -> str:
        """Relative sub-path under output/."""
        if self.model_type == "lstm":
            layers_suffix = f"_L{self.lstm_layers}"
            return os.path.join(
                self.exp_name(),
                f"Tier{self.param_tier}_{self.tier_name}_Ep{self.epoch}"
                f"_Bs{self.batch_size}_Rho{self.rho}_H{self.hidden_size}{layers_suffix}"
                f"_Seed{self.seed}",
            )
        # Mamba: condensed, self-describing folder name (matches plot_figures regex)
        # Tier{N}_Ep[epoch]_Variant-[variant]_V[version]_head[headdim]_H[hidden]_L[layers]_dConv[d_conv][_Rho{rho}]
        # _Rho suffix is appended only for non-default rho (365) so existing
        # runs keep their paths; rho sweeps get distinct folders.
        variant = self.mamba_variant or "standard"
        rho_suffix = "" if self.rho == 365 else f"_Rho{self.rho}"
        return os.path.join(
            self.exp_name(),
            f"Tier{self.param_tier}_Ep{self.epoch}"
            f"_Variant-{variant}"
            f"_V{self.mamba_version}"
            f"_head{self.mamba_headdim}"
            f"_H{self.hidden_size}"
            f"_L{self.mamba_layers}"
            f"_dConv{self.mamba_d_conv}{rho_suffix}"
            # Seed is always last so the folder is self-describing; parsers
            # (plot_figures.TIER_DIR_RE, sign_test_all_pairs.RUN_RE) treat it as
            # an optional trailing field.
            f"_Seed{self.seed}",
        )

    def output_path(self) -> str:
        """Full path to the run's output directory."""
        return os.path.join(self.script_dir, self.output_dir, self.save_path(), "All")

    def remove_other_epoch_runs(self) -> list[Path]:
        """Remove older epoch variants of this exact experiment configuration.

        A model/configuration has one active output folder: ``Ep20`` when
        training for 20 epochs, ``Ep30`` when training for 30 epochs, etc.
        Different tiers, batch sizes, sequence lengths, and architectures are
        left untouched.
        """
        current = Path(self.output_path()).parent
        experiment_root = current.parent
        if not experiment_root.is_dir():
            return []

        # Match this run's folder name while allowing only the epoch number to
        # differ. The seed stays inside ``tail`` so only the same seed matches:
        # different seeds must never delete each other's results.
        epoch_marker = f"_Ep{self.epoch}"
        if epoch_marker not in current.name:
            return []
        head, _, tail = current.name.partition(epoch_marker)
        pattern = re.compile(re.escape(head) + r"_Ep\d+" + re.escape(tail) + r"$")
        removed = []
        for candidate in sorted(experiment_root.iterdir()):
            if candidate.is_dir() and candidate != current and pattern.fullmatch(candidate.name):
                shutil.rmtree(candidate)
                removed.append(candidate)
        return removed

    def _resolve(self, name: str) -> Path:
        """Absolute path for a data file, independent of the working directory."""
        p = Path(name)
        return p if p.is_absolute() else self.script_dir / p

    def gauge_id_path(self) -> Path:
        """Standalone CSV of USGS gauge IDs (not stored in the pickle files)."""
        return self._resolve(self.gauge_id_file)

    def train_path(self) -> Path:
        """Path to the preprocessed training dataset."""
        return self._resolve(self.train_file)

    def validation_path(self) -> Path:
        """Path to the preprocessed validation dataset."""
        return self._resolve(self.validation_file)


def set_seed(cfg: ExperimentConfig):
    """Make the run as reproducible as the installed kernels allow.

    Bitwise reproducibility is only reachable for the cuDNN LSTM path: the
    mamba_ssm Triton kernels accumulate gradients with atomics, which no flag
    can order. ``deterministic_strict`` selects between raising on
    nondeterministic operations and only warning about them (discovery mode).
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)  # every device, not just current
        if torch.cuda.device_count() > 1:
            print(
                f"[determinism] {torch.cuda.device_count()} GPUs visible; "
                "multi-GPU reductions are not bitwise reproducible."
            )

    # Autotuning and TF32 both defeat seed control.
    torch.backends.cudnn.deterministic = bool(cfg.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    if cfg.deterministic:
        torch.use_deterministic_algorithms(
            True, warn_only=not cfg.deterministic_strict
        )
    else:
        torch.use_deterministic_algorithms(False)
