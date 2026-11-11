"""Mamba-based streamflow model — drop-in replacement for CudnnLstmModel/CpuLstmModel.

Supports Mamba 1 (selective SSM), Mamba 2 (SSD), and Mamba 3 (trapezoidal SSM + RoPE).
Interface matches hydroDL LSTM models used in the CAMELS tutorial:
    model = MambaStreamflowModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE, mambaVersion=3)
    yp = model(x)   # x: [seq_len, batch, nx] -> yp: [seq_len, batch, ny]
"""

import math

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba, Mamba2, Mamba3
except ImportError:
    try:
        from mamba_ssm import Mamba, Mamba2
        Mamba3 = None
    except ImportError:
        try:
            from mamba_ssm import Mamba
            Mamba2 = None
            Mamba3 = None
        except ImportError as e:
            raise ImportError(
                "mamba_ssm is required for MambaStreamflowModel. "
                "Install with: pip install mamba-ssm causal-conv1d"
            ) from e

# The mamba_ssm RMSNorm implementation is Triton/CUDA-only.  Importing it on
# a CPU host succeeds, but its first forward pass fails while selecting a CUDA
# device.  Use PyTorch's portable implementation unless CUDA is available.
if torch.cuda.is_available():
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
    except ImportError:
        try:
            from mamba_ssm import RMSNorm
        except ImportError:
            RMSNorm = nn.RMSNorm
else:
    RMSNorm = nn.RMSNorm


def _validate_mamba_version(mamba_version):
    if mamba_version not in (1, 2, 3):
        raise ValueError(
            f"Unsupported mambaVersion={mamba_version!r}; expected 1, 2, or 3."
        )


class _CpuSequenceBlock(nn.Module):
    """Portable sequence mixer used when CUDA-only Mamba kernels are unavailable."""

    def __init__(self, hidden_size, d_conv, expand):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=d_conv,
            padding=d_conv - 1, groups=hidden_size,
        )
        self.in_proj = nn.Linear(hidden_size, 2 * hidden_size * expand)
        self.out_proj = nn.Linear(hidden_size * expand, hidden_size)

    def forward(self, x):
        # x is [batch, sequence, features]. Trim the causal-convolution tail.
        conv = self.conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        value, gate = self.in_proj(conv).chunk(2, dim=-1)
        return self.out_proj(torch.nn.functional.silu(value) * torch.sigmoid(gate))


class MambaStreamflowModel(nn.Module):
    def __init__(
        self,
        *,
        nx,
        ny,
        hiddenSize,
        numLayers=2,
        dState=64,
        dConv=4,
        expand=2,
        headdim=64,
        mambaVersion=3,
        dropout=0.1,
        **kwargs,
    ):
        _validate_mamba_version(mambaVersion)
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.mambaVersion = mambaVersion
        self.cpu_fallback = not torch.cuda.is_available()

        d_inner = expand * hiddenSize
        original_headdim = headdim
        invalid_v3_headdim = (
            mambaVersion == 3 and (
                headdim is None or headdim < 32 or headdim not in (32, 64)
            )
        )
        if headdim is None or d_inner % headdim != 0 or invalid_v3_headdim:
            # Mamba-3 kernels require power-of-two head dimensions >= 32.
            for candidate in [64, 32]:
                if d_inner % candidate == 0:
                    headdim = candidate
                    break
            else:
                headdim = 32
            if original_headdim is not None:
                print(f"[WARNING] headdim={original_headdim} is incompatible with d_inner={d_inner} "
                      f"or Mamba-{mambaVersion}. Automatically adjusted to headdim={headdim}.")

        # Mamba-3's backward kernel requires headdim >= 32 and d_inner to be
        # divisible by headdim. Pad only the internal SSM width; the external
        # model input/output dimensions remain unchanged.
        mamba_hidden_size = hiddenSize
        if mambaVersion == 3 and not self.cpu_fallback:
            padded_inner = math.ceil(d_inner / headdim) * headdim
            mamba_hidden_size = math.ceil(padded_inner / expand)
            padded_inner = expand * mamba_hidden_size
            if mamba_hidden_size != hiddenSize:
                print(f"[INFO] Padding Mamba-3 internal hidden size from {hiddenSize} "
                      f"to {mamba_hidden_size} for headdim={headdim}.")

        self.linearIn = nn.Linear(nx, mamba_hidden_size)
        self.normIn = RMSNorm(mamba_hidden_size)

        if self.cpu_fallback:
            # mamba_ssm and causal-conv1d ship CUDA-only kernels.  Keep the
            # runner usable on CPU-only WSL installations with a causal,
            # gated sequence mixer that has the same input/output contract.
            self.layers = nn.ModuleList(
                [_CpuSequenceBlock(hiddenSize, dConv, expand) for _ in range(numLayers)]
            )
            self.name = f"Mamba{mambaVersion}StreamflowModel_CPUFallback"
        elif mambaVersion == 3:
            if Mamba3 is None:
                raise ImportError("Mamba3 is not available in current mamba_ssm installation.")
            self.layers = nn.ModuleList(
                [
                    Mamba3(
                        d_model=mamba_hidden_size,
                        d_state=dState,
                        expand=expand,
                        headdim=headdim,
                        dtype=torch.float32,
                        dropout=0.0,
                    )
                    for _ in range(numLayers)
                ]
            )
            self.name = "Mamba3StreamflowModel"
        elif mambaVersion == 2:
            if Mamba2 is None:
                raise ImportError("Mamba2 is not available in current mamba_ssm installation.")
            self.layers = nn.ModuleList(
                [
                    Mamba2(
                        d_model=hiddenSize,
                        d_state=dState,
                        d_conv=dConv,
                        expand=expand,
                        headdim=headdim,
                    )
                    for _ in range(numLayers)
                ]
            )
            self.name = "Mamba2StreamflowModel"
        else:
            self.layers = nn.ModuleList(
                [
                    Mamba(
                        d_model=hiddenSize,
                        d_state=dState if dState <= 32 else 16,
                        d_conv=dConv,
                        expand=expand,
                    )
                    for _ in range(numLayers)
                ]
            )
            self.name = "MambaStreamflowModel"

        self.norms = nn.ModuleList(
            [RMSNorm(mamba_hidden_size) for _ in range(numLayers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.linearOut = nn.Linear(mamba_hidden_size, ny)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        # x: [seq_len, batch, nx] (same convention as hydroDL LSTM models)
        h = self.normIn(self.linearIn(x))
        # Mamba expects [batch, seq, dim]
        h = h.permute(1, 0, 2).contiguous()
        for layer, norm in zip(self.layers, self.norms):
            layer_input = norm(h)
            if self.mambaVersion == 3 and not self.cpu_fallback:
                layer_input = layer_input.to(next(layer.in_proj.parameters()).dtype)
            h = h + self.dropout(layer(layer_input)).to(h.dtype)
        y = self.linearOut(h)
        # back to [seq_len, batch, ny]
        return y.permute(1, 0, 2).contiguous()


class MambaStreamflowModelFiLMHybrid(nn.Module):
    """Mamba model with hybrid FiLM: full concat through Mamba, static-only FiLM modulation.

    Unlike MambaStreamflowModelFiLM which splits dynamics/statics before Mamba,
    this hybrid version keeps the original concat path (Mamba sees all 40 features)
    and adds FiLM modulation afterward using only static attributes.

    This provides Mamba with full information while still allowing basin-level
    gain/bias adjustment via FiLM for improved KGE β.

    Uses MambaResidualBlock for consistent architecture with residual variant.
    """

    def __init__(
        self,
        *,
        nx,
        ny,
        hiddenSize,
        numLayers=2,
        dState=64,
        dConv=4,
        expand=2,
        headdim=64,
        mambaVersion=3,
        dropout=0.1,
        n_dynamic=5,
        n_static=35,
        residual_scale_init=0.1,
        **kwargs,
    ):
        _validate_mamba_version(mambaVersion)
        if mambaVersion != 3:
            raise ValueError(
                f"MambaStreamflowModelFiLMHybrid only supports mambaVersion=3, got {mambaVersion}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MambaStreamflowModelFiLMHybrid requires CUDA (no CPU fallback)"
            )
        if Mamba3 is None:
            raise ImportError("Mamba3 is not available in current mamba_ssm installation.")

        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.mambaVersion = mambaVersion
        self.n_dynamic = n_dynamic
        self.n_static = n_static
        self.cpu_fallback = False

        # FULL concat into the sequence path (like standard Mamba)
        self.linearIn = nn.Linear(nx, hiddenSize)  # nx = 40 (full concat)
        self.normIn = RMSNorm(hiddenSize)

        # Extra static branch for FiLM only
        self.film = StaticFiLM(n_static, hiddenSize, dropout=min(dropout, 0.2))

        d_inner = expand * hiddenSize
        original_headdim = headdim
        invalid_v3_headdim = (
            headdim is None or headdim < 32 or headdim not in (32, 64)
        )
        if headdim is None or d_inner % headdim != 0 or invalid_v3_headdim:
            for candidate in [64, 32]:
                if d_inner % candidate == 0:
                    headdim = candidate
                    break
            else:
                headdim = 32
            if original_headdim is not None:
                print(f"[WARNING] headdim={original_headdim} is incompatible with d_inner={d_inner} "
                      f"or Mamba-{mambaVersion}. Automatically adjusted to headdim={headdim}.")

        # Handle padding for Mamba-3
        mamba_hidden_size = hiddenSize
        padded_inner = math.ceil(d_inner / headdim) * headdim
        mamba_hidden_size = math.ceil(padded_inner / expand)
        padded_inner = expand * mamba_hidden_size
        if mamba_hidden_size != hiddenSize:
            print(f"[INFO] Padding Mamba-3 internal hidden size from {hiddenSize} "
                  f"to {mamba_hidden_size} for headdim={headdim}.")

        # Create residual blocks
        self.blocks = nn.ModuleList(
            [
                MambaResidualBlock(
                    hidden_size=mamba_hidden_size,
                    d_state=dState,
                    expand=expand,
                    headdim=headdim,
                    dropout=dropout,
                    residual_scale_init=residual_scale_init,
                )
                for _ in range(numLayers)
            ]
        )

        self.linearOut = nn.Linear(mamba_hidden_size, ny)
        self.name = "Mamba3StreamflowModelFiLMHybrid"

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        # x: [seq_len, batch, nx] = [T, B, 40] (already concat'd by select_subset)
        # Extract static attributes for FiLM only
        x_stat = x[0, :, self.n_dynamic :]  # [B, n_static] (same every timestep)

        # Process FULL concat through Mamba (uses ALL 40 features)
        h = self.normIn(self.linearIn(x))  # [T, B, H]
        h = h.permute(1, 0, 2).contiguous()  # [B, T, H]

        # Apply residual blocks
        for block in self.blocks:
            h = block(h)

        # Apply FiLM modulation with static attributes only
        h = self.film(h, x_stat)  # [B, T, H]

        y = self.linearOut(h)  # [B, T, ny]
        # back to [seq_len, batch, ny]
        return y.permute(1, 0, 2).contiguous()


class StaticFiLM(nn.Module):
    """Feature-wise Linear Modulation: static attributes produce per-channel γ and β to modulate hidden states."""

    def __init__(self, n_static, hidden_size, dropout=0.1):
        super().__init__()
        mid = max(hidden_size // 2, 32)
        self.mlp = nn.Sequential(
            nn.Linear(n_static, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 2 * hidden_size),  # → γ, β
        )
        # Initialize as identity: γ≈0, β≈0 → h'≈h
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h, s):
        # h: [B, T, H],  s: [B, n_static]
        gamma, beta = self.mlp(s).chunk(2, dim=-1)  # [B, H] each
        return h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class MambaStreamflowModelFiLM(nn.Module):
    """Mamba model with FiLM modulation for static/dynamic attribute separation.

    Splits input into dynamic (time-varying) and static (basin attributes) features.
    Dynamics go through Mamba layers; statics modulate via FiLM after the stack.
    This improves KGE β by letting SSM focus on temporal patterns while static MLP
    learns basin-level offsets/scales.

    Uses MambaResidualBlock for consistent architecture with residual variant.
    """

    def __init__(
        self,
        *,
        nx,
        ny,
        hiddenSize,
        numLayers=2,
        dState=64,
        dConv=4,
        expand=2,
        headdim=64,
        mambaVersion=3,
        dropout=0.1,
        n_dynamic=5,
        n_static=35,
        residual_scale_init=0.1,
        **kwargs,
    ):
        _validate_mamba_version(mambaVersion)
        if mambaVersion != 3:
            raise ValueError(
                f"MambaStreamflowModelFiLM only supports mambaVersion=3, got {mambaVersion}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MambaStreamflowModelFiLM requires CUDA (no CPU fallback)"
            )
        if Mamba3 is None:
            raise ImportError("Mamba3 is not available in current mamba_ssm installation.")

        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.mambaVersion = mambaVersion
        self.n_dynamic = n_dynamic
        self.n_static = n_static
        self.cpu_fallback = False

        # Dynamics only → sequence path
        self.dyn_in = nn.Linear(n_dynamic, hiddenSize)
        self.norm_in = RMSNorm(hiddenSize)

        # Statics once → FiLM modulation
        self.film = StaticFiLM(n_static, hiddenSize, dropout=dropout)

        d_inner = expand * hiddenSize
        original_headdim = headdim
        invalid_v3_headdim = (
            headdim is None or headdim < 32 or headdim not in (32, 64)
        )
        if headdim is None or d_inner % headdim != 0 or invalid_v3_headdim:
            for candidate in [64, 32]:
                if d_inner % candidate == 0:
                    headdim = candidate
                    break
            else:
                headdim = 32
            if original_headdim is not None:
                print(f"[WARNING] headdim={original_headdim} is incompatible with d_inner={d_inner} "
                      f"or Mamba-{mambaVersion}. Automatically adjusted to headdim={headdim}.")

        # Handle padding for Mamba-3
        mamba_hidden_size = hiddenSize
        padded_inner = math.ceil(d_inner / headdim) * headdim
        mamba_hidden_size = math.ceil(padded_inner / expand)
        padded_inner = expand * mamba_hidden_size
        if mamba_hidden_size != hiddenSize:
            print(f"[INFO] Padding Mamba-3 internal hidden size from {hiddenSize} "
                  f"to {mamba_hidden_size} for headdim={headdim}.")

        # Create residual blocks
        self.blocks = nn.ModuleList(
            [
                MambaResidualBlock(
                    hidden_size=mamba_hidden_size,
                    d_state=dState,
                    expand=expand,
                    headdim=headdim,
                    dropout=dropout,
                    residual_scale_init=residual_scale_init,
                )
                for _ in range(numLayers)
            ]
        )

        self.linearOut = nn.Linear(mamba_hidden_size, ny)
        self.name = "Mamba3StreamflowModelFiLM"

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        # x: [seq_len, batch, nx] = [T, B, 40] (concatenated dynamics + statics)
        # Split into dynamic and static features
        x_dyn = x[..., : self.n_dynamic]  # [T, B, n_dynamic]
        x_stat = x[0, :, self.n_dynamic :]  # [B, n_static] (same every timestep)

        # Process dynamics through Mamba
        h = self.norm_in(self.dyn_in(x_dyn))  # [T, B, H]
        h = h.permute(1, 0, 2).contiguous()  # [B, T, H]

        # Apply residual blocks
        for block in self.blocks:
            h = block(h)

        # Apply FiLM modulation with static attributes
        h = self.film(h, x_stat)  # [B, T, H]

        y = self.linearOut(h)  # [B, T, ny]
        # back to [seq_len, batch, ny]
        return y.permute(1, 0, 2).contiguous()


class StaticPrefixEncoder(nn.Module):
    """Encodes static basin attributes into a handful of "context tokens"
    that get prepended to the dynamic sequence, instead of being re-injected
    as constant features at every timestep.
    """

    def __init__(self, n_static, hidden_size, n_prefix_tokens=2, dropout=0.1):
        super().__init__()
        self.n_prefix_tokens = n_prefix_tokens
        self.hidden_size = hidden_size
        mid = max(hidden_size, 32)
        self.mlp = nn.Sequential(
            nn.Linear(n_static, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, n_prefix_tokens * hidden_size),
        )

    def forward(self, s):
        # s: [B, n_static] -> [B, n_prefix_tokens, hidden_size]
        out = self.mlp(s)
        return out.view(s.shape[0], self.n_prefix_tokens, self.hidden_size)


class StaticAdapter(nn.Module):
    """Small bottleneck layers that inject static information into the sequence.

    Uses a parameter-efficient adapter pattern where static attributes are
    projected to generate gating parameters that modulate the adapter pathway.
    """

    def __init__(self, hidden_size, n_static, bottleneck=32, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck = bottleneck
        
        # Static attributes generate gating parameters for the adapter
        self.static_gate = nn.Sequential(
            nn.Linear(n_static, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_size),
            nn.Sigmoid()
        )
        
        # Adapter pathway with bottleneck
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_size)
        )
        # Initialize adapter to be near-identity initially
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, h, static):
        # h: [B, T, H], static: [B, n_static]
        gate = self.static_gate(static)  # [B, hidden_size]
        gate = gate.unsqueeze(1)  # [B, 1, H]
        adapted = self.adapter(h)  # [B, T, H]
        return h + adapted * gate  # [B, T, H]


class MambaResidualBlock(nn.Module):
    """Dedicated residual block for Mamba layers with scaling and dropout.

    This abstraction makes it easier to experiment with:
    - residual scaling (learnable scale parameter)
    - layer dropout
    - FiLM conditioning
    - gated residuals
    - per-layer diagnostics
    """

    def __init__(
        self,
        hidden_size,
        d_state,
        expand,
        headdim,
        dropout=0.1,
        residual_scale_init=0.1,
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.mamba = Mamba3(
            d_model=hidden_size,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            # dtype=torch.float32,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Learnable residual scale parameter
        self.scale = nn.Parameter(torch.tensor(residual_scale_init))

    def forward(self, h):
        """Apply residual connection with scaled Mamba update.

        Args:
            h: Input tensor [batch, seq, hidden_size]

        Returns:
            Output tensor [batch, seq, hidden_size] with residual connection
        """
        # Normalize input
        h_norm = self.norm(h)
        # Convert dtype for Mamba3 if needed
        h_norm = h_norm.to(next(self.mamba.in_proj.parameters()))
        # Apply Mamba
        update = self.mamba(h_norm)
        # Convert back to original dtype
        update = update.to(h)
        # Apply dropout and scaling, then residual connection
        return h + self.scale * self.dropout(update)


class MambaStreamflowModelAdapter(nn.Module):
    """Mamba model with adapter layers for static attribute injection.

    Splits input into dynamic (time-varying) and static (basin attributes) features.
    Dynamics go through Mamba layers; statics are injected via parameter-efficient
    adapter layers that modulate the sequence at each layer.

    This provides a lightweight alternative to FiLM with potentially better
    parameter efficiency and different inductive biases.
    """

    def __init__(
        self,
        *,
        nx,
        ny,
        hiddenSize,
        numLayers=2,
        dState=64,
        dConv=4,
        expand=2,
        headdim=64,
        mambaVersion=3,
        dropout=0.1,
        n_dynamic=5,
        n_static=35,
        bottleneck=32,
        residual_scale_init=0.1,
        **kwargs,
    ):
        _validate_mamba_version(mambaVersion)
        if mambaVersion != 3:
            raise ValueError(
                f"MambaStreamflowModelAdapter only supports mambaVersion=3, got {mambaVersion}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MambaStreamflowModelAdapter requires CUDA (no CPU fallback)"
            )
        if Mamba3 is None:
            raise ImportError("Mamba3 is not available in current mamba_ssm installation.")

        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.mambaVersion = mambaVersion
        self.n_dynamic = n_dynamic
        self.n_static = n_static
        self.cpu_fallback = False

        d_inner = expand * hiddenSize
        original_headdim = headdim
        invalid_v3_headdim = (
            headdim is None or headdim < 32 or headdim not in (32, 64)
        )
        if headdim is None or d_inner % headdim != 0 or invalid_v3_headdim:
            for candidate in [64, 32]:
                if d_inner % candidate == 0:
                    headdim = candidate
                    break
            else:
                headdim = 32
            if original_headdim is not None:
                print(f"[WARNING] headdim={original_headdim} is incompatible with d_inner={d_inner} "
                      f"or Mamba-{mambaVersion}. Automatically adjusted to headdim={headdim}.")

        # Handle padding for Mamba-3
        mamba_hidden_size = hiddenSize
        padded_inner = math.ceil(d_inner / headdim) * headdim
        mamba_hidden_size = math.ceil(padded_inner / expand)
        padded_inner = expand * mamba_hidden_size
        if mamba_hidden_size != hiddenSize:
            print(f"[INFO] Padding Mamba-3 internal hidden size from {hiddenSize} "
                  f"to {mamba_hidden_size} for headdim={headdim}.")

        # Dynamics only → sequence path
        self.dyn_in = nn.Linear(n_dynamic, mamba_hidden_size)
        self.norm_in = RMSNorm(mamba_hidden_size)

        # Statics injected via adapter layers
        self.adapters = nn.ModuleList(
            [
                StaticAdapter(mamba_hidden_size, n_static, bottleneck, dropout)
                for _ in range(numLayers)
            ]
        )

        # Create residual blocks
        self.blocks = nn.ModuleList(
            [
                MambaResidualBlock(
                    hidden_size=mamba_hidden_size,
                    d_state=dState,
                    expand=expand,
                    headdim=headdim,
                    dropout=dropout,
                    residual_scale_init=residual_scale_init,
                )
                for _ in range(numLayers)
            ]
        )

        self.linearOut = nn.Linear(mamba_hidden_size, ny)
        self.name = "Mamba3StreamflowModelAdapter"

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        # x: [seq_len, batch, nx] = [T, B, 40] (concatenated dynamics + statics)
        # Split into dynamic and static features
        x_dyn = x[..., : self.n_dynamic]  # [T, B, n_dynamic]
        x_stat = x[0, :, self.n_dynamic :]  # [B, n_static] (same every timestep)

        # Process dynamics through Mamba
        h = self.norm_in(self.dyn_in(x_dyn))  # [T, B, H]
        h = h.permute(1, 0, 2).contiguous()  # [B, T, H]

        # Apply residual blocks with adapter injection
        for block, adapter in zip(self.blocks, self.adapters):
            h = block(h)
            h = adapter(h, x_stat)  # Inject static info via adapter

        y = self.linearOut(h)  # [B, T, ny]
        # back to [seq_len, batch, ny]
        return y.permute(1, 0, 2).contiguous()


class MambaStreamflowModelResidualBlocks(nn.Module):
    """Mamba model with dedicated residual block abstraction.

    Uses MambaResidualBlock for each layer, providing:
    - Learnable residual scaling
    - Layer-wise dropout
    - Cleaner architecture for experimentation
    - Easier to add FiLM, gated residuals, diagnostics

    CUDA-only (no CPU fallback).
    """

    def __init__(
        self,
        *,
        nx,
        ny,
        hiddenSize,
        numLayers=2,
        dState=64,
        dConv=4,
        expand=2,
        headdim=64,
        mambaVersion=3,
        dropout=0.1,
        residual_scale_init=0.1,
        **kwargs,
    ):
        _validate_mamba_version(mambaVersion)
        if mambaVersion != 3:
            raise ValueError(
                f"MambaStreamflowModelResidualBlocks only supports mambaVersion=3, got {mambaVersion}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MambaStreamflowModelResidualBlocks requires CUDA (no CPU fallback)"
            )
        if Mamba3 is None:
            raise ImportError("Mamba3 is not available in current mamba_ssm installation.")

        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.mambaVersion = mambaVersion
        self.cpu_fallback = False

        # Compute headdim with same logic as base model
        d_inner = expand * hiddenSize
        original_headdim = headdim
        invalid_v3_headdim = (
            headdim is None or headdim < 32 or headdim not in (32, 64)
        )
        if headdim is None or d_inner % headdim != 0 or invalid_v3_headdim:
            for candidate in [64, 32]:
                if d_inner % candidate == 0:
                    headdim = candidate
                    break
            else:
                headdim = 32
            if original_headdim is not None:
                print(f"[WARNING] headdim={original_headdim} is incompatible with d_inner={d_inner} "
                      f"or Mamba-{mambaVersion}. Automatically adjusted to headdim={headdim}.")

        # Handle padding for Mamba-3
        mamba_hidden_size = hiddenSize
        padded_inner = math.ceil(d_inner / headdim) * headdim
        mamba_hidden_size = math.ceil(padded_inner / expand)
        padded_inner = expand * mamba_hidden_size
        if mamba_hidden_size != hiddenSize:
            print(f"[INFO] Padding Mamba-3 internal hidden size from {hiddenSize} "
                  f"to {mamba_hidden_size} for headdim={headdim}.")

        self.linearIn = nn.Linear(nx, mamba_hidden_size)
        self.normIn = RMSNorm(mamba_hidden_size)

        # Create residual blocks
        self.blocks = nn.ModuleList(
            [
                MambaResidualBlock(
                    hidden_size=mamba_hidden_size,
                    d_state=dState,
                    expand=expand,
                    headdim=headdim,
                    dropout=dropout,
                    residual_scale_init=residual_scale_init,
                )
                for _ in range(numLayers)
            ]
        )

        self.linearOut = nn.Linear(mamba_hidden_size, ny)
        self.name = "Mamba3StreamflowModelResidualBlocks"

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        # x: [seq_len, batch, nx]
        h = self.normIn(self.linearIn(x))
        # Mamba expects [batch, seq, dim]
        h = h.permute(1, 0, 2)

        # Apply residual blocks
        for block in self.blocks:
            h = block(h)

        y = self.linearOut(h)
        # back to [seq_len, batch, ny]
        return y.permute(1, 0, 2)


class MambaStreamflowModelPrefixTokens(nn.Module):
    """Mamba model that primes the SSM state with basin identity via prefix
    tokens, instead of concatenating static attributes onto every timestep.

    Static attributes are encoded once into `n_prefix_tokens` embedding
    vectors and prepended to the dynamic-forcing sequence (like a prompt) —
    the SSM scans [prefix tokens, day 1, day 2, ..., day T] left to right, so
    every dynamic timestep already has basin context "in state" by the time
    it's processed, instead of having to re-derive a constant static signal
    at every one of T steps. The prefix positions are dropped from the
    output before the final projection.

    Interface matches MambaStreamflowModel:
        model = MambaStreamflowModelPrefixTokens(nx=nx, ny=ny, hiddenSize=H,
                                                  n_dynamic=5, n_static=35)
        yp = model(x)   # x: [seq_len, batch, nx] -> yp: [seq_len, batch, ny]
    """

    def __init__(
        self,
        *,
        nx,
        ny,
        hiddenSize,
        numLayers=2,
        dState=64,
        dConv=4,
        expand=2,
        headdim=64,
        mambaVersion=3,
        dropout=0.1,
        n_dynamic=5,
        n_static=35,
        n_prefix_tokens=2,
        **kwargs,
    ):
        _validate_mamba_version(mambaVersion)
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.mambaVersion = mambaVersion
        self.n_dynamic = n_dynamic
        self.n_static = n_static
        self.n_prefix_tokens = n_prefix_tokens
        self.cpu_fallback = not torch.cuda.is_available()

        # Dynamics only go through the per-timestep input projection —
        # statics never touch this path.
        self.dyn_in = nn.Linear(n_dynamic, hiddenSize)
        self.norm_in = RMSNorm(hiddenSize)

        # Statics, encoded once per basin into a few "context tokens".
        self.static_encoder = StaticPrefixEncoder(
            n_static, hiddenSize, n_prefix_tokens=n_prefix_tokens,
            dropout=min(dropout, 0.2),
        )

        d_inner = expand * hiddenSize
        original_headdim = headdim
        if headdim is None or d_inner % headdim != 0:
            # Try common headdim values in descending order
            for candidate in [64, 32, 16, 8]:
                if d_inner % candidate == 0:
                    headdim = candidate
                    break
            else:
                # Fallback to the largest divisor <= 32
                headdim = 8  # Always works as a fallback
            if original_headdim is not None:
                print(f"[WARNING] headdim={original_headdim} is incompatible with d_inner={d_inner}. "
                      f"Automatically adjusted to headdim={headdim} for divisibility.")

        if self.cpu_fallback:
            self.layers = nn.ModuleList(
                [_CpuSequenceBlock(hiddenSize, dConv, expand) for _ in range(numLayers)]
            )
            self.name = f"Mamba{mambaVersion}StreamflowModelPrefixTokens_CPUFallback"
        elif mambaVersion == 3:
            if Mamba3 is None:
                raise ImportError("Mamba3 is not available in current mamba_ssm installation.")
            self.layers = nn.ModuleList(
                [
                    Mamba3(
                        d_model=hiddenSize,
                        d_state=dState,
                        expand=expand,
                        headdim=headdim,
                        dtype=torch.float32,
                        dropout=0.0,
                    )
                    for _ in range(numLayers)
                ]
            )
            self.name = "Mamba3StreamflowModelPrefixTokens"
        elif mambaVersion == 2:
            if Mamba2 is None:
                raise ImportError("Mamba2 is not available in current mamba_ssm installation.")
            self.layers = nn.ModuleList(
                [
                    Mamba2(
                        d_model=hiddenSize,
                        d_state=dState,
                        d_conv=dConv,
                        expand=expand,
                        headdim=headdim,
                    )
                    for _ in range(numLayers)
                ]
            )
            self.name = "Mamba2StreamflowModelPrefixTokens"
        else:
            self.layers = nn.ModuleList(
                [
                    Mamba(
                        d_model=hiddenSize,
                        d_state=dState if dState <= 32 else 16,
                        d_conv=dConv,
                        expand=expand,
                    )
                    for _ in range(numLayers)
                ]
            )
            self.name = "MambaStreamflowModelPrefixTokens"

        self.norms = nn.ModuleList([RMSNorm(hiddenSize) for _ in range(numLayers)])
        self.dropout = nn.Dropout(dropout)
        self.norm_out = RMSNorm(hiddenSize)
        self.linearOut = nn.Linear(hiddenSize, ny)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        # x: [seq_len, batch, nx] = [T, B, 40] (concatenated dynamics + statics)
        x_dyn = x[..., : self.n_dynamic]           # [T, B, n_dynamic]
        x_stat = x[0, :, self.n_dynamic :]         # [B, n_static] (same every timestep)

        h = self.norm_in(self.dyn_in(x_dyn))       # [T, B, H]
        h = h.permute(1, 0, 2).contiguous()        # [B, T, H]

        prefix = self.static_encoder(x_stat)       # [B, n_prefix_tokens, H]
        h = torch.cat([prefix, h], dim=1)          # [B, n_prefix_tokens + T, H]

        for layer, norm in zip(self.layers, self.norms):
            layer_input = norm(h)
            if self.mambaVersion == 3 and not self.cpu_fallback:
                layer_input = layer_input.to(next(layer.in_proj.parameters()).dtype)
            h = h + self.dropout(layer(layer_input)).to(h.dtype)

        # Drop the prefix positions — keep only the T dynamic timesteps.
        h = h[:, self.n_prefix_tokens :, :]        # [B, T, H]

        y = self.linearOut(self.norm_out(h))       # [B, T, ny]
        return y.permute(1, 0, 2).contiguous()
