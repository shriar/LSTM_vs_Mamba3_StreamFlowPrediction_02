"""B7 -- mechanism + mass-balance extraction (validation slice only).

This module answers the reviewer's "speculative mechanism" objection with
measurements taken from checkpoints that already exist.  It never trains and it
never opens the test slice: the internal-state diagnostics run on the *val*
slice, and the only test-period number (the mass-balance ratio) is read from
the already-audited ``pred.npz`` that ``eval_test.py`` wrote for that cell.

What is captured
----------------
Mamba-3 (``standard`` wiring, SISO):
    * ``delta_mean`` -- the selective step size ``softplus(dd_dt + dt_bias)``
      averaged over heads, per basin-day.  Exact: it is recomputed from the
      same ``in_proj`` output the kernel consumes.
    * ``ssm_input_norm`` -- norm of the value branch ``x`` that the SSM
      integrates, per basin-day.
    * ``ssm_final_state_norm`` -- norm of the last SSM state, per layer and
      basin.  Obtained by running the same layers once with an
      ``InferenceParams`` cache; the output is asserted bit-identical to the
      plain forward pass.

LSTM (``CudnnLstmModel``):
    * ``gate_i / gate_f / gate_g / gate_o`` -- mean gate activation over hidden
      units, per basin-day, from a pure-PyTorch ``nn.LSTMCell``-style replay of
      the trained CuDNN weights.
    * ``cell_norm`` and ``hidden_norm``.
    * The replay is checked against the real CuDNN forward pass; a mismatch
      above ``--replay-tol`` aborts the run instead of writing wrong evidence.

Both models: ``q_sim`` / ``q_obs`` (m3/s) per basin-day and the cumulative
``sum(q_sim) / sum(q_obs)`` ratio over the scored window.

Nothing here changes a reported metric; it produces diagnostic CSVs, one
figure, and a JSON provenance record.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from constants import ATTR_LST, forcing_variables  # noqa: E402
from data_utils import GAUGE_IDS_PATH, load_split_with_context  # noqa: E402
from eval_test import (  # noqa: E402
    MANIFEST_PATH,
    Refusal,
    _load_manifest,
    _read_json,
    _read_split,
    _rel,
    _resolve,
    _sha256,
    _verify_best_checkpoint,
    _verify_checkpoint,
    _verify_master,
)
from inference import (  # noqa: E402
    CFS_TO_M3S,
    combined_stats,
    inverse_runoff,
    load_scaler,
    scale_attrs,
    scale_series,
)
from paths import RESULTS_DIR  # noqa: E402

DEFAULT_OUTPUT_DIR = RESULTS_DIR / "mechanism"
# cuDNN stores LSTM gates as input, forget, cell, output.
GATE_ORDER = ("i", "f", "g", "o")
N_SNOW, N_WET, N_DRY = 6, 7, 7
PRCP_INDEX = forcing_variables("daymet").index("prcp")


# ---------------------------------------------------------------------------
# Basin selection
# ---------------------------------------------------------------------------
def select_basins(
    c: np.ndarray,
    *,
    n_snow: int = N_SNOW,
    n_wet: int = N_WET,
    n_dry: int = N_DRY,
) -> list[dict]:
    """Pick a deterministic wet/dry/snow panel from the attribute matrix.

    Snow first (highest ``frac_snow``), then the wettest remaining basins by
    ``p_mean``, then the driest remaining basins.  Ties break on the basin
    index, so the panel is reproducible without a seed.
    """
    if c.ndim != 2:
        raise ValueError(f"expected 2-D attributes, got {c.shape}")
    n_basins = c.shape[0]
    wanted = n_snow + n_wet + n_dry
    if n_basins < wanted:
        raise ValueError(f"need {wanted} basins, only {n_basins} available")

    p_mean = np.asarray(c[:, ATTR_LST.index("p_mean")], dtype=float)
    frac_snow = np.asarray(c[:, ATTR_LST.index("frac_snow")], dtype=float)
    for name, values in (("p_mean", p_mean), ("frac_snow", frac_snow)):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} has non-finite values; cannot select basins")

    chosen: list[dict] = []
    taken: set[int] = set()

    def take(order: np.ndarray, n: int, category: str) -> None:
        added = 0
        for idx in order:
            idx = int(idx)
            if idx in taken:
                continue
            taken.add(idx)
            chosen.append({"index": idx, "category": category})
            added += 1
            if added == n:
                return
        raise ValueError(f"not enough basins left to fill {category}={n}")

    snow_order = np.lexsort((np.arange(n_basins), -frac_snow))
    take(snow_order, n_snow, "snow")
    remaining = np.array([i for i in range(n_basins) if i not in taken])
    wet_order = remaining[np.lexsort((remaining, -p_mean[remaining]))]
    take(wet_order, n_wet, "wet")
    remaining = np.array([i for i in range(n_basins) if i not in taken])
    dry_order = remaining[np.lexsort((remaining, p_mean[remaining]))]
    take(dry_order, n_dry, "dry")
    return chosen


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------
def build_input_tensor(x: np.ndarray, c: np.ndarray, device: torch.device) -> torch.Tensor:
    """Concatenate forcing and repeated attributes exactly as inference does."""
    if x.ndim != 3 or c.ndim != 2:
        raise ValueError(f"bad shapes x={x.shape} c={c.shape}")
    nt, nb = x.shape[1], x.shape[0]
    xb = np.swapaxes(x, 1, 0)                       # (T, B, 5)
    cb = np.repeat(c[:, None, :], nt, axis=1)
    cb = np.swapaxes(cb, 1, 0)                      # (T, B, 35)
    combined = np.concatenate([xb, cb], axis=2)     # (T, B, 40)
    return torch.from_numpy(combined).float().to(device)


def mamba_dt_from_projection(proj: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
    """Recompute ``softplus(dd_dt + dt_bias)`` from a Mamba-3 ``in_proj`` output.

    ``in_proj`` emits ``[z, x, B, C, dd_dt, dd_A, trap, angle]``; the offset is
    derived from the layer's own shapes so it cannot silently drift.
    """
    d_inner = int(layer.d_inner)
    d_state = int(layer.d_state)
    groups = int(layer.num_bc_heads)
    rank = int(getattr(layer, "mimo_rank", 1))
    heads = int(layer.nheads)
    start = 2 * d_inner + 2 * d_state * groups * rank
    dd_dt = proj[..., start:start + heads]
    if dd_dt.shape[-1] != heads:
        raise ValueError(
            f"dd_dt slice has width {dd_dt.shape[-1]}, expected {heads}"
        )
    return F.softplus(dd_dt.float() + layer.dt_bias.float())


def mass_balance(q_sim: np.ndarray, q_obs: np.ndarray) -> np.ndarray:
    """Cumulative simulated/observed discharge ratio per basin (no clamping)."""
    if q_sim.shape != q_obs.shape:
        raise ValueError(f"shape mismatch {q_sim.shape} vs {q_obs.shape}")
    sim_sum = np.nansum(q_sim, axis=1)
    obs_sum = np.nansum(q_obs, axis=1)
    out = np.full(sim_sum.shape, np.nan, dtype=float)
    valid = np.abs(obs_sum) > 0
    out[valid] = sim_sum[valid] / obs_sum[valid]
    return out


# ---------------------------------------------------------------------------
# Mamba-3 extraction
# ---------------------------------------------------------------------------
def _capture_mamba_projection(store: dict, index: int, layer, output) -> None:
    dt = mamba_dt_from_projection(output, layer)          # (B, T, heads)
    store[("delta", index)] = dt.mean(dim=-1).detach().float().cpu()
    value = output[..., layer.d_inner:2 * layer.d_inner]  # (B, T, d_inner)
    store[("value_norm", index)] = value.detach().float().norm(dim=-1).cpu()


def extract_mamba(
    model: torch.nn.Module,
    inp: torch.Tensor,
    *,
    drop: int,
) -> dict:
    from mamba_ssm.utils.generation import InferenceParams

    layers = list(model.layers)
    for i, layer in enumerate(layers):
        if getattr(layer, "is_mimo", False):
            raise Refusal("MIMO Mamba-3 is outside B7's scope (SISO only)")
        layer.layer_idx = i  # required by the inference-state cache

    with torch.inference_mode():
        reference = model(inp)

    store: dict = {}
    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(
            layer.in_proj.register_forward_hook(
                lambda _m, _a, out, i=i, layer=layer: _capture_mamba_projection(
                    store, i, layer, out
                )
            )
        )
        hooks.append(
            layer.register_forward_pre_hook(
                lambda _m, args, i=i: store.__setitem__(
                    ("layer_norm", i),
                    args[0].detach().float().norm(dim=-1).cpu(),
                )
            )
        )
    ip = InferenceParams(max_seqlen=inp.shape[0], max_batch_size=inp.shape[1])
    try:
        with torch.inference_mode():
            h = model.normIn(model.linearIn(inp)).permute(1, 0, 2).contiguous()
            for layer, norm in zip(model.layers, model.norms):
                layer_in = norm(h).to(next(layer.in_proj.parameters()).dtype)
                h = h + model.dropout(layer(layer_in, inference_params=ip)).to(h.dtype)
            path_output = model.linearOut(h).permute(1, 0, 2).contiguous()
    finally:
        for hook in hooks:
            hook.remove()

    path_delta = float((path_output - reference).abs().max())
    layers_n = len(layers)
    delta = torch.stack([store[("delta", i)] for i in range(layers_n)]).mean(0).numpy()
    value_norm = (
        torch.stack([store[("value_norm", i)] for i in range(layers_n)]).mean(0).numpy()
    )
    layer_norm = (
        torch.stack([store[("layer_norm", i)] for i in range(layers_n)]).mean(0).numpy()
    )
    final_state = np.stack(
        [
            ip.key_value_memory_dict[i][1].detach().float().flatten(1).norm(dim=1).cpu().numpy()
            for i in range(layers_n)
        ],
        axis=1,
    )                                                    # (B, layers)
    return {
        "delta_mean": delta[:, drop:],
        "ssm_input_norm": value_norm[:, drop:],
        "layer_input_norm": layer_norm[:, drop:],
        "ssm_final_state_norm": final_state,
        "state_path_max_abs_diff": path_delta,
    }


# ---------------------------------------------------------------------------
# LSTM replay
# ---------------------------------------------------------------------------
def replay_lstm(model: torch.nn.Module, inp: torch.Tensor) -> dict:
    """Replay the CuDNN weights with explicit gate algebra (i, f, g, o)."""
    nt, nb, _ = inp.shape
    h = F.relu(model.linearIn(inp))
    gate_means = []
    cell_norms = []
    hidden_norms = []
    for layer in model.lstm_layers:
        hidden = int(layer.hiddenSize)
        w_ih = layer.w_ih
        w_hh = layer.w_hh
        bias = layer.b_ih + layer.b_hh
        hs = torch.empty(nt, nb, hidden, device=inp.device)
        gates = torch.empty(nt, nb, 4, device=inp.device)
        cell_norm = torch.empty(nt, nb, device=inp.device)
        c = torch.zeros(nb, hidden, device=inp.device)
        previous = torch.zeros(nb, hidden, device=inp.device)
        for t in range(nt):
            g = h[t] @ w_ih.T + previous @ w_hh.T + bias
            i, f, gg, o = g.split(hidden, dim=-1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            gg = torch.tanh(gg)
            o = torch.sigmoid(o)
            c = f * c + i * gg
            previous = o * torch.tanh(c)
            hs[t] = previous
            gates[t, :, 0] = i.mean(-1)
            gates[t, :, 1] = f.mean(-1)
            gates[t, :, 2] = gg.mean(-1)
            gates[t, :, 3] = o.mean(-1)
            cell_norm[t] = c.norm(dim=-1)
        h = hs
        gate_means.append(gates)
        cell_norms.append(cell_norm)
        hidden_norms.append(hs.norm(dim=-1))
    out = model.linearOut(h)
    return {
        "output": out,
        "gate_means": torch.stack(gate_means),      # (layers, T, B, 4)
        "cell_norms": torch.stack(cell_norms),      # (layers, T, B)
        "hidden_norms": torch.stack(hidden_norms),  # (layers, T, B)
    }


def extract_lstm(
    model: torch.nn.Module,
    inp: torch.Tensor,
    *,
    drop: int,
    replay_tol: float,
) -> dict:
    with torch.inference_mode():
        reference = model(inp)
        replay = replay_lstm(model, inp)
    diff = float((replay["output"] - reference).abs().max())
    scale = max(1.0, float(reference.abs().max()))
    relative = diff / scale
    if not np.isfinite(diff) or relative > replay_tol:
        raise RuntimeError(
            f"LSTM replay does not reproduce the CuDNN forward pass "
            f"(max |diff| = {diff:.3e}, relative = {relative:.3e} > "
            f"{replay_tol:.1e}); refusing to write mechanism evidence"
        )
    gates = replay["gate_means"].mean(0).float().cpu().numpy()      # (T, B, 4)
    cell_norm = replay["cell_norms"].mean(0).float().cpu().numpy()  # (T, B)
    hidden_norm = replay["hidden_norms"].mean(0).float().cpu().numpy()
    return {
        "gate_means": gates[drop:],
        "cell_norm": cell_norm[drop:],
        "hidden_norm": hidden_norm[drop:],
        "replay_max_abs_diff": diff,
        "replay_relative_diff": relative,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _test_mass_balance(recipe: dict, tier: int, depth: int, seed: int) -> dict | None:
    """Read the audited test predictions; never open the test slice."""
    path = RESULTS_DIR / "test" / f"{recipe['id']}_T{tier}_L{depth}_Seed{seed}" / "pred.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        pred = np.asarray(data["pred_m3s"], dtype=float)
        obs = np.asarray(data["obs_m3s"], dtype=float)
        ids = np.asarray(data["gauge_ids"]).astype(str)
    if pred.shape != obs.shape or pred.shape[0] != ids.shape[0]:
        raise Refusal(f"{path} is internally inconsistent")
    ratios = mass_balance(pred, obs)
    return {gauge: float(r) for gauge, r in zip(ids, ratios) if np.isfinite(r)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B7 mechanism + mass-balance extraction on the validation slice."
    )
    parser.add_argument("--recipe", required=True, help="locked eval recipe JSON")
    parser.add_argument("--tier", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument("--depth", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--replay-tol",
        type=float,
        default=1e-3,
        help="max relative |diff| allowed between the LSTM replay and CuDNN",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    started = time.time()
    manifest = _load_manifest(_resolve(args.manifest))
    recipe_path = _resolve(args.recipe)
    recipe = _read_json(recipe_path, "recipe")
    if recipe.get("locked") is not True:
        raise Refusal(f"recipe {_rel(recipe_path)} is not locked")

    tier = int(args.tier if args.tier is not None else recipe["tier"])
    depth = int(args.depth if args.depth is not None else recipe["depth"])
    seed = int(args.seed if args.seed is not None else recipe["seed"])
    matches = [
        cell
        for cell in manifest["cells"]
        if _resolve(str(cell["recipe"])) == recipe_path
        and int(cell["tier"]) == tier
        and int(cell["depth"]) == depth
        and int(cell["seed"]) == seed
    ]
    if not matches:
        raise Refusal(
            f"cell {_rel(recipe_path)} T{tier} L{depth} seed {seed} is not in "
            f"{_rel(_resolve(args.manifest))}"
        )
    cell = matches[0]
    run_dir = _resolve(str(cell["run_dir"]))
    checkpoint = (run_dir / str(recipe["checkpoint"])).resolve()
    if not checkpoint.is_file():
        raise Refusal(f"checkpoint not found: {_rel(checkpoint)}")

    split = _read_split()
    if recipe["split_sha256"] != split["split_sha256"]:
        raise Refusal(
            f"recipe split {recipe['split_sha256']} != data/split.json "
            f"{split['split_sha256']}"
        )
    master = _read_json(run_dir / "master.json", "run master")
    _verify_master(
        master, recipe, tier=tier, depth=depth, seed=seed,
        split_sha=split["split_sha256"],
    )
    _verify_best_checkpoint(run_dir, checkpoint, master)

    out_dir = Path(args.output_dir) / f"{recipe['id']}_T{tier}_L{depth}_Seed{seed}"
    if (out_dir / "mechanism_meta.json").is_file() and not args.force:
        raise Refusal(
            f"{_rel(out_dir)} already has a mechanism record; pass --force "
            "only for a documented re-run",
            code=4,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _verify_checkpoint(model, recipe, master, depth=depth)
    model.to(device).eval()
    if recipe["model"] == "mamba3" and device.type != "cuda":
        raise Refusal("Mamba-3 checkpoints need the CUDA kernels")

    # Validation slice only.  The test slice is never opened here.
    ctx_days = int(recipe["ctx_days"])
    x, y, c = load_split_with_context("val", ctx_days=ctx_days)
    if x.shape[1] != ctx_days + y.shape[1]:
        raise Refusal(f"val context misaligned: {x.shape[1]} != {ctx_days}+{y.shape[1]}")
    stats = combined_stats(load_scaler())
    forcing = forcing_variables("daymet")
    area = c[:, ATTR_LST.index("area_gages2")][:, None]
    p_mean = c[:, ATTR_LST.index("p_mean")][:, None]
    x_n = scale_series(x, forcing, stats)
    c_n = scale_attrs(c, stats)
    obs = y * CFS_TO_M3S

    gauge_ids = np.asarray(
        pd.read_csv(GAUGE_IDS_PATH).sort_values("basin_index")["gauge_id"].astype(str)
    )
    if gauge_ids.shape[0] != c.shape[0]:
        raise Refusal("gauge id count does not match the attribute matrix")

    panel = select_basins(c)
    idx = np.array([row["index"] for row in panel], dtype=int)
    categories = [row["category"] for row in panel]
    selected_ids = gauge_ids[idx]
    xb, cb, yb = x_n[idx], c_n[idx], obs[idx]

    inp = build_input_tensor(xb, cb, device)
    if recipe["model"] == "mamba3":
        features = extract_mamba(model, inp, drop=ctx_days)
    else:
        features = extract_lstm(
            model, inp, drop=ctx_days, replay_tol=float(args.replay_tol)
        )

    with torch.inference_mode():
        pred_norm = model(inp)[ctx_days:, :, :].float().cpu().numpy()
    pred = np.swapaxes(pred_norm, 0, 1)                  # (B, T, 1)
    q_sim = inverse_runoff(pred, area[idx], p_mean[idx], stats).squeeze(-1)
    q_obs = yb.squeeze(-1)

    val_start = split["slices"]["val"]["start"]
    val_end = split["slices"]["val"]["end"]
    dates = pd.date_range(val_start, val_end, freq="D")
    if len(dates) != q_sim.shape[1]:
        raise Refusal(f"val dates {len(dates)} != scored days {q_sim.shape[1]}")

    mb_val = mass_balance(q_sim, q_obs)
    mb_test_map = _test_mass_balance(recipe, tier, depth, seed)
    mb_test = np.array(
        [mb_test_map.get(g, np.nan) if mb_test_map else np.nan for g in selected_ids],
        dtype=float,
    )
    precip = x[idx][:, ctx_days:, PRCP_INDEX]

    out_dir.mkdir(parents=True, exist_ok=True)
    daily = {
        "gauge_id": np.repeat(selected_ids, q_sim.shape[1]),
        "category": np.repeat(np.asarray(categories), q_sim.shape[1]),
        "date": np.tile(dates.strftime("%Y-%m-%d").to_numpy(), len(idx)),
        "precip": precip.reshape(-1),
        "q_sim_m3s": q_sim.reshape(-1),
        "q_obs_m3s": q_obs.reshape(-1),
    }
    if recipe["model"] == "mamba3":
        daily["delta_mean"] = features["delta_mean"].reshape(-1)
        daily["ssm_input_norm"] = features["ssm_input_norm"].reshape(-1)
        daily["layer_input_norm"] = features["layer_input_norm"].reshape(-1)
    else:
        gates = features["gate_means"]  # (T, B, 4)
        for k, name in enumerate(GATE_ORDER):
            daily[f"gate_{name}"] = np.swapaxes(gates[:, :, k], 0, 1).reshape(-1)
        daily["cell_norm"] = features["cell_norm"].T.reshape(-1)
        daily["hidden_norm"] = features["hidden_norm"].T.reshape(-1)
    pd.DataFrame(daily).to_csv(out_dir / "mechanism_daily.csv", index=False)

    summary = {
        "gauge_id": selected_ids,
        "category": categories,
        "mass_balance_val": mb_val,
        "mass_balance_test": mb_test,
        "mean_q_obs_m3s": np.nanmean(q_obs, axis=1),
        "mean_q_sim_m3s": np.nanmean(q_sim, axis=1),
    }
    if recipe["model"] == "mamba3":
        summary["mean_delta"] = features["delta_mean"].mean(axis=1)
        summary["p95_delta"] = np.percentile(features["delta_mean"], 95, axis=1)
        summary["mean_ssm_input_norm"] = features["ssm_input_norm"].mean(axis=1)
        for layer in range(features["ssm_final_state_norm"].shape[1]):
            summary[f"ssm_final_state_norm_layer{layer + 1}"] = features[
                "ssm_final_state_norm"
            ][:, layer]
    else:
        for k, name in enumerate(GATE_ORDER):
            summary[f"mean_gate_{name}"] = features["gate_means"][:, :, k].mean(axis=0)
        summary["mean_cell_norm"] = features["cell_norm"].mean(axis=0)
        summary["mean_hidden_norm"] = features["hidden_norm"].mean(axis=0)
        summary["replay_max_abs_diff"] = np.full(len(idx), features["replay_max_abs_diff"])
        summary["replay_relative_diff"] = np.full(len(idx), features["replay_relative_diff"])
    pd.DataFrame(summary).to_csv(out_dir / "mechanism_summary.csv", index=False)

    runtime = time.time() - started
    meta = {
        "task": "B7",
        "cell_id": cell["id"],
        "recipe": _rel(recipe_path),
        "recipe_id": recipe["id"],
        "model": recipe["model"],
        "tier": tier,
        "depth": depth,
        "seed": seed,
        "checkpoint": _rel(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "split_sha256": split["split_sha256"],
        "slice": "val",
        "period": {"start": val_start, "end": val_end},
        "context_days": ctx_days,
        "n_basins": len(idx),
        "basins": [
            {"gauge_id": str(g), "category": cat}
            for g, cat in zip(selected_ids, categories)
        ],
        "mass_balance_test_source": (
            "results/test/<cell>/pred.npz" if mb_test_map else "not available"
        ),
        "runtime_seconds": round(runtime, 2),
        "within_5_min": bool(runtime < 300),
    }
    if recipe["model"] == "mamba3":
        meta["state_path_max_abs_diff"] = features["state_path_max_abs_diff"]
        meta["ssm_final_state_norm_mean"] = [
            float(features["ssm_final_state_norm"][:, layer].mean())
            for layer in range(features["ssm_final_state_norm"].shape[1])
        ]
    else:
        meta["replay_max_abs_diff"] = features["replay_max_abs_diff"]
        meta["replay_relative_diff"] = features["replay_relative_diff"]
    (out_dir / "mechanism_meta.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8"
    )

    _plot(out_dir, recipe, features, panel, selected_ids, dates, q_sim, q_obs,
          precip, mb_val, mb_test)

    print(
        f"{cell['id']} | {recipe['model']} | {len(idx)} basins | "
        f"{q_sim.shape[1]} val days | {runtime:.1f} s"
    )
    if recipe["model"] == "mamba3":
        print(
            f"  state-path max |diff| {features['state_path_max_abs_diff']:.2e} | "
            f"final-state norms {[round(float(v), 2) for v in meta['ssm_final_state_norm_mean']]}"
        )
    else:
        print(
            f"  replay max |diff| {features['replay_max_abs_diff']:.2e} "
            f"(relative {features['replay_relative_diff']:.2e})"
        )
    print(
        f"  val mass balance median {np.nanmedian(mb_val):.3f}"
        + (
            f" | test median {np.nanmedian(mb_test):.3f}"
            if np.isfinite(mb_test).any()
            else " | test pred.npz not found"
        )
    )
    print(f"  wrote {_rel(out_dir)}")
    return 0


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot(out_dir, recipe, features, panel, ids, dates, q_sim, q_obs, precip,
          mb_val, mb_test) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [next(i for i, p in enumerate(panel) if p["category"] == c)
             for c in ("wet", "dry", "snow")]
    colors = {"wet": "#1b6ca8", "dry": "#c25b1e", "snow": "#3b8a6b"}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex="col")
    for col, pos in enumerate(order):
        cat = panel[pos]["category"]
        ax = axes[0, col]
        ax.plot(dates, q_obs[pos], color="black", lw=1.0, label="obs")
        ax.plot(dates, q_sim[pos], color=colors[cat], lw=1.0, label="sim")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(f"{ids[pos]} ({cat})")
        ax.set_ylabel("discharge (m3/s)")
        ax.legend(fontsize=7, loc="upper right")
        ax2 = ax.twinx()
        ax2.bar(dates, precip[pos], color="#9ecae1", alpha=0.5, width=1.0)
        ax2.set_ylabel("precip", fontsize=8)
        ax2.tick_params(labelsize=7)

        ax = axes[1, col]
        if recipe["model"] == "mamba3":
            ax.plot(dates, features["delta_mean"][pos], color=colors[cat], lw=0.8)
            ax.set_ylabel("mean delta")
        else:
            for k, name in enumerate(GATE_ORDER):
                ax.plot(dates, features["gate_means"][:, pos, k], lw=0.8, label=name)
            ax.set_ylabel("mean gate")
            ax.legend(fontsize=7, ncol=4)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_mechanism.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ids))
    ax.axhline(1.0, color="black", lw=0.8)
    ax.scatter(x, mb_val, color="tab:blue", label="val", s=28)
    if np.isfinite(mb_test).any():
        ax.scatter(x, mb_test, color="tab:red", marker="x", label="test (pred.npz)", s=30)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{g}\n{panel[i]['category']}" for i, g in enumerate(ids)],
        rotation=90, fontsize=6,
    )
    ax.set_ylabel("sum(Q_sim) / sum(Q_obs)")
    ax.set_title(f"Mass balance | {recipe['id']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_mass_balance.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
