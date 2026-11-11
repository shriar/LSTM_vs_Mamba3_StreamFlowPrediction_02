"""Numerical helpers shared by validation scoring and test evaluation.

Keeping these in one torch-only module is deliberate: ``train_val.py`` and
``eval_test.py`` must scale inputs, invert the runoff transform and compute
metrics identically, or the validation choice and the test number would not be
on the same scale.  Nothing here reads a split or a checkpoint; callers pass
arrays and models in.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from hydroDL.post import stat

from constants import ATTR_LST, forcing_variables
from data_utils import PREPROCESSED_DIR, basin_norm, transform_by_statistics

CFS_TO_M3S = 0.0283168


def load_scaler() -> dict:
    """Load the train-fitted scaler written by ``data_utils.py --build``."""
    path = PREPROCESSED_DIR / "scaler.json"
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing; run first: python src/data_utils.py --build"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def combined_stats(scaler: dict) -> dict:
    """Merge attribute and series statistics into one name -> stats mapping."""
    stats = dict(scaler.get("attribute_statistics") or {})
    stats.update(scaler.get("series_statistics") or {})
    missing = [
        name
        for name in list(forcing_variables()) + ["runoff"] + ATTR_LST
        if name not in stats
    ]
    if missing:
        raise SystemExit(f"scaler.json is missing statistics for {missing[:5]}")
    return stats


def scale_series(x: np.ndarray, names: list[str], stats: dict) -> np.ndarray:
    """Normalise 3-D forcing/runoff arrays; missing inputs become 0 after scaling."""
    out = transform_by_statistics(x, names, stats, [], True)
    out[np.isnan(out)] = 0.0
    return out


def scale_attrs(c: np.ndarray, stats: dict) -> np.ndarray:
    """Normalise the 2-D static attribute matrix; missing values become 0."""
    out = transform_by_statistics(c, ATTR_LST, stats, [], True)
    out[np.isnan(out)] = 0.0
    return out


def target_normalised(
    y_cfs: np.ndarray,
    area: np.ndarray,
    p_mean: np.ndarray,
    stats: dict,
) -> np.ndarray:
    """Convert cfs -> basin-dimensionless runoff -> train-fitted normalisation."""
    dimensionless = basin_norm(y_cfs, area, p_mean, True)
    mean, std = stats["runoff"][2:]
    return (dimensionless - mean) / std


def inverse_runoff(
    pred_norm: np.ndarray,
    area: np.ndarray,
    p_mean: np.ndarray,
    stats: dict,
) -> np.ndarray:
    """Invert the target normalisation and return streamflow in m3/s."""
    mean, std = stats["runoff"][2:]
    dimensionless = pred_norm * std + mean
    cfs = basin_norm(dimensionless, area, p_mean, False)
    return cfs * CFS_TO_M3S


def model_device(model: torch.nn.Module) -> torch.device:
    """Device of a loaded module, independent of the caller's config object."""
    return next(model.parameters()).device


def predict_array(
    model: torch.nn.Module,
    x: np.ndarray,
    c: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Batched forward pass returning ``(basins, time, ny)`` in model space.

    The model is put in eval mode for the pass and restored afterwards, so a
    callback inside training does not leave dropout disabled by accident.
    """
    ngrid, nt, _ = x.shape
    ny = int(getattr(model, "ny", 1))
    out = np.empty((ngrid, nt, ny), dtype=np.float32)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, ngrid, batch_size):
                stop = min(start + batch_size, ngrid)
                xb = np.swapaxes(x[start:stop], 1, 0)          # (nt, b, nx)
                cb = np.repeat(c[start:stop][:, None, :], nt, axis=1)
                cb = np.swapaxes(cb, 1, 0)                     # (nt, b, nc)
                inputs = torch.from_numpy(
                    np.concatenate([xb, cb], axis=2)
                ).float().to(model_device(model))
                out[start:stop] = (
                    model(inputs).detach().cpu().numpy().swapaxes(0, 1)
                )
    finally:
        model.train(was_training)
    return out


def nse_loss(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    std_array: np.ndarray,
) -> float:
    """Masked ``NSELossBatch`` in numpy, on the shared normalisation."""
    p = pred_norm[:, :, 0]
    t = target_norm[:, :, 0]
    mask = ~np.isnan(t)
    if not mask.any():
        return float("nan")
    std = std_array[:, 0][:, None] if std_array.ndim == 2 else std_array
    residual = (p - t) / (np.broadcast_to(std, t.shape) + 0.1)
    return float(np.mean(residual[mask] ** 2))


def score_slice(
    model: torch.nn.Module,
    x: np.ndarray,
    c: np.ndarray,
    y_norm: np.ndarray,
    obs_m3s: np.ndarray,
    *,
    drop: int,
    label_drop: int,
    area: np.ndarray,
    p_mean: np.ndarray,
    stats: dict,
    std_array: np.ndarray | None,
    batch_size: int,
) -> dict:
    """Forward, invert, and score one slice.

    ``drop`` removes warm-up predictions.  ``label_drop`` removes the matching
    warm-up labels; validation/test slices are stored already scored-length, so
    their callers pass 0.
    """
    batch = max(1, int(batch_size))
    pred_norm = predict_array(model, x, c, batch)[:, drop:, :]
    target_norm = y_norm[:, label_drop:, :]
    pred_m3s = inverse_runoff(pred_norm, area, p_mean, stats)
    obs = obs_m3s[:, label_drop:, :]
    if pred_m3s.shape[1] != obs.shape[1]:
        raise RuntimeError(
            f"prediction/observation misalignment: {pred_m3s.shape} vs {obs.shape}"
        )

    metric_dict = stat.statError(pred_m3s.squeeze(-1), obs.squeeze(-1))
    nse = metric_dict.get("NSE")
    kge = metric_dict.get("KGE")
    return {
        "loss": (
            nse_loss(pred_norm, target_norm, std_array)
            if std_array is not None
            else float("nan")
        ),
        "median_nse": float(np.nanmedian(nse)) if nse is not None else float("nan"),
        "median_kge": float(np.nanmedian(kge)) if kge is not None else float("nan"),
        "n_basins_scored": int(np.sum(~np.isnan(nse))) if nse is not None else 0,
        "pred_m3s": pred_m3s,
        "obs_m3s": obs,
        "metrics": metric_dict,
    }
