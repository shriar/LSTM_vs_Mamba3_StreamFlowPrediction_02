"""B12 -- zero-flow & negative-Q rule audit (audit item 2.8).

The rule as implemented, everywhere:

  * **No clamping.** ``inference.inverse_runoff`` maps the network output to
    m3/s linearly; negative predictions are never raised to zero. Every
    reported metric -- NSE, KGE (hydroDL ``statError``), FHV, FLV -- consumes
    the same raw series. Nothing downstream clamps either.
  * **One epsilon.** FLV carries a ``+0.0001`` denominator for zero-flow
    basins, identical in hydroDL ``statError`` and
    ``evaluation.calculate_hydrologic_signatures``. FHV carries none, in both.
  * Display-only clamping exists in ``plot_figures`` (log-axis flow-duration
    panels use ``FIG8_EPSILON``); it never enters a number.

This script quantifies what the no-clamp rule means on a real test cell: it
recomputes the headline metrics on raw vs clamped predictions and reports the
deltas, so the paper can state the effect instead of assuming it.

Usage: python src/audit_zero_flow.py <run_dir_with_pred.npz>  (default: the
G1 cell). Exit 0 always -- this audit reports, it does not gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOL = 0.0


def _load(run_dir: Path):
    data = np.load(run_dir / "pred.npz")
    return (
        data["pred_m3s"].astype(np.float64),
        data["obs_m3s"].astype(np.float64),
    )


def main(argv: list[str] | None = None) -> int:
    run_dir = Path(
        argv[0] if argv else "results/test/g1_T1_L1_Seed1"
    ).resolve()
    pred, obs = _load(run_dir)

    n_neg = int(np.sum(pred < 0))
    frac = n_neg / pred.size
    most_negative = float(pred.min())

    raw = stat_error(pred, obs)
    clamped_pred = np.maximum(pred, 0.0)
    clamped = stat_error(clamped_pred, obs)

    print(f"run dir           : {run_dir}")
    print(f"series            : {pred.shape[0]} basins x {pred.shape[1]} days")
    print(f"negative preds    : {n_neg} ({frac:.6%}), min {most_negative:.4f} m3/s")
    print(f"{'metric':<14}{'raw':>12}{'clamped':>12}{'delta':>12}")
    for name in ("NSE", "KGE", "FHV", "FLV"):
        r, c = raw[name], clamped[name]
        ok = np.isfinite(r) & np.isfinite(c)
        med_r = float(np.nanmedian(r[ok]))
        med_c = float(np.nanmedian(c[ok]))
        print(f"{name:<14}{med_r:>12.4f}{med_c:>12.4f}{med_c - med_r:>+12.4f}")

    # consistency: the epsilon rule is byte-identical in both implementations?
    from evaluation import calculate_hydrologic_signatures

    _, fhv_e, flv_e = calculate_hydrologic_signatures(obs, pred)
    d_flv = np.nanmedian(np.abs(flv_e - raw["FLV"]))
    d_fhv = np.nanmedian(np.abs(fhv_e - raw["FHV"]))
    print(f"evaluation.py vs hydroDL on the same series: "
          f"median |dFHV| = {d_fhv:.2e}, median |dFLV| = {d_flv:.2e}")
    print("rule: no clamping in any metric; single FLV epsilon (+1e-4); "
          "raw negatives retained and quantified above")
    return 0


def stat_error(pred: np.ndarray, obs: np.ndarray) -> dict[str, np.ndarray]:
    """Headline metrics + the two flow biases, per basin."""
    from hydroDL.post import stat

    out = stat.statError(pred, obs)
    # FHV/FLV in hydroDL are PBiashigh/PBiaslow
    return {
        "NSE": out["NSE"],
        "KGE": out["KGE"],
        "FHV": out["FHV"],
        "FLV": out["FLV"],
    }


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
