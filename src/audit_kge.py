"""B15 -- seasonal KGE audit (audit item 1.2).

The paper declares one KGE formula (Gupta et al. 2009:

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2),
    alpha = std_sim / std_obs,  beta = mean_sim / mean_obs)

and three code paths must agree on it:

  1. the headline metric, ``hydroDL.post.stat.statError()['KGE']``,
     called from ``inference.score_slice`` for every val/test number;
  2. the Fig 9 seasonal scorer, ``plot_figures.seasonal_basin_scores``;
  3. this script's own reference implementation, written from the equation
     above rather than copied from either code path.

Run ``python src/audit_kge.py`` -- exit 0 when the maximum absolute
discrepancy is <= 1e-12 (the paths differ only in float rounding: statError
correlates with ``scipy.stats.pearsonr``, the seasonal scorer with
``np.corrcoef``). The script also prints the 2012-form KGE to document that
it is a *different* number, i.e. the audit would catch a swap.
"""

from __future__ import annotations

import sys

import numpy as np

TOLERANCE = 1e-12


def kge_gupta2009(obs: np.ndarray, sim: np.ndarray) -> float:
    """Reference implementation, straight from the §3.4 equation.

    Days where either series is NaN are dropped first, exactly like
    ``statError`` does with its ``ind`` index.
    """
    o = np.asarray(obs, dtype=float)
    s = np.asarray(sim, dtype=float)
    valid = np.isfinite(o) & np.isfinite(s)
    o, s = o[valid], s[valid]
    if o.size < 2:
        return float("nan")
    mean_o, mean_s = np.mean(o), np.mean(s)
    std_o, std_s = np.std(o), np.std(s)
    if mean_o <= 0 or mean_s <= 0 or std_o <= 0 or std_s <= 0:
        return float("nan")
    r = float(np.corrcoef(o, s)[0, 1])
    return float(
        1.0 - np.sqrt((r - 1.0) ** 2 + (std_s / std_o - 1.0) ** 2
                      + (mean_s / mean_o - 1.0) ** 2)
    )


def _synthetic_basins(n_basins: int = 12, n_days: int = 1500, seed: int = 20090721):
    """Deterministic log-normal daily series with NaN gaps and seasonality."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    seasonal = 1.0 + 0.6 * np.sin(2 * np.pi * t / 365.0)
    obs = np.zeros((n_basins, n_days))
    sim = np.zeros((n_basins, n_days))
    for b in range(n_basins):
        scale = 10.0 ** (b - n_basins / 2)  # wet to dry basins
        obs[b] = scale * seasonal * np.exp(rng.normal(0.0, 0.35, n_days))
        noise = rng.normal(0.0, 0.05 + 0.05 * b / n_basins, n_days)
        sim[b] = obs[b] * np.exp(noise) + scale * rng.normal(0.0, 0.02, n_days)
    # realistic gaps: 3% of days missing, never the same day in every basin
    gap = rng.random((n_basins, n_days)) < 0.03
    obs[gap] = np.nan
    sim[gap] = np.nan
    # a pathological basin: constant zero flow
    obs[-1] = 0.0
    sim[-1] = 0.0
    return obs, sim


def main() -> int:
    from hydroDL.post import stat
    from plot_figures import seasonal_basin_scores

    obs, sim = _synthetic_basins()
    n_basins, n_days = obs.shape
    month = (
        (np.arange(n_days) % 365) * 12 // 365 + 1
    )  # Jan..Dec, good enough to build masks
    season_months = {
        "Winter": {12, 1, 2}, "Spring": {3, 4, 5},
        "Summer": {6, 7, 8}, "Autumn": {9, 10, 11},
    }

    # 1. headline path vs reference, full series, all basins.
    # statError emits warnings on the deliberate constant-zero basin; they are
    # the expected NaN result, not a defect, so keep the audit output clean.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        headline = stat.statError(sim, obs)["KGE"]
        kge12 = stat.statError(sim, obs)["KGE12"]
    ref_full = np.array(
        [kge_gupta2009(obs[b], sim[b]) for b in range(n_basins)]
    )
    ok = np.isfinite(ref_full)
    d_head = np.abs(headline[ok] - ref_full[ok]).max()

    # 2. seasonal path vs reference, every season, every basin
    d_season = 0.0
    n_scored = 0
    for months in season_months.values():
        mask = np.isin(month, list(months))
        for b in range(n_basins):
            valid = mask & np.isfinite(obs[b]) & np.isfinite(sim[b])
            if valid.sum() < 2:
                continue
            got = seasonal_basin_scores(
                obs[b], sim[b], mask, np.inf, -np.inf, ["KGE"]
            )["KGE"]
            want = kge_gupta2009(obs[b][valid], sim[b][valid])
            if np.isnan(got) and np.isnan(want):
                continue
            d_season = max(d_season, abs(got - want))
            n_scored += 1

    print(f"basins x days: {n_basins} x {n_days}")
    print(f"headline statError.KGE vs reference : max |d| = {d_head:.3e}")
    print(f"seasonal scorer vs reference        : max |d| = {d_season:.3e} "
          f"({n_scored} basin-seasons)")
    finite = ok & np.isfinite(kge12)
    print(f"2012-form KGE (must differ)         : max |KGE12-KGE| = "
          f"{np.abs(kge12[finite] - headline[finite]).max():.3e}")

    worst = max(d_head, d_season)
    if worst <= TOLERANCE:
        print(f"PASS: all KGE paths agree with the Gupta (2009) formula "
              f"to <= {TOLERANCE:g} (zero to machine precision)")
        return 0
    print(f"FAIL: max discrepancy {worst:.3e} exceeds {TOLERANCE:g}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
