"""Paired per-basin significance tests for the LSTM vs. Mamba-3 comparison.

Superseded for reporting by ``stats_multiseed.py`` (task B6), which runs the
same constructions on the current per-cell result files with seed aggregation,
bootstrap CIs and TOST, and which is the only producer of the reported
statistics. This module is retained as the documented definition of the tests
and as the single implementation of ``blocked_permutation_p``. Its run paths
point at the pre-revision 30-epoch / H80 runs and no longer exist.

Reproduces the basin counts reported in the Results section (461 of 671 for
NSE, 390 of 671 for KGE) and reports three tests of the win share, that is the
fraction of basins on which the LSTM scores higher.

  1. Sign test on the paired win/loss counts. Under the null hypothesis that
     the two architectures are equally good, each basin is an independent coin
     flip, so the win count is Binomial(n, 0.5). Needs no distributional
     assumption and is immune to the handful of basins where Mamba-3 collapses,
     which inflate the mean paired difference well above the median.

  2. Paired Wilcoxon signed-rank test on the metric values. Reported for
     reference only. Like the sign test it assumes independent pairs, so for
     spatially correlated basins it is anticonservative and should not be the
     headline p-value.

  3. Cluster-level blocked permutation test. Neighbouring CAMELS basins share
     climate, soils and a single regional model, so treating 671 basins as 671
     independent draws understates the null variance. Here each spatial block
     is one unit: every basin in a block is relabelled together, so a block
     that uniformly favours the LSTM contributes as a single unit rather than
     as many weakly correlated ones. The null win share is centred on 0.5 with
     a variance set by block sizes and internal homogeneity, and the p-value is
     two-sided.

Why joint flips rather than a within-block permutation: permuting labels inside
a block leaves that block's win count unchanged, so the global win count is
invariant and the null is degenerate, returning p = 1 for any data set.
Independent per-basin flips are the opposite failure, since they reproduce the
plain sign test and ignore the blocks entirely. Joint flipping is the
construction that genuinely widens the null for clustered data.

The p-value depends on the block size, so every grid size is reported and the
spread across them is the honest statement of precision. No single grid size is
privileged, and a p-value quoted at one block size alone would be arbitrary.

Run from the experiment directory:

    python src/sign_test_lstm_vs_mamba.py

Writes paired_basin_metrics_sign_test.csv next to this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

# Configuration pair reported in the Results section: best LSTM (1 layer,
# Tier 1) against best Mamba-3 (3 layers, Tier 1), both at rho = 365,
# 30 epochs, seed 111111.
from paths import DATA_DIR, OUTPUT_ROOT

OUTPUT_DIR = OUTPUT_ROOT
LSTM_RUN = OUTPUT_DIR / "LSTM_1L" / "Tier1_200K_Ep30_Bs128_Rho365_H162_L1" / "All"
MAMBA_RUN = (OUTPUT_DIR / "Mamba3_3L"
             / "Tier1_Ep30_Variant-adapter_V3_head32_H80_L3_dConv4" / "All")
TOPO_FILE = DATA_DIR / "camels_attributes_v2.0" / "camels_topo.txt"

METRICS = ("NSE", "KGE")
TIE_TOL = 1e-9
BLOCK_SIZES_DEG = (3, 4, 5)
# 200,000 permutations so that a p-value near 3e-4 rests on roughly 60
# exceedances rather than 6, which makes the reported value stable enough to
# interpret. The minimum resolvable p is 1/(N_PERM + 1) = 5e-6.
N_PERM = 200_000
RNG_SEED = 20260928


def load_paired() -> pd.DataFrame:
    """Join the two runs on gauge_id, one row per basin."""
    paths = {"lstm": LSTM_RUN / "basin_metrics.csv",
             "mamba": MAMBA_RUN / "basin_metrics.csv"}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name} metrics: {path}")

    cols = ["gauge_id", *METRICS]
    frames = []
    for key, path in paths.items():
        frame = pd.read_csv(path, usecols=cols, dtype={"gauge_id": str})
        frame["gauge_id"] = frame["gauge_id"].str.strip().str.zfill(8)
        frames.append(frame.rename(columns={m: f"{m}_{key}" for m in METRICS}))

    paired = frames[0].merge(frames[1], on="gauge_id", how="inner")
    for metric in METRICS:
        paired[f"d{metric}"] = paired[f"{metric}_lstm"] - paired[f"{metric}_mamba"]
    return paired


def blocked_permutation_p(paired: pd.DataFrame, metric: str, cell_deg: int,
                          coords: pd.DataFrame) -> tuple[float, int, float, float]:
    """Two-sided p for the win share under a cluster-level joint-flip null.

    Every basin in a block is relabelled together, so the null respects the
    homogeneity of neighbouring basins. Per-basin independent flips would
    reproduce the plain sign test and make the block partition irrelevant.

    Returns (p_value, n_blocks, observed_win_share, null_sd).
    """
    codes, _ = pd.factorize(coords[f"bx{cell_deg}"].to_numpy(), sort=True)
    n_blocks = int(codes.max()) + 1

    win = (paired[f"d{metric}"].to_numpy() > TIE_TOL).astype(np.float64)
    sizes = np.bincount(codes, minlength=n_blocks).astype(np.float64)
    wins = np.bincount(codes, weights=win, minlength=n_blocks)

    n = len(paired)
    observed = wins.sum() / n

    # Keep the block as is, or swap every win in it for a loss and vice versa.
    # Expected wins per block under the null is sizes / 2, so the null share is
    # centred on 0.5.
    rng = np.random.default_rng(RNG_SEED)
    flip = rng.random((N_PERM, n_blocks)) < 0.5
    null_share = np.where(flip, sizes - wins, wins).sum(axis=1) / n

    # Two-sided: as extreme as the observed share under a null centred on 0.5.
    deviation = abs(observed - 0.5)
    p_value = (np.sum(np.abs(null_share - 0.5) >= deviation - 1e-12) + 1) / (N_PERM + 1)
    return p_value, n_blocks, observed, float(null_share.std())


def main() -> None:
    paired = load_paired()
    n = len(paired)
    print(f"paired basins: {n}\n")

    # Grid cells for the blocked permutation, from CAMELS gauge coordinates.
    coords = pd.DataFrame(index=paired.index)
    if TOPO_FILE.exists():
        topo = pd.read_csv(TOPO_FILE, sep=";", dtype=str,
                           usecols=["gauge_id", "gauge_lat", "gauge_lon"])
        topo["gauge_id"] = topo["gauge_id"].str.strip().str.zfill(8)
        topo["gauge_lat"] = pd.to_numeric(topo["gauge_lat"], errors="coerce")
        topo["gauge_lon"] = pd.to_numeric(topo["gauge_lon"], errors="coerce")
        coords = (paired[["gauge_id"]]
                  .merge(topo, on="gauge_id", how="left")
                  .set_index(paired.index)[["gauge_lat", "gauge_lon"]])
        for cell in BLOCK_SIZES_DEG:
            coords[f"bx{cell}"] = (np.floor((coords.gauge_lon + 125.0) / cell).astype("Int64")
                                   * 1000
                                   + np.floor((coords.gauge_lat - 25.0) / cell).astype("Int64"))
        have_coords = int(coords["gauge_lat"].notna().sum())
        print(f"basins with CAMELS coordinates: {have_coords} of {n}")
    else:
        print(f"coordinates unavailable ({TOPO_FILE}), blocked test skipped")
    coords = coords.dropna(subset=[f"bx{c}" for c in BLOCK_SIZES_DEG[:1]])

    summary = []
    for metric in METRICS:
        diff = paired[f"d{metric}"].to_numpy()
        a, b = paired[f"{metric}_lstm"].to_numpy(), paired[f"{metric}_mamba"].to_numpy()

        wins = int((diff > TIE_TOL).sum())
        losses = int((diff < -TIE_TOL).sum())
        ties = n - wins - losses
        sign_p = binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue
        wilcoxon_p = wilcoxon(a, b).pvalue

        print(f"=== {metric} (paired difference = LSTM - Mamba-3) ===")
        print(f"  median            LSTM {np.median(a):.4f}   Mamba-3 {np.median(b):.4f}")
        print(f"  median paired d   {np.median(diff):+.4f}")
        print(f"  mean   paired d   {diff.mean():+.4f}  (sd {diff.std():.4f})")
        print(f"  range             {diff.min():+.3f} to {diff.max():+.3f}")
        print(f"  percentiles       10% {np.percentile(diff, 10):+.3f}"
              f"   25% {np.percentile(diff, 25):+.3f}"
              f"   50% {np.median(diff):+.3f}"
              f"   75% {np.percentile(diff, 75):+.3f}"
              f"   90% {np.percentile(diff, 90):+.3f}")
        print(f"  LSTM better at    {wins}   Mamba-3 better at {losses}   ties {ties}")
        print(f"  sign test         p = {sign_p:.3e}")
        print(f"  Wilcoxon          p = {wilcoxon_p:.3e}")
        print(f"  catastrophic (value < 0), 2x2 contingency table:")
        l_fail, m_fail = a < 0, b < 0
        both_fail = int((l_fail & m_fail).sum())
        l_only = int((l_fail & ~m_fail).sum())
        m_only = int((~l_fail & m_fail).sum())
        neither = int((~l_fail & ~m_fail).sum())
        print(f"      {'':>14}  Mamba-3 < 0   Mamba-3 >= 0     total")
        print(f"    {'LSTM < 0':>14}  {both_fail:>10}   {l_only:>11}   {both_fail + l_only:>9}")
        print(f"    {'LSTM >= 0':>14}  {m_only:>10}   {neither:>11}   {m_only + neither:>9}")
        print(f"    {'total':>14}  {both_fail + m_only:>10}   {l_only + neither:>11}   {n:>9}")
        # The two inclusive totals overlap at `both_fail`, so comparing the totals
        # directly is misleading. McNemar's exact test uses only the discordant
        # basins, which is the quantity that actually differs between models.
        mcnemar_p = binomtest(m_only, m_only + l_only, 0.5,
                              alternative="two-sided").pvalue
        print(f"    McNemar exact p = {mcnemar_p:.3f} on {m_only + l_only} discordant"
              f" basins (Mamba-3 {m_only} vs LSTM {l_only})")
        if mcnemar_p > 0.05:
            print(f"    -> NOT significant; the failure counts are consistent with"
                  f" chance. Do not report as a difference.")
        print(f"  (report the median, not the mean: the top basins carry a large share"
              f" of the mean)")

        blocked = []
        for cell in BLOCK_SIZES_DEG:
            col = f"bx{cell}"
            if col not in coords or coords[col].isna().any():
                continue
            sub = paired.loc[coords.index]
            p_block, n_blocks, obs_share, null_sd = blocked_permutation_p(
                sub, metric, cell, coords)
            print(f"  blocked {cell}x{cell} deg  {n_blocks:3d} blocks"
                  f"   observed {100*obs_share:.1f}%"
                  f"   null sd {null_sd:.4f}   two-sided p = {p_block:.3e}")
            blocked.append(p_block)
            summary.append({"metric": metric, "test": f"blocked_{cell}deg",
                            "n_blocks": n_blocks, "p": p_block})
        if blocked:
            print(f"  -> p across block sizes: {min(blocked):.3e} to "
                  f"{max(blocked):.3e}  (block-size dependent, not a single value;"
                  f" floor is {1/(N_PERM+1):.3e})")

        summary.append({"metric": metric, "test": "sign", "n_blocks": np.nan,
                        "p": sign_p})
        summary.append({"metric": metric, "test": "wilcoxon_indep_assuming",
                        "n_blocks": np.nan, "p": wilcoxon_p})
        print()

    out = Path(__file__).resolve().parent / "paired_basin_metrics_sign_test.csv"
    paired.to_csv(out, index=False)
    print(f"saved {out}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
