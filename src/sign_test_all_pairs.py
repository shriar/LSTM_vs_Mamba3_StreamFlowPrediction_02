"""Paired basin-level tests for all nine matched LSTM / Mamba-3 pairs.

Superseded for reporting by ``stats_multiseed.py`` (task B6), which covers the
same nine pairs plus the cross-depth headline pair, aggregates seeds, adds
bootstrap CIs and TOST, and reads the current result files. This module keeps
the documented construction; its run paths point at the pre-revision 30-epoch /
H80 runs and no longer exist.

Extends sign_test_lstm_vs_mamba.py, which covers only the single pair reported
in the headline sentence (1-layer Tier 1 LSTM against 3-layer Tier 1 Adapter),
to the full 3x3 matched ladder of three depths by three capacity tiers.

For every matched (depth, tier) cell this reports, per metric:

  * the median of each architecture and their difference
  * the number and share of basins on which the LSTM scores higher
  * the sign-test p-value, which assumes the 671 basins are independent
  * the cluster-level blocked permutation p-value, which does not

The blocked test is the one to quote. It relabels whole spatial blocks jointly
so that the null reflects the fact that neighbouring basins share climate,
soils and a single regional model. It is two-sided, and it is reported at three
block sizes because the result depends on the block size; the spread across
those sizes is the honest precision statement, and no single size is privileged.

Test construction and the reasons the two obvious alternatives fail are
documented in sign_test_lstm_vs_mamba.py. The test itself is imported from
there so that there is only one implementation.

Run from the experiment directory:

    python src/sign_test_all_pairs.py

Writes paired_pvalues_all_pairs.csv next to this script.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sign_test_lstm_vs_mamba import (BLOCK_SIZES_DEG, TIE_TOL, N_PERM,
                                     blocked_permutation_p)

from paths import DATA_DIR, OUTPUT_ROOT, RESULTS_DIR

OUTPUT_DIR = OUTPUT_ROOT
TOPO_FILE = DATA_DIR / "camels_attributes_v2.0" / "camels_topo.txt"
OUT_CSV = RESULTS_DIR / "paired_pvalues_all_pairs.csv"

METRICS = ("NSE", "KGE")
DEPTHS = (1, 2, 3)
TIERS = (1, 2, 3)

RUN_RE = re.compile(r"^Tier(?P<tier>\d)_.*_L(?P<depth>\d)(?:_dConv4)?(?:_Rho\d+)?(?:_Seed\d+)?$")
ARCH_RE = re.compile(r"^(?P<arch>LSTM|Mamba3)_(?P<depth>\d)L$")


def discover_runs() -> dict[tuple[int, int, str], Path]:
    """Map (depth, tier, architecture) to the run directory holding its metrics."""
    runs: dict[tuple[int, int, str], Path] = {}
    for arch_dir in sorted(OUTPUT_DIR.iterdir()):
        if not arch_dir.is_dir():
            continue
        m = ARCH_RE.match(arch_dir.name)
        if not m:
            continue
        arch = "lstm" if m.group("arch") == "LSTM" else "mamba"
        for run_dir in sorted(arch_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            r = RUN_RE.match(run_dir.name)
            if not r:
                continue
            depth = int(r.group("depth"))
            tier = int(r.group("tier"))
            if depth != int(m.group("depth")):
                continue
            metrics_file = run_dir / "All" / "basin_metrics.csv"
            if metrics_file.exists():
                runs[(depth, tier, arch)] = run_dir / "All"
    return runs


def load_pair(lstm_dir: Path, mamba_dir: Path) -> pd.DataFrame:
    """Join one matched pair on gauge_id, with paired differences attached."""
    frames = []
    for key, path in (("lstm", lstm_dir / "basin_metrics.csv"),
                      ("mamba", mamba_dir / "basin_metrics.csv")):
        f = pd.read_csv(path, usecols=["gauge_id", *METRICS], dtype={"gauge_id": str})
        f["gauge_id"] = f["gauge_id"].str.strip().str.zfill(8)
        frames.append(f.rename(columns={m: f"{m}_{key}" for m in METRICS}))
    paired = frames[0].merge(frames[1], on="gauge_id", how="inner")
    for metric in METRICS:
        paired[f"d{metric}"] = paired[f"{metric}_lstm"] - paired[f"{metric}_mamba"]
    return paired


def block_grid(paired: pd.DataFrame) -> pd.DataFrame:
    """Spatial block id for each basin, one column per block size in degrees.

    The index matches `paired`. Basins without coordinates are dropped, so the
    caller selects the same rows from both frames with `paired.loc[coords.index]`.

    Coordinates are joined **by gauge_id**, never by row position: the earlier
    version assigned `lat.to_numpy()` positionally, which coincided with the
    correct join only because both files happened to be sorted by gauge_id.
    Any reordering would silently relabel the blocks.
    """
    topo = pd.read_csv(TOPO_FILE, sep=";", dtype=str,
                       usecols=["gauge_id", "gauge_lat", "gauge_lon"])
    topo["gauge_id"] = topo["gauge_id"].str.strip().str.zfill(8)
    topo["gauge_lat"] = pd.to_numeric(topo["gauge_lat"], errors="coerce")
    topo["gauge_lon"] = pd.to_numeric(topo["gauge_lon"], errors="coerce")

    merged = paired[["gauge_id"]].copy()
    merged["gauge_id"] = merged["gauge_id"].astype(str).str.strip().str.zfill(8)
    merged = merged.merge(topo, on="gauge_id", how="left")
    merged.index = paired.index
    for cell in BLOCK_SIZES_DEG:
        merged[f"bx{cell}"] = (np.floor((merged.gauge_lon + 125.0) / cell)
                               * 1000
                               + np.floor((merged.gauge_lat - 25.0) / cell))
    keep = merged.dropna(subset=[f"bx{c}" for c in BLOCK_SIZES_DEG]).index
    return merged.loc[keep]


def analyse(paired: pd.DataFrame, coords: pd.DataFrame) -> list[dict]:
    """All statistics for one matched pair, one record per metric."""
    sub = paired.loc[coords.index]
    n = len(sub)
    rows = []
    for metric in METRICS:
        diff = sub[f"d{metric}"].to_numpy()
        a = sub[f"{metric}_lstm"].to_numpy()
        b = sub[f"{metric}_mamba"].to_numpy()
        wins = int((diff > TIE_TOL).sum())
        losses = int((diff < -TIE_TOL).sum())
        sign_p = stats.binomtest(wins, wins + losses, 0.5,
                                 alternative="two-sided").pvalue

        blocked = {}
        for cell in BLOCK_SIZES_DEG:
            p_b, n_blocks, _obs, _sd = blocked_permutation_p(
                sub, metric, cell, coords)
            blocked[cell] = p_b

        rows.append({
            "metric": metric,
            "n_basins": n,
            "median_lstm": float(np.median(a)),
            "median_mamba": float(np.median(b)),
            "median_delta": float(np.median(diff)),
            "wins_lstm": wins,
            "losses_mamba": losses,
            "win_share_pct": 100.0 * wins / n,
            "sign_p": sign_p,
            **{f"blocked_p_{cell}deg": blocked[cell] for cell in BLOCK_SIZES_DEG},
            "blocked_p_min": min(blocked.values()),
            "blocked_p_max": max(blocked.values()),
        })
    return rows


def fmt_p(p: float) -> str:
    if p <= 1.0 / (N_PERM + 1) + 1e-12:
        return "<=5e-06"
    return f"{p:.2e}"


def main() -> None:
    runs = discover_runs()
    print(f"discovered {len(runs)} run directories "
          f"(expect 18: 9 pairs x 2 architectures)\n")

    missing = [(d, t) for d in DEPTHS for t in TIERS
               if (d, t, "lstm") not in runs or (d, t, "mamba") not in runs]
    if missing:
        raise SystemExit(f"incomplete ladder, missing pairs: {missing}")

    all_rows = []
    for depth in DEPTHS:
        for tier in TIERS:
            paired = load_pair(runs[(depth, tier, "lstm")],
                               runs[(depth, tier, "mamba")])
            coords = block_grid(paired)
            for row in analyse(paired, coords):
                row["depth"] = depth
                row["tier"] = tier
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)

    print("=== medians and win shares for all nine matched pairs ===")
    print(f"{'pair':>7} {'metric':>5} {'LSTM':>7} {'Mamba':>7} {'delta':>7} "
          f"{'wins':>5} {'share':>7} {'sign p':>10} {'blocked p (3/4/5 deg)':>34}")
    for depth in DEPTHS:
        for tier in TIERS:
            for metric in METRICS:
                r = df[(df.depth == depth) & (df.tier == tier)
                       & (df.metric == metric)].iloc[0]
                b = (f"{fmt_p(r[f'blocked_p_{c}deg'])}"
                     for c in BLOCK_SIZES_DEG)
                print(f"{depth}L T{tier} {metric:>5} {r.median_lstm:7.4f} "
                      f"{r.median_mamba:7.4f} {r.median_delta:+7.4f} "
                      f"{r.wins_lstm:5d} {r.win_share_pct:6.1f}% "
                      f"{r.sign_p:10.2e}   {' / '.join(b)}")

    print("\n=== summary ===")
    n_at_floor = int((df.blocked_p_min <= 1.0 / (N_PERM + 1) + 1e-12).sum())
    print(f"  metric-pair combinations at the resolution floor: {n_at_floor}"
          f" of {len(df)}")
    worst = df.loc[df.blocked_p_max.idxmax()]
    print(f"  least significant combination: {int(worst.depth)}L T{int(worst.tier)}"
          f" {worst.metric}, p up to {worst.blocked_p_max:.2e}")
    for metric in METRICS:
        s = df[df.metric == metric]
        print(f"  {metric}: win share {s.win_share_pct.min():.1f}% to "
              f"{s.win_share_pct.max():.1f}%, blocked p "
              f"{s.blocked_p_min.min():.2e} to {s.blocked_p_max.max():.2e}")
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
