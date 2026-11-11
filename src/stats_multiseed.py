"""Multi-seed + spatial-block statistics for the LSTM vs Mamba-3 ladder.

Regenerates every statistic the paper reports from the saved result files:

  * seed aggregation      -- mean/SD and median/IQR of the per-seed basin median,
                            per (tier, depth, architecture);
  * paired basin tests    -- sign test, Wilcoxon, paired t, and the joint
                            spatial-block permutation test, per (tier, depth);
  * bootstrap CI          -- 10 000-draw basin bootstrap for the reported gap;
  * TOST                  -- equivalence test at a +/- 0.01 NSE/KGE margin.

The reported gap is the mean over seeds of the per-seed basin median (the same
estimand as the seed ladder), so the printed medians and the printed delta
subtract exactly. The bootstrap resamples basins and recomputes that average.
The median of the per-basin paired difference is kept in the CSV as
``median_basin_delta`` for the sign/Wilcoxon tests but is *not* the reported
gap and must not be subtracted from the printed medians.

Two Mamba-3 basins blow up in one seed each (NSE -923 and -6.9e5). The
``n_blowups``/``*_ex_blowups`` columns count them (seed-averaged NSE below
-10 in either model) and show the medians move by <=0.0009 when they are
dropped. All mean-based quantities (``mean_delta``, ``paired_t_p``,
``tost_mean_p``) are dominated by those basins and are reference-only.

It reads only ``results/test/<cell>/basin_test_metrics.csv`` (the per-basin
test scores written by ``eval_test.py``) plus the CAMELS topo file for the
spatial blocks. It never trains, never scores predictions, and never opens a
split loader.

Statistical notes
-----------------
* Seeds are aggregated *after* the per-seed basin median, because the basin
  median is the reported statistic; a mean of per-seed means would answer a
  different question.
* Paired tests use seed-averaged per-basin scores (mean over the available
  seeds of each architecture). This keeps one test per (tier, depth) pair and
  avoids pretending that seeds are independent replicates of the basins.
* The block permutation relabels whole spatial blocks jointly. Independent
  per-basin flips reproduce the plain sign test and make the partition
  irrelevant; within-block permutation is degenerate (the block's win count is
  invariant, so the null never moves). Both alternatives are documented in
  ``sign_test_lstm_vs_mamba.py``, whose implementation is imported here so
  there is one construction, not two.
* Basin bootstrap is i.i.d. in basins by design (the plan asks for a
  10 000-draw basin bootstrap). Spatial dependence is handled by the block
  permutation test, not by the bootstrap.

Run from the experiment directory:

    python src/stats_multiseed.py

Writes:
    results/stats_seed_values.csv        one row per (cell, seed)
    results/stats_seed_aggregate.csv     one row per (tier, depth, arch)
    results/stats_pairwise.csv           one row per (tier, depth, metric)
    results/tables/tab_seed_ladder.tex
    results/tables/tab_pairwise_tests.tex
    results/tables/tab_headline.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EXP = Path(__file__).resolve().parent.parent
if str(EXP / "src") not in sys.path:
    sys.path.insert(0, str(EXP / "src"))

from sign_test_lstm_vs_mamba import (  # noqa: E402  (single implementation)
    BLOCK_SIZES_DEG,
    N_PERM,
    TIE_TOL,
    blocked_permutation_p,
)

RESULTS = EXP / "results"
TABLES = RESULTS / "tables"
TOPO_FILE = EXP / "data/camels_attributes_v2.0/camels_topo.txt"

METRICS = ("NSE", "KGE")
TIER_NAMES = {1: "T1", 2: "T2", 3: "T3"}
N_BOOT = 10_000
BOOT_SEED = 20261105
TOST_MARGIN = 0.01
BLOWUP_NSE = -10.0
ARCH_LABEL = {"lstm": "LSTM", "mamba3": "Mamba-3"}
ARCH_ORDER = ("lstm", "mamba3")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_topo() -> pd.DataFrame:
    topo = pd.read_csv(TOPO_FILE, sep=";", dtype=str,
                       usecols=["gauge_id", "gauge_lat", "gauge_lon"])
    topo["gauge_id"] = topo["gauge_id"].str.strip().str.zfill(8)
    topo["gauge_lat"] = pd.to_numeric(topo["gauge_lat"], errors="coerce")
    topo["gauge_lon"] = pd.to_numeric(topo["gauge_lon"], errors="coerce")
    return topo


def load_cells(tiers: tuple[int, ...] | None = None) -> list[dict]:
    """Every completed C1 cell, with its per-basin test scores."""
    manifest = json.loads((EXP / "recipes/eval_manifest.json").read_text())
    cells = []
    for cell in manifest["cells"]:
        if not cell["id"].startswith(("C1a-", "C1b-", "C1c-")):
            continue
        if tiers and cell["tier"] not in tiers:
            continue
        arch = "mamba3" if "mamba" in cell["recipe"] else "lstm"
        recipe_id = json.loads((EXP / cell["recipe"]).read_text())["id"]
        test_dir = (EXP / "results/test"
                    / f"{recipe_id}_T{cell['tier']}_L{cell['depth']}_Seed{cell['seed']}")
        basin_file = test_dir / "basin_test_metrics.csv"
        if not basin_file.exists():
            continue
        frame = pd.read_csv(basin_file, usecols=["gauge_id", *METRICS],
                            dtype={"gauge_id": str})
        frame["gauge_id"] = frame["gauge_id"].str.strip().str.zfill(8)
        for metric in METRICS:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        cells.append({
            "id": cell["id"], "arch": arch, "tier": cell["tier"],
            "depth": cell["depth"], "seed": cell["seed"],
            "frame": frame, "test_dir": test_dir,
        })
    return cells


def seed_values(cells: list[dict]) -> pd.DataFrame:
    """One row per cell: the per-seed basin median for each metric."""
    rows = []
    for cell in cells:
        frame = cell["frame"]
        row = {"cell": cell["id"], "tier": cell["tier"], "depth": cell["depth"],
               "arch": cell["arch"], "seed": cell["seed"]}
        n = 0
        for metric in METRICS:
            values = frame[metric].dropna().to_numpy()
            n = len(values)
            row[f"median_{metric.lower()}"] = float(np.median(values)) if n else math.nan
            row[f"min_{metric.lower()}"] = float(np.min(values)) if n else math.nan
        nse = frame["NSE"].dropna().to_numpy()
        row["n_blowups"] = int((nse < BLOWUP_NSE).sum())
        row["n_basins"] = n
        _check_against_metrics(row, cell["test_dir"])
        rows.append(row)
    return pd.DataFrame(rows)


def _check_against_metrics(row: dict, test_dir: Path) -> None:
    """Warn if the per-basin median disagrees with test_metrics.csv."""
    metrics_file = test_dir / "test_metrics.csv"
    if not metrics_file.exists():
        return
    with metrics_file.open(newline="") as fh:
        meta = next(csv.DictReader(fh))
    for metric in METRICS:
        expected = float(meta[f"median_{metric.lower()}"])
        got = row[f"median_{metric.lower()}"]
        if not (math.isnan(got) or math.isnan(expected)) and abs(got - expected) > 1e-9:
            print(f"WARNING: {row['cell']} {metric} median {got!r} != "
                  f"test_metrics.csv {expected!r}", file=sys.stderr)


def aggregate_seeds(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/SD and median/IQR over seeds of the per-seed basin median."""
    rows = []
    for (tier, depth, arch), group in seed_df.groupby(["tier", "depth", "arch"]):
        row = {"tier": tier, "depth": depth, "arch": arch,
               "n_seeds": int(group["seed"].nunique())}
        for metric in METRICS:
            values = group[f"median_{metric.lower()}"].dropna().to_numpy()
            row[f"{metric.lower()}_mean"] = float(np.mean(values))
            row[f"{metric.lower()}_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else math.nan)
            row[f"{metric.lower()}_median"] = float(np.median(values))
            row[f"{metric.lower()}_iqr_low"] = float(np.percentile(values, 25))
            row[f"{metric.lower()}_iqr_high"] = float(np.percentile(values, 75))
        row["seeds"] = ",".join(str(s) for s in sorted(group["seed"].unique()))
        row["min_nse"] = (float(group["min_nse"].min())
                          if "min_nse" in group else math.nan)
        row["n_blowups"] = (int(group["n_blowups"].sum())
                            if "n_blowups" in group else 0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["tier", "depth", "arch"]).reset_index(drop=True)


def seed_matrix(cells: list[dict], tier: int, depth: int,
                arch: str) -> pd.DataFrame | None:
    """Per-basin, per-seed scores: index gauge_id, columns ``NSE_<seed>``."""
    group = [c for c in cells
             if c["tier"] == tier and c["depth"] == depth and c["arch"] == arch]
    if not group:
        return None
    acc: pd.DataFrame | None = None
    for cell in group:
        frame = cell["frame"].set_index("gauge_id")[list(METRICS)]
        frame.columns = [f"{metric}_{cell['seed']}" for metric in METRICS]
        acc = frame if acc is None else acc.join(frame, how="inner")
    return acc


def seed_averaged(cells: list[dict], tier: int, depth: int, arch: str) -> pd.DataFrame | None:
    """Per-basin metric mean over the seeds of one (tier, depth, arch) group."""
    matrix = seed_matrix(cells, tier, depth, arch)
    if matrix is None:
        return None
    out = pd.DataFrame(index=matrix.index)
    for metric in METRICS:
        columns = [c for c in matrix.columns if c.startswith(metric + "_")]
        out[metric] = matrix[columns].mean(axis=1)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def bootstrap_draws(a_seeds: list[np.ndarray], b_seeds: list[np.ndarray],
                    n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                    chunk: int = 1000) -> np.ndarray:
    """Basin bootstrap of the seed-mean medians.

    Each draw resamples the 671 basins once, recomputes every seed's basin
    median, and averages over seeds. ``delta_medians`` is therefore the same
    estimand as the reported gap (mean over seeds of the per-seed basin median),
    so the table's absolute medians and its delta add up.

    Resampling the same basin indices for every seed and for both architectures
    preserves the paired structure; it does not assume seeds are replicates.
    """
    n = len(a_seeds[0])
    rng = np.random.default_rng(seed)
    draws = []
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(size, n))
        med_a = np.mean([np.median(values[idx], axis=1) for values in a_seeds], axis=0)
        med_b = np.mean([np.median(values[idx], axis=1) for values in b_seeds], axis=0)
        draws.append(np.column_stack([med_a, med_b, med_a - med_b]))
        done += size
    return np.vstack(draws)


BOOT_NAMES = ("median_lstm", "median_mamba", "delta_medians")


def bootstrap_cis(draws: np.ndarray) -> dict[str, tuple[float, float]]:
    """Percentile CIs for both seed-mean medians and their difference."""
    return {name: (float(np.percentile(draws[:, i], 2.5)),
                   float(np.percentile(draws[:, i], 97.5)))
            for i, name in enumerate(BOOT_NAMES)}


def bootstrap_tost(draws: np.ndarray, margin: float = TOST_MARGIN) -> float:
    """Equivalence p-value for the reported gap by CI inversion.

    ``p < 0.05`` iff the 90 % bootstrap interval of the gap lies strictly inside
    +/- margin. This is the mean-free analogue of TOST and is the one used for
    inference, because two Mamba-3 basins blow up and make any mean-based
    statistic meaningless.
    """
    delta = draws[:, BOOT_NAMES.index("delta_medians")]
    return float(max(np.mean(delta <= -margin), np.mean(delta >= margin)))


def tost(delta: np.ndarray, margin: float = TOST_MARGIN) -> tuple[float, float, float]:
    """Mean-based two one-sided tests, reported for reference only.

    Returns (p_tost, mean_delta, se). ``p_tost < 0.05`` means the difference is
    statistically equivalent to zero at the stated margin.
    """
    n = len(delta)
    mean = float(np.mean(delta))
    se = float(np.std(delta, ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    if not math.isfinite(se) or se == 0.0:
        return (0.0 if abs(mean) < margin else 1.0), mean, se
    p_lower = float(stats.t.sf((mean + margin) / se, n - 1))
    p_upper = float(stats.t.cdf((mean - margin) / se, n - 1))
    return max(p_lower, p_upper), mean, se


def spatial_blocks(gauge_ids: pd.Series, topo: pd.DataFrame) -> pd.DataFrame:
    """Block ids per basin, joined by gauge_id (never by row position)."""
    joined = pd.DataFrame({"gauge_id": gauge_ids.to_numpy()}).merge(
        topo, on="gauge_id", how="left")
    joined.index = gauge_ids.index
    for cell in BLOCK_SIZES_DEG:
        joined[f"bx{cell}"] = (np.floor((joined.gauge_lon + 125.0) / cell)
                               * 1000
                               + np.floor((joined.gauge_lat - 25.0) / cell))
    keep = joined.dropna(subset=[f"bx{c}" for c in BLOCK_SIZES_DEG]).index
    return joined.loc[keep]


def paired_tests(cells: list[dict], tier: int, depth_lstm: int,
                 depth_mamba: int, topo: pd.DataFrame) -> list[dict]:
    """All statistics for one (tier, LSTM depth, Mamba depth) pair."""
    lstm_matrix = seed_matrix(cells, tier, depth_lstm, "lstm")
    mamba_matrix = seed_matrix(cells, tier, depth_mamba, "mamba3")
    if lstm_matrix is None or mamba_matrix is None:
        return []
    lstm_seeds = sorted({int(c.split("_")[1]) for c in lstm_matrix.columns
                         if c.startswith("NSE_")})
    mamba_seeds = sorted({int(c.split("_")[1]) for c in mamba_matrix.columns
                          if c.startswith("NSE_")})
    lstm_matrix = lstm_matrix.rename(columns={c: c + "_l" for c in lstm_matrix.columns})
    mamba_matrix = mamba_matrix.rename(columns={c: c + "_m" for c in mamba_matrix.columns})
    joint = lstm_matrix.join(mamba_matrix, how="inner")
    if joint.empty:
        return []
    coords = spatial_blocks(pd.Series(joint.index, index=joint.index), topo)

    rows = []
    for metric in METRICS:
        a_seeds = [joint[f"{metric}_{s}_l"].to_numpy() for s in lstm_seeds]
        b_seeds = [joint[f"{metric}_{s}_m"].to_numpy() for s in mamba_seeds]
        a = np.mean(np.column_stack(a_seeds), axis=1)
        b = np.mean(np.column_stack(b_seeds), axis=1)
        keep = np.isfinite(a) & np.isfinite(b)
        a_seeds = [values[keep] for values in a_seeds]
        b_seeds = [values[keep] for values in b_seeds]
        a, b = a[keep], b[keep]
        delta = a - b
        n = len(delta)

        wins = int((delta > TIE_TOL).sum())
        losses = int((delta < -TIE_TOL).sum())
        sign_p = stats.binomtest(wins, wins + losses, 0.5,
                                 alternative="two-sided").pvalue
        wilcoxon_p = float(stats.wilcoxon(a, b).pvalue)
        t_p = float(stats.ttest_rel(a, b).pvalue)
        p_tost_mean, mean_delta, se = tost(delta)

        # Reported gap: mean over seeds of the per-seed basin median. This is
        # the same estimand as the seed-ladder table, so the printed medians and
        # the printed delta subtract exactly.
        median_lstm = float(np.mean([np.median(values) for values in a_seeds]))
        median_mamba = float(np.mean([np.median(values) for values in b_seeds]))
        delta_medians = median_lstm - median_mamba
        draws = bootstrap_draws(a_seeds, b_seeds)
        cis = bootstrap_cis(draws)
        p_tost = bootstrap_tost(draws)

        blow = (a < BLOWUP_NSE) | (b < BLOWUP_NSE)
        delta_rob = delta[~blow]
        n_blow = int(blow.sum())
        delta_medians_rob = float(
            np.mean([np.median(values[~blow]) for values in a_seeds])
            - np.mean([np.median(values[~blow]) for values in b_seeds])
        ) if n_blow else delta_medians

        work = pd.DataFrame({
            "d" + metric: delta,
        }, index=joint.index[keep])
        blocked = {}
        if len(coords):
            sub = work.loc[coords.index]
            for cell in BLOCK_SIZES_DEG:
                p_b, n_blocks, _obs, _sd = blocked_permutation_p(
                    sub, metric, cell, coords)
                blocked[cell] = {"p": p_b, "n_blocks": n_blocks}

        rows.append({
            "tier": tier, "depth_lstm": depth_lstm, "depth_mamba": depth_mamba,
            "depth": depth_lstm if depth_lstm == depth_mamba else math.nan,
            "metric": metric, "n_basins": n,
            "median_lstm": median_lstm, "median_mamba": median_mamba,
            "delta_medians": delta_medians,
            "median_basin_delta": float(np.median(delta)),
            "mean_delta": mean_delta, "se_mean_delta": se,
            "wins_lstm": wins, "losses_mamba": losses, "ties": n - wins - losses,
            "win_share_pct": 100.0 * wins / n,
            "sign_p": float(sign_p), "wilcoxon_p": wilcoxon_p, "paired_t_p": t_p,
            "n_blowups": n_blow,
            "blowup_gauges": ";".join(joint.index[keep][blow]),
            "delta_medians_ex_blowups": delta_medians_rob,
            "win_share_ex_blowups_pct": (100.0 * float((delta_rob > TIE_TOL).sum()) / len(delta_rob)
                                          if len(delta_rob) else math.nan),
            "blocked_p_min": min(v["p"] for v in blocked.values()) if blocked else math.nan,
            "blocked_p_max": max(v["p"] for v in blocked.values()) if blocked else math.nan,
            "n_blocks": ";".join(str(blocked[c]["n_blocks"]) for c in BLOCK_SIZES_DEG),
            **{f"blocked_p_{c}deg": blocked[c]["p"] for c in BLOCK_SIZES_DEG},
            "ci_median_lstm_low": cis["median_lstm"][0],
            "ci_median_lstm_high": cis["median_lstm"][1],
            "ci_median_mamba_low": cis["median_mamba"][0],
            "ci_median_mamba_high": cis["median_mamba"][1],
            "ci_delta_low": cis["delta_medians"][0],
            "ci_delta_high": cis["delta_medians"][1],
            "tost_p": p_tost,
            "tost_equivalent": bool(p_tost < 0.05),
            "tost_mean_p": p_tost_mean,
        })
    return rows
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return "--"
    if p <= 1.0 / (N_PERM + 1) + 1e-12:
        return f"$<${1.0 / (N_PERM + 1):.0e}"
    return f"{p:.2e}"


def latex_seed_ladder(agg: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Seed ladder: per-seed basin-median NSE and KGE aggregated "
        r"over seeds (mean $\pm$ SD). The last column is the worst single-basin "
        r"NSE seen in any seed of the cell.}",
        r"\label{tab:seed_ladder}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Tier & Depth & Model & $n$ & NSE (mean $\pm$ SD) & KGE (mean $\pm$ SD) "
        r"& Min NSE \\",
        r"\midrule",
    ]
    for _, r in agg.sort_values(["tier", "depth", "arch"]).iterrows():
        nse = (f"{r.nse_mean:.4f} $\\pm$ {r.nse_sd:.4f}"
               if math.isfinite(r.nse_sd) else f"{r.nse_mean:.4f}")
        kge = (f"{r.kge_mean:.4f} $\\pm$ {r.kge_sd:.4f}"
               if math.isfinite(r.kge_sd) else f"{r.kge_mean:.4f}")
        lines.append(f"{TIER_NAMES[int(r.tier)]} & {int(r.depth)} & "
                     f"{ARCH_LABEL[r.arch]} & {int(r.n_seeds)} & {nse} & {kge} & "
                     f"{r.min_nse:.4g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_pairwise(pairs: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Paired basin tests per matched (tier, depth) pair on "
        r"seed-averaged per-basin scores: median difference (LSTM $-$ Mamba-3), "
        r"LSTM win share, sign / Wilcoxon / paired-$t$ p-values, joint "
        r"spatial-block permutation p (range over 3--5$^\circ$ blocks), and "
        r"TOST equivalence at $\pm$0.01 by bootstrap CI inversion for the "
        r"median. Blow-ups counts basins with seed-averaged NSE $<-10$ in "
        r"either model; they are retained in the median statistics and dropped "
        r"in the \texttt{ex\_blowups} sensitivity columns of the CSV. The "
        r"paired-$t$ column is mean-based and is dominated by those basins.}",
        r"\label{tab:pairwise_tests}",
        r"\begin{tabular}{llrrrrrrrlr}",
        r"\toprule",
        r"Tier & Depth & Metric & $\Delta$ median & Win \% & Sign $p$ & "
        r"Wilcoxon $p$ & Blocked $p$ & TOST $p$ & Equiv. & Blow-ups \\",
        r"\midrule",
    ]
    for _, r in pairs.sort_values(["tier", "depth_lstm", "metric"]).iterrows():
        blocked = (f"{fmt_p(r.blocked_p_min)}--{fmt_p(r.blocked_p_max)}"
                   if math.isfinite(r.blocked_p_min) else "--")
        lines.append(
            f"{TIER_NAMES[int(r.tier)]} & {int(r.depth_lstm)} & {r.metric} & "
            f"{r.delta_medians:+.4f} & {r.win_share_pct:.1f} & {fmt_p(r.sign_p)} & "
            f"{fmt_p(r.wilcoxon_p)} & {blocked} & {fmt_p(r.tost_p)} & "
            f"{'yes' if r.tost_equivalent else 'no'} & {int(r.n_blowups)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_headline(head: pd.DataFrame) -> str:
    head = head[head.metric == "NSE"].sort_values("tier")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Headline pair per tier: LSTM 1 layer vs Mamba-3 at its "
        r"deepest depth. Values are the mean over seeds of the per-seed basin "
        r"median (the seed-ladder estimand), and $\Delta$ is the difference of "
        r"those two printed values. CI is the 10\,000-draw basin bootstrap of "
        r"$\Delta$ (2.5--97.5\,\%).}",
        r"\label{tab:headline}",
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"Tier & LSTM median & Mamba-3 median & $\Delta$ median [95\,\% CI] & TOST $p$ \\",
        r"\midrule",
    ]
    for _, r in head.iterrows():
        lines.append(
            f"{TIER_NAMES[int(r.tier)]} & {r.median_lstm:.4f} & {r.median_mamba:.4f} & "
            f"{r.delta_medians:+.4f} [{r.ci_delta_low:+.4f}, "
            f"{r.ci_delta_high:+.4f}] & {fmt_p(r.tost_p)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiers", default=None,
                        help="comma-separated tier filter, e.g. 1,2")
    args = parser.parse_args(argv)
    tiers = tuple(int(t) for t in args.tiers.split(",")) if args.tiers else None

    topo = load_topo()
    cells = load_cells(tiers)
    if not cells:
        print("no completed cells found", file=sys.stderr)
        return 2

    seed_df = seed_values(cells)
    agg = aggregate_seeds(seed_df)

    pair_rows: list[dict] = []
    headline_rows: list[dict] = []
    depths = sorted(seed_df["depth"].unique())
    for tier in sorted(seed_df["tier"].unique()):
        for depth in depths:
            have_l = ((seed_df.tier == tier) & (seed_df.depth == depth)
                      & (seed_df.arch == "lstm")).any()
            have_m = ((seed_df.tier == tier) & (seed_df.depth == depth)
                      & (seed_df.arch == "mamba3")).any()
            if have_l and have_m:
                pair_rows.extend(paired_tests(cells, tier, depth, depth, topo))
        mamba_depths = sorted(seed_df.loc[(seed_df.tier == tier)
                                          & (seed_df.arch == "mamba3"), "depth"].unique())
        have_l1 = ((seed_df.tier == tier) & (seed_df.depth == 1)
                   & (seed_df.arch == "lstm")).any()
        if mamba_depths and have_l1:
            headline_rows.extend(
                paired_tests(cells, tier, 1, int(max(mamba_depths)), topo))
    pairs = pd.DataFrame(pair_rows)
    headline = pd.DataFrame(headline_rows)

    RESULTS.mkdir(exist_ok=True)
    seed_df.to_csv(RESULTS / "stats_seed_values.csv", index=False)
    agg.to_csv(RESULTS / "stats_seed_aggregate.csv", index=False)
    if not pairs.empty:
        pairs.to_csv(RESULTS / "stats_pairwise.csv", index=False)
    if not headline.empty:
        headline.to_csv(RESULTS / "stats_headline.csv", index=False)

    TABLES.mkdir(exist_ok=True)
    (TABLES / "tab_seed_ladder.tex").write_text(latex_seed_ladder(agg))
    if not pairs.empty:
        (TABLES / "tab_pairwise_tests.tex").write_text(latex_pairwise(pairs))
    if not headline.empty:
        (TABLES / "tab_headline.tex").write_text(latex_headline(headline))

    print(f"cells: {len(cells)}  groups: {len(agg)}  pairs: {len(pairs) // len(METRICS)}")
    print("\n=== seed ladder (basin-median NSE, mean over seeds) ===")
    print(agg.pivot_table(index=["tier", "depth"], columns="arch",
                          values="nse_mean").round(4).to_string())
    if not pairs.empty:
        print("\n=== same-depth paired tests (NSE) ===")
        cols = ["tier", "depth_lstm", "delta_medians", "win_share_pct", "sign_p",
                "wilcoxon_p", "blocked_p_max", "ci_delta_low",
                "ci_delta_high", "tost_p", "tost_equivalent"]
        print(pairs[pairs.metric == "NSE"][cols].to_string(index=False,
                                                          float_format=lambda v: f"{v:.4g}"))
    if not headline.empty:
        print("\n=== headline pair: LSTM 1L vs Mamba-3 deepest (NSE) ===")
        cols = ["tier", "depth_mamba", "median_lstm", "median_mamba",
                "delta_medians", "ci_delta_low", "ci_delta_high",
                "win_share_pct", "blocked_p_max", "tost_p", "tost_equivalent"]
        print(headline[headline.metric == "NSE"][cols].to_string(
            index=False, float_format=lambda v: f"{v:.4g}"))
    missing = []
    for tier in (1, 2, 3):
        for depth in (1, 2, 3):
            for arch in ARCH_ORDER:
                if not ((seed_df.tier == tier) & (seed_df.depth == depth)
                        & (seed_df.arch == arch)).any():
                    missing.append(f"T{tier} L{depth} {arch}")
    if missing:
        print(f"\nmissing cells (not yet evaluated): {', '.join(missing)}")
    print(f"\nwrote {RESULTS}/stats_*.csv and {TABLES}/tab_*.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
