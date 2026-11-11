"""ECDF / distributional comparison for the LSTM vs. Mamba-3 result.

Backs the claims in Results subsection "Primary Benchmark Results" that
Figure 7 (fig:ecdf) previously only asserted by eye. The sentence in the
manuscript used to read:

    "the LSTM curve lies consistently to the right of the Mamba-3 curve across
     the 10th to 90th percentiles for both NSE and KGE, demonstrating that the
     median advantage reflects a population-wide shift"

Checking that numerically shows the first half is true only for NSE: on KGE
the Mamba-3 quantile is the higher of the two at the 14th, 15th, 16th, 20th and
21st percentiles, so the two ECDFs cross inside the stated window. This script
locates every crossing and prints the percentile table that Table
tab:ecdf_quantiles reports.

Scope note: this script deliberately does NOT compute significance. That is the
job of sign_test_lstm_vs_mamba.py, which does it properly. A plain Wilcoxon over
all 671 basins is reported there only for reference and is anticonservative,
because neighbouring CAMELS basins share climate, soils and a single regional
model. Use the spatially blocked permutation test from that script instead.

Run from the experiment directory:

    python ecdf_percentile_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Same configuration pair reported in the Results section: best LSTM (1 layer,
# Tier 1) against best Mamba-3 (3 layers, Tier 1), both at rho = 365,
# 30 epochs, seed 111111.
from paths import OUTPUT_ROOT

OUTPUT_DIR = OUTPUT_ROOT
LSTM_RUN = OUTPUT_DIR / "LSTM_1L" / "Tier1_200K_Ep30_Bs128_Rho365_H162_L1" / "All"
MAMBA_RUN = (OUTPUT_DIR / "Mamba3_3L"
             / "Tier1_Ep30_Variant-adapter_V3_head32_H80_L3_dConv4" / "All")

METRICS = ("NSE", "KGE")

# Percentiles tabulated in the manuscript.
TABLE_PCTS = (10, 25, 50, 75, 90)

# The window the original claim referred to.
CLAIM_LO, CLAIM_HI = 10, 90

# Published in the manuscript, used here as a self-check that we have opened
# the right files.
EXPECTED = {
    "NSE": {"wins": 461, "pct": 68.7, "median_lstm": 0.7515, "median_mam": 0.7272},
    "KGE": {"wins": 390, "pct": 58.1, "median_lstm": 0.7792, "median_mam": 0.7636},
}

TIE_TOL = 1e-9

# Boundaries for the paired-difference histogram, in metric units.
DIFF_BINS = ((-1.00, -0.05), (-0.05, -0.01), (-0.01, 0.01),
             (0.01, 0.05), (0.05, 1.00))


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


def check_against_published(paired: pd.DataFrame) -> None:
    """Fail loudly if the loaded files are not the ones the paper reports."""
    n = len(paired)
    print(f"paired basins: {n}   unique gauge_id: {paired.gauge_id.nunique()}")
    if n != 671:
        raise SystemExit(f"expected 671 basins, got {n}")

    for metric, exp in EXPECTED.items():
        med_l = paired[f"{metric}_lstm"].median()
        med_m = paired[f"{metric}_mamba"].median()
        wins = int((paired[f"d{metric}"] > TIE_TOL).sum())
        pct = 100.0 * wins / n
        ok = (wins == exp["wins"]
              and abs(med_l - exp["median_lstm"]) < 5e-4
              and abs(med_m - exp["median_mam"]) < 5e-4)
        print(f"  {metric}: wins {wins} (paper {exp['wins']}), "
              f"medians {med_l:.4f}/{med_m:.4f} "
              f"(paper {exp['median_lstm']}/{exp['median_mam']})  "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"{metric} does not match the published values")


def report_percentiles(paired: pd.DataFrame) -> None:
    """The table reported as tab:ecdf_quantiles."""
    print("\n=== per-basin percentiles (Table tab:ecdf_quantiles) ===")
    for metric in METRICS:
        a = paired[f"{metric}_lstm"].to_numpy()
        b = paired[f"{metric}_mamba"].to_numpy()
        print(f"  {metric}")
        print("    pct   lstm    mamba    diff")
        for p in TABLE_PCTS:
            qa, qb = np.percentile(a, p), np.percentile(b, p)
            print(f"    {p:>3}  {qa:.4f}  {qb:.4f}  {qa - qb:+.4f}")


def locate_crossings(paired: pd.DataFrame) -> None:
    """Test the original 'consistently to the right' claim percentile by
    percentile, which is the only way to check a claim about ECDF shape."""
    grid = np.arange(1, 100)
    print("\n=== where does the Mamba-3 quantile exceed the LSTM? ===")
    for metric in METRICS:
        qa = np.percentile(paired[f"{metric}_lstm"].to_numpy(), grid)
        qb = np.percentile(paired[f"{metric}_mamba"].to_numpy(), grid)
        diff = qa - qb
        above = grid[diff <= 0]
        inside = (grid >= CLAIM_LO) & (grid <= CLAIM_HI)
        n_in = int(inside.sum())
        n_ok = int((diff[inside] > 0).sum())

        print(f"  {metric}:")
        print(f"    Mamba-3 higher at percentiles {list(map(int, above))}")
        print(f"    LSTM higher at {int((diff > 0).sum())}/99 percentiles overall")
        print(f"    within the claimed {CLAIM_LO}-{CLAIM_HI} window: "
              f"{n_ok}/{n_in} favour the LSTM")
        if n_ok < n_in:
            j = int(np.argmin(np.where(inside, diff, np.inf)))
            print(f"    -> claim FALSE in this window; worst margin "
                  f"{diff[j]:+.4f} at the {grid[j]}th percentile")
        else:
            print(f"    -> claim holds across the whole window; "
                  f"narrowest margin {diff[inside].min():+.4f}")


def report_outcome_mix(paired: pd.DataFrame) -> None:
    """How lopsided is the basin-level outcome? Supports 'broad but not
    universal', and is the honest replacement for 'population-wide'."""
    n = len(paired)
    print("\n=== basin-level outcome mix ===")
    for metric in METRICS:
        d = paired[f"d{metric}"].to_numpy()
        print(f"  {metric}  (Mamba-3 better in {int((d < -TIE_TOL).sum())} of {n})")
        for lo, hi in DIFF_BINS:
            count = int(((d > lo) & (d <= hi)).sum())
            print(f"    ({lo:+.2f}, {hi:+.2f}]: {count:>3} basins "
                  f"({100 * count / n:5.1f}%)")


def note_on_medians() -> None:
    print("\n=== on the 'not driven by outliers' wording ===")
    print("  The headline metric is a median, which is insensitive to extreme")
    print("  values by construction, so 'the median could be an artefact of")
    print("  outlier basins' is not a coherent concern. A symmetric trimmed")
    print("  median returns the same median and therefore proves nothing; the")
    print("  basin-level win counts above are the relevant evidence instead.")


def main() -> None:
    paired = load_paired()
    check_against_published(paired)
    report_percentiles(paired)
    locate_crossings(paired)
    report_outcome_mix(paired)
    note_on_medians()
    print("\nfor p-values see sign_test_lstm_vs_mamba.py (spatially blocked "
          "permutation test)")


if __name__ == "__main__":
    main()
