"""Aggregate one benchmark block (C1a, C1b, C1c, ...) into a seed-table CSV.

Reads the manifest cells whose id starts with ``<prefix>-``, pulls each cell's
``results/test/<recipe>_T<tier>_L<depth>_Seed<seed>/test_metrics.csv`` and the
training run's best epoch / epoch count / median epoch time from
``val_metrics.csv`` + ``master.json``, then writes
``results/<prefix-lower>_seed_table.csv`` (or ``--out``).

The test number is read only from the per-cell ``test_metrics.csv`` that
``eval_test.py`` wrote; this script never opens a run directory's predictions,
never scores anything, and never touches the split loaders.

Usage
-----
    python scripts/aggregate_seed_table.py C1a
    python scripts/aggregate_seed_table.py C1b --out results/c1b_seed_table.csv

Writes one row per cell and prints the per-group means and the headline gap
(LSTM 1 layer vs Mamba-3 at the deepest Mamba depth in the block).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent


def wallclock(run_dir: str):
    """Return (best_epoch, epochs_run, median_epoch_seconds) for a run dir."""
    run = EXP / run_dir
    try:
        master = json.loads((run / "master.json").read_text())
    except FileNotFoundError:
        return None, None, None
    selection = master.get("selection") or {}
    with (run / "val_metrics.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    # Skip epoch 1: it carries the CUDA/Triton warm-up.
    seconds = [float(r["epoch_seconds"]) for r in rows[1:]]
    return selection.get("best_epoch"), len(rows), statistics.median(seconds)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prefix", help="cell-id prefix, e.g. C1a")
    parser.add_argument("--out", default=None, help="output CSV path")
    args = parser.parse_args(argv)

    manifest = json.loads((EXP / "recipes/eval_manifest.json").read_text())
    cells = sorted(
        [c for c in manifest["cells"] if c["id"].startswith(args.prefix + "-")],
        key=lambda c: (c["recipe"], c["depth"], c["seed"]),
    )
    if not cells:
        print(f"no cells with prefix {args.prefix}-", file=sys.stderr)
        return 2

    rows_out: list[dict] = []
    missing: list[str] = []
    print(f"{'cell':22s} {'NSE':>8s} {'KGE':>8s} | best_ep n_ep med_ep_s")
    for cell in cells:
        recipe_id = json.loads((EXP / cell["recipe"]).read_text())["id"]
        test_dir = EXP / "results/test" / (
            f"{recipe_id}_T{cell['tier']}_L{cell['depth']}_Seed{cell['seed']}"
        )
        try:
            with (test_dir / "test_metrics.csv").open(newline="") as fh:
                row = list(csv.DictReader(fh))[0]
        except FileNotFoundError:
            missing.append(cell["id"])
            continue

        best_epoch, epochs_run, median_epoch = wallclock(cell["run_dir"])
        short = "-".join(cell["id"].split("-")[1:])
        rows_out.append({
            "cell": cell["id"],
            "model": "mamba3" if "mamba" in cell["recipe"] else "lstm",
            "tier": cell["tier"],
            "depth": cell["depth"],
            "seed": cell["seed"],
            "median_nse": float(row["median_nse"]),
            "mean_nse": float(row.get("mean_nse", "nan")),
            "median_kge": float(row["median_kge"]),
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "median_epoch_s": round(median_epoch, 1) if median_epoch else "",
            "test_dir": str(test_dir.relative_to(EXP)).replace("\\", "/"),
            "recipe": cell["recipe"],
            "run_dir": cell["run_dir"],
        })
        print(
            f"{short:22s} {float(row['median_nse']):8.4f} "
            f"{float(row['median_kge']):8.4f} | {best_epoch} {epochs_run} "
            f"{median_epoch:.1f}s"
        )

    if missing:
        print("\nMISSING test_metrics.csv for: " + ", ".join(missing))

    out = Path(args.out) if args.out else EXP / "results" / f"{args.prefix.lower()}_seed_table.csv"
    if not out.is_absolute():
        out = EXP / out
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nwrote {out} ({len(rows_out)} rows)")

    def group(model: str, depth: int):
        nse = [r["median_nse"] for r in rows_out if r["model"] == model and r["depth"] == depth]
        kge = [r["median_kge"] for r in rows_out if r["model"] == model and r["depth"] == depth]
        if not nse:
            return None
        return (len(nse), statistics.mean(nse), statistics.pstdev(nse),
                statistics.mean(kge), statistics.pstdev(kge))

    print("\n--- per group: n, NSE mean | sd, KGE mean | sd ---")
    for depth in sorted({r["depth"] for r in rows_out}):
        for model in ("lstm", "mamba3"):
            stats = group(model, depth)
            if stats:
                n, nse_mean, nse_sd, kge_mean, kge_sd = stats
                print(f"{model:7s} L{depth}: n={n} NSE {nse_mean:.4f} | {nse_sd:.4f}   "
                      f"KGE {kge_mean:.4f} | {kge_sd:.4f}")

    depths = [r["depth"] for r in rows_out if r["model"] == "mamba3"]
    if depths:
        headline_depth = max(depths)
        lstm_l1 = group("lstm", 1)
        mamba_headline = group("mamba3", headline_depth)
        if lstm_l1 and mamba_headline:
            print(f"\nheadline gap LSTM-L1 minus Mamba3-L{headline_depth}: "
                  f"{lstm_l1[1] - mamba_headline[1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
