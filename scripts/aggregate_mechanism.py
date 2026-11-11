"""C11 -- aggregate the per-seed mechanism records into paper-ready tables.

B7 writes one ``mechanism_summary.csv`` per cell.  C11 runs six cells (LSTM
1L and Mamba-3 3L, seeds 1-3) and this script turns them into:

  * ``c11_seed_table.csv``     -- one row per cell, headline mechanism numbers;
  * ``c11_seed_aggregate.csv`` -- mean / SD / min / max across the 3 seeds.

The SD is the sample SD (ddof=1); a single seed yields NaN, matching the B6
rule that no CI may be stronger than its seed count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

MANIFEST = PROJECT_ROOT / "recipes" / "eval_manifest.json"
MECHANISM_DIR = RESULTS_DIR / "mechanism"

CELLS = {
    ("lstm", 1, 1, 1),
    ("lstm", 1, 1, 2),
    ("lstm", 1, 1, 3),
    ("mamba3", 1, 3, 1),
    ("mamba3", 1, 3, 2),
    ("mamba3", 1, 3, 3),
}

# Metrics that exist for both models.
COMMON = ("mass_balance_val", "mass_balance_test")
# Per-model metrics, in the order A10 will cite them.
MODEL_METRICS = {
    "lstm": (
        "mean_gate_i", "mean_gate_f", "mean_gate_g", "mean_gate_o",
        "mean_cell_norm", "mean_hidden_norm", "replay_relative_diff",
    ),
    "mamba3": (
        "mean_delta", "p95_delta", "mean_ssm_input_norm",
        "ssm_final_state_norm_layer1", "ssm_final_state_norm_layer2",
        "ssm_final_state_norm_layer3",
    ),
}


def _cell_dir(recipe: str, tier: int, depth: int, seed: int) -> Path:
    return MECHANISM_DIR / f"{recipe}_T{tier}_L{depth}_Seed{seed}"


def collect(mechanism_dir: Path = MECHANISM_DIR) -> list[dict]:
    """Read every configured cell; raise if one is missing."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_recipe = {
        (Path(c["recipe"]).name.replace(".json", ""), int(c["tier"]),
         int(c["depth"]), int(c["seed"])): c
        for c in manifest["cells"]
    }
    rows = []
    missing = []
    for key in sorted(CELLS):
        cell = by_recipe.get(key)
        if cell is None:
            raise SystemExit(f"cell {key} is not in the manifest")
        recipe_id = key[0]
        path = _cell_dir(recipe_id, key[1], key[2], key[3]) / "mechanism_summary.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            missing.append(f"{path} (empty)")
            continue
        row = {
            "model": recipe_id,
            "tier": key[1],
            "depth": key[2],
            "seed": key[3],
            "cell_id": cell["id"],
            "n_basins": int(len(frame)),
            "mass_balance_val": float(frame["mass_balance_val"].median()),
            "mass_balance_test": float(frame["mass_balance_test"].median()),
        }
        for category in ("wet", "dry", "snow"):
            part = frame[frame["category"] == category]
            row[f"mass_balance_val_{category}"] = (
                float(part["mass_balance_val"].median()) if len(part) else np.nan
            )
        for metric in MODEL_METRICS[recipe_id]:
            if metric in frame.columns:
                row[metric] = float(frame[metric].mean())
            else:
                row[metric] = np.nan
        rows.append(row)
    if missing:
        raise SystemExit(
            "missing mechanism cells (run scripts/run_c11.sh first):\n  "
            + "\n  ".join(missing)
        )
    return rows


def summarise(rows: list[dict]) -> pd.DataFrame:
    """Mean / SD / min / max across seeds, per model and metric."""
    table = pd.DataFrame(rows)
    out = []
    for model, group in table.groupby("model", sort=True):
        metrics = [m for m in (*COMMON, *MODEL_METRICS.get(model, ())) if m in group]
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                mean = sd = lo = hi = np.nan
            else:
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if values.size > 1 else np.nan
                lo, hi = float(values.min()), float(values.max())
            out.append({
                "model": model,
                "metric": metric,
                "n_seeds": int(values.size),
                "mean": mean,
                "sd": sd,
                "min": lo,
                "max": hi,
            })
    return pd.DataFrame(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate C11 mechanism cells.")
    parser.add_argument("--out-dir", default=str(MECHANISM_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect()
    seed_table = pd.DataFrame(rows)
    seed_table.to_csv(out_dir / "c11_seed_table.csv", index=False)
    aggregate = summarise(rows)
    aggregate.to_csv(out_dir / "c11_seed_aggregate.csv", index=False)

    print(f"cells: {len(seed_table)}")
    for model, group in seed_table.groupby("model"):
        print(
            f"  {model:7s} seeds {group['seed'].tolist()} | "
            f"mass balance val {group['mass_balance_val'].mean():.3f} "
            f"(test {group['mass_balance_test'].mean():.3f})"
        )
    print(f"wrote {out_dir / 'c11_seed_table.csv'} and c11_seed_aggregate.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
