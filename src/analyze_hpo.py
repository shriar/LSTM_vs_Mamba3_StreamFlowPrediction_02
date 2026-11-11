"""Read-only analysis of the tracked HPO grids (C5 matched, C7 sensitivity).

Three things the revision needs, all computed from files that are already on
disk and tracked -- no training, no test split:

  * **matched** -- per-trial val medians for both architectures, the draw
    statistics around the submitted recipe, and the Spearman rank correlation
    of each axis with val NSE. This is the R2#3 answer ("was the Mamba run
    under-tuned relative to the LSTM?") and it must be read as a *distribution
    over draws*, not as a best-of-13 point estimate.
  * **volatility** -- per-epoch val-NSE jumps for every finished run, so the
    "sawtooth" in the training curve (R3#4) can be stated as a number and
    compared across architectures instead of eyeballed from a figure.
  * **sensitivity** -- the 3x2x2 C7 factorial: main effects of eta0, grad clip
    and beta2 on val NSE and on the volatility metrics.

Volatility definitions, fixed before C7 was read: a *drop* is a decrease of
more than 0.015 in val median NSE between consecutive epochs; the *best-last
gap* is the best epoch minus the final epoch; the *post-best range* is
max-min over the epochs at or after the best one. They describe different
failures (jumpiness during training vs. drift after the peak) and the plan
quotes each separately.

Run from the experiment directory:

    python src/analyze_hpo.py                  # everything that exists
    python src/analyze_hpo.py matched
    python src/analyze_hpo.py volatility
    python src/analyze_hpo.py sensitivity
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

from scipy import stats

MATCHED_ROOT = Path("results/hpo/matched")
SENSITIVITY_ROOT = Path("results/hpo/sensitivity")
AXES = ("eta0", "eta_min", "lambda", "clip", "beta2", "dropout")
DROP = 0.015
BASELINE = "trial_0"


def load_trials(grid_root: Path) -> list[dict]:
    """Every finished trial under *grid_root*/<model>/, in model then id order."""
    trials = []
    for model_dir in sorted(path for path in grid_root.iterdir() if path.is_dir()):
        for path in sorted(model_dir.glob("*/trial.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "ok":
                continue
            payload["model"] = payload.get("model", model_dir.name)
            trials.append(payload)
    return trials


def val_curve(run_dir: str) -> list[float]:
    """Per-epoch validation median NSE, in epoch order."""
    path = sorted(Path(run_dir).rglob("val_metrics.csv"))[0]
    with path.open(encoding="utf-8") as handle:
        return [float(row["val_median_nse"]) for row in csv.DictReader(handle)]


def volatility(curve: list[float], drop: float = DROP) -> dict:
    drops = [curve[i - 1] - curve[i] for i in range(1, len(curve))]
    best = max(curve)
    at = curve.index(best)
    after = curve[at:]
    return {
        "epochs": len(curve),
        "best": best,
        "best_epoch": at + 1,
        "last": curve[-1],
        "gap": best - curve[-1],
        "drops": sum(1 for value in drops if value > drop),
        "max_drop": max(drops),
        "post_best_range": max(after) - min(after),
    }


def axes_of(payload: dict) -> dict:
    axes = payload.get("axes")
    if axes:
        return axes
    params = payload["params"]
    return {
        "eta0": params.get("mamba_lr") or params.get("lstm_lr"),
        "eta_min": params.get("mamba_min_lr") or params.get("lstm_min_lr"),
        "lambda": params.get("mamba_weight_decay") or params.get("lstm_weight_decay"),
        "clip": params.get("mamba_grad_clip") or params.get("lstm_grad_clip"),
        "beta2": params.get("mamba_beta2") or params.get("lstm_beta2"),
        "dropout": params.get("mamba_dropout") or params.get("lstm_dropout"),
    }


def summarise(values: list[float]) -> str:
    return (f"median {statistics.median(values):.3f}  "
            f"mean {statistics.mean(values):.3f}  "
            f"range {min(values):.3f}-{max(values):.3f}")


def report_matched() -> None:
    trials = load_trials(MATCHED_ROOT)
    if not trials:
        print("matched: nothing on disk")
        return
    by_model: dict[str, list[dict]] = {}
    for trial in trials:
        by_model.setdefault(trial["model"], []).append(trial)

    for model, rows in by_model.items():
        rows.sort(key=lambda r: -r["val_median_nse"])
        axes = {r["trial_id"]: axes_of(r) for r in rows}
        print(f"\n=== matched / {model} ({len(rows)} trials) ===")
        print(f"{'trial':<9}{'valNSE':>9}{'d vs base':>10}{'valKGE':>9}{'best':>6}"
              f"{'eta0':>9}{'clip':>6}{'beta2':>7}")
        base = next((r for r in rows if r["trial_id"] == BASELINE), None)
        for row in rows:
            ax = axes[row["trial_id"]]
            delta = "" if base is None else f"{row['val_median_nse'] - base['val_median_nse']:+.4f}"
            print(f"{row['trial_id']:<9}{row['val_median_nse']:>9.4f}{delta:>10}"
                  f"{row['val_median_kge']:>9.4f}{row['best_epoch']:>6}"
                  f"{ax['eta0']:>9.4g}{ax['clip']:>6.2f}{ax['beta2']:>7.3f}")

        draws = [r["val_median_nse"] for r in rows if r["trial_id"] != BASELINE]
        if base is not None and draws:
            above = sum(1 for value in draws if value > base["val_median_nse"])
            sd = statistics.stdev(draws)
            print(f"draws (n={len(draws)}) vs {BASELINE}: median "
                  f"{statistics.median(draws):.4f} "
                  f"({statistics.median(draws) - base['val_median_nse']:+.4f}), "
                  f"sd {sd:.4f}, above {above}/{len(draws)}, "
                  f"best {max(draws):.4f} ({max(draws) - base['val_median_nse']:+.4f})")
        scores = [r["val_median_nse"] for r in rows]
        rhos = "  ".join(
            f"{axis} {stats.spearmanr([axes[r['trial_id']][axis] for r in rows], scores).statistic:+.2f}"
            for axis in AXES
        )
        print(f"rho vs val NSE: {rhos}")


def report_volatility() -> None:
    for root, label in ((MATCHED_ROOT, "matched"),
                        (SENSITIVITY_ROOT, "sensitivity")):
        if not root.exists():
            continue
        trials = load_trials(root)
        if not trials:
            continue
        by_model: dict[str, list[dict]] = {}
        for trial in trials:
            stats_row = volatility(val_curve(trial["run_dir"]))
            by_model.setdefault(trial["model"], []).append(stats_row)
        print(f"\n=== volatility / {label} ===")
        for model, rows in by_model.items():
            print(f"  {model}: {len(rows)} runs")
            for key, name in (("drops", "drops > 0.015"), ("max_drop", "max drop"),
                              ("gap", "best-last gap"), ("post_best_range", "post-best range")):
                values = [row[key] for row in rows]
                print(f"    {name:<16} {summarise(values)}")


def report_sensitivity() -> None:
    trials = load_trials(SENSITIVITY_ROOT)
    if not trials:
        print("\nsensitivity: nothing on disk")
        return
    rows = []
    for trial in trials:
        axes = axes_of(trial)
        rows.append({
            "cell": trial["trial_id"],
            "eta0": axes["eta0"],
            "clip": axes["clip"],
            "beta2": axes["beta2"],
            "nse": trial["val_median_nse"],
            "kge": trial["val_median_kge"],
            "best_epoch": trial["best_epoch"],
            "vol": volatility(val_curve(trial["run_dir"])),
        })
    rows.sort(key=lambda r: -r["nse"])

    print(f"\n=== sensitivity / {len(rows)} cells ===")
    print(f"{'cell':<28}{'eta0':>9}{'clip':>6}{'beta2':>7}{'valNSE':>9}{'valKGE':>9}"
          f"{'best':>6}{'drops':>7}{'gap':>8}{'post':>8}")
    for row in rows:
        vol = row["vol"]
        print(f"{row['cell']:<28}{row['eta0']:>9.4g}{row['clip']:>6.2f}"
              f"{row['beta2']:>7.3f}{row['nse']:>9.4f}{row['kge']:>9.4f}"
              f"{row['best_epoch']:>6}{vol['drops']:>7}{vol['gap']:>8.4f}"
              f"{vol['post_best_range']:>8.4f}")

    print("\nmain effects (mean over the cells present at each level)")
    for axis, levels in (("eta0", sorted({r["eta0"] for r in rows})),
                         ("clip", sorted({r["clip"] for r in rows})),
                         ("beta2", sorted({r["beta2"] for r in rows}))):
        for level in levels:
            members = [r for r in rows if abs(r[axis] - level) < 1e-12]
            print(f"  {axis}={level:<8} n={len(members):<2} "
                  f"nse {statistics.mean(m['nse'] for m in members):.4f}  "
                  f"drops {statistics.mean(m['vol']['drops'] for m in members):.1f}  "
                  f"maxdrop {statistics.mean(m['vol']['max_drop'] for m in members):.4f}  "
                  f"post {statistics.mean(m['vol']['post_best_range'] for m in members):.4f}")

    base = [r for r in rows if abs(r["eta0"] - 0.003) < 1e-12
            and abs(r["clip"] - 0.5) < 1e-12 and abs(r["beta2"] - 0.95) < 1e-12]
    if base:
        anchor = base[0]
        print(f"\nsubmitted cell {anchor['cell']}: val NSE {anchor['nse']:.4f} "
              f"(KGE {anchor['kge']:.4f}, best ep {anchor['best_epoch']})")
        for row in rows:
            if row is not anchor:
                print(f"  {row['cell']:<28} dNSE {row['nse'] - anchor['nse']:+.4f}  "
                      f"dKGE {row['kge'] - anchor['kge']:+.4f}")


def main() -> None:
    wanted = set(sys.argv[1:]) or {"matched", "volatility", "sensitivity"}
    if "matched" in wanted:
        report_matched()
    if "volatility" in wanted:
        report_volatility()
    if "sensitivity" in wanted:
        report_sensitivity()


if __name__ == "__main__":
    main()
