"""D4.5 -- generate the data-driven LaTeX table fragments.

Writes body-only fragments (no caption/label, so the manuscript keeps its
captions) into ``Writing/Submission_WRR/05_LaTeX_Source/tables/`` plus a ``_data.csv``
beside every fragment in ``experiment/results/tables/``.

Tables generated here:
  tab_param_matching        widths and parameters per depth x tier
  tab_hyperparams           the two locked recipes
  tab_paired_pvalues        per-pair gaps, win share, tests, CIs
  tab_ecdf_quantiles        per-basin NSE/KGE percentiles, seed-averaged
  tab_benchmark_full        3 tiers x 3 depths, mean +- SD
  tab_appendix_training_time measured epoch seconds per tier
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from figure_source import cell_for, model_params, result_files  # noqa: E402
from paths import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

OUT = PROJECT_ROOT.parent / "Writing" / "Submission_WRR" / "05_LaTeX_Source" / "tables"
DATA = RESULTS_DIR / "tables"
EPOCH_SECONDS = {1: (28.9, 32.7), 2: (29.1, 55.6), 3: (45.0, 186.6)}
SEEDS = {(1, 1): [1, 2, 3, 4, 5], (1, 3): [1, 2, 3, 4, 5]}


def _fmt_int(value: int) -> str:
    return f"{int(value):,}".replace(",", "\\,")


def _p(value: float) -> str:
    if not np.isfinite(value):
        return "---"
    if value < 5e-6:
        return "$<$5e-6"
    return f"{value:.2e}" if value < 0.01 else f"{value:.3f}"


SPECS = {
    "tab_param_matching": "lllcccc",
    "tab_hyperparams": "lcc",
    "tab_paired_pvalues": "lcccccc",
    "tab_ecdf_quantiles": "lcccc",
    "tab_benchmark_full": "lcccccc",
    "tab_appendix_training_time": "lccc",
}


def _write(name: str, body: str, data: pd.DataFrame | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = SPECS.get(name, "lccc")
    wrapped = f"\\begin{{tabular}}{{{spec}}}\n{body.rstrip()}\n\\end{{tabular}}"
    (OUT / f"{name}.tex").write_text(wrapped + "\n", encoding="utf-8")
    if data is not None:
        DATA.mkdir(parents=True, exist_ok=True)
        data.to_csv(DATA / f"{name}_data.csv", index=False)
    print(f"  wrote tables/{name}.tex")


# ---------------------------------------------------------------------------
def param_matching() -> None:
    rows = []
    for depth in (1, 2, 3):
        for tier in (1, 2, 3):
            m = model_params(cell_for("recipes/mamba3.json", tier, depth, 1))
            l = model_params(cell_for("recipes/lstm.json", tier, depth, 1))
            rows.append({
                "depth": depth, "tier": tier,
                "mamba_hidden": m["hidden_size"], "lstm_hidden": l["hidden_size"],
                "mamba_params": m["n_parameters"], "lstm_params": l["n_parameters"],
                "abs_diff": l["n_parameters"] - m["n_parameters"],
                "rel_pct": 100 * (l["n_parameters"] - m["n_parameters"]) / m["n_parameters"],
            })
    table = pd.DataFrame(rows)
    lines = [
        "\\toprule",
        "\\textbf{Depth} & \\textbf{Tier} & \\textbf{Architecture} & \\textbf{$H$} & "
        "\\textbf{Parameters} & \\textbf{Absolute diff} & \\textbf{Relative diff} \\\\",
        "\\midrule",
    ]
    for depth in (1, 2, 3):
        part = table[table.depth == depth]
        lines.append(f"\\multirow{{{2 * len(part)}}}{{*}}{{\\textbf{{{depth}-Layer}}}}")
        for i, (_, row) in enumerate(part.iterrows()):
            lines.append(
                f"& Tier {int(row.tier)} & Mamba-3 (standard) & {int(row.mamba_hidden)} & "
                f"{_fmt_int(row.mamba_params)} & Baseline & Baseline \\\\"
            )
            lines.append(
                f"& & CuDNN LSTM & {int(row.lstm_hidden)} & {_fmt_int(row.lstm_params)} & "
                f"{int(row.abs_diff):+,} & {row.rel_pct:+.2f}\\% \\\\".replace(
                    f"{int(row.abs_diff):+,}", f"{int(row.abs_diff):+,}".replace(",", "\\,")
                )
            )
            if i < len(part) - 1:
                lines.append("\\cmidrule{2-7}")
        if depth < 3:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    _write("tab_param_matching", "\n".join(lines), table)


def hyperparams() -> None:
    import json

    lstm = json.loads((PROJECT_ROOT / "recipes" / "lstm.json").read_text())["hyperparameters"]
    mamba = json.loads((PROJECT_ROOT / "recipes" / "mamba3.json").read_text())["hyperparameters"]

    rows = [
        ("Optimizer", lstm["lstm_optimizer"], mamba["mamba_optimizer"]),
        ("Initial learning rate $\\eta_0$", f"{lstm['lr']:.3g}", f"{mamba['mamba_lr']:.3g}"),
        ("Minimum learning rate", f"{lstm['lstm_min_lr']:.3g}", f"{mamba['mamba_min_lr']:.3g}"),
        ("Schedule", lstm["lstm_scheduler"], mamba["mamba_scheduler"]),
        ("Warmup", "none", f"{mamba['mamba_warmup_steps']} steps"),
        ("Weight decay $\\lambda$", f"{lstm['lstm_weight_decay']:.3g}", f"{mamba['mamba_weight_decay']:.3g}"),
        ("Gradient-clip norm", f"{lstm['lstm_grad_clip']:.2f}", f"{mamba['mamba_grad_clip']:.2f}"),
        ("$\\beta_1,\\ \\beta_2$", f"{lstm['lstm_beta1']:.2f}, {lstm['lstm_beta2']:.3f}",
         f"{mamba['mamba_beta1']:.2f}, {mamba['mamba_beta2']:.3f}"),
        ("Dropout", f"{lstm['lstm_dropout']:.2f}", f"{mamba['mamba_dropout']:.2f}"),
        ("Batch size", "128", "128"),
        ("Lookback $\\rho$", "365 days", "365 days"),
        ("Spin-up $\\tau$", "365 days", "365 days"),
        ("Epochs (budget)", "60", "60"),
    ]
    lines = ["\\toprule", "\\textbf{Hyperparameter} & \\textbf{CuDNN LSTM} & \\textbf{Mamba-3} \\\\", "\\midrule"]
    lines += [f"{name} & {a} & {b} \\\\" for name, a, b in rows]
    lines.append("\\bottomrule")
    _write("tab_hyperparams", "\n".join(lines),
           pd.DataFrame(rows, columns=["hyperparameter", "lstm", "mamba3"]))


def paired_pvalues() -> None:
    table = pd.read_csv(RESULTS_DIR / "stats_pairwise.csv")
    table = table[table.metric == "NSE"].copy()
    lines = [
        "\\toprule",
        "\\textbf{Tier} & \\textbf{Depth} & $\\Delta$\\textbf{NSE} & \\textbf{Win share} & "
        "\\textbf{Sign $p$} & \\textbf{Blocked $p$} & \\textbf{95\\,\\% CI} \\\\",
        "\\midrule",
    ]
    for tier in (1, 2, 3):
        part = table[table.tier == tier]
        lines.append(f"\\multirow{{{len(part)}}}{{*}}{{T{tier}}}")
        for i, (_, row) in enumerate(part.iterrows()):
            blocked = (f"{row.blocked_p_min:.2e}--{row.blocked_p_max:.2e}"
                       if row.blocked_p_max >= 0.0001 else "$<$5e-6")
            lines.append(
                f"& {int(row.depth_lstm)} & {row.delta_medians:+.4f} & "
                f"{row.win_share_pct:.1f}\\% & {_p(row.sign_p)} & {blocked} & "
                f"[{row.ci_delta_low:+.4f}, {row.ci_delta_high:+.4f}] \\\\"
            )
        if tier < 3:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    _write("tab_paired_pvalues", "\n".join(lines), table)


def ecdf_quantiles() -> None:
    rows = []
    percentiles = (5, 10, 25, 50, 75, 90, 95)
    for model, depth in (("lstm", 1), ("mamba3", 3)):
        frames = []
        for seed in SEEDS[(1, depth)]:
            frame = pd.read_csv(result_files(model, 1, depth, seed)["basin_metrics"])
            frames.append(frame.set_index("gauge_id")[["NSE", "KGE"]])
        stacked = pd.concat(frames, axis=0).groupby(level=0).mean()
        for metric in ("NSE", "KGE"):
            for q in percentiles:
                rows.append({"model": model, "metric": metric, "percentile": q,
                             "value": float(np.nanpercentile(stacked[metric], q))})
    table = pd.DataFrame(rows)
    lines = [
        "\\toprule",
        "\\textbf{Percentile} & \\textbf{LSTM NSE} & \\textbf{Mamba-3 NSE} & "
        "\\textbf{LSTM KGE} & \\textbf{Mamba-3 KGE} \\\\",
        "\\midrule",
    ]
    for q in percentiles:
        vals = {}
        for model in ("lstm", "mamba3"):
            for metric in ("NSE", "KGE"):
                hit = table[(table.model == model) & (table.metric == metric)
                            & (table.percentile == q)]
                vals[(model, metric)] = float(hit["value"].iloc[0])
        lines.append(
            f"P{q} & {vals[('lstm','NSE')]:.4f} & {vals[('mamba3','NSE')]:.4f} & "
            f"{vals[('lstm','KGE')]:.4f} & {vals[('mamba3','KGE')]:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    _write("tab_ecdf_quantiles", "\n".join(lines), table)


def benchmark_full() -> None:
    table = pd.read_csv(RESULTS_DIR / "stats_seed_aggregate.csv")
    lines = [
        "\\toprule",
        "\\textbf{Tier} & \\textbf{Depth} & \\textbf{LSTM NSE} & \\textbf{Mamba-3 NSE} & "
        "\\textbf{$\\Delta$NSE} & \\textbf{LSTM KGE} & \\textbf{Mamba-3 KGE} \\\\",
        "\\midrule",
    ]
    for tier in (1, 2, 3):
        part = table[table.tier == tier].sort_values("depth")
        depths = sorted(part.depth.unique())
        lines.append(f"\\multirow{{{len(depths)}}}{{*}}{{T{tier}}}")
        for i, depth in enumerate(depths):
            l = part[(part.depth == depth) & (part.arch == "lstm")].iloc[0]
            m = part[(part.depth == depth) & (part.arch == "mamba3")].iloc[0]
            lines.append(
                f"& {int(depth)} & {l.nse_mean:.4f}$\\pm${l.nse_sd:.4f} & "
                f"{m.nse_mean:.4f}$\\pm${m.nse_sd:.4f} & {l.nse_mean - m.nse_mean:+.4f} & "
                f"{l.kge_mean:.4f} & {m.kge_mean:.4f} \\\\"
            )
            if i < len(depths) - 1:
                lines.append("\\cmidrule{2-7}")
        if tier < 3:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    _write("tab_benchmark_full", "\n".join(lines), table)


def appendix_training_time() -> None:
    rows = [{"tier": t, "lstm_s": l, "mamba_s": m, "ratio": m / l}
            for t, (l, m) in EPOCH_SECONDS.items()]
    table = pd.DataFrame(rows)
    lines = ["\\toprule",
             "\\textbf{Tier} & \\textbf{LSTM (s/epoch)} & \\textbf{Mamba-3 (s/epoch)} & "
             "\\textbf{Ratio} \\\\", "\\midrule"]
    for _, row in table.iterrows():
        lines.append(f"T{int(row.tier)} & {row.lstm_s:.1f} & {row.mamba_s:.1f} & "
                     f"{row.ratio:.2f}$\\times$ \\\\")
    lines.append("\\bottomrule")
    _write("tab_appendix_training_time", "\n".join(lines), table)


def main() -> int:
    param_matching()
    hyperparams()
    paired_pvalues()
    ecdf_quantiles()
    benchmark_full()
    appendix_training_time()
    print(f"wrote 6 fragments to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
