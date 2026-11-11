"""D5 -- numerical consistency audit.

Two passes over the manuscript:

1. **Expected values** -- headline numbers recomputed from the CSVs; the audit
   reports whether each appears in the text and whether the string matches.
2. **Stale values** -- numbers that the regenerated artefacts no longer
   support; any occurrence is a defect with a location.

Outputs ``results/audit/D5_TRACE.csv`` (one row per claim) and
``results/audit/D5_SUMMARY.md`` (severity-ranked worklist for A15).

This is a trace, not a proof: a "present" verdict means the string occurs, not
that the surrounding sentence is correct.  Prose judgement stays with A15.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

WRITING = PROJECT_ROOT.parent / "Writing" / "Submission_WRR" / "05_LaTeX_Source"
SECTIONS = WRITING / "sections"
OUT = RESULTS_DIR / "audit"


def _manuscript_files() -> list[Path]:
    files = sorted(SECTIONS.glob("*.tex"))
    abstract = WRITING / "Abstract.tex"
    if abstract.is_file():
        files.append(abstract)
    return files


def _registry() -> list[dict]:
    """Authoritative values, each with the strings the text may legitimately use."""
    head = pd.read_csv(RESULTS_DIR / "stats_headline.csv")
    pair = pd.read_csv(RESULTS_DIR / "stats_pairwise.csv")
    cap = pd.read_csv(RESULTS_DIR / "table_effective_capacity.csv")
    mech = pd.read_csv(RESULTS_DIR / "mechanism" / "c11_seed_aggregate.csv")

    rows: list[dict] = []
    for tier in (1, 2, 3):
        hit = head[(head.tier == tier) & (head.metric == "NSE")].iloc[0]
        rows.append({
            "id": f"headline_gap_T{tier}",
            "claim": f"Tier-{tier} headline NSE gap",
            "expected": float(hit.delta_medians),
            "strings": [f"{hit.delta_medians:.4f}", f"{hit.delta_medians:.3f}"],
            "source": "results/stats_headline.csv",
            "severity": "high",
        })
        rows.append({
            "id": f"headline_ci_T{tier}",
            "claim": f"Tier-{tier} gap 95% CI",
            "expected": [float(hit.ci_delta_low), float(hit.ci_delta_high)],
            "strings": [f"{hit.ci_delta_low:.4f}", f"{hit.ci_delta_high:.4f}"],
            "source": "results/stats_headline.csv",
            "severity": "high",
        })
    for _, row in pair[pair.metric == "NSE"].iterrows():
        rows.append({
            "id": f"win_share_T{int(row.tier)}_L{int(row.depth_lstm)}",
            "claim": f"win share T{int(row.tier)} depth {int(row.depth_lstm)}",
            "expected": float(row.win_share_pct),
            "strings": [f"{row.win_share_pct:.1f}"],
            "source": "results/stats_pairwise.csv",
            "severity": "medium",
        })
    headline_cap = cap[(cap.model == "mamba3") & (cap.variant == "standard")].iloc[0]
    rows.append({
        "id": "params_headline",
        "claim": "headline parameter counts",
        "expected": int(headline_cap.params_total),
        "strings": [f"{int(headline_cap.params_total):,}".replace(",", "\\,"), f"{int(headline_cap.params_total):,}".replace(",", "{,}"), f"{int(headline_cap.params_total):,}"],
        "source": "results/table_effective_capacity.csv",
        "severity": "high",
    })
    lstm_cap = cap[cap.model == "lstm"].iloc[0]
    rows.append({
        "id": "params_lstm",
        "claim": "LSTM headline parameter count",
        "expected": int(lstm_cap.params_total),
        "strings": [f"{int(lstm_cap.params_total):,}".replace(",", "\\,"), f"{int(lstm_cap.params_total):,}".replace(",", "{,}"), f"{int(lstm_cap.params_total):,}"],
        "source": "results/table_effective_capacity.csv",
        "severity": "high",
    })
    for metric, label in (("mass_balance_val", "val mass balance"), ("mean_delta", "mean delta")):
        for model in ("lstm", "mamba3"):
            hit = mech[(mech.model == model) & (mech.metric == metric)]
            if hit.empty:
                continue
            rows.append({
                "id": f"{metric}_{model}",
                "claim": f"{label} ({model})",
                "expected": float(hit["mean"].iloc[0]),
                "strings": [f"{hit['mean'].iloc[0]:.3f}", f"{hit['mean'].iloc[0]:.2f}"],
                "source": "results/mechanism/c11_seed_aggregate.csv",
                "severity": "medium" if "mass" in metric else "low",
            })
    return rows


STALE = [
    ("22.7", "old FLV median (LSTM); regenerated data give ~47", "high"),
    ("25.2", "old FLV median (Mamba-3); regenerated data give ~47", "high"),
    ("18 configurations", "benchmark is 3 tiers x 3 depths", "high"),
    ("18-run", "benchmark is 3 tiers x 3 depths", "medium"),
    ("0.50\\%", "parity claim; real same-depth max is 3.65%", "high"),
    ("0.49\\%", "parity claim; real same-depth max is 3.65%", "high"),
    ("Adapter", "the headline wiring is standard; Adapter is not reported", "high"),
    ("1095", "rho=1095 was not run (C6 parked)", "high"),
    ("111111", "single fixed seed is obsolete; 5 seeds at the headline", "medium"),
    ("0.752", "old single-seed LSTM NSE", "medium"),
    ("0.741", "old Adapter best NSE", "medium"),
    ("30-epoch", "the budget is 60 epochs", "high"),
    ("30 epochs", "the budget is 60 epochs", "high"),
]


def _scan(files: list[Path], needle: str) -> list[tuple[str, int, str]]:
    hits = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(re.escape(needle), text):
            line = text.count("\n", 0, match.start()) + 1
            start = max(0, match.start() - 70)
            excerpt = text[start:match.start() + len(needle) + 70].replace("\n", " ")
            hits.append((path.name, line, re.sub(r"\s+", " ", excerpt)))
    return hits


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = _manuscript_files()
    registry = _registry()

    trace: list[dict] = []
    for row in registry:
        found = []
        locations = []
        for needle in row["strings"]:
            for name, line, excerpt in _scan(files, needle):
                found.append(needle)
                locations.append(f"{name}:{line}")
        trace.append({
            "id": row["id"],
            "claim": row["claim"],
            "expected": str(row["expected"]),
            "strings_searched": " | ".join(row["strings"]),
            "found": "yes" if found else "no",
            "locations": "; ".join(locations[:4]),
            "verdict": "present" if found else "ABSENT",
            "severity": row["severity"],
            "source": row["source"],
        })

    stale_rows = []
    for needle, why, severity in STALE:
        hits = _scan(files, needle)
        stale_rows.append({
            "id": f"stale:{needle}",
            "claim": why,
            "expected": "0 occurrences",
            "strings_searched": needle,
            "found": str(len(hits)),
            "locations": "; ".join(f"{n}:{l}" for n, l, _ in hits[:5]),
            "verdict": "STALE" if hits else "clean",
            "severity": severity,
            "source": "D4 artefacts (figures/tables)",
        })

    all_rows = trace + stale_rows
    pd.DataFrame(all_rows).to_csv(OUT / "D5_TRACE.csv", index=False)

    counts = pd.DataFrame(all_rows).groupby("verdict").size().to_dict()
    stale_high = [r for r in stale_rows if r["verdict"] == "STALE" and r["severity"] == "high"]
    lines = [
        "# D5 — numerical consistency audit",
        "",
        f"Manuscript files scanned: {len(files)} · rows: {len(all_rows)} · {counts}",
        "",
        "## High-severity stale content (must be rewritten by A15)",
        "",
    ]
    for row in stale_high:
        lines.append(f"- **{row['strings_searched']}** — {row['claim']}  \n  at {row['locations']}")
    lines += [
        "",
        "## Registry values that are absent from the text",
        "",
    ]
    for row in trace:
        if row["verdict"] == "ABSENT":
            lines.append(f"- {row['claim']} = {row['expected']} ({row['source']})")
    lines += [
        "",
        "## Reading this file",
        "",
        "`present` means the string occurs somewhere, not that its sentence is correct.",
        "`ABSENT` means the regenerated value is not quoted in prose; it may still sit in a",
        "generated table, so ABSENT is a prompt to check the sentence, not a defect by itself.",
        "`STALE` means the number is contradicted by the regenerated artefacts.",
        "Prose judgement and replacement remain with A15; this file is its worklist.",
        "",
        "## Legitimate exceptions",
        "",
        "- `1095` must not appear anywhere: the paper does not report that horizon and",
        "  does not name it. The explanation belongs in the response letter (D1).",
        "- `30-epoch` mentions that describe the *submitted* paper's protocol may stay if the",
        "  sentence says so explicitly; all others must become 60 epochs.",
    ]
    (OUT / "D5_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"rows: {len(all_rows)}  verdicts: {counts}")
    print(f"wrote {OUT / 'D5_TRACE.csv'} and D5_SUMMARY.md")
    print(f"high-severity stale strings: {len(stale_high)}")
    for row in stale_high:
        print(f"  {row['strings_searched']:>18}  {row['found']:>3} hits  {row['locations'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
