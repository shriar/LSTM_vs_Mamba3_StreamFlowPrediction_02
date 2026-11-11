"""Verify that every finding is recorded in MANUSCRIPT_NOTES.md.

Fails if a required phrase is missing, so the ledger cannot silently lose a
finding.  Run: python scripts/check_findings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT.parent / "MANUSCRIPT_NOTES.md"

FINDINGS = [
    ("C1c", "gap does not narrow monotonically", "0.0259 / 0.0168 / 0.0244"),
    ("C1c", "T3 depth-3 block test marginal", "0.042"),
    ("C1c", "blow-up totals across 21 cells", "15 events in 12/21"),
    ("B7", "LSTM replay fidelity", "1.9e-4"),
    ("B7", "Mamba cache path identical", "bit-identical"),
    ("C11", "mass-balance gap was seed noise", "1.035"),
    ("C11", "layer-2 state norm seed-sensitive", "8.87"),
    ("parked", "rho=1095 work parked", "rho=1095"),
    ("G3", "Branch 1 -> JEM", "Journal of Environmental Management"),
    ("D4", "val panels were labelled test", "Median test"),
    ("D4", "Fig 4 empty panels", "empty panels"),
    ("D4", "Fig 6 blow-up counts", "7 basins"),
    ("D4", "Fig 9 delta colour scale", "colour scale"),
    ("D4", "FLV denominator instability", "denominator-unstable"),
    ("D4", "parity restated at 3.65 %", "3.65"),
    ("D4", "three tables deleted", "tab:epoch_robustness"),
    ("D4", "prose remnants owned by A15", "A15"),
    ("D4", "checker green", "8/8"),
    ("D5", "audit outputs exist", "D5_TRACE.csv"),
    ("D5", "nine high-severity stale categories", "Nine high-severity stale"),
    ("D5", "prose quotes old numbers, not new", "absent from the prose"),
    ("1095 policy", "zero mentions target recorded", "target is now zero"),
    ("1095 policy", "hard gate exists", "no 1095 mention"),
]


def main() -> int:
    text = NOTES.read_text(encoding="utf-8")
    missing = []
    for task, label, needle in FINDINGS:
        if needle not in text:
            missing.append(f"{task}: {label} (expected {needle!r})")
    print(f"findings checked: {len(FINDINGS)}; missing: {len(missing)}")
    for row in missing:
        print(f"  MISSING {row}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
