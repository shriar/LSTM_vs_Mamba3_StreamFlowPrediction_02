"""Repair the cell identity in artefacts written before the D5 audit fix.

``eval_test._write_outputs`` used the shared recipe's tier/depth/seed, so every
non-headline cell was mislabelled in ``test_metrics.csv`` and ``audit.jsonl``.
The metrics themselves are correct; only the three identity columns are wrong.

This script rewrites those three fields from the manifest cell id.  It is
idempotent and prints what it changed.  Run once, from the experiment dir:

    python scripts/repair_cell_identity.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from paths import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

MANIFEST = PROJECT_ROOT / "recipes" / "eval_manifest.json"
TEST_DIR = RESULTS_DIR / "test"
AUDIT = TEST_DIR / "audit.jsonl"


def _cells() -> dict[str, dict]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {str(c["id"]): c for c in payload["cells"]}


def repair_csv(path: Path, cells: dict[str, dict], dry: bool) -> bool:
    frame = pd.read_csv(path)
    if frame.empty or "cell_id" not in frame:
        return False
    cell = cells.get(str(frame.iloc[0]["cell_id"]))
    if cell is None:
        print(f"  skip {path}: unknown cell id")
        return False
    before = (int(frame.iloc[0]["tier"]), int(frame.iloc[0]["depth"]), int(frame.iloc[0]["seed"]))
    after = (int(cell["tier"]), int(cell["depth"]), int(cell["seed"]))
    if before == after:
        return False
    frame.loc[:, "tier"] = after[0]
    frame.loc[:, "depth"] = after[1]
    frame.loc[:, "seed"] = after[2]
    if not dry:
        frame.to_csv(path, index=False)
    print(f"  {path.parent.name}: {before} -> {after}")
    return True


def repair_audit(path: Path, cells: dict[str, dict], dry: bool) -> int:
    if not path.is_file():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    out, changed = [], 0
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        cell = cells.get(str(rec.get("cell_id")))
        if cell is None:
            out.append(line)
            continue
        before = (rec.get("tier"), rec.get("depth"), rec.get("seed"))
        after = (int(cell["tier"]), int(cell["depth"]), int(cell["seed"]))
        if before != after:
            rec["tier"], rec["depth"], rec["seed"] = after
            changed += 1
        out.append(json.dumps(rec, sort_keys=True))
    if changed and not dry:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cells = _cells()
    csvs = 0
    for path in sorted(TEST_DIR.glob("*/test_metrics.csv")):
        csvs += int(repair_csv(path, cells, args.dry_run))
    audit = repair_audit(AUDIT, cells, args.dry_run)
    print(f"csv files repaired: {csvs}; audit lines repaired: {audit}"
          + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
