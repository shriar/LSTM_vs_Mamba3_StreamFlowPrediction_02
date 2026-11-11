"""D4.4 -- copy regenerated figures into the manuscript's figure folder.

The manuscript includes ``figures/results/<stem>``.  This script copies the
PNG, the vector PDF and the ``_data.csv`` companion for every regenerated
stem, so the LaTeX build reads the new artefacts.  Static figures (map,
architecture) are never touched.

Usage: python scripts/sync_figures.py [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paths import FIGURES_DIR, PROJECT_ROOT  # noqa: E402

TARGET = PROJECT_ROOT.parent / "Writing" / "Submission_WRR" / "05_LaTeX_Source" / "figures" / "results"
SOURCES = (FIGURES_DIR / "paper",)
STATIC = {"fig01_camels_conus_hydroclimatic_4panel", "fig02a_lstm_architecture",
          "fig02b_mamba_architecture"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    TARGET.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source in SOURCES:
        if not source.is_dir():
            continue
        for png in sorted(source.glob("*.png")):
            stem = png.stem
            if stem in STATIC:
                continue
            for ext in (".png", ".pdf"):
                src = source / f"{stem}{ext}"
                if src.is_file():
                    dst = TARGET / f"{stem}{ext}"
                    if not args.dry_run:
                        shutil.copy2(src, dst)
                    copied += 1
            data = source / f"{stem}_data.csv"
            if data.is_file():
                if not args.dry_run:
                    shutil.copy2(data, TARGET / data.name)
                copied += 1
        summary = source / "ladder_summary.csv"
        if summary.is_file() and not args.dry_run:
            shutil.copy2(summary, TARGET / summary.name)
    print(f"synced {copied} files -> {TARGET}"
          + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
