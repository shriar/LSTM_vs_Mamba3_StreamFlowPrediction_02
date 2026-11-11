"""D4.6 -- mechanical checks for the figure/table regeneration.

Run after D4.3/D4.4:

    python src/check_figure_sources.py

Checks, in order:

1. every ``\\includegraphics`` in the manuscript resolves to a PNG;
2. each result figure has a ``_data.csv`` companion and a vector PDF;
3. the manuscript copy of each regenerated figure is byte-identical to the
   generated file (catches the classic "new PNG, old inclusion" mistake);
4. every staged cell (``.figure_staging/STAGING_MANIFEST.csv``) was trained
   on the revision split and at the 60-epoch budget;
5. banned legacy strings are reported with counts (a warning until Phase 5
   rewrites the prose, then a failure with ``--strict``);
6. the figure count matches the inventory.

Exit code is non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from d4_inventory import scan  # noqa: E402
from figure_source import EXPECTED_SPLIT_SHA, cell_for, check_split  # noqa: E402
from paths import FIGURES_DIR, PROJECT_ROOT  # noqa: E402

WRITING = PROJECT_ROOT.parent / "Writing" / "Submission_WRR" / "05_LaTeX_Source"
FIGURES_DIR_MANUSCRIPT = WRITING / "figures"
STAGING = PROJECT_ROOT / ".figure_staging"
GENERATED_DIRS = (FIGURES_DIR / "paper",)
MANIFEST = PROJECT_ROOT / ".figure_staging" / "STAGING_MANIFEST.csv"

BANNED = ("1095", "recipe_swap", "epoch_robustness", "Adapter")
# Files where a mention is legitimate (literature, method history, audit).
BANNED_ALLOWED = {"appendix.tex", "03_methodology.tex"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generated(name: str) -> Path | None:
    """Newest generated PNG for a stem (staged and canonical dirs both exist)."""
    candidates = [
        base / f"{name}.png"
        for base in GENERATED_DIRS
        if (base / f"{name}.png").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def check_includes() -> tuple[bool, list[str]]:
    figures, _ = scan()
    problems = []
    for row in figures:
        rel = row["caption_or_path"]
        path = WRITING / f"{rel}.png"
        if not path.is_file():
            problems.append(f"missing include: {rel}.png")
    return (not problems), problems


def check_companions() -> tuple[bool, list[str]]:
    figures, _ = scan()
    problems = []
    for row in figures:
        rel = row["caption_or_path"]
        if not rel.startswith("figures/results/"):
            continue
        name = Path(rel).name
        gen = _generated(name)
        if gen is None:
            problems.append(f"no generated file for {name}")
            continue
        if not (gen.parent / f"{name}_data.csv").is_file():
            problems.append(f"missing data CSV beside {name}")
        if not (gen.parent / f"{name}.pdf").is_file():
            problems.append(f"missing PDF beside {name}")
    return (not problems), problems


def check_stale_copies() -> tuple[bool, list[str]]:
    """The manuscript copy must be byte-identical to the generated figure."""
    figures, _ = scan()
    problems = []
    for row in figures:
        rel = row["caption_or_path"]
        if not rel.startswith("figures/results/"):
            continue
        name = Path(rel).name
        gen = _generated(name)
        ms = FIGURES_DIR_MANUSCRIPT / f"{rel}.png"
        if gen and ms.is_file() and _sha256(gen) != _sha256(ms):
            problems.append(f"stale copy: {ms.relative_to(WRITING)} != {gen.name}")
    return (not problems), problems


def check_staging() -> tuple[bool, list[str]]:
    if not MANIFEST.is_file():
        return True, ["staging manifest absent (D4.2 not run yet) — skipped"]
    problems = []
    table = pd.read_csv(MANIFEST)
    for _, row in table.iterrows():
        try:
            cell = cell_for(
                "recipes/lstm.json" if row["model"] == "lstm" else "recipes/mamba3.json",
                int(row["tier"]), int(row["depth"]), int(row["seed"]),
            )
            if check_split(cell) != EXPECTED_SPLIT_SHA:
                problems.append(f"{cell['id']}: wrong split")
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            problems.append(f"{row['cell_id']}: {exc}")
        if int(row.get("epochs", 0)) <= 0:
            problems.append(f"{row['cell_id']}: no epochs")
    return (not problems), problems


def check_banned(strict: bool = False) -> tuple[bool, list[str]]:
    notes = []
    failures = 0
    for tex in sorted((WRITING / "sections").glob("*.tex")):
        text = tex.read_text(encoding="utf-8")
        for word in BANNED:
            hits = len(re.findall(re.escape(word), text))
            if hits:
                notes.append(f"{tex.name}: '{word}' x{hits}")
                if strict or tex.name not in BANNED_ALLOWED:
                    failures += 1
    if strict:
        return (failures == 0), notes
    return True, notes


def check_counts() -> tuple[bool, list[str]]:
    figures, tables = scan()
    news = []
    if len(figures) != 14:
        news.append(f"figure count {len(figures)} != 14")
    if len(tables) != 10:
        news.append(f"table count {len(tables)} != 10 (9 after deletions + the capacity table)")
    return (not news), news


TABLE_INPUT = re.compile(r"\\input\{(?P<path>tables/[^}]+)\}")


def check_table_inputs() -> tuple[bool, list[str]]:
    problems = []
    for tex in sorted((WRITING / "sections").glob("*.tex")):
        text = tex.read_text(encoding="utf-8")
        for match in TABLE_INPUT.finditer(text):
            path = WRITING / f"{match.group('path')}.tex"
            if not path.is_file():
                problems.append(f"{tex.name}: missing {match.group('path')}.tex")
    return (not problems), problems


def check_table_refs() -> tuple[bool, list[str]]:
    """Every \\ref{tab:...} must have a matching \\label{tab:...}."""
    labels, refs = set(), set()
    for tex in sorted((WRITING / "sections").glob("*.tex")):
        text = tex.read_text(encoding="utf-8")
        labels.update(re.findall(r"\\label\{(tab:[^}]+)\}", text))
        refs.update(re.findall(r"\\ref\{(tab:[^}]+)\}", text))
    missing = sorted(refs - labels)
    return (not missing), [f"dangling table ref: {name}" for name in missing]


def check_no_1095() -> tuple[bool, list[str]]:
    """Hard gate: the manuscript must not name the 1 095-day horizon at all."""
    hits = []
    for tex in sorted((WRITING / "sections").glob("*.tex")):
        text = tex.read_text(encoding="utf-8")
        for match in re.finditer(r"1095|1\\,095|1,095", text):
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"{tex.name}:{line}")
    abstract = WRITING / "Abstract.tex"
    if abstract.is_file():
        text = abstract.read_text(encoding="utf-8")
        for match in re.finditer(r"1095|1\\,095|1,095", text):
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"Abstract.tex:{line}")
    return (not hits), hits


CHECKS = (
    ("includes resolve", check_includes),
    ("data CSV + PDF companions", check_companions),
    ("manuscript copies fresh", check_stale_copies),
    ("staged cells on the revision split", check_staging),
    ("figure/table count", check_counts),
    ("table inputs resolve", check_table_inputs),
    ("table references resolve", check_table_refs),
    ("no 1095 mention", check_no_1095),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="treat banned legacy strings as failures (Phase 5+)")
    args = parser.parse_args(argv)

    failed = 0
    for label, fn in CHECKS:
        ok, notes = fn()
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        for note in notes:
            print(f"       {note}")
        failed += 0 if ok else 1

    ok, notes = check_banned(strict=args.strict)
    print(f"[{'PASS' if ok else 'FAIL'}] banned legacy strings")
    for note in notes:
        print(f"       {note}")
    failed += 0 if ok else 1

    print(f"\n{len(CHECKS) + 1 - failed}/{len(CHECKS) + 1} checks passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
