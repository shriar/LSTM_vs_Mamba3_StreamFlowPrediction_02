"""D4.0 -- inventory of every figure and table in the manuscript.

Reads ``sections/*.tex``, records each ``\\includegraphics`` and each
``\\begin{table}`` with its caption and label, and writes
``D4_INVENTORY.csv`` in the project root.  Read-only with respect to the
manuscript.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

SECTIONS = PROJECT_ROOT.parent / "Writing" / "Submission_WRR" / "05_LaTeX_Source" / "sections"
OUTPUT = PROJECT_ROOT.parent / "D4_INVENTORY.csv"

INCLUDEGRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{(?P<file>[^}]+)\}")
CAPTION = re.compile(r"\\caption\{(?P<text>[^{}]+)\}", re.S)
LABEL = re.compile(r"\\label\{(?P<label>[^}]+)\}")
FIG_LABEL = re.compile(r"\\label\{(?P<label>fig:[^}]+)\}")


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def scan(sections: Path = SECTIONS) -> tuple[list[dict], list[dict]]:
    figures: list[dict] = []
    tables: list[dict] = []
    files = sorted(sections.glob("*.tex"))
    if not files:
        raise SystemExit(f"no .tex files in {sections}")
    for tex in files:
        text = tex.read_text(encoding="utf-8")
        for match in INCLUDEGRAPHICS.finditer(text):
            window = text[match.end():match.end() + 1500]
            fig_label = FIG_LABEL.search(window)
            figures.append({
                "kind": "figure",
                "file": tex.name,
                "line": _line_number(text, match.start()),
                "label": fig_label.group("label") if fig_label else "",
                "caption_or_path": match.group("file"),
            })
        for match in re.finditer(r"\\begin\{table\}", text):
            block = text[match.start():match.start() + 4000]
            caption = CAPTION.search(block)
            label = LABEL.search(block)
            caption_text = caption.group("text") if caption else ""
            caption_text = re.sub(r"\s+", " ", caption_text).strip()
            tables.append({
                "kind": "table",
                "file": tex.name,
                "line": _line_number(text, match.start()),
                "label": label.group("label") if label else "",
                "caption_or_path": caption_text,
            })
    return figures, tables


def main() -> int:
    figures, tables = scan()
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["kind", "file", "line", "label", "caption_or_path"]
        )
        writer.writeheader()
        for row in figures + tables:
            writer.writerow(row)
    print(f"figures: {len(figures)}  tables: {len(tables)}")
    for row in figures:
        print(f"  FIG  {row['file']}:{row['line']:<4} {row['caption_or_path']}")
    for row in tables:
        print(f"  TAB  {row['file']}:{row['line']:<4} {row['label']}")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
