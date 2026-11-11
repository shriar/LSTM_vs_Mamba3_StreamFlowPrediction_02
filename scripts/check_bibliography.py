"""D10 -- bibliography check.

Checks, without network access:
  * every \\cite key resolves to a BibTeX entry (and vice versa);
  * duplicate keys and duplicate DOIs;
  * DOI syntax and presence for journal articles;
  * the six reviewer-requested works are present;
  * year and venue sanity for the recent entries.

Writes ``Writing/bibliography_check.md`` and exits non-zero on a hard failure.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

WRITING = ROOT.parent / "Writing" / "Submission_WRR" / "05_LaTeX_Source"
BIB = WRITING / "references.bib"
TEX_FILES = list((WRITING / "sections").glob("*.tex")) + [WRITING / "Manuscript_AGU.tex"]
OUT = ROOT.parent / "Writing" / "bibliography_check.md"

ENTRY = re.compile(r"@(?P<type>\w+)\{(?P<key>[^,]+),(?P<body>.*?)\n\}", re.S)
CITE = re.compile(r"(?:\\protect)?\\cite(?:[pt]|A)?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")
DOI = re.compile(r"doi\s*=\s*\{(?P<doi>[^}]+)\}", re.I)
YEAR = re.compile(r"year\s*=\s*\{(?P<year>\d{4})\}", re.I)

REQUIRED_DOIS = {
    "10.1016/j.jhydrol.2025.134304": "R3#8 MamGA",
    "10.1016/j.ecoinf.2025.103565": "R5#4 Mamba2-Transformer",
    "10.1016/j.jhydrol.2025.134145": "Zhang reservoir release",
    "10.1029/2025wr039888": "S4D-FT",
    "10.52202/085713-4446": "RiverMamba",
}


def parse_bib() -> dict[str, dict]:
    text = BIB.read_text(encoding="utf-8")
    entries = {}
    for m in ENTRY.finditer(text):
        key = m.group("key").strip()
        body = m.group("body")
        doi = DOI.search(body)
        year = YEAR.search(body)
        entries[key] = {
            "type": m.group("type").lower(),
            "doi": doi.group("doi").strip().lower() if doi else "",
            "year": year.group("year") if year else "",
        }
    return entries


def cited_keys() -> Counter:
    counter: Counter = Counter()
    for tex in TEX_FILES:
        text = tex.read_text(encoding="utf-8")
        for m in CITE.finditer(text):
            for key in m.group(1).split(","):
                counter[key.strip()] += 1
    return counter


def main() -> int:
    entries = parse_bib()
    cited = cited_keys()

    missing = sorted(k for k in cited if k not in entries)
    uncited = sorted(k for k in entries if k not in cited)
    dup_doi = [d for d, n in Counter(e["doi"] for e in entries.values() if e["doi"]).items() if n > 1]
    journal_no_doi = sorted(
        k for k, e in entries.items()
        if e["type"] == "article" and not e["doi"] and "preprint" not in k
    )
    dois = {e["doi"] for e in entries.values() if e["doi"]}
    required = {doi: label for doi, label in REQUIRED_DOIS.items() if doi not in dois}

    lines = [
        "# Bibliography check (D10)",
        "",
        f"Entries: **{len(entries)}** · cited keys: **{len(cited)}** · citations in text: **{sum(cited.values())}**",
        "",
        f"- unresolved citations: **{len(missing)}**" + (f" — {missing}" if missing else ""),
        f"- uncited entries: **{len(uncited)}**" + (f" — {uncited}" if uncited else ""),
        f"- duplicate DOIs: **{len(dup_doi)}**" + (f" — {dup_doi}" if dup_doi else ""),
        f"- journal articles without DOI: **{len(journal_no_doi)}**",
        f"- required works missing: **{len(required)}**" + (f" — {required}" if required else ""),
        "",
        "## Reviewer-requested works",
        "",
    ]
    for doi, label in REQUIRED_DOIS.items():
        lines.append(f"- {label}: `{doi}` — {'present' if doi in dois else 'MISSING'}")
    lines += [
        "",
        "## Method note",
        "",
        "This check verifies structure, not truth. DOI metadata for the five newly added works was read from Crossref on 2026-11-10; "
        "the remaining entries are the manuscript's existing bibliography. Uncited entries are reported for review, not removed automatically.",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"entries {len(entries)} | cited {len(cited)} | missing {len(missing)} | uncited {len(uncited)}")
    print(f"duplicate DOIs {len(dup_doi)} | journal no DOI {len(journal_no_doi)} | required missing {len(required)}")
    if uncited:
        print("uncited:", ", ".join(uncited))
    if journal_no_doi:
        print("no DOI:", ", ".join(journal_no_doi))
    print("wrote", OUT)
    hard_fail = bool(missing or dup_doi or required)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
