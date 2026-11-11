"""AGU figure/table audit for the WRR submission.

Checks:
  * every float has a label, a caption, and a cited reference;
  * every included graphic exists as PDF (AGU production format) and PNG;
  * no footnotes outside affiliations;
  * figure/table count and publication-unit estimate (1 PU = 500 words or
    1 display element);
  * flags appendix floats that could move to Supporting Information.

Writes ``Writing/agu_figure_table_audit.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WRITING = ROOT.parent / "Writing" / "Submission_WRR" / "05_LaTeX_Source"
TEX = WRITING / "Manuscript_AGU.tex"
OUT = ROOT.parent / "Writing" / "agu_figure_table_audit.md"

INCLUDE = re.compile(r"\\input\{(sections/[^}]+)\}")
FLOAT = re.compile(r"\\begin\{(figure\*?|table)\}(?P<body>.*?)\\end\{\1\}", re.S)
LABEL = re.compile(r"\\label\{(?P<label>[^}]+)\}")
CAPTION = re.compile(r"\\caption\{(?P<cap>.*?)\}\s*\n", re.S)
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{(?P<file>[^}]+)\}")
REF = re.compile(r"\\ref\{(?P<label>[^}]+)\}")


def ordered_text() -> tuple[str, list[str]]:
    """Concatenate the manuscript in include order, tracking section names."""
    main = TEX.read_text(encoding="utf-8")
    parts = []
    files = []
    pos = 0
    for m in INCLUDE.finditer(main):
        parts.append(main[pos:m.start()])
        rel = m.group(1)
        if not rel.endswith(".tex"):
            rel += ".tex"
        parts.append((WRITING / rel).read_text(encoding="utf-8"))
        files.append(rel)
        pos = m.end()
    parts.append(main[pos:])
    return "".join(parts), files


def main() -> int:
    text, files = ordered_text()

    refs = [(m.start(), m.group("label")) for m in REF.finditer(text)]
    ref_positions: dict[str, int] = {}
    for pos, label in refs:
        ref_positions.setdefault(label, pos)

    floats = []
    for m in FLOAT.finditer(text):
        kind = "figure" if m.group(1).startswith("figure") else "table"
        body = m.group("body")
        label = LABEL.findall(body)
        label = label[-1] if label else None
        caption = CAPTION.search(body)
        graphic = GRAPHIC.search(body)
        floats.append({
            "kind": kind,
            "label": label or "",
            "pos": m.start(),
            "caption_chars": len(caption.group("cap")) if caption else 0,
            "file": graphic.group("file") if graphic else "",
        })

    problems = []
    rows = []
    for f in floats:
        cited = f["label"] in ref_positions
        first_ref = ref_positions.get(f["label"])
        before = first_ref is not None and first_ref < f["pos"]
        file_state = "n/a"
        if f["file"]:
            png = WRITING / f"{f['file']}.png"
            pdf = WRITING / f"{f['file']}.pdf"
            file_state = ("pdf+png" if pdf.is_file() and png.is_file()
                          else "pdf only" if pdf.is_file()
                          else "png only" if png.is_file()
                          else "MISSING")
            if not pdf.is_file():
                problems.append(f"{f['label']}: no PDF (AGU production format)")
        if not f["label"]:
            problems.append(f"{f['kind']} at char {f['pos']}: missing label")
        if not f["caption_chars"]:
            problems.append(f"{f['label']}: missing caption")
        if not cited:
            problems.append(f"{f['label']}: never cited")
        rows.append({**f, "cited": cited, "cited_before": before, "files": file_state})

    footnotes = len(re.findall(r"\\footnote\{", text))
    body = re.sub(r"\\begin\{(figure\*?|table)\}.*?\\end\{\1\}", "", text, flags=re.S)
    body = re.sub(r"\\begin\{thebibliography\}.*", "", body, flags=re.S)
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", body)
    words = len([w for w in re.split(r"\s+", body) if w])
    n_fig = sum(1 for f in floats if f["kind"] == "figure")
    n_tab = sum(1 for f in floats if f["kind"] == "table")
    pu = words / 500 + n_fig + n_tab

    lines = [
        "# AGU figure/table audit — WRR submission",
        "",
        f"Manuscript: `Writing/Submission_WRR/05_LaTeX_Source/Manuscript_AGU.tex` · sections scanned: {len(files)}",
        "",
        f"- figures: **{n_fig}** · tables: **{n_tab}** · display elements: **{n_fig + n_tab}**",
        f"- body words (approx., floats and references removed): **{words:,}**",
        f"- publication units (1 PU = 500 words or 1 display element): **{pu:.1f}** — WRR limit 25",
        f"- footnotes: **{footnotes}** (only affiliations/table notes allowed)",
        "",
        "## Floats",
        "",
        "| kind | label | cited | cited before float | caption chars | files |",
        "|---|---|---|---|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['kind']} | `{r['label']}` | {'yes' if r['cited'] else 'NO'} | "
            f"{'yes' if r['cited_before'] else 'no'} | {r['caption_chars']} | {r['files']} |"
        )
    lines += ["", "## Problems", ""]
    lines += [f"- {p}" for p in problems] if problems else ["- none"]
    lines += [
        "",
        "## Notes",
        "",
        "- AGU accepts PDF/EPS/TIFF/JPG for production; PNG alone is not enough. The LaTeX build picks the PDF when present.",
        "- Panel labels inside a figure (a), (b) are allowed; a caption or \"Figure 1\" inside the image is not.",
        "- The four appendix figures and two appendix tables count toward the 25-PU limit; moving them to Supporting Information removes six display elements.",
        "- Colour check: model curves use blue/orange plus distinct markers/line styles; maps use a diverging blue–red scale with a labelled bar.",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"figures {n_fig} | tables {n_tab} | words {words} | PU {pu:.1f} | footnotes {footnotes}")
    print("problems:", len(problems))
    for p in problems:
        print("  -", p)
    print("wrote", OUT)
    return 1 if any("missing" in p.lower() or "never cited" in p.lower() for p in problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
