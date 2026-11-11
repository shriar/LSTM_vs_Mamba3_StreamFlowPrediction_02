"""Run every pre-registered test cell through ``eval_test.py``, once each.

This runner has no model, data or metric code.  It only reads
``recipes/eval_manifest.json`` and shells out to the single authorised test
entry point, so it cannot bypass the four locks in ``eval_test.py``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "recipes" / "eval_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "test"
DEFAULT_AUDIT = DEFAULT_OUTPUT / "audit.jsonl"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), PROJECT_ROOT, PROJECT_ROOT.parent):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every cell in the locked test manifest."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def cells(manifest_path: Path) -> list[dict]:
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    if doc.get("schema") != "eval_manifest/v1":
        raise SystemExit(f"unexpected manifest schema: {doc.get('schema')!r}")
    cells = doc.get("cells")
    if not isinstance(cells, list):
        raise SystemExit("manifest 'cells' must be a list")
    return cells


def command_for(cell: dict, args: argparse.Namespace, manifest: Path) -> list[str]:
    command = [
        sys.executable,
        str(SRC_DIR / "eval_test.py"),
        "--recipe", str(cell["recipe"]),
        "--tier", str(int(cell["tier"])),
        "--depth", str(int(cell["depth"])),
        "--seed", str(int(cell["seed"])),
        "--manifest", str(manifest),
        "--output-dir", str(_resolve(args.output_dir)),
        "--audit", str(_resolve(args.audit)),
        "--batch-size", str(args.batch_size),
    ]
    if args.force:
        command.append("--force")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = _resolve(args.manifest)
    if not manifest.is_file():
        print(f"manifest not found: {manifest}", file=sys.stderr)
        return 5
    try:
        queue = cells(manifest)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read manifest: {exc}", file=sys.stderr)
        return 5

    if not queue:
        print("manifest contains no cells; nothing to test.")
        return 0

    print(f"{len(queue)} pre-registered cell(s) in {manifest}")
    for index, cell in enumerate(queue, 1):
        command = command_for(cell, args, manifest)
        print(f"[{index}/{len(queue)}] {' '.join(shlex.quote(p) for p in command)}")
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        if completed.returncode != 0:
            print(
                f"cell {cell.get('id')} failed with exit code "
                f"{completed.returncode}; stopped.",
                file=sys.stderr,
            )
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
