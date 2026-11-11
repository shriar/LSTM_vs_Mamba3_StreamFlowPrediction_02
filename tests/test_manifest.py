"""Manifest integrity: eval_test match keys must be unique across all cells.

eval_test identifies a cell by (recipe, tier, depth, seed); a duplicate key
makes the intended cell ambiguous, and eval_test refuses it (exit 5). This
bit on 2026-10-29: the C6 cells shared keys with C1a (they differ only in
epochs/rho, which are not part of the key), so the first two C1a evals were
refused. The fix gave C6 its own recipe files; this test keeps the property.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "recipes" / "eval_manifest.json"


class ManifestIntegrity(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_match_keys_are_unique(self):
        keys = [
            (c["recipe"], c["tier"], c["depth"], c["seed"])
            for c in self.manifest["cells"]
        ]
        dupes = [k for k in set(keys) if keys.count(k) > 1]
        self.assertEqual(dupes, [])

    def test_run_dirs_are_unique(self):
        dirs = [c["run_dir"] for c in self.manifest["cells"]]
        dupes = [d for d in set(dirs) if dirs.count(d) > 1]
        self.assertEqual(dupes, [])

    def test_split_hash_matches(self):
        self.assertEqual(
            self.manifest["split_sha256"],
            "9efbe60e790965f203b037d7e630828f7fe36285d4c86ee5c3ddbd93ec13cc58",
        )


if __name__ == "__main__":
    unittest.main()
