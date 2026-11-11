"""Guards for the D5 numerical audit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import audit_numbers as an  # noqa: E402


class RegistryTest(unittest.TestCase):
    def test_registry_is_non_empty_and_finite(self):
        rows = an._registry()
        self.assertGreaterEqual(len(rows), 10)
        for row in rows:
            self.assertTrue(row["strings"], row["id"])
            self.assertTrue(row["source"].startswith("results/"), row["id"])

    def test_stale_list_covers_the_known_defects(self):
        needles = {needle for needle, _, _ in an.STALE}
        for required in ("Adapter", "1095", "22.7", "25.2", "0.50\\%"):
            self.assertIn(required, needles)

    def test_scan_reports_line_numbers(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.tex"
            path.write_text("line one\nAdapter here\n", encoding="utf-8")
            hits = an._scan([path], "Adapter")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][1], 2)


if __name__ == "__main__":
    unittest.main()
