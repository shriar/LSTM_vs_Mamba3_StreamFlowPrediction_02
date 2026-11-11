"""B15 regression: every KGE path matches the Gupta (2009) formula.

audit item 1.2 -- the headline metric (hydroDL statError), the Fig 9 seasonal
scorer and an independent reference must agree to machine precision, and the
2012 CV-ratio form must be a different number (so a silent swap cannot pass).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import audit_kge  # noqa: E402


class AuditKge(unittest.TestCase):
    def test_all_paths_match_gupta_2009(self):
        self.assertEqual(audit_kge.main(), 0)

    def test_reference_rejects_degenerate_series(self):
        # constant series -> std 0 -> NaN, matching both production paths
        self.assertTrue(np.isnan(audit_kge.kge_gupta2009(
            np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))))
        self.assertTrue(np.isnan(audit_kge.kge_gupta2009(
            np.array([np.nan, 1.0]), np.array([1.0, 2.0]))))

    def test_perfect_simulation_scores_one(self):
        rng = np.random.default_rng(7)
        o = rng.lognormal(0.0, 1.0, 400)
        self.assertAlmostEqual(audit_kge.kge_gupta2009(o, o.copy()), 1.0,
                               places=12)


if __name__ == "__main__":
    unittest.main()
