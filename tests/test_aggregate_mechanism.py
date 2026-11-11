"""Guards for the C11 mechanism aggregator (pure summarisation only)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import aggregate_mechanism as am  # noqa: E402


class SummariseTest(unittest.TestCase):
    def _rows(self):
        return [
            {"model": "lstm", "seed": 1, "mass_balance_val": 1.0,
             "mass_balance_test": 1.1, "mean_gate_i": 0.10},
            {"model": "lstm", "seed": 2, "mass_balance_val": 2.0,
             "mass_balance_test": 1.2, "mean_gate_i": 0.20},
            {"model": "lstm", "seed": 3, "mass_balance_val": 3.0,
             "mass_balance_test": 1.3, "mean_gate_i": 0.30},
        ]

    def test_mean_sd_min_max(self):
        table = am.summarise(self._rows())
        row = table[(table.model == "lstm") & (table.metric == "mass_balance_val")].iloc[0]
        self.assertEqual(int(row.n_seeds), 3)
        self.assertAlmostEqual(float(row["mean"]), 2.0)
        self.assertAlmostEqual(float(row.sd), 1.0)   # ddof=1
        self.assertAlmostEqual(float(row["min"]), 1.0)
        self.assertAlmostEqual(float(row["max"]), 3.0)

    def test_single_seed_has_nan_sd(self):
        table = am.summarise(self._rows()[:1])
        row = table[table.metric == "mass_balance_val"].iloc[0]
        self.assertEqual(int(row.n_seeds), 1)
        self.assertTrue(np.isnan(float(row.sd)))

    def test_one_row_per_model_metric(self):
        table = am.summarise(self._rows())
        keys = list(zip(table.model, table.metric))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn(("lstm", "mean_gate_i"), keys)


if __name__ == "__main__":
    unittest.main()
