"""Regression tests for the 2026-11-10 experiment-audit fixes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from config import ExperimentConfig  # noqa: E402
from figure_source import cell_for  # noqa: E402
from stage_figure_inputs import _legacy_exp_and_dir  # noqa: E402
from figure_source import model_params  # noqa: E402


class ConfigDefaultsTest(unittest.TestCase):
    def test_defaults_match_the_revision(self):
        cfg = ExperimentConfig()
        self.assertEqual(cfg.mamba_variant, "standard")
        self.assertEqual(cfg.t_valid, [19951001, 19991001])
        self.assertTrue(cfg.train_file.endswith("train.pkl"))
        self.assertTrue(cfg.validation_file.endswith("val.pkl"))


class CellIdentityTest(unittest.TestCase):
    def test_non_headline_metrics_carry_their_own_identity(self):
        """A shared recipe used to stamp every cell with tier 1 / depth 1 / seed 1."""
        path = ROOT / "results" / "test" / "lstm_T2_L2_Seed1" / "test_metrics.csv"
        if not path.is_file():
            self.skipTest("cell not present")
        row = pd.read_csv(path).iloc[0]
        self.assertEqual((int(row.tier), int(row.depth), int(row.seed)), (2, 2, 1))

    def test_mamba_three_layer_cell(self):
        path = ROOT / "results" / "test" / "mamba3_T1_L1_Seed1" / "test_metrics.csv"
        if not path.is_file():
            self.skipTest("cell not present")
        row = pd.read_csv(path).iloc[0]
        self.assertEqual((int(row.tier), int(row.depth), int(row.seed)), (1, 1, 1))


class StagingNameTest(unittest.TestCase):
    def test_lstm_tier_names_match_output(self):
        for tier, name in ((1, "200K"), (2, "500K"), (3, "1M")):
            cell = cell_for("recipes/lstm.json", tier, 1, 1)
            exp, tier_dir = _legacy_exp_and_dir(cell, model_params(cell))
            self.assertEqual(exp, "LSTM")
            self.assertTrue(tier_dir.name.startswith(f"Tier{tier}_{name}_"),
                            msg=f"{tier}: {tier_dir.name}")


if __name__ == "__main__":
    unittest.main()
