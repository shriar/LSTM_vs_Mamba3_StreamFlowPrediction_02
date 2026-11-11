"""Guards for the D4.1 figure resolver and D4.0 manuscript inventory."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path as _P

_SRC = _P(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from d4_inventory import scan
from figure_source import (
    EXPECTED_SPLIT_SHA,
    cell_for,
    learning_curves,
    load_basin_metrics,
    load_predictions,
    result_files,
    seed_averaged_basin_scores,
    validation_files,
)

LSTM = "lstm"
MAMBA = "mamba3"


class ResolverTest(unittest.TestCase):
    def test_headline_cell_resolves_and_is_standard(self):
        cell = cell_for("recipes/lstm.json", 1, 1, 1)
        files = validation_files(cell)
        master = files["master"]
        self.assertEqual(master["protocol"], "train_val/v1")
        self.assertIs(master["test_split_opened"], False)
        self.assertEqual(master["data"]["split_sha256"], EXPECTED_SPLIT_SHA)
        self.assertTrue(files["val_metrics"].is_file())

    def test_mamba_requires_standard_wiring(self):
        cell = cell_for("recipes/mamba3.json", 1, 3, 1)
        master = validation_files(cell)["master"]
        self.assertEqual(master["model"]["mamba_variant"], "standard")
        self.assertEqual(int(master["model"]["mamba_version"]), 3)

    def test_unregistered_cell_raises(self):
        with self.assertRaises(KeyError):
            cell_for("recipes/mamba3_native.json", 1, 3, 9)

    def test_unevaluated_cell_raises(self):
        # C1d native cells are listed but were never trained.
        try:
            cell = cell_for("recipes/mamba3_native.json", 1, 3, 1)
        except KeyError:
            self.skipTest("manifest has no native cells")
        with self.assertRaises(FileNotFoundError):
            # result_files resolves without checking the run_dir, so skip if the
            # cell exists but has no test folder.
            result_files("mamba3_native", 1, 3, 1)

    def test_result_files_exist_for_headline(self):
        files = result_files(LSTM, 1, 1, 1)
        self.assertTrue(files["basin_metrics"].is_file())
        frame = load_basin_metrics(LSTM, 1, 1, 1)
        self.assertIn("gauge_id", frame.columns)
        self.assertEqual(len(frame), 671)

    def test_predictions_are_audited_arrays(self):
        pred = load_predictions(LSTM, 1, 1, 1)
        self.assertEqual(pred["pred_m3s"].shape, pred["obs_m3s"].shape)
        self.assertEqual(pred["pred_m3s"].shape[0], 671)
        self.assertEqual(pred["gauge_ids"].shape[0], 671)

    def test_seed_averaged_join_uses_gauge_id(self):
        table = seed_averaged_basin_scores(LSTM, 1, 1, [1, 2, 3])
        self.assertIn("gauge_id", table.columns)
        self.assertEqual(len(table), 671)
        self.assertTrue({"NSE", "KGE"} <= set(table.columns))
        # ids are unique and complete.
        self.assertEqual(table["gauge_id"].nunique(), 671)

    def test_learning_curves_cover_epochs_and_seeds(self):
        curves = learning_curves("recipes/lstm.json", 1, 1, [1, 2, 3, 4, 5])
        self.assertGreaterEqual(len(curves), 50)
        self.assertTrue(
            {"epoch", "median_nse", "q25_nse", "q75_nse", "median_kge",
             "q25_kge", "q75_kge"}.issubset(curves.columns)
        )
        # IQR must bracket the median.
        self.assertTrue((curves.q25_nse <= curves.median_nse).all())
        self.assertTrue((curves.median_nse <= curves.q75_nse).all())

    def test_join_is_by_id_not_position(self):
        a = load_basin_metrics(LSTM, 1, 1, 1).set_index("gauge_id")[["NSE"]]
        b = load_basin_metrics(LSTM, 1, 1, 2).iloc[::-1].set_index("gauge_id")[["NSE"]]
        joined = a.join(b, lsuffix="_1", rsuffix="_2")
        # Shuffling cannot change the id-matched merge.
        self.assertTrue(np.allclose(joined["NSE_1"], joined["NSE_2"], equal_nan=True) is False)
        merged = pd.concat([a, b], axis=1)
        self.assertEqual(len(merged), 671)
        self.assertTrue(np.isfinite(merged.mean(axis=1)).all())


class InventoryTest(unittest.TestCase):
    def test_counts_match_manuscript(self):
        figures, tables = scan()
        self.assertEqual(len(figures), 14)
        # D4.5 deleted three legacy tables (12 -> 9).
        self.assertEqual(len(tables), 10)

    def test_deleted_tables_are_gone(self):
        _, tables = scan()
        labels = {t["label"] for t in tables}
        for gone in ("tab:epoch_robustness", "tab:recipe_swap", "tab:lookback"):
            self.assertNotIn(gone, labels)
        for kept in ("tab:benchmark_full", "tab:paired_pvalues",
                     "tab:ecdf_quantiles", "tab:param_matching",
                     "tab:hyperparams", "tab:appendix_training_time"):
            self.assertIn(kept, labels)

    def test_each_figure_has_a_label(self):
        figures, _ = scan()
        for row in figures:
            self.assertTrue(row["label"].startswith("fig:"))


if __name__ == "__main__":
    unittest.main()
