"""Guards for the multi-seed statistics module (B6).

The tests use small synthetic vectors so they run on CPU in a second; they
exercise the constructions that are easy to get subtly wrong (SD definition,
TOST direction, bootstrap CI, spatial join by id, joint block flips).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stats_multiseed as sm  # noqa: E402
from sign_test_lstm_vs_mamba import blocked_permutation_p  # noqa: E402


class SeedAggregateTest(unittest.TestCase):
    def test_mean_sd_median_iqr(self):
        frame = pd.DataFrame({
            "tier": [1, 1, 1], "depth": [1, 1, 1], "arch": ["lstm"] * 3,
            "seed": [1, 2, 3],
            "median_nse": [0.70, 0.72, 0.74],
            "median_kge": [0.60, 0.62, 0.64],
        })
        agg = sm.aggregate_seeds(frame)
        self.assertEqual(len(agg), 1)
        row = agg.iloc[0]
        self.assertEqual(int(row["n_seeds"]), 3)
        self.assertAlmostEqual(row["nse_mean"], 0.72)
        self.assertAlmostEqual(row["nse_sd"], float(np.std([0.70, 0.72, 0.74], ddof=1)))
        self.assertAlmostEqual(row["nse_median"], 0.72)
        self.assertAlmostEqual(row["nse_iqr_low"], float(np.percentile([0.70, 0.72, 0.74], 25)))

    def test_two_seeds_have_finite_sd(self):
        frame = pd.DataFrame({
            "tier": [2, 2], "depth": [3, 3], "arch": ["mamba3"] * 2,
            "seed": [1, 2], "median_nse": [0.70, 0.72], "median_kge": [0.60, 0.62],
        })
        agg = sm.aggregate_seeds(frame)
        self.assertEqual(int(agg.iloc[0]["n_seeds"]), 2)
        self.assertTrue(np.isfinite(agg.iloc[0]["nse_sd"]))
        self.assertAlmostEqual(agg.iloc[0]["nse_sd"], 0.02 / np.sqrt(2))

    def test_single_seed_has_nan_sd(self):
        frame = pd.DataFrame({
            "tier": [3], "depth": [2], "arch": ["lstm"],
            "seed": [1], "median_nse": [0.70], "median_kge": [0.60],
        })
        agg = sm.aggregate_seeds(frame)
        self.assertFalse(np.isfinite(agg.iloc[0]["nse_sd"]))


class TostTest(unittest.TestCase):
    def test_near_zero_difference_is_equivalent(self):
        delta = np.random.default_rng(0).normal(0.0, 0.001, 500)
        p, mean, se = sm.tost(delta, margin=0.01)
        self.assertLess(p, 0.05)
        self.assertGreater(se, 0.0)

    def test_clear_difference_is_not_equivalent(self):
        delta = np.random.default_rng(0).normal(0.02, 0.001, 500)
        p, _, _ = sm.tost(delta, margin=0.01)
        self.assertGreater(p, 0.05)

    def test_exactly_zero_difference(self):
        p, mean, se = sm.tost(np.zeros(100), margin=0.01)
        self.assertEqual(p, 0.0)
        self.assertEqual(mean, 0.0)
        self.assertEqual(se, 0.0)


class BootstrapTest(unittest.TestCase):
    def test_identical_vectors_have_zero_in_delta_ci(self):
        a = np.random.default_rng(1).normal(0.7, 0.1, 200)
        draws = sm.bootstrap_draws([a], [a.copy()], n_boot=400, seed=1, chunk=200)
        low, high = sm.bootstrap_cis(draws)["delta_medians"]
        self.assertLessEqual(low, 0.0)
        self.assertGreaterEqual(high, 0.0)

    def test_shifted_vectors_have_positive_delta_ci(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0.75, 0.05, 200)
        b = rng.normal(0.70, 0.05, 200)
        draws = sm.bootstrap_draws([a], [b], n_boot=400, seed=2, chunk=200)
        self.assertGreater(sm.bootstrap_cis(draws)["delta_medians"][0], 0.0)

    def test_median_tost_accepts_identical(self):
        a = np.random.default_rng(3).normal(0.7, 0.1, 300)
        draws = sm.bootstrap_draws([a], [a.copy()], n_boot=2000, seed=3)
        self.assertLess(sm.bootstrap_tost(draws), 0.05)

    def test_median_tost_rejects_shifted(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0.75, 0.05, 300)
        b = a - 0.05
        draws = sm.bootstrap_draws([a], [b], n_boot=2000, seed=4)
        self.assertGreater(sm.bootstrap_tost(draws), 0.05)


class SpatialJoinTest(unittest.TestCase):
    def test_blocks_join_by_gauge_id_not_row_position(self):
        topo = pd.DataFrame({
            "gauge_id": ["00000002", "00000001"],
            "gauge_lat": [40.0, 30.0],
            "gauge_lon": [-100.0, -110.0],
        })
        ids = pd.Series(["00000001", "00000002"])
        blocks = sm.spatial_blocks(ids, topo)
        self.assertAlmostEqual(float(blocks.loc[0, "gauge_lat"]), 30.0)
        self.assertAlmostEqual(float(blocks.loc[0, "gauge_lon"]), -110.0)
        self.assertAlmostEqual(float(blocks.loc[1, "gauge_lat"]), 40.0)

    def test_missing_coordinates_are_dropped(self):
        topo = pd.DataFrame({
            "gauge_id": ["00000001"], "gauge_lat": [30.0], "gauge_lon": [-110.0],
        })
        ids = pd.Series(["00000001", "00000009"])
        blocks = sm.spatial_blocks(ids, topo)
        self.assertEqual(list(blocks["gauge_id"]), ["00000001"])


class BlockedPermutationTest(unittest.TestCase):
    def test_single_block_makes_the_null_degenerate(self):
        # With one block the joint flip sends the win share to 0 or 1, so even
        # an all-tie data set is not extreme relative to that null: p = 1.
        paired = pd.DataFrame({"dNSE": np.zeros(8)})
        coords = pd.DataFrame({"bx3": [1000] * 8})
        p, n_blocks, share, _ = blocked_permutation_p(paired, "NSE", 3, coords)
        self.assertEqual(n_blocks, 1)
        self.assertGreater(p, 0.99)
        self.assertEqual(share, 0.0)

    def test_two_blocks_put_mass_on_half(self):
        # With two equal blocks a single flipped block gives share = 0.5, so an
        # all-tie data set has p about 0.5 -- the discrete null, not 1.
        paired = pd.DataFrame({"dNSE": np.zeros(8)})
        coords = pd.DataFrame({"bx3": [1000] * 4 + [2000] * 4})
        p, n_blocks, share, _ = blocked_permutation_p(paired, "NSE", 3, coords)
        self.assertEqual(n_blocks, 2)
        self.assertGreater(p, 0.4)
        self.assertLess(p, 0.6)


class SeedAveragingTest(unittest.TestCase):
    def test_seed_averaging_is_per_basin(self):
        cells = [
            {"tier": 1, "depth": 1, "arch": "lstm", "seed": 1,
             "frame": pd.DataFrame({"gauge_id": ["1", "2"], "NSE": [0.6, 0.8],
                                    "KGE": [0.5, 0.7]})},
            {"tier": 1, "depth": 1, "arch": "lstm", "seed": 2,
             "frame": pd.DataFrame({"gauge_id": ["1", "2"], "NSE": [0.7, 0.9],
                                    "KGE": [0.6, 0.8]})},
        ]
        averaged = sm.seed_averaged(cells, 1, 1, "lstm")
        self.assertEqual(len(averaged), 2)
        self.assertAlmostEqual(float(averaged.loc[0, "NSE"]), 0.65)
        self.assertAlmostEqual(float(averaged.loc[1, "NSE"]), 0.85)


class LegacyBlockGridTest(unittest.TestCase):
    def test_legacy_block_grid_joins_by_id(self):
        import sign_test_all_pairs as sap

        topo = pd.read_csv(sap.TOPO_FILE, sep=";", dtype=str,
                           usecols=["gauge_id", "gauge_lat", "gauge_lon"])
        topo["gauge_id"] = topo["gauge_id"].str.strip().str.zfill(8)
        # Reverse two ids so a positional join cannot coincide with the id join.
        ids = list(topo["gauge_id"].iloc[-2:])[::-1]
        paired = pd.DataFrame({"gauge_id": ids})
        blocks = sap.block_grid(paired)
        expected = paired.merge(topo, on="gauge_id", how="left")
        for row in range(len(ids)):
            self.assertAlmostEqual(float(blocks.loc[row, "gauge_lat"]),
                                   float(expected.loc[row, "gauge_lat"]))
            self.assertAlmostEqual(float(blocks.loc[row, "gauge_lon"]),
                                   float(expected.loc[row, "gauge_lon"]))


if __name__ == "__main__":
    unittest.main()
