"""B2c acceptance tests: the training path cannot reach the test slice.

Run from ``experiment/``:

    python -m unittest discover -s tests -v
"""

import builtins
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import train_val  # noqa: E402
from config import ExperimentConfig  # noqa: E402
from data_utils import (  # noqa: E402
    PREPROCESSED_DIR,
    SPLIT_PATH,
    fit_scaler,
    load_split,
    load_split_with_context,
)


class TestTestSliceBoundary(unittest.TestCase):
    def test_training_never_opens_the_test_pickle(self):
        """Monkeypatch ``open`` so any access to test.pkl fails the test."""
        real_open = builtins.open
        touched = []

        def guarded(file, *args, **kwargs):
            path = str(file).replace("\\", "/")
            if path.endswith("test.pkl"):
                touched.append(path)
                raise AssertionError(f"training path opened {path}")
            return real_open(file, *args, **kwargs)

        cfg, _ = train_val.build_config([
            "--model", "lstm", "--tier", "1", "--depth", "1",
            "--epochs", "1", "--seed", "1", "--basins", "4",
        ])
        builtins.open = guarded
        try:
            data = train_val.load_datasets(cfg, n_basins=4)
        finally:
            builtins.open = real_open

        self.assertEqual(touched, [])
        self.assertEqual(data["val_y"].shape[1], 1461)
        self.assertEqual(data["val_x"].shape[1], 730 + 1461)

    def test_load_split_test_is_closed_by_default(self):
        with self.assertRaises(PermissionError):
            load_split("test")


class TestScalerCutoff(unittest.TestCase):
    def test_scaler_is_fitted_on_train_only(self):
        payload = json.loads((PREPROCESSED_DIR / "scaler.json").read_text())
        fitted = payload["fitted_on"]
        self.assertEqual(fitted["slice"], "train")
        self.assertEqual(fitted["start"], "1980-10-01")
        self.assertEqual(fitted["end"], "1995-09-30")
        self.assertEqual(fitted["days"], 5478)

        # Recompute one statistic from the first 5478 days only.
        x = np.load(PREPROCESSED_DIR / "x.npy")
        values = x[:, :5478, 1].flatten()          # prcp
        values = values[~np.isnan(values)]
        expected = [
            np.percentile(values, 10), np.percentile(values, 90),
            np.mean(values), np.std(values),
        ]
        recorded = payload["series_statistics"]["prcp"]
        for got, want in zip(recorded, expected):
            self.assertTrue(np.isclose(got, want, rtol=0, atol=1e-12),
                            f"{got} != {want}")

    def test_fit_scaler_refuses_non_training_windows(self):
        x = np.load(PREPROCESSED_DIR / "x.npy")
        y = np.load(PREPROCESSED_DIR / "y.npy")
        c = np.load(PREPROCESSED_DIR / "c.npy")
        from data_utils import _date_axis

        with self.assertRaises(ValueError):
            fit_scaler(x, y, c, _date_axis(), train_only=False)


class TestValidationWindow(unittest.TestCase):
    def test_validation_slice_ends_1999_and_context_is_train_tail(self):
        split = json.loads(SPLIT_PATH.read_text())
        self.assertEqual(split["slices"]["val"]["start"], "1995-10-01")
        self.assertEqual(split["slices"]["val"]["end"], "1999-09-30")
        self.assertEqual(split["context"]["windows"]["val"]["end"], "1995-09-30")

        train_x, train_y, _ = load_split("train")
        val_x, val_y, _ = load_split_with_context("val", ctx_days=730)
        self.assertEqual(val_y.shape[1], 1461)
        self.assertEqual(val_x.shape[1], 730 + 1461)
        np.testing.assert_array_equal(val_x[:, :730], train_x[:, -730:])
        self.assertEqual(train_y.shape[1], 5478)
        self.assertEqual(split["slices"]["val"]["stop_index"], 6939)
        self.assertLess(split["slices"]["val"]["stop_index"], 12418)


class TestReportIsReadOnly(unittest.TestCase):
    def test_report_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "All"
            run.mkdir()
            (run / "val_metrics.csv").write_text(
                "epoch,train_loss,val_loss,train_median_nse,val_median_nse,"
                "val_median_kge,epoch_seconds\n"
                "1,0.4,0.5,0.3,0.35,0.45,2.0\n"
            )
            before = {
                p.name: (p.stat().st_size, p.stat().st_mtime_ns)
                for p in run.iterdir()
            }
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                train_val.report(tmp)
            after = {
                p.name: (p.stat().st_size, p.stat().st_mtime_ns)
                for p in run.iterdir()
            }
            self.assertEqual(before, after)
            self.assertIn("0.3500", captured.getvalue())


class TestWiringIsPinned(unittest.TestCase):
    def test_mamba_cli_forces_standard_v3(self):
        cfg, _ = train_val.build_config([
            "--model", "mamba3", "--tier", "1", "--depth", "3", "--epochs", "1",
        ])
        self.assertEqual(cfg.mamba_variant, "standard")
        self.assertEqual(cfg.mamba_version, 3)
        self.assertEqual(cfg.mamba_layers, 3)

    def test_recipe_cannot_switch_wiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ({"mamba_variant": "adapter"}, {"mamba_version": 2}):
                recipe = Path(tmp) / "bad.json"
                recipe.write_text(json.dumps({
                    "hyperparameters": bad,
                }))
                with self.assertRaises(SystemExit):
                    train_val.build_config([
                        "--model", "mamba3", "--tier", "1", "--depth", "3",
                        "--epochs", "1", "--config", str(recipe),
                    ])

    def test_unknown_recipe_key_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp) / "typo.json"
            recipe.write_text(json.dumps({"mamba_lrr": 1e-3}))
            with self.assertRaises(SystemExit):
                train_val.build_config([
                    "--model", "mamba3", "--tier", "1", "--depth", "3",
                    "--epochs", "1", "--config", str(recipe),
                ])

    def test_subset_run_gets_its_own_output_root(self):
        subset, _ = train_val.build_config([
            "--model", "lstm", "--tier", "1", "--depth", "1",
            "--epochs", "1", "--basins", "150",
        ])
        full, _ = train_val.build_config([
            "--model", "lstm", "--tier", "1", "--depth", "1", "--epochs", "1",
        ])
        self.assertEqual(Path(subset.output_dir).as_posix(), "output/subsets")
        self.assertEqual(full.output_dir, "output")
        self.assertNotEqual(
            Path(subset.output_path()).parent.parent,
            Path(full.output_path()).parent.parent,
        )

    def test_ctx_covers_rho_plus_warmup(self):
        cfg = ExperimentConfig()
        cfg.rho = 1095
        cfg.train_buff = 365
        cfg.ctx_days = 730
        train_val.validate_wiring(cfg)
        self.assertEqual(cfg.ctx_days, 1460)


class TestSelectionRule(unittest.TestCase):
    def test_patience_and_min_delta(self):
        selector = train_val.BestOnValidation()
        self.assertTrue(selector.update(1, 0.50, 0.60, 0.4))
        for epoch in range(2, 2 + train_val.PATIENCE):
            self.assertFalse(selector.update(epoch, 0.50, 0.60, 0.4))
        self.assertTrue(selector.should_stop)
        self.assertEqual(selector.best_epoch, 1)
        self.assertEqual(selector.n_selections, 1 + train_val.PATIENCE)
        self.assertAlmostEqual(selector.best_minus_last, 0.0)

    def test_nan_never_wins(self):
        selector = train_val.BestOnValidation()
        self.assertFalse(selector.update(1, float("nan"), float("nan"),
                                         float("nan")))
        self.assertEqual(selector.best_epoch, 0)
        self.assertTrue(np.isnan(selector.best_minus_last))


class TestAtomicBasinMetricsWrite(unittest.TestCase):
    """The per-basin file must survive a transient Windows file lock."""

    def test_replace_retries_transient_permission_error(self):
        calls = {"n": 0}
        real = os.replace

        def flaky(source, destination):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError(13, "locked by an indexer")
            real(source, destination)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "val.tmp"
            destination = tmpdir / "val.csv"
            source.write_text("gauge_id,NSE,KGE\n", encoding="utf-8")
            with mock.patch("train_val.os.replace", flaky):
                train_val._replace_with_retry(
                    source, destination, attempts=5, pause=0.0
                )
            self.assertTrue(destination.is_file())
            self.assertEqual(calls["n"], 3)

    def test_replace_raises_when_every_attempt_fails(self):
        with mock.patch(
            "train_val.os.replace",
            side_effect=PermissionError(13, "still locked"),
        ):
            with self.assertRaises(PermissionError):
                train_val._replace_with_retry(
                    Path("a.tmp"), Path("a.csv"), attempts=2, pause=0.0
                )


if __name__ == "__main__":
    unittest.main()
