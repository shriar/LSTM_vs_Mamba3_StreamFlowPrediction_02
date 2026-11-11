"""B8 acceptance tests: grid validation, resume, refusal, aggregation."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import hpo  # noqa: E402
from paths import PROJECT_ROOT  # noqa: E402

MATCHED = PROJECT_ROOT / "recipes" / "sweeps" / "matched.json"
SENSITIVITY = PROJECT_ROOT / "recipes" / "sweeps" / "sensitivity.json"


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        return fn(*args, **kwargs)


def fake_run(destination: Path, rows, basin_nse=None):
    """Create the file layout a finished train_val.py run leaves behind."""
    run = destination / "run" / "MODEL" / "RUN" / "All"
    run.mkdir(parents=True, exist_ok=True)
    with (run / "val_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "train_median_nse",
            "val_median_nse", "val_median_kge", "epoch_seconds",
        ])
        writer.writerows(rows)
    if basin_nse is not None:
        with (run / "val_basin_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["gauge_id", "NSE", "KGE"])
            for index, nse in enumerate(basin_nse):
                writer.writerow([f"{index + 1:08d}", f"{nse:.6f}", "0.5"])
    return run


def mark_done(destination: Path, **fields):
    """Record a finished trial the way ``hpo.py`` does (status ok in trial.json)."""
    payload = {"trial_id": destination.name, "status": "ok", "wall_clock": 10.0}
    payload.update(fields)
    (destination / "trial.json").write_text(json.dumps(payload), encoding="utf-8")


class TestTrackedGrids(unittest.TestCase):
    def test_matched_has_thirteen_trials_per_architecture(self):
        grid, _ = hpo.load_grid(MATCHED)
        self.assertEqual(sorted(grid["architectures"]), ["lstm", "mamba3"])
        for model, block in grid["architectures"].items():
            self.assertEqual(len(block["trials"]), 13, model)
            self.assertEqual(block["trials"][0]["id"], "trial_0")

    def test_trial_zero_is_the_submitted_recipe(self):
        grid, _ = hpo.load_grid(MATCHED)
        lstm = grid["architectures"]["lstm"]
        self.assertEqual(lstm["trials"][0]["params"], lstm["baseline"])
        mamba = grid["architectures"]["mamba3"]
        self.assertEqual(mamba["trials"][0]["params"], mamba["baseline"])

    def test_sensitivity_is_the_full_factorial(self):
        grid, _ = hpo.load_grid(SENSITIVITY)
        trials = grid["architectures"]["mamba3"]["trials"]
        self.assertEqual(len(trials), 12)
        cells = {
            (t["params"]["mamba_lr"], t["params"]["mamba_grad_clip"],
             t["params"]["mamba_beta2"])
            for t in trials
        }
        expected = {
            (lr, clip, beta2)
            for lr in (5e-4, 1.5e-3, 3e-3)
            for clip in (0.5, 1.0)
            for beta2 in (0.95, 0.999)
        }
        self.assertEqual(cells, expected)

    def test_no_grid_carries_a_test_metric(self):
        for path in (MATCHED, SENSITIVITY):
            grid, _ = hpo.load_grid(path)
            self.assertFalse(hpo._carries_test_metric(grid), path.name)
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("test_metrics", text)
            self.assertNotIn("test_nse", text)


class TestGridValidation(unittest.TestCase):
    def _write(self, payload) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, tmp)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def test_test_metric_key_is_refused(self):
        payload = {
            "schema": hpo.SCHEMA,
            "id": "bad",
            "architectures": {
                "lstm": {
                    "tier": 1, "depth": 1, "epochs": 1,
                    "baseline": {"lr": 1e-3},
                    "trials": [{"id": "trial_0", "params": {"lr": 1e-3}}],
                    "test_nse": 0.9,
                }
            },
        }
        with self.assertRaises(SystemExit) as caught:
            hpo.load_grid(self._write(payload))
        self.assertIn("test metric", str(caught.exception))

    def test_duplicate_trial_id_is_refused(self):
        payload = {
            "schema": hpo.SCHEMA,
            "id": "dup",
            "architectures": {
                "lstm": {
                    "tier": 1, "depth": 1, "epochs": 1,
                    "baseline": {"lr": 1e-3},
                    "trials": [
                        {"id": "trial_0", "params": {"lr": 1e-3}},
                        {"id": "trial_0", "params": {"lr": 2e-3}},
                    ],
                }
            },
        }
        with self.assertRaises(SystemExit) as caught:
            hpo.load_grid(self._write(payload))
        self.assertIn("duplicate", str(caught.exception))

    def test_unknown_parameter_is_refused(self):
        payload = {
            "schema": hpo.SCHEMA,
            "id": "typo",
            "architectures": {
                "lstm": {
                    "tier": 1, "depth": 1, "epochs": 1,
                    "baseline": {"lr": 1e-3},
                    "trials": [{"id": "trial_0", "params": {"lr_typo": 1e-3}}],
                }
            },
        }
        with self.assertRaises(SystemExit) as caught:
            hpo.load_grid(self._write(payload))
        self.assertIn("unknown", str(caught.exception))


class TestDriverBehaviour(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.grid = {
            "schema": hpo.SCHEMA,
            "id": "unit",
            "name": "unit grid",
            "basins": 150,
            "seed": 1,
            "architectures": {
                "lstm": {
                    "tier": 1, "depth": 1, "epochs": 2,
                    "baseline": {"lr": 2e-3, "lstm_grad_clip": 1.0},
                    "trials": [
                        {"id": "trial_0", "params": {"lr": 2e-3}},
                        {"id": "trial_1", "params": {"lr": 5e-4}},
                    ],
                }
            },
        }
        self.grid_path = self.root / "grid.json"
        self.grid_path.write_text(json.dumps(self.grid), encoding="utf-8")

    def test_dry_run_writes_nothing(self):
        code = quiet(
            hpo.main,
            [str(self.grid_path), "--output-root", str(self.root / "out"), "--dry-run"],
        )
        self.assertEqual(code, 0)
        self.assertFalse((self.root / "out").exists())

    def test_resume_skips_finished_trial_and_ranks_summary(self):
        out = self.root / "out"
        first = out / "unit" / "lstm" / "trial_0"
        second = out / "unit" / "lstm" / "trial_1"
        fake_run(first, [[1, 0.3, 0.4, 0.60, 0.52, 0.51, 10.0],
                         [2, 0.2, 0.3, 0.65, 0.58, 0.55, 10.0]],
                 basin_nse=[0.1, 0.5, 0.9, 0.95])
        fake_run(second, [[1, 0.3, 0.4, 0.60, 0.71, 0.70, 10.0]])
        mark_done(first, best_epoch=2, val_median_nse=0.58,
                  val_median_kge=0.55, val_nse_iqr=0.5125)
        mark_done(second, best_epoch=1, val_median_nse=0.71,
                  val_median_kge=0.70, val_nse_iqr=0.1)
        code = quiet(hpo.main, [str(self.grid_path), "--output-root", str(out)])
        self.assertEqual(code, 0)
        summary = out / "unit" / "lstm" / "summary.csv"
        self.assertTrue(summary.is_file())
        with summary.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["trial_id"] for r in rows], ["trial_1", "trial_0"])
        self.assertEqual(rows[1]["val_median_nse"], "0.58")
        self.assertAlmostEqual(float(rows[1]["val_nse_iqr"]), 0.5125, places=6)
        self.assertEqual(rows[1]["eta0"], "0.002")

    def test_summary_covers_trials_from_earlier_invocations(self):
        out = self.root / "out"
        fake_run(out / "unit" / "lstm" / "trial_0",
                 [[1, 0.30, 0.40, 0.60, 0.52, 0.50, 9.0]])
        fake_run(out / "unit" / "lstm" / "trial_1",
                 [[1, 0.30, 0.40, 0.60, 0.61, 0.60, 9.0]])
        mark_done(out / "unit" / "lstm" / "trial_0", best_epoch=1,
                  val_median_nse=0.52, val_median_kge=0.50, val_nse_iqr=0.1)
        mark_done(out / "unit" / "lstm" / "trial_1", best_epoch=1,
                  val_median_nse=0.61, val_median_kge=0.60, val_nse_iqr=0.1)
        code = quiet(
            hpo.main,
            [str(self.grid_path), "--output-root", str(out),
             "--only", "lstm/trial_1"],
        )
        self.assertEqual(code, 0)
        with (out / "unit" / "lstm" / "summary.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["trial_id"] for r in rows], ["trial_1", "trial_0"])

    def test_second_grid_does_not_overwrite_first_grid_summary(self):
        out = self.root / "out"
        fake_run(out / "unit" / "lstm" / "trial_0",
                 [[1, 0.30, 0.40, 0.60, 0.52, 0.50, 9.0]])
        mark_done(out / "unit" / "lstm" / "trial_0", best_epoch=1,
                  val_median_nse=0.52, val_median_kge=0.50, val_nse_iqr=0.1)
        quiet(hpo.main, [str(self.grid_path), "--output-root", str(out),
                         "--only", "lstm/trial_0"])
        first = out / "unit" / "lstm" / "summary.csv"
        self.assertTrue(first.is_file())

        other = self.root / "other.json"
        grid = dict(self.grid)
        grid["id"] = "other"
        other.write_text(json.dumps(grid), encoding="utf-8")
        fake_run(out / "other" / "lstm" / "trial_0",
                 [[1, 0.30, 0.40, 0.60, 0.71, 0.70, 9.0]])
        mark_done(out / "other" / "lstm" / "trial_0", best_epoch=1,
                  val_median_nse=0.71, val_median_kge=0.70, val_nse_iqr=0.1)
        quiet(hpo.main, [str(other), "--output-root", str(out),
                         "--only", "lstm/trial_0"])

        with first.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["trial_id"] for r in rows], ["trial_0"])
        self.assertEqual(rows[0]["val_median_nse"], "0.52")
        second = out / "other" / "lstm" / "summary.csv"
        self.assertTrue(second.is_file())
        with second.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["val_median_nse"], "0.71")

    def test_crashed_trial_is_restarted_with_force(self):
        out = self.root / "out"
        destination = out / "unit" / "lstm" / "trial_0"
        fake_run(destination, [[1, 0.30, 0.40, 0.60, 0.58, 0.55, 9.0]])
        calls = []
        original = hpo.run_trial

        def fake(command):
            calls.append(list(command))
            return 0, 5.0

        hpo.run_trial = fake
        self.addCleanup(setattr, hpo, "run_trial", original)
        code = quiet(
            hpo.main,
            [str(self.grid_path), "--output-root", str(out),
             "--only", "lstm/trial_0"],
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("--force", calls[0])
        payload = json.loads(
            (destination / "trial.json").read_text(encoding="utf-8")
        )
        self.assertTrue(payload["restarted"])
        self.assertFalse(payload["forced"])

    def test_finished_trial_is_not_retrained(self):
        out = self.root / "out"
        destination = out / "unit" / "lstm" / "trial_0"
        fake_run(destination, [[1, 0.30, 0.40, 0.60, 0.58, 0.55, 9.0]])
        mark_done(destination, best_epoch=1, val_median_nse=0.58,
                  val_median_kge=0.55, val_nse_iqr=0.1)

        def explode(command):
            raise AssertionError(f"finished trial retrained: {command}")

        original = hpo.run_trial
        hpo.run_trial = explode
        self.addCleanup(setattr, hpo, "run_trial", original)
        code = quiet(
            hpo.main,
            [str(self.grid_path), "--output-root", str(out),
             "--only", "lstm/trial_0"],
        )
        self.assertEqual(code, 0)

    def test_trial_is_complete_only_for_ok_status(self):
        destination = self.root / "trial"
        destination.mkdir()
        self.assertFalse(hpo.trial_is_complete(destination))
        (destination / "trial.json").write_text(
            json.dumps({"status": "failed(1)"}), encoding="utf-8"
        )
        self.assertFalse(hpo.trial_is_complete(destination))
        (destination / "trial.json").write_text(
            json.dumps({"status": "ok"}), encoding="utf-8"
        )
        self.assertTrue(hpo.trial_is_complete(destination))

    def test_partial_folder_is_labelled_incomplete(self):
        out = self.root / "out"
        destination = out / "unit" / "lstm" / "trial_0"
        fake_run(destination, [[1, 0.30, 0.40, 0.60, 0.61, 0.60, 9.0]])
        rows = hpo.collect_summary_rows(
            out, hpo.load_grid(self.grid_path)[0], "lstm"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "incomplete")

    def test_collect_rows_recovers_axes_without_trial_json(self):
        out = self.root / "out"
        destination = out / "unit" / "lstm" / "trial_1"
        fake_run(destination, [[1, 0.30, 0.40, 0.60, 0.61, 0.60, 9.0]])
        (destination / "recipe.json").write_text(json.dumps({
            "hyperparameters": {"lr": 0.0005, "lstm_grad_clip": 0.8,
                                "lstm_beta2": 0.99},
        }), encoding="utf-8")
        rows = hpo.collect_summary_rows(out, hpo.load_grid(self.grid_path)[0], "lstm")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["eta0"], 0.0005)
        self.assertEqual(rows[0]["clip"], 0.8)
        self.assertEqual(rows[0]["val_median_nse"], 0.61)

    def test_test_artefact_in_trial_folder_is_refused(self):
        out = self.root / "out"
        destination = out / "unit" / "lstm" / "trial_0"
        destination.mkdir(parents=True)
        (destination / "test_metrics.csv").write_text("median_nse\n0.9\n")
        code = quiet(hpo.main, [str(self.grid_path), "--output-root", str(out)])
        self.assertEqual(code, 5)

    def test_percentile_matches_numpy(self):
        values = [0.1, 0.2, 0.35, 0.5, 0.9]
        for q in (25.0, 50.0, 75.0):
            self.assertAlmostEqual(
                hpo._percentile(values, q), float(np.percentile(values, q)), places=10
            )

    def test_driver_holds_no_training_code(self):
        source = Path(hpo.__file__).read_text(encoding="utf-8")
        for forbidden in ("import torch", "load_split", "statError", "eval_test"):
            self.assertNotIn(forbidden, source)


class TestTrialSelection(unittest.TestCase):
    """A 13+13 grid must be splittable by architecture."""

    def setUp(self):
        self.grid, self.path = hpo.load_grid(MATCHED)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def test_model_filter_selects_one_architecture(self):
        for model, other in (("lstm", "mamba3"), ("mamba3", "lstm")):
            planned = hpo.plan_trials(self.grid, self.path, self.out, models={model})
            self.assertEqual(len(planned), 13, model)
            self.assertEqual({item["model"] for item in planned}, {model})
            self.assertNotIn(
                other, {item["model"] for item in planned}
            )

    def test_qualified_only_selects_one_architecture(self):
        planned = hpo.plan_trials(
            self.grid, self.path, self.out, only={"mamba3/trial_3"}
        )
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0]["model"], "mamba3")
        self.assertEqual(planned[0]["trial"]["id"], "trial_3")

    def test_bare_only_still_matches_every_architecture(self):
        planned = hpo.plan_trials(self.grid, self.path, self.out, only={"trial_3"})
        self.assertEqual(
            sorted((item["model"], item["trial"]["id"]) for item in planned),
            [("lstm", "trial_3"), ("mamba3", "trial_3")],
        )

    def test_mixed_qualified_and_bare_tokens(self):
        planned = hpo.plan_trials(
            self.grid, self.path, self.out,
            only={"mamba3/trial_0", "trial_0"},
        )
        self.assertEqual(
            sorted((item["model"], item["trial"]["id"]) for item in planned),
            [("lstm", "trial_0"), ("mamba3", "trial_0")],
        )

    def test_qualifier_and_model_filter_agree(self):
        qualified = hpo.plan_trials(
            self.grid, self.path, self.out, only={"lstm/trial_5"}
        )
        filtered = hpo.plan_trials(
            self.grid, self.path, self.out, only={"trial_5"}, models={"lstm"}
        )
        self.assertEqual(
            [item["destination"] for item in qualified],
            [item["destination"] for item in filtered],
        )

    def test_unknown_trial_and_model_are_refused(self):
        with self.assertRaises(SystemExit) as caught:
            hpo.plan_trials(self.grid, self.path, self.out, only={"trial_99"})
        self.assertIn("unknown trial id", str(caught.exception))
        with self.assertRaises(SystemExit) as caught:
            hpo.plan_trials(self.grid, self.path, self.out, only={"gru/trial_0"})
        self.assertIn("unknown trial id", str(caught.exception))
        with self.assertRaises(SystemExit) as caught:
            hpo.plan_trials(self.grid, self.path, self.out, models={"gru"})
        self.assertIn("unknown model", str(caught.exception))

    def test_cli_reports_bad_selection_as_exit_two(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            buffer
        ):
            code = hpo.main([
                str(MATCHED), "--output-root", str(self.out / "out"),
                "--only", "trial_99", "--dry-run",
            ])
        self.assertEqual(code, 2)
        self.assertIn("unknown trial id", buffer.getvalue())

    def test_dry_run_with_model_filter_plans_only_that_half(self):
        buffer = io.StringIO()
        out_root = self.out / "out"
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(
            io.StringIO()
        ):
            code = hpo.main([
                str(MATCHED), "--output-root", str(out_root),
                "--models", "lstm", "--dry-run",
            ])
        self.assertEqual(code, 0)
        printed = buffer.getvalue()
        self.assertIn("13 trial(s) planned", printed)
        self.assertNotIn("mamba3/", printed)
        self.assertFalse(out_root.exists())


if __name__ == "__main__":
    unittest.main()
