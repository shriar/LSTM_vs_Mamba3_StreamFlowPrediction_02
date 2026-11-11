"""B2d acceptance tests: manifest locks, one audit line per cell, no eval in training."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import eval_test  # noqa: E402
import run_all_tiers  # noqa: E402
import run_test_all  # noqa: E402
import train_val  # noqa: E402
from paths import PROJECT_ROOT  # noqa: E402

SPLIT_SHA = "9efbe60e790965f203b037d7e630828f7fe36285d4c86ee5c3ddbd93ec13cc58"
FULL_RUN = PROJECT_ROOT / (
    "output/LSTM_1L/Tier1_200K_Ep1_Bs128_Rho365_H162_L1_Seed1/All"
)
CHECKPOINT = FULL_RUN / "model_best.pt"


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def base_recipe(**overrides) -> dict:
    recipe = {
        "schema": "eval_recipe/v1",
        "id": "unit",
        "name": "unit test recipe",
        "model": "lstm",
        "tier": 1,
        "depth": 1,
        "seed": 1,
        "epochs": 1,
        "batch_size": 128,
        "rho": 365,
        "train_buff": 365,
        "ctx_days": 730,
        "hyperparameters": {
            "lr": 0.002,
            "lstm_min_lr": 0.0001,
            "lstm_optimizer": "adamw",
            "lstm_weight_decay": 0.00001,
            "lstm_beta1": 0.9,
            "lstm_beta2": 0.999,
            "lstm_grad_clip": 1.0,
            "lstm_scheduler": "cosine",
            "lstm_dropout": 0.2,
            "use_amp": False,
        },
        "run_dir": "output/unit-does-not-exist/All",
        "checkpoint": "model_best.pt",
        "split_sha256": SPLIT_SHA,
        "locked": True,
        "lock_date": "2026-10-28",
    }
    recipe.update(overrides)
    return recipe


def base_manifest(recipe_path: Path, tier=1, depth=1, seed=1) -> dict:
    return {
        "schema": "eval_manifest/v1",
        "created_utc": "2026-10-28T00:00:00Z",
        "updated_utc": "2026-10-28T00:00:00Z",
        "split_sha256": SPLIT_SHA,
        "cells": [
            {
                "id": "UNIT",
                "recipe": str(recipe_path),
                "tier": tier,
                "depth": depth,
                "seed": seed,
                "run_dir": "output/unit-does-not-exist/All",
            }
        ],
    }


class TestRefusals(unittest.TestCase):
    def _run(self, tmp: Path, recipe: dict, manifest: dict | None = None,
             tier=1, depth=1, seed=1) -> tuple[int, str, str]:
        recipe_path = write_json(tmp / "recipe.json", recipe)
        manifest_path = write_json(
            tmp / "manifest.json",
            manifest if manifest is not None else base_manifest(recipe_path, tier, depth, seed),
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = eval_test.main([
                "--recipe", str(recipe_path),
                "--tier", str(tier),
                "--depth", str(depth),
                "--seed", str(seed),
                "--manifest", str(manifest_path),
                "--output-dir", str(tmp / "out"),
                "--audit", str(tmp / "out" / "audit.jsonl"),
            ])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_unlisted_cell_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            recipe = base_recipe()
            empty = {
                "schema": "eval_manifest/v1",
                "split_sha256": SPLIT_SHA,
                "cells": [],
            }
            code, _, err = self._run(tmp, recipe, empty)
            self.assertEqual(code, 2)
            self.assertIn("not listed", err)
            self.assertFalse((tmp / "out" / "audit.jsonl").exists())

    def test_unlocked_recipe_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            code, _, err = self._run(tmp, base_recipe(locked=False))
            self.assertEqual(code, 2)
            self.assertIn("not locked", err)
            self.assertFalse((tmp / "out" / "audit.jsonl").exists())

    def test_split_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            code, _, err = self._run(
                tmp, base_recipe(split_sha256="0" * 64)
            )
            self.assertEqual(code, 3)
            self.assertIn("split_sha256", err)
            self.assertFalse((tmp / "out" / "audit.jsonl").exists())

    def test_recipe_run_dir_stays_binding_for_its_own_cell(self):
        # The recipe's own (tier, depth, seed) with a diverging manifest
        # run_dir is still refused: a stale recipe must not evaluate
        # silently against a different run.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            recipe = base_recipe(run_dir="output/recipe-says-here/All")
            manifest = base_manifest(
                tmp / "recipe.json", tier=1, depth=1, seed=1
            )
            manifest["cells"][0]["run_dir"] = "output/manifest-says-here/All"
            code, _, err = self._run(tmp, recipe, manifest)
            self.assertEqual(code, 5)
            self.assertIn("recipe's own cell", err)

    def test_shared_recipe_uses_the_manifest_cell_run_dir(self):
        # C1a shares one locked recipe across nine cells.  Selecting a
        # non-headline cell must evaluate that cell's run_dir (proven
        # against the recipe by _verify_master), not the recipe's own.
        # No run exists here, so the refusal chain must get past the
        # run-dir check and stop at the missing checkpoint instead.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            recipe = base_recipe()          # headline identity T1/L1/S1
            manifest = base_manifest(
                tmp / "recipe.json", tier=1, depth=1, seed=1
            )
            cell = dict(manifest["cells"][0])
            cell["id"] = "C1A-L2-S2"
            cell["tier"], cell["depth"], cell["seed"] = 1, 2, 2
            cell["run_dir"] = "output/unit-does-not-exist-L2-S2/All"
            manifest["cells"] = [cell]
            code, _, err = self._run(tmp, recipe, manifest, tier=1, depth=2, seed=2)
            self.assertEqual(code, 5)
            self.assertNotIn("run_dir", err)
            self.assertIn("checkpoint not found", err)

    def test_wrong_seed_run_is_refused_by_master_verification(self):
        # A real run whose master.json records seed 1 must be refused when
        # the manifest cell claims seed 2 -- the identity check lives in
        # _verify_master, so a wrong-seed run can never be evaluated.
        if not CHECKPOINT.is_file():
            self.skipTest("B2c 1-epoch acceptance run is absent")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            recipe = base_recipe()          # seed 1, matches the real run
            manifest = base_manifest(
                tmp / "recipe.json", tier=1, depth=1, seed=1
            )
            cell = dict(manifest["cells"][0])
            cell["id"] = "C1A-L1-S2"
            cell["seed"] = 2
            cell["run_dir"] = FULL_RUN.relative_to(PROJECT_ROOT).as_posix()
            manifest["cells"] = [cell]
            code, _, err = self._run(tmp, recipe, manifest, tier=1, depth=1, seed=2)
            self.assertEqual(code, 5)
            self.assertIn("seed", err)


@unittest.skipUnless(CHECKPOINT.is_file(), "B2c 1-epoch acceptance run is absent")
class TestOneCellEndToEnd(unittest.TestCase):
    def _recipe_from_master(self) -> dict:
        master = json.loads((FULL_RUN / "master.json").read_text(encoding="utf-8"))
        training = master["training"]
        model = master["model"]
        if model["type"] == "mamba3":
            hyper = {
                "mamba_lr": training["lr"],
                "mamba_min_lr": training["min_lr"],
                "mamba_weight_decay": training["weight_decay"],
                "mamba_beta1": training["betas"][0],
                "mamba_beta2": training["betas"][1],
                "mamba_warmup_steps": training["warmup_steps"],
                "mamba_scheduler": training["scheduler"],
                "mamba_optimizer": training["optimizer"],
                "mamba_grad_clip": training["grad_clip"],
                "mamba_dropout": model.get("dropout", 0.11),
                "mamba_version": model["mamba_version"],
                "mamba_variant": model["mamba_variant"],
                "use_amp": training["use_amp"],
            }
        else:
            hyper = {
                "lr": training["lr"],
                "lstm_min_lr": training["min_lr"],
                "lstm_optimizer": training["optimizer"],
                "lstm_weight_decay": training["weight_decay"],
                "lstm_beta1": training["betas"][0],
                "lstm_beta2": training["betas"][1],
                "lstm_grad_clip": training["grad_clip"],
                "lstm_scheduler": training["scheduler"],
                "lstm_dropout": model.get("dropout", 0.2),
                "use_amp": training["use_amp"],
            }
        return {
            "schema": "eval_recipe/v1",
            "id": "smoke1",
            "model": model["type"],
            "tier": model["param_tier"],
            "depth": model["depth"],
            "seed": master["determinism"]["seed"],
            "epochs": training["epochs_requested"],
            "batch_size": training["batch_size"],
            "rho": training["rho"],
            "train_buff": training["train_buff"],
            "ctx_days": master["data"]["context_days"],
            "hyperparameters": hyper,
            "run_dir": FULL_RUN.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint": "model_best.pt",
            "split_sha256": master["data"]["split_sha256"],
            "locked": True,
        }

    def test_one_audit_line_and_second_call_refused(self):
        before = {
            path for path in (PROJECT_ROOT / "output").rglob("*") if path.is_file()
        }
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            recipe = self._recipe_from_master()
            recipe_path = write_json(tmp / "recipe.json", recipe)
            manifest = base_manifest(
                recipe_path,
                tier=int(recipe["tier"]),
                depth=int(recipe["depth"]),
                seed=int(recipe["seed"]),
            )
            manifest["cells"][0]["id"] = "SMOKE"
            manifest["cells"][0]["run_dir"] = recipe["run_dir"]
            manifest_path = write_json(tmp / "manifest.json", manifest)
            audit = tmp / "out" / "audit.jsonl"
            argv = [
                "--recipe", str(recipe_path),
                "--tier", str(recipe["tier"]),
                "--depth", str(recipe["depth"]),
                "--seed", str(recipe["seed"]),
                "--manifest", str(manifest_path),
                "--output-dir", str(tmp / "out"),
                "--audit", str(audit),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(eval_test.main(argv), 0)
            cell_dir = tmp / "out" / (
                f"smoke1_T{recipe['tier']}_L{recipe['depth']}_Seed{recipe['seed']}"
            )
            self.assertTrue((cell_dir / "test_metrics.csv").is_file())
            self.assertTrue((cell_dir / "basin_test_metrics.csv").is_file())
            self.assertTrue((cell_dir / "pred.npz").is_file())
            lines = audit.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["cell_id"], "SMOKE")
            self.assertFalse(record["forced"])
            self.assertEqual(record["audit_line_number"], 1)
            with np.load(cell_dir / "pred.npz") as payload:
                self.assertEqual(payload["pred_m3s"].shape, (671, 5479))
                self.assertEqual(payload["obs_m3s"].shape, (671, 5479))
                self.assertEqual(payload["gauge_ids"].shape, (671,))

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(eval_test.main(argv), 4)
            self.assertEqual(
                len(audit.read_text(encoding="utf-8").splitlines()), 1
            )

        after = {
            path for path in (PROJECT_ROOT / "output").rglob("*") if path.is_file()
        }
        self.assertEqual(before, after, "eval_test wrote under output/")


class TestTrackedRecipes(unittest.TestCase):
    def test_recipes_load_and_run_dir_is_reproducible(self):
        for name in ("g1", "lstm", "mamba3"):
            recipe_path = PROJECT_ROOT / "recipes" / f"{name}.json"
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            config, _ = train_val.config_from_args(
                train_val.parse_args(["--config", str(recipe_path)])
            )
            self.assertEqual(
                Path(config.output_path()).resolve(),
                (PROJECT_ROOT / recipe["run_dir"]).resolve(),
                f"{name} recipe run_dir does not match ExperimentConfig",
            )


class TestRunnerBoundaries(unittest.TestCase):
    def test_training_runner_has_no_test_or_eval_code(self):
        source = Path(run_all_tiers.__file__).read_text(encoding="utf-8")
        self.assertIn("train_val.py", source)
        for forbidden in (
            "import torch",
            "from data_utils",
            "load_split",
            "statError",
            "evaluate_final",
            "eval_only",
        ):
            self.assertNotIn(forbidden, source)

    def test_queue_is_one_train_val_call_per_cell(self):
        args = run_all_tiers.parse_args([
            "--only-lstm", "--tiers", "2",
            "--lstm-layers", "1", "2", "--seeds", "1", "2",
            "--epochs", "1",
        ])
        queue = run_all_tiers.build_queue(args)
        self.assertEqual(len(queue), 4)
        for command in queue:
            self.assertIn("train_val.py", command[1])
            self.assertIn("lstm", command)
            self.assertNotIn("eval_test.py", command)

    def test_test_runner_dry_run_walks_the_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            manifest = {
                "schema": "eval_manifest/v1",
                "cells": [
                    {
                        "id": "A",
                        "recipe": "recipes/a.json",
                        "tier": 1,
                        "depth": 1,
                        "seed": 1,
                        "run_dir": "output/a/All",
                    },
                    {
                        "id": "B",
                        "recipe": "recipes/b.json",
                        "tier": 2,
                        "depth": 3,
                        "seed": 2,
                        "run_dir": "output/b/All",
                    },
                ],
            }
            path = write_json(tmp / "manifest.json", manifest)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = run_test_all.main([
                    "--manifest", str(path),
                    "--output-dir", str(tmp / "out"),
                    "--audit", str(tmp / "out" / "audit.jsonl"),
                    "--dry-run",
                ])
            text = stdout.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("2 pre-registered", text)
            self.assertEqual(text.count("eval_test.py"), 2)


if __name__ == "__main__":
    unittest.main()
