"""B14 tests: provenance block in master.json (git SHA, seed, split, hashes)."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import train_val  # noqa: E402
from config import ExperimentConfig  # noqa: E402
from paths import PROJECT_ROOT  # noqa: E402


class TestCanonicalHash(unittest.TestCase):
    def test_hash_is_independent_of_key_order(self):
        a = {"lr": 0.002, "seed": 1, "nested": {"b": [1, 2], "a": None}}
        b = {"seed": 1, "nested": {"a": None, "b": [1, 2]}, "lr": 0.002}
        self.assertEqual(train_val._sha256_json(a), train_val._sha256_json(b))

    def test_hash_changes_with_any_field(self):
        base = {"lr": 0.002, "seed": 1}
        for changed in ({"lr": 0.0021}, {"seed": 2}, {"lr": 0.002, "seed": 1, "x": 0}):
            self.assertNotEqual(
                train_val._sha256_json(base), train_val._sha256_json(changed)
            )

    def test_hash_is_64_hex(self):
        digest = train_val._sha256_json({"a": 1})
        self.assertEqual(len(digest), 64)
        int(digest, 16)


class TestResolvedConfig(unittest.TestCase):
    def test_covers_every_init_field_plus_basins(self):
        cfg = ExperimentConfig(model_type="lstm", param_tier=1)
        spec = train_val._resolved_config(cfg, 150)
        self.assertEqual(spec["basins"], 150)
        for name in ("lr", "seed", "batch_size", "rho", "ctx_days", "epoch"):
            self.assertIn(name, spec)
        self.assertNotIn("script_dir", spec)
        json.dumps(spec)  # must be JSON-clean

    def test_values_are_plain_python(self):
        cfg = ExperimentConfig(model_type="mamba", param_tier=2)
        spec = train_val._resolved_config(cfg, 671)
        for value in spec.values():
            self.assertIsInstance(value, (bool, int, float, str, list, type(None)))


class TestCommandSanitising(unittest.TestCase):
    def test_absolute_paths_become_relative_or_external(self):
        argv = sys.argv
        try:
            sys.argv = [
                str(PROJECT_ROOT / "src" / "train_val.py"),
                "--config", str(PROJECT_ROOT / "recipes" / "g1.json"),
                "--output-dir", "/home/somebody/private/run",
            ]
            text = train_val._relative_command()
        finally:
            sys.argv = argv
        self.assertTrue(text.startswith("python "))
        self.assertNotIn("/home/", text)
        self.assertIn("src/train_val.py", text)
        self.assertIn("recipes/g1.json", text)
        self.assertIn("<external>/run", text)


class TestDirtyPaths(unittest.TestCase):
    def test_parse_keeps_root_relative_paths_and_renames(self):
        text = (
            " M experiment/src/train_val.py\n"
            "?? experiment/Writing/Jounal_Writting_latex/Abstract.tex\n"
            "R  experiment/old.py -> experiment/new.py\n"
        )
        self.assertEqual(
            train_val._parse_status(text),
            [
                "experiment/Writing/Jounal_Writting_latex/Abstract.tex",
                "experiment/new.py",
                "experiment/src/train_val.py",
            ],
        )

    def test_paths_are_repository_relative_and_exist(self):
        root = train_val._repo_root()
        paths = train_val._dirty_paths()
        self.assertIsInstance(paths, list)
        for entry in paths:
            self.assertFalse(Path(entry).is_absolute(), entry)
            self.assertNotIn("\\", entry)
            self.assertTrue((root / entry).exists(), entry)


class TestMasterCarriesProvenance(unittest.TestCase):
    """One real 1-epoch / 2-basin run, then inspect what it wrote."""

    def test_provenance_block_and_no_test_artefact(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp) / "unit_recipe.json"
            recipe.write_text(json.dumps({
                "schema": "eval_recipe/v1",
                "id": "unit",
                "model": "lstm",
                "tier": 1,
                "depth": 1,
                "epochs": 1,
                "seed": 5,
                "hyperparameters": {"lr": 0.0015},
                "locked": False,
            }), encoding="utf-8")
            out_root = Path(tmp) / "out"
            with contextlib.redirect_stdout(io.StringIO()):
                train_val.main([
                    "--config", str(recipe),
                    "--basins", "2",
                    "--output-dir", str(out_root),
                ])
            masters = sorted(out_root.rglob("master.json"))
            self.assertEqual(len(masters), 1, masters)
            payload = json.loads(masters[0].read_text(encoding="utf-8"))

            prov = payload["provenance"]
            for key in ("git_sha", "seed", "split_dates", "split_sha256",
                        "data_manifest_sha256", "config", "config_sha256",
                        "recipe", "git_dirty", "git_dirty_paths"):
                self.assertIn(key, prov)
            self.assertEqual(prov["seed"], 5)
            self.assertEqual(sorted(prov["split_dates"]), ["test", "train", "val"])
            self.assertEqual(len(prov["split_sha256"]), 64)
            self.assertEqual(len(prov["data_manifest_sha256"]), 64)
            self.assertEqual(
                prov["config_sha256"], train_val._sha256_json(prov["config"])
            )
            self.assertEqual(prov["config"]["basins"], 2)
            self.assertEqual(prov["config"]["lr"], 0.0015)
            self.assertEqual(len(prov["recipe"]["sha256"]), 64)
            self.assertEqual(prov["recipe"]["path"], "unit_recipe.json")

            self.assertNotIn("/home/", payload["command"])
            self.assertNotIn(os.sep + "home" + os.sep, payload["command"])
            self.assertIn("--basins 2", payload["command"])
            self.assertFalse(payload["test_split_opened"])
            self.assertEqual(list(masters[0].parent.glob("*test*")), [])
            self.assertTrue((masters[0].parent / "val_basin_metrics.csv").is_file())

    def test_second_seed_changes_config_hash_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "out"
            hashes = []
            for seed in (11, 12):
                with contextlib.redirect_stdout(io.StringIO()):
                    train_val.main([
                        "--model", "lstm", "--tier", "1", "--depth", "1",
                        "--epochs", "1", "--seed", str(seed), "--basins", "2",
                        "--output-dir", str(out_root),
                    ])
                master = sorted(out_root.rglob(f"*Seed{seed}/All/master.json"))[0]
                hashes.append(json.loads(master.read_text())["provenance"])
            self.assertNotEqual(hashes[0]["config_sha256"], hashes[1]["config_sha256"])
            self.assertEqual(hashes[0]["split_sha256"], hashes[1]["split_sha256"])
            self.assertEqual(
                hashes[0]["data_manifest_sha256"], hashes[1]["data_manifest_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
