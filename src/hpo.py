"""B8 - validation-only HPO driver: one ``train_val.py`` subprocess per trial.

This module holds no torch, data, metric or evaluation code.  It reads a
tracked grid, writes one recipe per trial, calls ``train_val.py`` once with its
own run folder, and aggregates the validation scores.  It cannot open the test
slice: there is no ``allow_test`` path here, and it refuses a trial folder that
already contains a test artefact.

Usage (run from ``experiment/``)::

    python src/hpo.py recipes/sweeps/matched.json --dry-run
    python src/hpo.py recipes/sweeps/matched.json --models lstm
    python src/hpo.py recipes/sweeps/sensitivity.json --only lr0p0005_clip0p5_b20p95
    python src/hpo.py recipes/sweeps/matched.json --only mamba3/trial_3 lstm/trial_7

Resumable by construction: a trial whose ``trial.json`` records ``status: ok``
is skipped, so an interrupted sweep continues where it stopped.  A folder left
behind by a crashed run (partial ``val_metrics.csv``, no ``trial.json``) is
restarted with ``--force`` instead of being mistaken for a finished trial.
``--force`` on the command line re-runs a finished trial too (and is recorded in
its ``trial.json``).

Outputs::

    results/hpo/<grid>/<model>/<trial_id>/recipe.json   generated, not tracked
    results/hpo/<grid>/<model>/<trial_id>/run/...       train_val.py output
    results/hpo/<grid>/<model>/<trial_id>/trial.json    status + scores
    results/hpo/<grid>/<model>/summary.csv                ranked table
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
TRAIN_SCRIPT = SRC_DIR / "train_val.py"
DEFAULT_OUTPUT_ROOT = Path("results") / "hpo"

SCHEMA = "hpo_grid/v1"

# Grid axis -> ExperimentConfig field, per architecture.  The summary quotes the
# axis names (eta0, eta_min, lambda, clip, beta2, dropout); the recipe quotes the
# field names train_val.py expects.
AXIS_FIELDS = {
    "lstm": {
        "eta0": "lr",
        "eta_min": "lstm_min_lr",
        "lambda": "lstm_weight_decay",
        "clip": "lstm_grad_clip",
        "beta2": "lstm_beta2",
        "dropout": "lstm_dropout",
    },
    "mamba3": {
        "eta0": "mamba_lr",
        "eta_min": "mamba_min_lr",
        "lambda": "mamba_weight_decay",
        "clip": "mamba_grad_clip",
        "beta2": "mamba_beta2",
        "dropout": "mamba_dropout",
    },
}
SUMMARY_COLUMNS = [
    "trial_id", "model", "eta0", "eta_min", "lambda", "clip", "beta2",
    "dropout", "best_epoch", "val_median_nse", "val_median_kge",
    "val_nse_iqr", "wall_clock", "status", "run_dir",
]


class GridError(SystemExit):
    """A grid that cannot be run as written (exit code 2)."""

    def __init__(self, message: str, code: int = 2):
        super().__init__(code)
        self.message = message

    def __str__(self) -> str:
        return self.message


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (the numpy default), no numpy needed."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_grid(path: str | Path) -> tuple[dict, Path]:
    grid_path = Path(path)
    if not grid_path.is_file():
        raise GridError(f"grid not found: {grid_path}")
    try:
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GridError(f"grid {grid_path} is not valid JSON: {exc}") from exc
    if not isinstance(grid, dict) or grid.get("schema") != SCHEMA:
        raise GridError(
            f"grid {grid_path} must be a JSON object with schema {SCHEMA!r}"
        )
    if _carries_test_metric(grid):
        raise GridError(
            f"grid {grid_path} carries a test metric; grids are validation-only"
        )
    archs = grid.get("architectures")
    if not isinstance(archs, dict) or not archs:
        raise GridError(f"grid {grid_path} has no 'architectures' object")
    for model, block in archs.items():
        if model not in AXIS_FIELDS:
            raise GridError(f"grid {grid_path}: unknown model {model!r}")
        trials = block.get("trials")
        if not isinstance(trials, list) or not trials:
            raise GridError(f"grid {grid_path}: {model} has no trials")
        ids = [str(t.get("id") or "") for t in trials]
        if any(not tid for tid in ids):
            raise GridError(f"grid {grid_path}: {model} trial without an id")
        if len(set(ids)) != len(ids):
            raise GridError(f"grid {grid_path}: duplicate trial id in {model}")
        for trial in trials:
            params = trial.get("params")
            if not isinstance(params, dict) or not params:
                raise GridError(
                    f"grid {grid_path}: {model}/{trial.get('id')} has no params"
                )
            unknown = sorted(set(params) - set(AXIS_FIELDS[model].values()))
            if unknown:
                raise GridError(
                    f"grid {grid_path}: {model}/{trial['id']} has unknown "
                    f"parameter(s) {unknown}; known: "
                    f"{sorted(AXIS_FIELDS[model].values())}"
                )
    return grid, grid_path


def _carries_test_metric(node) -> bool:
    """True if any key anywhere in the grid mentions a test metric."""
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = str(key).lower()
            if "test" in lowered:
                return True
            if _carries_test_metric(value):
                return True
    elif isinstance(node, list):
        return any(_carries_test_metric(item) for item in node)
    return False


def trial_recipe(grid: dict, model: str, trial: dict, grid_path: Path) -> dict:
    block = grid["architectures"][model]
    params = dict(block.get("baseline") or {})
    params.update(trial["params"])
    return {
        "schema": "hpo_trial/v1",
        "id": f"{grid['id']}/{model}/{trial['id']}",
        "name": f"{grid['name']} :: {model} :: {trial['id']}",
        "source": (
            f"generated by src/hpo.py from {grid_path.as_posix()} "
            f"(sha256 {_sha256(grid_path)[:16]}); do not edit by hand"
        ),
        "notes": trial.get("origin", ""),
        "model": model,
        "tier": int(block["tier"]),
        "depth": int(block["depth"]),
        "seed": int(grid.get("seed", 1)),
        "epochs": int(block["epochs"]),
        "hyperparameters": params,
    }


def trial_dir(output_root: Path, grid: dict, model: str, trial_id: str) -> Path:
    return output_root / str(grid["id"]) / model / trial_id


def trial_is_complete(destination: Path) -> bool:
    """A trial counts as done only when ``trial.json`` says ``status: ok``.

    ``val_metrics.csv`` alone is not enough: a run that crashed mid-sweep
    leaves a two-row table behind, and treating that as finished would silently
    drop the trial from the comparison.
    """
    payload_path = destination / "trial.json"
    if not payload_path.is_file():
        return False
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "ok"


def find_val_metrics(root: Path) -> Path | None:
    matches = sorted(root.rglob("val_metrics.csv"))
    return matches[0] if matches else None


def find_test_artefacts(root: Path) -> list[Path]:
    names = ("test_metrics.csv", "basin_test_metrics.csv", "pred.npz")
    return [p for p in root.rglob("*") if p.is_file() and p.name in names]


def best_epoch_row(csv_path: Path) -> dict:
    best = None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                nse = float(row["val_median_nse"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isnan(nse):
                continue
            if best is None or nse > float(best["val_median_nse"]):
                best = row
    if best is None:
        raise GridError(f"{csv_path} has no finite val_median_nse row")
    return best


def val_iqr(run_root: Path, metrics_path: Path) -> float:
    for candidate in sorted(run_root.rglob("val_basin_metrics.csv")):
        values = []
        with candidate.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    nse = float(row["NSE"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isnan(nse):
                    values.append(nse)
        return _percentile(values, 75.0) - _percentile(values, 25.0)
    # Fall back to what master.json recorded when the run was written.
    for master in sorted(metrics_path.parent.glob("master.json")):
        payload = json.loads(master.read_text(encoding="utf-8"))
        value = (payload.get("selection") or {}).get("best_val_nse_iqr")
        if value is not None:
            return float(value)
    return float("nan")


def axes_from_params(model: str, params: dict) -> dict:
    fields = AXIS_FIELDS[model]
    return {axis: params.get(field) for axis, field in fields.items()}


def run_trial(command: list[str]) -> tuple[int, float]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return completed.returncode, time.perf_counter() - started


def plan_trials(grid: dict, grid_path: Path, output_root: Path,
                only: set[str] | None = None,
                models: set[str] | None = None) -> list[dict]:
    """Build the run list for a grid.

    ``models`` keeps only the named architectures.  ``only`` accepts either a
    bare trial id (``trial_3``, which selects that trial in every architecture)
    or a qualified one (``mamba3/trial_3``), so a 13+13 grid can be split by
    architecture without listing every id.
    """
    unknown_models = set(models or ()) - set(grid["architectures"])
    if unknown_models:
        raise GridError(
            f"unknown model(s) {sorted(unknown_models)}; grid has "
            f"{sorted(grid['architectures'])}"
        )

    planned = []
    matched: set[str] = set()
    for model, block in grid["architectures"].items():
        if models and model not in models:
            continue
        for trial in block["trials"]:
            trial_id = trial["id"]
            qualified = f"{model}/{trial_id}"
            if only and trial_id not in only and qualified not in only:
                continue
            matched.add(trial_id)
            matched.add(qualified)
            destination = trial_dir(output_root, grid, model, trial_id)
            recipe = trial_recipe(grid, model, trial, grid_path)
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--model", model,
                "--tier", str(recipe["tier"]),
                "--depth", str(recipe["depth"]),
                "--epochs", str(recipe["epochs"]),
                "--seed", str(recipe["seed"]),
                "--basins", str(int(grid["basins"])),
                "--config", str(destination / "recipe.json"),
                "--output-dir", str(destination / "run"),
            ]
            planned.append({
                "model": model,
                "trial": trial,
                "recipe": recipe,
                "destination": destination,
                "command": command,
            })
    if only:
        missing = set(only) - matched
        if missing:
            raise GridError(
                f"unknown trial id(s) {sorted(missing)}; use a bare id "
                "('trial_3') or a qualified one ('mamba3/trial_3')"
            )
    return planned


def write_summary(rows: list[dict], grid_id: str, model: str, output_root: Path) -> Path:
    # The summary sits beside the trial folders it describes.  It used to be
    # output_root/<model>/summary.csv, which meant a second grid on the same
    # model (sensitivity after matched) silently overwrote the first grid's
    # ranking while leaving its trial folders intact.
    path = output_root / grid_id / model / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows,
        key=lambda r: (
            -(r["val_median_nse"] if r["val_median_nse"] is not None else -math.inf)
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ("" if row.get(key) is None else row.get(key))
                for key in SUMMARY_COLUMNS
            })
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tracked validation-only HPO grid via train_val.py."
    )
    parser.add_argument("grid", help="recipes/sweeps/{matched,sensitivity}.json")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="[MODEL/]TRIAL_ID",
        help="run only these trials; 'trial_3' selects it in every "
             "architecture, 'mamba3/trial_3' in one",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        help="run only these architectures (e.g. --models lstm)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        grid, grid_path = load_grid(args.grid)
        output_root = Path(args.output_root)
        planned = plan_trials(
            grid, grid_path, output_root,
            set(args.only) if args.only else None,
            set(args.models) if args.models else None,
        )
    except GridError as exc:
        print(exc, file=sys.stderr)
        return int(exc.code)

    print(f"{len(planned)} trial(s) planned from {grid_path}.")
    failures = 0
    for index, item in enumerate(planned, 1):
        model = item["model"]
        trial = item["trial"]
        destination = item["destination"]
        params = item["recipe"]["hyperparameters"]
        axes = axes_from_params(model, params)

        artefacts = find_test_artefacts(destination) if destination.is_dir() else []
        if artefacts:
            print(
                f"refusing {model}/{trial['id']}: test artefact(s) present "
                f"({', '.join(p.name for p in artefacts)}); grids are "
                "validation-only.",
                file=sys.stderr,
            )
            return 5

        existing = find_val_metrics(destination) if destination.is_dir() else None
        complete = trial_is_complete(destination)
        if complete and not args.force:
            print(f"[{index}/{len(planned)}] {model}/{trial['id']} already done; skip")
            continue
        restart = destination.is_dir() and not complete
        if restart and existing is not None:
            print(
                f"[{index}/{len(planned)}] {model}/{trial['id']} incomplete "
                "(val_metrics.csv without an ok trial.json); restarting"
            )
        if args.force or restart:
            item["command"] = [*item["command"], "--force"]

        print(f"[{index}/{len(planned)}] {model}/{trial['id']} "
              f"eta0={axes['eta0']} clip={axes['clip']} beta2={axes['beta2']}",
              flush=True)
        print("   " + " ".join(item["command"]), flush=True)
        if args.dry_run:
            continue

        destination.mkdir(parents=True, exist_ok=True)
        (destination / "recipe.json").write_text(
            json.dumps(item["recipe"], indent=2) + "\n", encoding="utf-8"
        )

        code, seconds = run_trial(item["command"])
        metrics = find_val_metrics(destination)
        status = "ok" if code == 0 and metrics is not None else f"failed({code})"
        payload = dict(
            trial_id=trial["id"],
            model=model,
            grid=grid["id"],
            params=params,
            axes=axes,
            recipe="recipe.json",
            run_dir=str(destination / "run"),
            status=status,
            exit_code=code,
            wall_clock=round(seconds, 1),
            forced=bool(args.force),
            restarted=bool(restart),
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        if metrics is not None:
            row = best_epoch_row(metrics)
            payload["best_epoch"] = int(float(row["epoch"]))
            payload["val_median_nse"] = float(row["val_median_nse"])
            payload["val_median_kge"] = float(row["val_median_kge"])
            payload["val_nse_iqr"] = val_iqr(destination, metrics)
        (destination / "trial.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        if status != "ok":
            failures += 1
            print(f"   trial failed (exit {code})", file=sys.stderr)
            if not args.keep_going:
                return code or 1

    if args.dry_run:
        return 0

    # The summary describes the state of the sweep, not just this invocation:
    # a grid run in halves must not lose the half that ran earlier.
    for model in sorted({item["model"] for item in planned}):
        subset = collect_summary_rows(output_root, grid, model)
        path = write_summary(subset, str(grid["id"]), model, output_root)
        print(f"\n{model}: {path} ({len(subset)} trial(s) on disk)")
        for rank, row in enumerate(
            sorted(subset, key=lambda r: -(r["val_median_nse"] or -math.inf)), 1
        ):
            print(
                f"  {rank:2d}. {row['trial_id']:<34} "
                f"val NSE {row['val_median_nse']}  "
                f"eta0={row['eta0']} clip={row['clip']} beta2={row['beta2']}"
            )
    return 1 if failures else 0


def collect_summary_rows(output_root: Path, grid: dict, model: str) -> list[dict]:
    """One row per trial folder that exists on disk for this grid/model."""
    base = output_root / str(grid["id"]) / model
    if not base.is_dir():
        return []
    grid_params = {
        trial["id"]: trial.get("params") or {}
        for trial in grid["architectures"][model]["trials"]
    }
    rows = []
    for destination in sorted(p for p in base.iterdir() if p.is_dir()):
        payload: dict = {}
        payload_path = destination / "trial.json"
        if payload_path.is_file():
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        params = payload.get("params")
        if not params:
            recipe_path = destination / "recipe.json"
            if recipe_path.is_file():
                params = json.loads(
                    recipe_path.read_text(encoding="utf-8")
                ).get("hyperparameters")
        if not params:
            # Last resort: the grid itself, for a folder whose recipe was lost.
            params = grid_params.get(destination.name) or {}
        axes = payload.get("axes") or axes_from_params(model, params or {})
        row = _trial_row(
            model, {"id": destination.name}, axes, destination,
            find_val_metrics(destination),
        )
        if payload.get("status"):
            row["status"] = payload["status"]
        if payload.get("wall_clock") is not None:
            row["wall_clock"] = payload["wall_clock"]
        rows.append(row)
    return rows


def _trial_row(model: str, trial: dict, axes: dict, destination: Path,
               metrics: Path | None) -> dict:
    row = {
        "trial_id": trial["id"],
        "model": model,
        **axes,
        "best_epoch": None,
        "val_median_nse": None,
        "val_median_kge": None,
        "val_nse_iqr": None,
        "wall_clock": None,
        "status": "missing",
        "run_dir": str(destination / "run"),
    }
    payload_path = destination / "trial.json"
    if payload_path.is_file():
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        row["status"] = payload.get("status", "missing")
        row["wall_clock"] = payload.get("wall_clock")
        row["best_epoch"] = payload.get("best_epoch")
        row["val_median_nse"] = payload.get("val_median_nse")
        row["val_median_kge"] = payload.get("val_median_kge")
        row["val_nse_iqr"] = payload.get("val_nse_iqr")
        return row
    if metrics is not None:
        best = best_epoch_row(metrics)
        # No trial.json: the run died before recording itself, so its score is
        # a partial curve, not a result.
        row["status"] = "incomplete"
        row["best_epoch"] = int(float(best["epoch"]))
        row["val_median_nse"] = float(best["val_median_nse"])
        row["val_median_kge"] = float(best["val_median_kge"])
        row["val_nse_iqr"] = val_iqr(destination, metrics)
    return row


if __name__ == "__main__":
    raise SystemExit(main())
