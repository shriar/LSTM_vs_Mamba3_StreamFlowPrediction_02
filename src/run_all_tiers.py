"""Training-only batch runner for LSTM / Mamba-3 CAMELS runs.

This module deliberately contains no data, torch, metric or evaluation code.
Every run is delegated to ``src/train_val.py`` in a fresh process, so the
training path cannot open the test pickle.  Test evaluation is a separate
command: ``src/run_test_all.py`` -> ``src/eval_test.py``.

Examples (run from ``experiment/``)::

    # all tiers, one LSTM depth and one Mamba-3 depth, seed 1
    python src/run_all_tiers.py

    # C1b-style LSTM depth ladder, two seeds
    python src/run_all_tiers.py --only-lstm --tiers 2 \
        --lstm-layers 1 2 3 --seeds 1 2

    # one locked headline cell
    python src/run_all_tiers.py --only-mamba --tiers 1 \
        --mamba-layers 3 --seeds 1 --mamba-config recipes/mamba3.json

Use ``--dry-run`` to print the exact train_val commands without launching.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import TypeVar

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
TRAIN_SCRIPT = SRC_DIR / "train_val.py"
T = TypeVar("T")


def _dedupe(values: list[T]) -> list[T]:
    seen: set[T] = set()
    ordered: list[T] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all requested LSTM/Mamba-3 tiers via train_val.py."
    )
    parser.add_argument(
        "--models", "--model", dest="models", nargs="+", default=None,
        help="'lstm', 'mamba3' (alias 'mamba'), or 'all'. Default: all",
    )
    parser.add_argument("--only-lstm", action="store_true")
    parser.add_argument("--only-mamba", action="store_true")
    parser.add_argument(
        "--tiers", type=int, nargs="+", choices=(1, 2, 3), default=(1, 2, 3)
    )
    parser.add_argument(
        "--depths", type=int, nargs="+", choices=(1, 2, 3), default=None,
        help="depth ladder applied to both models; overrides per-model lists",
    )
    parser.add_argument(
        "--lstm-layers", type=int, nargs="+", choices=(1, 2, 3), default=(1,)
    )
    parser.add_argument(
        "--mamba-layers", type=int, nargs="+", choices=(1, 2, 3), default=(3,)
    )
    parser.add_argument(
        "--epochs", "--epoch", dest="epochs", type=int, default=30
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rho", type=int, default=None)
    parser.add_argument(
        "--lstm-config", default=None, help="locked LSTM eval/training recipe"
    )
    parser.add_argument(
        "--mamba-config", default=None, help="locked Mamba-3 eval/training recipe"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--keep-going", action="store_true",
        help="continue after a failed run instead of stopping",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _models(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    if args.only_lstm and args.only_mamba:
        parser.error("--only-lstm and --only-mamba are mutually exclusive")
    requested = args.models
    if args.only_lstm:
        requested = ["lstm"]
    elif args.only_mamba:
        requested = ["mamba3"]
    if requested is None:
        requested = ["lstm", "mamba3"]
    models: list[str] = []
    for raw in requested:
        name = str(raw).lower()
        if name == "all":
            models.extend(["lstm", "mamba3"])
        elif name == "mamba":
            models.append("mamba3")
        elif name in {"lstm", "mamba3"}:
            models.append(name)
        else:
            parser.error(f"unknown model {raw!r}; use lstm, mamba3/mamba, or all")
    return _dedupe(models)


def _depth_ladder(
    args: argparse.Namespace, model: str
) -> list[int]:
    if args.depths:
        return _dedupe([int(d) for d in args.depths])
    return _dedupe(
        [int(d) for d in (args.lstm_layers if model == "lstm" else args.mamba_layers)]
    )


def build_queue(args: argparse.Namespace) -> list[list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    models = _models(args, parser)
    seeds = list(args.seeds or [])
    if args.seed is not None:
        seeds.append(args.seed)
    if not seeds:
        seeds = [1]
    seeds = _dedupe([int(s) for s in seeds])
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    queue: list[list[str]] = []
    for model in models:
        config = args.lstm_config if model == "lstm" else args.mamba_config
        for tier in _dedupe([int(t) for t in args.tiers]):
            for depth in _depth_ladder(args, model):
                for seed in seeds:
                    command = [
                        sys.executable,
                        str(TRAIN_SCRIPT),
                        "--model", model,
                        "--tier", str(tier),
                        "--depth", str(depth),
                        "--epochs", str(int(args.epochs)),
                        "--seed", str(seed),
                    ]
                    if config:
                        command += ["--config", str(config)]
                    if args.batch_size is not None:
                        command += ["--batch-size", str(int(args.batch_size))]
                    if args.rho is not None:
                        command += ["--rho", str(int(args.rho))]
                    if args.output_dir:
                        command += ["--output-dir", str(args.output_dir)]
                    if args.force:
                        command.append("--force")
                    queue.append(command)
    return queue


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        queue = build_queue(args)
    except SystemExit as exc:
        return int(exc.code)

    print(f"{len(queue)} training run(s) queued.")
    failures = 0
    for index, command in enumerate(queue, 1):
        print(f"[{index}/{len(queue)}] {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, cwd=PROJECT_ROOT)
        except KeyboardInterrupt:
            print("interrupted; stopping.", file=sys.stderr)
            return 130
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            failures += 1
            print(
                f"run failed with exit code {completed.returncode} "
                f"after {elapsed:.1f}s",
                file=sys.stderr,
            )
            if not args.keep_going:
                return completed.returncode
        else:
            print(f"run finished in {elapsed:.1f}s", flush=True)
    if failures:
        print(f"{failures} run(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
