"""B9 regression: the effective-capacity arithmetic, without a GPU.

The profiler's value is the two numbers the capacity claim (A8) quotes --
218,053 parameters vs 18,432 SSM-state elements -- so the arithmetic that
produces them must be checkable on a CPU-only checkout.  The GPU part
(VRAM, throughput) is measurement, not arithmetic, and is covered by the
committed ``results/table_effective_capacity.csv`` plus its provenance
columns.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import effective_capacity as ec  # noqa: E402


class HeadlineLstmArithmetic(unittest.TestCase):
    def test_macs_match_the_hand_computation(self):
        # linearIn 40x162 + one 4-gate cell 4*162*(162+162) + head 162*1
        expected = 40 * 162 + 4 * 162 * 324 + 162
        self.assertEqual(ec.lstm_macs_per_basin_day(40, 162, 1, 1), expected)
        self.assertEqual(expected, 216_594)

    def test_flops_are_twice_the_macs(self):
        macs = ec.lstm_macs_per_basin_day(40, 162, 1, 1)
        self.assertEqual(2 * macs * ec.SEQ_DAYS, 2 * 216_594 * 365)


class HeadlineMambaArithmetic(unittest.TestCase):
    def test_scan_term_is_two_macs_per_state_element(self):
        self.assertEqual(ec.scan_macs_per_basin_day(192, 64, 3),
                         2 * 3 * 192 * 64)
        self.assertEqual(ec.scan_macs_per_basin_day(192, 64, 3), 73_728)

    def test_state_asymmetry(self):
        lstm = ec.state_sizes("lstm", 162, 1, 64)
        mamba = ec.state_sizes("mamba", 96, 3, 64, 4, 2)
        self.assertEqual(lstm["state_ssm_total"], 162)
        self.assertEqual(mamba["state_ssm_total"], 18_432)   # 3 * 96 * 64
        # every buffer: LSTM h+c, Mamba SSM + 3 layers * 192 channels * 3 taps
        self.assertEqual(lstm["state_total_incl_aux"], 324)
        self.assertEqual(mamba["state_total_incl_aux"], 18_432 + 1_728)
        self.assertGreater(mamba["state_ssm_total"] / lstm["state_ssm_total"], 100)


class ModuleWalk(unittest.TestCase):
    def test_counts_dense_and_depthwise_layers(self):
        import torch.nn as nn

        module = nn.Sequential(
            nn.Linear(40, 96),
            nn.Conv1d(192, 192, kernel_size=4, groups=192),
            nn.Linear(96, 1),
        )
        self.assertEqual(ec.module_macs(module), 40 * 96 + 192 * 4 + 96 * 1)


class SplitGuard(unittest.TestCase):
    def test_profiler_cannot_open_the_test_split(self):
        import re

        source = (SRC / "effective_capacity.py").read_text(encoding="utf-8")
        # The docstring may mention the name; the code may not import it, and
        # nothing here may reach the split loader that takes allow_test.
        self.assertNotRegex(source, r"(?m)^\s*(from|import)\s+eval_test")
        self.assertNotIn("allow_test", source)
        self.assertNotIn("load_split", source)


if __name__ == "__main__":
    unittest.main()
