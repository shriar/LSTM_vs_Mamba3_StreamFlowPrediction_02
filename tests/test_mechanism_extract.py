"""Guards for the B7 mechanism extraction module.

Pure-function tests run on CPU.  One test replays a tiny CuDNN LSTM on the
GPU to pin the gate order (i, f, g, o); it is skipped where CUDA is
unavailable.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mechanism_extract as me  # noqa: E402


class SelectBasinsTest(unittest.TestCase):
    def test_panel_counts_and_disjoint(self):
        real = me.ATTR_LST
        c = np.zeros((40, len(real)))
        c[:, real.index("p_mean")] = np.linspace(500, 2500, 40)
        c[:, real.index("frac_snow")] = np.linspace(0, 0.7, 40)
        panel = me.select_basins(c, n_snow=6, n_wet=7, n_dry=7)
        self.assertEqual(len(panel), 20)
        cats = [row["category"] for row in panel]
        self.assertEqual(cats.count("snow"), 6)
        self.assertEqual(cats.count("wet"), 7)
        self.assertEqual(cats.count("dry"), 7)
        self.assertEqual(len({row["index"] for row in panel}), 20)

    def test_deterministic(self):
        c = np.zeros((30, len(me.ATTR_LST)))
        c[:, me.ATTR_LST.index("p_mean")] = np.arange(30, dtype=float) + 500
        c[:, me.ATTR_LST.index("frac_snow")] = (np.arange(30) % 5) / 10.0
        first = me.select_basins(c)
        second = me.select_basins(c)
        self.assertEqual(
            [(r["index"], r["category"]) for r in first],
            [(r["index"], r["category"]) for r in second],
        )

    def test_too_few_basins_raises(self):
        c = np.zeros((10, len(me.ATTR_LST)))
        with self.assertRaises(ValueError):
            me.select_basins(c)


class InputTensorTest(unittest.TestCase):
    def test_shape_and_channel_order(self):
        x = np.arange(2 * 5 * 5, dtype=float).reshape(2, 5, 5)
        c = np.arange(2 * 35, dtype=float).reshape(2, 35) + 1000.0
        out = me.build_input_tensor(x, c, torch.device("cpu")).numpy()
        self.assertEqual(out.shape, (5, 2, 40))
        np.testing.assert_array_equal(out[:, :, :5], np.swapaxes(x, 0, 1))
        np.testing.assert_array_equal(out[:, :, 5:], np.swapaxes(
            np.repeat(c[:, None, :], 5, axis=1), 0, 1))


class DeltaSliceTest(unittest.TestCase):
    def _layer(self):
        return SimpleNamespace(
            d_inner=192, d_state=64, num_bc_heads=1, mimo_rank=1,
            nheads=6, dt_bias=torch.zeros(6),
        )

    def test_offset_shape_and_value(self):
        layer = self._layer()
        start = 2 * 192 + 2 * 64
        width = start + 6 + 4
        proj = torch.zeros(4, 10, width)
        proj[..., start:start + 6] = 1.0
        dt = me.mamba_dt_from_projection(proj, layer)
        self.assertEqual(tuple(dt.shape), (4, 10, 6))
        self.assertTrue(torch.all(dt > 0))
        self.assertAlmostEqual(float(dt.mean()), float(torch.nn.functional.softplus(torch.tensor(1.0))), places=5)

    def test_short_projection_raises(self):
        layer = self._layer()
        with self.assertRaises(ValueError):
            me.mamba_dt_from_projection(torch.zeros(2, 5, 10), layer)


class MassBalanceTest(unittest.TestCase):
    def test_exact_ratio(self):
        q_sim = np.array([[2.0, 4.0], [1.0, 1.0]])
        q_obs = np.array([[1.0, 2.0], [1.0, 1.0]])
        np.testing.assert_allclose(me.mass_balance(q_sim, q_obs), [2.0, 1.0])

    def test_zero_observed_is_nan(self):
        out = me.mass_balance(np.array([[1.0]]), np.array([[0.0]]))
        self.assertTrue(np.isnan(out[0]))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            me.mass_balance(np.zeros((2, 3)), np.zeros((2, 4)))


@unittest.skipUnless(torch.cuda.is_available(), "needs the CUDA LSTM kernels")
class ReplayOrderTest(unittest.TestCase):
    def test_ifgo_reproduces_cudnn(self):
        from CudnnLstmModel import CudnnLstmModel

        torch.manual_seed(0)
        dev = torch.device("cuda")
        model = CudnnLstmModel(nx=40, ny=1, hiddenSize=8, nLayer=1).to(dev).eval()
        inp = torch.randn(30, 4, 40, device=dev)
        with torch.inference_mode():
            reference = model(inp)
            replay = me.replay_lstm(model, inp)["output"]
        diff = float((replay - reference).abs().max())
        scale = max(1.0, float(reference.abs().max()))
        self.assertLess(diff / scale, 1e-3)


if __name__ == "__main__":
    unittest.main()
