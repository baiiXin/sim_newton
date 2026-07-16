from __future__ import annotations

import unittest
from unittest.mock import patch

from cloth06_probe_memory_and_throughput import (
    _recommend,
    _row_configuration,
    _validate_controller_args,
    parse_args as parse_memory_probe_args,
)
from tshirt_config import DEFAULT_EVALUATION
from validation_protocol import CHECKPOINT_VALIDATION, FAST_MONITOR, checkpoint_rank


class PipelineContractTests(unittest.TestCase):
    def test_requested_inner_iteration_caps_are_frozen(self) -> None:
        self.assertEqual(FAST_MONITOR.inner_steps, 15)
        self.assertEqual(CHECKPOINT_VALIDATION.inner_steps, 50)
        self.assertEqual(DEFAULT_EVALUATION.convergence_residual_ratio, 1e-3)
        self.assertEqual(DEFAULT_EVALUATION.two_order_single_step_ratio, 1e-2)
        self.assertFalse(FAST_MONITOR.early_stop)
        self.assertFalse(CHECKPOINT_VALIDATION.early_stop)

    def test_checkpoint_rank_is_stability_first(self) -> None:
        base = {
            "failed_motion_count": 0,
            "survival_frame_p05": 100,
            "residual_ratio_p95": 0.1,
            "single_step_le_two_orders_frame_count": 20,
            "energy_increase_fraction": 0.0,
        }
        unstable = {**base, "failed_motion_count": 1, "residual_ratio_p95": 1e-8}
        self.assertLess(checkpoint_rank(base), checkpoint_rank(unstable))

    def test_memory_probe_exposes_model_pool_and_batch_configuration(self) -> None:
        argv = [
            "cloth06_probe_memory_and_throughput.py",
            "--activation", "silu",
            "--depth", "2",
            "--width", "1024",
            "--use-bias",
            "--pool-size", "64",
            "--batch-sizes", "4", "8", "16",
        ]
        with patch("sys.argv", argv):
            args = parse_memory_probe_args()
        _validate_controller_args(args)
        row = _row_configuration(args, batch_size=8)
        self.assertEqual(row["activation"], "silu")
        self.assertEqual(row["depth"], 2)
        self.assertEqual(row["width"], 1024)
        self.assertTrue(row["use_bias"])
        self.assertEqual(row["pool_size"], 64)
        self.assertEqual(row["batch_size"], 8)

    def test_memory_recommendation_preserves_training_shape(self) -> None:
        row = {
            "status": "success",
            "batch_size": 16,
            "pool_size": 64,
            "device": "cuda:0",
            "dtype": "float64",
            "seed": 7,
            "activation": "gelu",
            "depth": 3,
            "width": 512,
            "use_bias": False,
            "peak_reserved_fraction": 0.5,
            "peak_reserved_gib": 12.0,
            "motions_per_second": 20.0,
        }
        recommendation = _recommend([row], headroom=0.85)
        self.assertIsNotNone(recommendation)
        assert recommendation is not None
        for key in ("activation", "depth", "width", "use_bias", "pool_size"):
            self.assertEqual(recommendation[key], row[key])
        self.assertEqual(recommendation["recommended_batch_size"], row["batch_size"])


if __name__ == "__main__":
    unittest.main()
