from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
