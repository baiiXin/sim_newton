from __future__ import annotations

import unittest

from cloth07_evaluate_best_checkpoint import (
    grouped_test_summaries,
    summarize_motion_rows,
)
from cloth04_reference_free_validation import ValidationResult


class FinalEvaluationTests(unittest.TestCase):
    def make_row(
        self,
        *,
        scenario_id: int,
        group: str,
        failed: bool,
        survival_frames: int,
        residual_ratio: float,
        energy_fraction: float,
    ) -> dict:
        return {
            "scenario_id": scenario_id,
            "scenario_group": group,
            "boundary_id": "corner_tl",
            "material_id": "baseline",
            "failed": failed,
            "failure_frame": survival_frames if failed else None,
            "survival_frames": survival_frames,
            "failure_reason": "residual" if failed else "",
            "residual_ratio_p95_diagnostic": residual_ratio,
            "residual_ratio_selection": float("inf") if failed else residual_ratio,
            "final_residual": 2.0 + scenario_id,
            "energy_increase_fraction": energy_fraction,
            "minimum_edge_ratio": 0.8,
            "maximum_edge_ratio": 1.2,
            "maximum_constraint_error": 0.0,
        }

    def test_summary_is_stability_aware(self) -> None:
        rows = [
            self.make_row(
                scenario_id=0,
                group="id_combination",
                failed=False,
                survival_frames=500,
                residual_ratio=0.2,
                energy_fraction=0.1,
            ),
            self.make_row(
                scenario_id=1,
                group="id_combination",
                failed=True,
                survival_frames=20,
                residual_ratio=0.01,
                energy_fraction=0.0,
            ),
        ]
        summary = summarize_motion_rows(rows, rollout_frames=500)
        self.assertEqual(summary["motion_count"], 2)
        self.assertEqual(summary["failed_motion_count"], 1)
        self.assertEqual(summary["survival_rate"], 0.5)
        self.assertEqual(summary["residual_ratio_p95"], float("inf"))

    def test_grouped_test_summary_contains_all_and_each_group(self) -> None:
        rows = [
            self.make_row(
                scenario_id=0,
                group="id_combination",
                failed=False,
                survival_frames=500,
                residual_ratio=0.2,
                energy_fraction=0.1,
            ),
            self.make_row(
                scenario_id=1,
                group="magnitude_ood",
                failed=False,
                survival_frames=500,
                residual_ratio=0.4,
                energy_fraction=0.2,
            ),
        ]
        result = ValidationResult(
            protocol={"id": "test"},
            summary={"rollout_frames": 500},
            per_motion=rows,
            curves={},
            raw={},
        )
        grouped = grouped_test_summaries(result, inner_steps=10)
        self.assertEqual(
            {row["group"] for row in grouped},
            {"all", "id_combination", "magnitude_ood"},
        )
        self.assertTrue(all(row["inner_steps"] == 10 for row in grouped))


if __name__ == "__main__":
    unittest.main()
