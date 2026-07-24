from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from cloth02_batched_physics import FrozenMotionBatch, load_physics
    from cloth03_training_pool import (
        LearnedOptimizerMLP,
        ModelSpec,
        OnlineTrainingPool,
        apply_model_update,
        training_step,
    )
    from cloth09_rollout_single_motion import SingleMotionSettings, run_solver_rollout
    from cloth25_rollout_newton_single_motion import _minimum_residual
    from cloth26_rollout_newton_best_iterate import _initial_iterate


@unittest.skipIf(torch is None, "PyTorch is not installed in this runtime")
class TrainingPoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.physics = load_physics(dtype=torch.float64)

    def test_online_pool_is_reproducible_but_does_not_store_dataset(self) -> None:
        first = OnlineTrainingPool(
            physics=self.physics, seed=77, pool_size=4, batch_size=4, k_buckets=(1,)
        )
        second = OnlineTrainingPool(
            physics=self.physics, seed=77, pool_size=4, batch_size=4, k_buckets=(1,)
        )
        torch.testing.assert_close(first.p, second.p, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first.v, second.v, rtol=0.0, atol=0.0)
        self.assertFalse(first.manifest()["training_samples_persisted"])

    def test_zero_output_model_respects_fixed_gate(self) -> None:
        pool = OnlineTrainingPool(
            physics=self.physics, seed=91, pool_size=4, batch_size=4, k_buckets=(1,)
        )
        model = LearnedOptimizerMLP(
            physics=self.physics,
            model_spec=ModelSpec(activation="relu", depth=1, width=16, use_bias=False),
        )
        batch = pool.ask()
        y_next, delta, _ = apply_model_update(
            model,
            batch.y,
            batch.q,
            batch.fixed_targets,
            previous_residual=batch.previous_residual,
            previous_update=batch.previous_update,
        )
        torch.testing.assert_close(delta, torch.zeros_like(delta))
        torch.testing.assert_close(
            y_next[:, self.physics.fixed_mask], batch.fixed_targets[:, self.physics.fixed_mask]
        )

    def test_one_training_step_backpropagates(self) -> None:
        pool = OnlineTrainingPool(
            physics=self.physics, seed=101, pool_size=4, batch_size=4, k_buckets=(1,)
        )
        model = LearnedOptimizerMLP(
            physics=self.physics,
            model_spec=ModelSpec(activation="relu", depth=1, width=16, use_bias=False),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        metrics = training_step(model=model, optimizer=optimizer, pool=pool)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertGreater(metrics["gradient_norm_before_clip"], 0.0)

    def test_single_motion_uses_the_full_inner_iteration_budget(self) -> None:
        rest = self.physics.rest_positions.unsqueeze(0)
        motion = FrozenMotionBatch(
            motion_ids=("fixed_budget_contract",),
            positions=rest,
            velocities=torch.zeros_like(rest),
            seeds=torch.tensor((0,), dtype=torch.long),
        )
        result = run_solver_rollout(
            solver="gd_fixed",
            physics=self.physics,
            motion=motion,
            settings=SingleMotionSettings(
                rollout_frames=1,
                inner_steps=3,
                residual_ratio_tolerance=2.0,
                fixed_gd_step_size=0.0,
                trajectory_stride=1,
                early_stop=False,
            ),
        )
        self.assertEqual(int(result.curves["inner_steps"][0]), 3)
        self.assertTrue(result.summary["fixed_inner_iteration_budget"])

    def test_network_line_search_safely_rejects_zero_direction(self) -> None:
        rest = self.physics.rest_positions.unsqueeze(0)
        motion = FrozenMotionBatch(
            motion_ids=("line_search_zero_direction",),
            positions=rest,
            velocities=torch.zeros_like(rest),
            seeds=torch.tensor((0,), dtype=torch.long),
        )
        model = LearnedOptimizerMLP(
            physics=self.physics,
            model_spec=ModelSpec(activation="relu", depth=1, width=16, use_bias=False),
        )
        result = run_solver_rollout(
            solver="network",
            physics=self.physics,
            motion=motion,
            model=model,
            settings=SingleMotionSettings(
                rollout_frames=1,
                inner_steps=3,
                network_line_search=True,
                trajectory_stride=1,
            ),
        )
        self.assertEqual(result.summary["line_search_accepted_step_count"], 0)
        self.assertEqual(result.summary["line_search_rejected_step_count"], 3)
        self.assertEqual(int(result.curves["inner_steps"][0]), 3)
        self.assertFalse(result.summary["failed"])

    def test_preconditioned_minres_solves_spd_system(self) -> None:
        diagonal = torch.tensor((2.0, 5.0), dtype=torch.float64)
        right_hand_side = torch.tensor((2.0, -5.0), dtype=torch.float64)
        result = _minimum_residual(
            hvp=lambda value: diagonal * value,
            preconditioner=lambda value: value / diagonal,
            right_hand_side=right_hand_side,
            max_iterations=4,
            relative_tolerance=1e-12,
            absolute_tolerance=1e-14,
        )
        self.assertTrue(result.converged)
        self.assertFalse(result.breakdown)
        torch.testing.assert_close(
            result.step,
            torch.tensor((1.0, -1.0), dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_minres_solves_indefinite_system(self) -> None:
        diagonal = torch.tensor((2.0, -3.0, 5.0), dtype=torch.float64)
        right_hand_side = torch.tensor((1.0, -2.0, 3.0), dtype=torch.float64)
        result = _minimum_residual(
            hvp=lambda value: diagonal * value,
            preconditioner=lambda value: value,
            right_hand_side=right_hand_side,
            max_iterations=10,
            relative_tolerance=1e-12,
            absolute_tolerance=1e-14,
        )
        self.assertTrue(result.converged)
        self.assertFalse(result.breakdown)
        self.assertLess(result.minimum_curvature, 0.0)
        torch.testing.assert_close(
            diagonal * result.step,
            right_hand_side,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_inertia_initial_iterate_preserves_free_velocity(self) -> None:
        positions = self.physics.rest_positions.unsqueeze(0).clone()
        velocities = torch.ones_like(positions)
        fixed_targets = positions.clone()
        guess = _initial_iterate(
            physics=self.physics,
            positions=positions,
            velocities=velocities,
            fixed_targets=fixed_targets,
            mode="inertia",
        )
        expected = self.physics.project_positions(
            positions + self.physics.dt * velocities,
            fixed_targets,
        )
        torch.testing.assert_close(guess, expected)
        _, next_velocities = self.physics.advance_state(
            positions, guess, fixed_targets
        )
        gate = self.physics.free_update_gate(1, dtype=positions.dtype)
        torch.testing.assert_close(next_velocities, velocities * gate)


if __name__ == "__main__":
    unittest.main()
