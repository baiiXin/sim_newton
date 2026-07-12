from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from cloth03_training_pool import (
    LearnedOptimizerMLP,
    LiveTrainingPool,
    ModelSpec,
    apply_model_update,
    normalized_one_step_energy_loss,
    training_step,
)
from cloth04_reference_free_validation import (
    FailureThresholds,
    checkpoint_rank,
    run_reference_free_validation,
    save_validation_result,
)
from scenario_catalogue import build_catalogues
from validation_protocol import FAST_MONITOR


class TrainingPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalogues = build_catalogues()
        cls.scenarios = tuple(catalogues["train_c1_1024"][:32])
        cls.validation = tuple(catalogues["validation_128"][:8])

    def make_pool(
        self,
        *,
        pool_size: int = 8,
        batch_size: int = 4,
        k_buckets=(1, 2),
        lifetime: int = 4,
    ) -> LiveTrainingPool:
        return LiveTrainingPool(
            scenarios=self.scenarios,
            device="cpu",
            dtype=torch.float64,
            pool_size=pool_size,
            batch_size=batch_size,
            k_buckets=k_buckets,
            max_lifetime_physical_steps=lifetime,
        )

    def make_model(self, pool: LiveTrainingPool) -> LearnedOptimizerMLP:
        return LearnedOptimizerMLP(
            full_state_dim=pool.parameter_bank.full_state_dim,
            residual_length_scale=5e-2,
            model_spec=ModelSpec(
                activation="identity",
                depth=1,
                width=16,
                use_bias=False,
            ),
            dtype=torch.float64,
        )

    def test_balanced_batch_scheduler(self) -> None:
        pool = self.make_pool()
        for _ in range(5):
            batch = pool.ask()
            values, counts = torch.unique(batch.k_values, return_counts=True)
            self.assertEqual(values.tolist(), [1, 2])
            self.assertEqual(counts.tolist(), [2, 2])
            self.assertEqual(len(set(batch.row_indices.tolist())), 4)

    def test_zero_initialized_model_respects_fixed_gate(self) -> None:
        pool = self.make_pool()
        batch = pool.ask()
        model = self.make_model(pool)
        y_next, delta, _ = apply_model_update(
            model,
            batch.y,
            batch.q,
            batch.params,
            target_positions=batch.target_positions,
            previous_residual=batch.previous_residual,
            previous_update=batch.previous_update,
        )
        self.assertTrue(torch.allclose(delta, torch.zeros_like(delta)))
        expected = batch.y.reshape(batch.params.batch_size, -1)
        self.assertTrue(torch.allclose(y_next, expected))
        gate = (~batch.params.fixed_mask).unsqueeze(-1).expand(-1, -1, 3)
        delta_points = delta.reshape(batch.params.batch_size, -1, 3)
        self.assertTrue(
            torch.equal(
                delta_points[~gate],
                torch.zeros_like(delta_points[~gate]),
            )
        )

    def test_energy_change_loss_backpropagates(self) -> None:
        pool = self.make_pool()
        batch = pool.ask()
        model = self.make_model(pool)
        y_next, delta, _ = apply_model_update(
            model,
            batch.y,
            batch.q,
            batch.params,
            target_positions=batch.target_positions,
            previous_residual=batch.previous_residual,
            previous_update=batch.previous_update,
        )
        result = normalized_one_step_energy_loss(
            y_before=batch.y,
            y_after=y_next,
            q=batch.q,
            delta=delta,
            params=batch.params,
            target_positions=batch.target_positions,
        )
        result.loss.backward()
        self.assertTrue(torch.isfinite(result.loss))
        self.assertIsNotNone(model.output_layer.weight.grad)
        self.assertGreater(float(model.output_layer.weight.grad.abs().sum()), 0.0)

    def test_one_training_step_changes_output_layer(self) -> None:
        pool = self.make_pool()
        model = self.make_model(pool)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.output_layer.weight.detach().clone()
        metrics = training_step(
            model=model,
            optimizer=optimizer,
            pool=pool,
            gradient_clip_norm=10.0,
        )
        self.assertFalse(torch.equal(before, model.output_layer.weight.detach()))
        self.assertEqual(metrics["batch_size"], 4)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    def test_lifetime_reset_reassigns_scenarios(self) -> None:
        pool = self.make_pool(
            pool_size=4,
            batch_size=4,
            k_buckets=(1,),
            lifetime=1,
        )
        batch = pool.ask()
        old = pool.scenario_indices.clone()
        zeros = torch.zeros_like(batch.previous_update)
        pool.tell(
            batch,
            y_next=batch.y.reshape(batch.params.batch_size, -1),
            delta=zeros,
            current_residual=zeros,
            energy_after=torch.zeros(batch.params.batch_size, dtype=torch.float64),
            residual_after=torch.ones(batch.params.batch_size, dtype=torch.float64),
        )
        self.assertFalse(torch.equal(old, pool.scenario_indices))
        self.assertEqual(pool.reset_counts["resets_lifetime"], 4)
        self.assertEqual(pool.total_completed_physical_frames, 4)

    def test_pool_state_roundtrip(self) -> None:
        pool = self.make_pool()
        _ = pool.ask()
        state = pool.state_dict()
        restored = self.make_pool()
        restored.load_state_dict(state)
        for name in (
            "scenario_indices",
            "p",
            "v",
            "q",
            "y",
            "target_positions",
            "previous_residual",
            "previous_update",
            "inner_iteration",
            "physical_step",
            "age_physical_step",
            "seen_scenarios",
        ):
            self.assertTrue(torch.equal(getattr(pool, name), getattr(restored, name)))
        self.assertEqual(pool.batch_cursors, restored.batch_cursors)
        self.assertEqual(pool.scenario_cursor, restored.scenario_cursor)

    def test_checkpoint_rank_is_stability_first(self) -> None:
        stable = {
            "failed_motion_count": 0,
            "survival_frame_p05": 50,
            "residual_ratio_p95": 100.0,
            "energy_increase_fraction": 1.0,
        }
        unstable = {
            "failed_motion_count": 1,
            "survival_frame_p05": 50,
            "residual_ratio_p95": 1e-12,
            "energy_increase_fraction": 0.0,
        }
        self.assertLess(checkpoint_rank(stable), checkpoint_rank(unstable))

    def test_short_reference_free_validation_and_save(self) -> None:
        pool = self.make_pool()
        model = self.make_model(pool)
        protocol = replace(
            FAST_MONITOR,
            motion_count=4,
            rollout_frames=2,
            inner_steps=1,
            interval_updates=1,
            render_plots=False,
        )
        result = run_reference_free_validation(
            model=model,
            scenarios=self.validation,
            protocol=protocol,
            device="cpu",
            dtype=torch.float64,
            batch_size=2,
            thresholds=FailureThresholds(
                max_residual=1e30,
                max_abs_position=1e30,
                min_edge_ratio=1e-12,
                max_edge_ratio=1e12,
            ),
        )
        self.assertEqual(len(result.per_motion), 4)
        self.assertEqual(tuple(result.raw["residual_ratio"].shape), (4, 2))
        self.assertEqual(tuple(result.curves["inner_residual"].shape), (4, 2, 2))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_validation_result(
                result=result,
                output_root=root,
                update_count=3,
                wall_clock_seconds=1.5,
                render_plots=False,
            )
            protocol_root = root / "validation" / protocol.id
            self.assertTrue((protocol_root / "history.csv").exists())
            self.assertTrue((protocol_root / "per_motion.csv").exists())
            self.assertTrue((protocol_root / "curves.pt").exists())
            self.assertTrue(
                (
                    protocol_root
                    / "runs"
                    / "update_000000003"
                    / "curves.pt"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
