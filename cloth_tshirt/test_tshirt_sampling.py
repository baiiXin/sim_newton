from __future__ import annotations

import unittest

import numpy as np

from tshirt_config import (
    DEFAULT_DYNAMICS,
    DEFAULT_FIXED_DATA_DIR,
    DEFAULT_MODEL_SEED,
    DEFAULT_OBJ_PATH,
)
from tshirt_mesh import connected_component_sizes, load_tshirt_mesh
from tshirt_sampling import (
    build_fixed_model_spec,
    build_inference_motion,
    build_typical_motions,
    deformation_quality,
    sample_random_motion,
)


class TShirtSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = load_tshirt_mesh(DEFAULT_OBJ_PATH)
        cls.model = build_fixed_model_spec(cls.mesh, model_seed=DEFAULT_MODEL_SEED)

    def test_mesh_matches_export_audit(self) -> None:
        self.assertEqual(self.mesh.num_vertices, 4424)
        self.assertEqual(self.mesh.num_faces, 8710)
        self.assertEqual(self.mesh.edges.shape, (13136, 2))
        self.assertEqual(self.mesh.hinge_indices.shape, (12994, 4))
        self.assertEqual(self.mesh.boundary_edges.shape, (142, 2))
        self.assertEqual(connected_component_sizes(self.mesh.num_vertices, self.mesh.edges), (4424,))

    def test_fixed_model_has_four_distinct_shoulder_vertices(self) -> None:
        fixed = self.model.fixed_indices
        self.assertEqual(len(fixed), 4)
        self.assertEqual(len(set(fixed)), 4)
        positions = self.mesh.vertices[np.asarray(fixed)]
        self.assertTrue(np.any(positions[:, 0] < 0.0))
        self.assertTrue(np.any(positions[:, 0] > 0.0))
        self.assertTrue(np.ptp(positions[positions[:, 0] < 0.0, 2]) > 0.02)
        self.assertTrue(np.ptp(positions[positions[:, 0] > 0.0, 2]) > 0.02)

    def test_random_motion_is_reproducible_and_admissible(self) -> None:
        first = sample_random_motion(
            self.mesh,
            self.model,
            DEFAULT_DYNAMICS,
            seed=12345,
            motion_id="a",
            split="test",
        )
        second = sample_random_motion(
            self.mesh,
            self.model,
            DEFAULT_DYNAMICS,
            seed=12345,
            motion_id="a",
            split="test",
        )
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(first.velocities, second.velocities)
        quality = deformation_quality(self.mesh, first.positions)
        self.assertGreaterEqual(quality["min_area_ratio"], DEFAULT_DYNAMICS.min_area_ratio)
        self.assertGreaterEqual(quality["min_singular_value"], DEFAULT_DYNAMICS.min_singular_value)
        np.testing.assert_array_equal(
            first.velocities[np.asarray(self.model.fixed_indices)],
            np.zeros((4, 3)),
        )

    def test_typical_set_contains_horizontal_gravity_release(self) -> None:
        states = build_typical_motions(
            self.mesh,
            self.model,
            DEFAULT_DYNAMICS,
            seed=20260723,
        )
        self.assertEqual(len(states), 4)
        horizontal = states[0]
        self.assertIn("horizontal_gravity_release", horizontal.motion_id)
        np.testing.assert_allclose(horizontal.velocities, 0.0)
        self.assertGreater(
            states[2].metadata["high_frequency_velocity_rms_requested"], 2.0
        )

    def test_frozen_evaluation_sizes_are_fixed_but_training_is_not_archived(self) -> None:
        for filename, count in (
            ("validation_32.npz", 32),
            ("test_64.npz", 64),
            ("typical_single_motions_4.npz", 4),
        ):
            with np.load(DEFAULT_FIXED_DATA_DIR / filename) as archive:
                self.assertEqual(archive["positions"].shape, (count, 4424, 3))
                self.assertEqual(archive["velocities"].shape, (count, 4424, 3))
        self.assertFalse((DEFAULT_FIXED_DATA_DIR / "train.npz").exists())

    def test_inference_motion_has_user_controlled_high_frequency_component(self) -> None:
        motion = build_inference_motion(
            self.mesh,
            self.model,
            DEFAULT_DYNAMICS,
            seed=42,
            pose="horizontal",
            high_frequency_velocity_rms=2.0,
            position_perturb_rms_edge_fraction=0.05,
        )
        self.assertEqual(motion.metadata["pose"], "horizontal")
        self.assertEqual(motion.metadata["high_frequency_velocity_rms_requested"], 2.0)
        self.assertGreater(motion.metadata["velocity_rms"], 1.8)
        self.assertGreaterEqual(
            motion.metadata["min_area_ratio"], DEFAULT_DYNAMICS.min_area_ratio
        )


if __name__ == "__main__":
    unittest.main()
