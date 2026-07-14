from __future__ import annotations

import unittest

import numpy as np

from scenario_catalogue import (
    BOUNDARY_BY_ID,
    ScenarioSpec,
    audit_catalogues,
    build_catalogues,
    build_initial_state,
    dirichlet_targets,
    resolve_boundary_indices,
)
from validation_protocol import (
    CHECKPOINT_VALIDATION,
    VALIDATION_OUTPUTS,
    VALIDATION_PROTOCOLS,
)


class ScenarioCatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalogues = build_catalogues()
        cls.audit = audit_catalogues(cls.catalogues)

    def test_exact_counts(self) -> None:
        self.assertEqual(len(self.catalogues["train_c1_1024"]), 1024)
        self.assertEqual(len(self.catalogues["train_c2_2048"]), 2048)
        self.assertEqual(len(self.catalogues["train_c3_3072"]), 3072)
        self.assertEqual(len(self.catalogues["validation_128"]), 128)
        self.assertEqual(len(self.catalogues["test_256"]), 256)

    def test_nested_training_prefixes(self) -> None:
        c1 = [item.signature() for item in self.catalogues["train_c1_1024"]]
        c2 = [item.signature() for item in self.catalogues["train_c2_2048"]]
        c3 = [item.signature() for item in self.catalogues["train_c3_3072"]]
        self.assertEqual(c2[: len(c1)], c1)
        self.assertEqual(c3[: len(c2)], c2)

    def test_audit_passes(self) -> None:
        self.assertTrue(self.audit["passed"], self.audit["tests"])
        self.assertEqual(
            self.audit["catalogues"]["train_c1_1024"]["pairwise_coverage"]["ratio"],
            1.0,
        )

    def test_test_group_sizes(self) -> None:
        test = self.catalogues["test_256"]
        counts = {}
        for item in test:
            counts[item.group] = counts.get(item.group, 0) + 1
        self.assertEqual(counts, {
            "id_combination": 96,
            "magnitude_ood": 48,
            "boundary_material_ood": 48,
            "orientation_ood": 32,
            "hard_combined_ood": 32,
        })

    def test_all_scenarios_have_fixed_vertices(self) -> None:
        for catalogue_name, scenarios in self.catalogues.items():
            for scenario in scenarios:
                boundary = BOUNDARY_BY_ID[scenario.boundary_id]
                indices = resolve_boundary_indices(boundary, 15, 15)
                self.assertGreater(
                    len(indices),
                    0,
                    msg=f"{catalogue_name}/{scenario.scenario_id} has no fixed vertex",
                )

    def test_resolution_independent_boundary_mapping(self) -> None:
        four = BOUNDARY_BY_ID["four_corners"]
        self.assertEqual(resolve_boundary_indices(four, 5, 5), (0, 4, 20, 24))
        self.assertEqual(resolve_boundary_indices(four, 15, 15), (0, 14, 210, 224))

    def test_moving_dirichlet_starts_without_position_jump(self) -> None:
        scenario = ScenarioSpec(
            scenario_id=0,
            split="test",
            group="unit",
            difficulty="unit",
            shape_id="dome_up",
            strain_id="stretch_x",
            velocity_id="translate_pos_x_high",
            boundary_id="pair_diagonal_main",
            dirichlet_id="circle_vertical_pos",
            material_id="material_baseline",
        )
        state = build_initial_state(scenario)
        indices = state["fixed_indices"]
        targets, velocities = dirichlet_targets(
            scenario,
            state["positions"],
            t=0.0,
            fixed_indices=indices,
        )
        self.assertTrue(np.allclose(targets, state["positions"][list(indices)]))
        self.assertTrue(np.isfinite(velocities).all())

    def test_dual_validation_contract(self) -> None:
        self.assertEqual(len(VALIDATION_PROTOCOLS), 2)
        selectors = [item for item in VALIDATION_PROTOCOLS if item.selects_checkpoint]
        self.assertEqual(selectors, [CHECKPOINT_VALIDATION])
        for protocol in VALIDATION_PROTOCOLS:
            self.assertTrue(protocol.save_per_motion)
            self.assertTrue(protocol.save_aggregate_curves)
            self.assertTrue(protocol.render_plots)
            self.assertIn(protocol.id, VALIDATION_OUTPUTS)


if __name__ == "__main__":
    unittest.main()
