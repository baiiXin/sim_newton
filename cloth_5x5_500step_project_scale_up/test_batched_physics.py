from __future__ import annotations

import unittest

import torch

from cloth02_batched_physics import (
    advance_state,
    build_batched_parameters,
    build_cloth_topology,
    dirichlet_targets,
    make_q,
    project_positions,
    stationarity_residual,
    variational_energy,
)
from scenario_templates import ScenarioSpec


def make_scenario(
    scenario_id: int,
    *,
    boundary_id: str,
    dirichlet_id: str,
    material_id: str,
    velocity_id: str = "velocity_zero",
) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=scenario_id,
        split="train",
        group="unit",
        difficulty="unit",
        shape_id="plane",
        strain_id="strain_none",
        velocity_id=velocity_id,
        boundary_id=boundary_id,
        dirichlet_id=dirichlet_id,
        material_id=material_id,
    )


class BatchedPhysicsTests(unittest.TestCase):
    def test_topology_counts_and_area(self) -> None:
        topology = build_cloth_topology()
        self.assertEqual(topology.num_vertices, 25)
        self.assertEqual(topology.num_edges, 56)
        self.assertAlmostEqual(sum(topology.vertex_areas), 49.0)
        self.assertEqual(sum(topology.edge_is_diagonal), 16)

    def test_variable_fixed_masks_and_materials(self) -> None:
        params = build_batched_parameters([
            make_scenario(
                0,
                boundary_id="single_corner_tl",
                dirichlet_id="circle_horizontal_pos",
                material_id="material_baseline",
            ),
            make_scenario(
                1,
                boundary_id="pair_diagonal_main",
                dirichlet_id="twist_pos",
                material_id="material_heavy",
            ),
            make_scenario(
                2,
                boundary_id="four_corners",
                dirichlet_id="static",
                material_id="material_shear_stiff",
            ),
        ])
        self.assertEqual(tuple(params.masses.shape), (3, 25))
        self.assertEqual(params.fixed_mask.sum(dim=1).tolist(), [1, 2, 4])
        self.assertGreater(params.masses[1].sum(), params.masses[0].sum())

        diagonal = torch.as_tensor(params.topology.edge_is_diagonal)
        diagonal_mean = params.spring_stiffness[2, diagonal].mean()
        structural_mean = params.spring_stiffness[2, ~diagonal].mean()
        self.assertGreater(diagonal_mean, structural_mean)

    def test_projection_leaves_free_vertices_unchanged(self) -> None:
        params = build_batched_parameters([
            make_scenario(
                0,
                boundary_id="pair_left_corners",
                dirichlet_id="static",
                material_id="material_baseline",
            )
        ])
        trial = params.initial_positions + 1.0
        projected = project_positions(trial, params)
        self.assertTrue(torch.allclose(
            projected[params.fixed_mask],
            params.initial_positions[params.fixed_mask],
        ))
        self.assertTrue(torch.allclose(
            projected[~params.fixed_mask],
            trial[~params.fixed_mask],
        ))

    def test_energy_gradient_matches_analytic_residual(self) -> None:
        torch.manual_seed(42)
        params = build_batched_parameters([
            make_scenario(
                0,
                boundary_id="pair_left_corners",
                dirichlet_id="static",
                material_id="material_baseline",
            ),
            make_scenario(
                1,
                boundary_id="four_corners",
                dirichlet_id="static",
                material_id="material_heavy",
            ),
        ])
        q = make_q(
            params.initial_positions,
            params.initial_velocities,
            params,
        )
        y = (
            params.initial_positions
            + 0.01 * torch.randn_like(params.initial_positions)
        ).requires_grad_(True)
        energy = variational_energy(y, q, params).sum()
        autograd_gradient = torch.autograd.grad(energy, y)[0]
        analytic_gradient = stationarity_residual(y, q, params)
        self.assertTrue(torch.allclose(
            autograd_gradient,
            analytic_gradient,
            rtol=1e-10,
            atol=1e-9,
        ), float((autograd_gradient - analytic_gradient).abs().max().item()))
        self.assertEqual(
            float(analytic_gradient[params.fixed_mask].abs().max().item()),
            0.0,
        )

    def test_moving_targets_start_without_position_jump(self) -> None:
        params = build_batched_parameters([
            make_scenario(
                0,
                boundary_id="single_corner_tl",
                dirichlet_id="circle_horizontal_pos",
                material_id="material_baseline",
            ),
            make_scenario(
                1,
                boundary_id="pair_diagonal_main",
                dirichlet_id="twist_pos",
                material_id="material_baseline",
            ),
        ])
        targets_zero, velocities_zero = dirichlet_targets(params, 0.0)
        self.assertTrue(torch.allclose(targets_zero, params.initial_positions))
        self.assertTrue(torch.isfinite(velocities_zero).all())

        targets_later, velocities_later = dirichlet_targets(params, 0.1)
        self.assertFalse(torch.allclose(
            targets_later[params.fixed_mask],
            params.initial_positions[params.fixed_mask],
        ))
        self.assertTrue(torch.isfinite(velocities_later).all())

    def test_advance_state_enforces_prescribed_boundary(self) -> None:
        params = build_batched_parameters([
            make_scenario(
                0,
                boundary_id="single_edge_top",
                dirichlet_id="circle_vertical_pos",
                material_id="material_baseline",
            )
        ])
        solved = params.initial_positions + 1.0
        target, target_velocity = dirichlet_targets(params, params.dt)
        next_positions, next_velocities = advance_state(
            params.initial_positions,
            solved,
            params,
            next_time=params.dt,
        )
        self.assertTrue(torch.allclose(
            next_positions[params.fixed_mask],
            target[params.fixed_mask],
        ))
        self.assertTrue(torch.allclose(
            next_velocities[params.fixed_mask],
            target_velocity[params.fixed_mask],
        ))

    def test_batch_index_selection_preserves_metadata(self) -> None:
        params = build_batched_parameters([
            make_scenario(
                5,
                boundary_id="single_center",
                dirichlet_id="static",
                material_id="material_soft",
            ),
            make_scenario(
                7,
                boundary_id="four_corners",
                dirichlet_id="static",
                material_id="material_stiff",
            ),
        ])
        selected = params.index_select([1])
        self.assertEqual(selected.scenario_ids.tolist(), [7])
        self.assertEqual(int(selected.fixed_mask.sum().item()), 4)
        self.assertEqual(selected.boundary_ids, ("four_corners",))
        self.assertEqual(selected.material_ids, ("material_stiff",))


if __name__ == "__main__":
    unittest.main()
