from __future__ import annotations

import unittest

import torch

from cloth5x5.config import default_physical_config
from cloth5x5.constants import (
    FREE_STATE_DIM,
    NUM_FREE_PARTICLES,
    NUM_SPRINGS,
    NUM_TRIANGLES,
    SPRING_EDGES,
    TRIANGLE_FACES,
    TORCH_DTYPE,
    build_triangular_cloth_topology,
)
from cloth5x5.physics import (
    free_state_from_full,
    full_positions_from_free,
    spring_lengths_from_free,
    stationarity_residual,
    variational_energy,
)
from cloth5x5.solvers import run_solver_steps


class TopologyTests(unittest.TestCase):
    def test_topology_counts(self) -> None:
        edges, faces = build_triangular_cloth_topology()
        self.assertEqual(edges, SPRING_EDGES)
        self.assertEqual(faces, TRIANGLE_FACES)
        self.assertEqual(len(SPRING_EDGES), NUM_SPRINGS)
        self.assertEqual(len(TRIANGLE_FACES), NUM_TRIANGLES)
        self.assertEqual(NUM_SPRINGS, 56)
        self.assertEqual(NUM_TRIANGLES, 32)
        self.assertEqual(NUM_FREE_PARTICLES, 23)
        self.assertEqual(FREE_STATE_DIM, 69)


class PhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.physical = default_physical_config()
        p = torch.tensor(self.physical.p0, dtype=TORCH_DTYPE)
        self.y = free_state_from_full(p).reshape(1, -1)
        self.q = self.y.clone()
        self.masses = torch.ones(1, NUM_FREE_PARTICLES, dtype=TORCH_DTYPE)

    def test_fixed_vertices_reconstruct_exactly(self) -> None:
        reconstructed = full_positions_from_free(self.y, self.physical).reshape(-1, 3)
        base = torch.tensor(self.physical.p0, dtype=TORCH_DTYPE)
        self.assertLess(float(torch.max(torch.abs(reconstructed - base)).item()), 1e-12)

    def test_energy_gradient_matches_stationarity_residual(self) -> None:
        y = self.y.clone().requires_grad_(True)
        energy = variational_energy(y, self.q, self.masses, self.physical).sum()
        energy.backward()
        assert y.grad is not None
        analytic = stationarity_residual(y.detach(), self.q, self.masses, self.physical)
        self.assertLess(
            float(torch.max(torch.abs(y.grad - analytic)).item()),
            1e-8,
        )

    def test_spring_lengths_are_positive(self) -> None:
        lengths = spring_lengths_from_free(self.y, self.physical)
        self.assertTrue(bool(torch.all(lengths > 0).item()))


class SolverStepperTests(unittest.TestCase):
    def test_run_solver_steps_newton_reduces_energy(self) -> None:
        physical = default_physical_config()
        p = torch.tensor(physical.p0, dtype=TORCH_DTYPE)
        y0 = free_state_from_full(p).reshape(1, -1)
        q = y0.clone()
        masses = torch.ones(1, NUM_FREE_PARTICLES, dtype=TORCH_DTYPE)
        e0 = float(variational_energy(y0, q, masses, physical).item())
        y1, _ = run_solver_steps("full_newton", y0, q, masses, physical, 1)
        e1 = float(variational_energy(y1, q, masses, physical).item())
        self.assertLessEqual(e1, e0 + 1e-10)


if __name__ == "__main__":
    unittest.main()
