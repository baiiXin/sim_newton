from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # Local documentation/runtime images may omit PyTorch.
    torch = None

if torch is not None:
    from cloth02_batched_physics import load_physics


@unittest.skipIf(torch is None, "PyTorch is not installed in this runtime")
class TShirtPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.physics = load_physics(dtype=torch.float64)

    def test_rest_internal_energy_is_zero(self) -> None:
        rest = self.physics.rest_positions.unsqueeze(0)
        membrane = self.physics.membrane_energy(rest)
        bending = self.physics.bending_energy(rest)
        self.assertLess(float(torch.abs(membrane).max()), 1e-9)
        self.assertLess(float(torch.abs(bending).max()), 1e-12)

    def test_rigid_rotation_does_not_create_internal_energy(self) -> None:
        rest = self.physics.rest_positions
        center = rest.mean(dim=0, keepdim=True)
        angle = torch.tensor(0.73, dtype=rest.dtype)
        c, s = torch.cos(angle), torch.sin(angle)
        rotation = torch.stack(
            (
                torch.stack((c, -s, torch.tensor(0.0, dtype=rest.dtype))),
                torch.stack((s, c, torch.tensor(0.0, dtype=rest.dtype))),
                torch.tensor((0.0, 0.0, 1.0), dtype=rest.dtype),
            )
        )
        rotated = (rest - center) @ rotation.T + center
        self.assertLess(float(torch.abs(self.physics.membrane_energy(rotated)).max()), 1e-8)
        self.assertLess(float(torch.abs(self.physics.bending_energy(rotated)).max()), 1e-9)

    def test_autograd_residual_matches_directional_difference(self) -> None:
        generator = torch.Generator().manual_seed(17)
        rest = self.physics.rest_positions.unsqueeze(0)
        perturb = 1e-4 * torch.randn(rest.shape, dtype=rest.dtype, generator=generator)
        y = (rest + perturb).requires_grad_(True)
        q = rest.clone()
        targets = rest.clone()
        residual = self.physics.stationarity_residual(y, q, targets)
        direction = torch.randn(y.shape, dtype=y.dtype, generator=generator)
        direction[:, self.physics.fixed_mask] = 0.0
        direction /= torch.linalg.vector_norm(direction)
        epsilon = 1e-7
        plus = self.physics.variational_energy(y.detach() + epsilon * direction, q, targets)
        minus = self.physics.variational_energy(y.detach() - epsilon * direction, q, targets)
        finite_difference = (plus - minus) / (2.0 * epsilon)
        analytic = torch.sum(residual * direction, dim=(-2, -1))
        torch.testing.assert_close(analytic, finite_difference, rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(
            residual[:, self.physics.fixed_mask],
            torch.zeros_like(residual[:, self.physics.fixed_mask]),
        )

    def test_block_hessian_is_spd_and_fixed_gate_is_zero(self) -> None:
        rest = self.physics.rest_positions.unsqueeze(0)
        q = rest + 1e-3
        residual = self.physics.stationarity_residual(rest, q, rest)
        blocks = self.physics.block_diagonal_hessian(rest)
        eigenvalues = torch.linalg.eigvalsh(blocks)
        self.assertGreater(float(eigenvalues.min()), 0.0)
        direction = self.physics.block_hessian_preconditioned_residual(rest, residual)
        torch.testing.assert_close(
            direction[:, self.physics.fixed_mask],
            torch.zeros_like(direction[:, self.physics.fixed_mask]),
        )


if __name__ == "__main__":
    unittest.main()

