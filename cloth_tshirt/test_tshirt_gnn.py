"""Unit tests for the first plain T-shirt GNN baseline."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from cloth16_gnn_model import GNNModelSpec, LearnedOptimizerGNN


class _FakePhysics:
    def __init__(self) -> None:
        self.num_vertices = 4
        self.dtype = torch.float64
        self.device = torch.device("cpu")
        self.edges = torch.tensor(((0, 1), (1, 2), (2, 3)), dtype=torch.long)
        self.model = SimpleNamespace(mesh_sha256="fake")

    def check_state(self, value: torch.Tensor, name: str) -> torch.Tensor:
        expected = (value.shape[0], self.num_vertices, 3)
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        return value

    def stationarity_residual(
        self, y: torch.Tensor, q: torch.Tensor, fixed_targets: torch.Tensor
    ) -> torch.Tensor:
        del fixed_targets
        return y - q

    def mass_preconditioned_residual(self, residual: torch.Tensor) -> torch.Tensor:
        raise AssertionError("The GNN baseline must not mass-precondition its residual")

    def free_update_gate(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        gate = torch.ones((batch_size, self.num_vertices, 3), dtype=dtype)
        gate[:, -1] = 0.0
        return gate


class _NegativeDecoder(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return -torch.ones((*hidden.shape[:-1], 3), dtype=hidden.dtype, device=hidden.device)


class TShirtGNNTests(unittest.TestCase):
    def setUp(self) -> None:
        self.physics = _FakePhysics()
        self.model = LearnedOptimizerGNN(physics=self.physics, model_spec=GNNModelSpec(width=8))
        self.y = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3) / 10.0
        self.q = torch.zeros_like(self.y)
        self.fixed = torch.zeros_like(self.y)

    def test_default_contract(self) -> None:
        spec = GNNModelSpec()
        self.assertEqual(spec.width, 128)
        self.assertEqual(spec.depth, 2)
        self.assertEqual(spec.message_passing_steps, 15)
        self.assertFalse(spec.use_bias)
        self.assertIs(self.model.edge_mlp, self.model.edge_mlp)
        self.assertIs(self.model.node_mlp, self.model.node_mlp)
        self.assertFalse(any(layer.bias is not None for layer in self.model.modules() if isinstance(layer, nn.Linear)))

    def test_raw_residual_is_used(self) -> None:
        current = self.model.current_residual(self.y, self.q, self.fixed)
        torch.testing.assert_close(current, self.y.reshape(2, -1))
        alias = self.model.current_preconditioned_residual(self.y, self.q, self.fixed)
        torch.testing.assert_close(alias, current)

    def test_zero_initialized_update_and_shapes(self) -> None:
        delta, current = self.model(self.y, self.q, self.fixed)
        self.assertEqual(tuple(delta.shape), (2, 12))
        self.assertEqual(tuple(current.shape), (2, 12))
        torch.testing.assert_close(delta, torch.zeros_like(delta))

    def test_decoder_output_has_no_relu_or_scale(self) -> None:
        self.model.decoder = _NegativeDecoder()
        delta, _ = self.model(self.y, self.q, self.fixed)
        points = delta.reshape(2, 4, 3)
        torch.testing.assert_close(points[:, :3], -torch.ones_like(points[:, :3]))
        torch.testing.assert_close(points[:, 3], torch.zeros_like(points[:, 3]))

    def test_invalid_baseline_variants_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GNNModelSpec(activation="gelu")
        with self.assertRaises(ValueError):
            GNNModelSpec(depth=3)
        with self.assertRaises(ValueError):
            GNNModelSpec(use_bias=True)


if __name__ == "__main__":
    unittest.main()
