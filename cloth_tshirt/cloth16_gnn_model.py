"""Shared-weight message-passing GNN baseline for the T-shirt learned optimizer."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DEFAULT_GNN_WIDTH = 128
DEFAULT_MESSAGE_PASSING_STEPS = 5


@dataclass(frozen=True)
class GNNModelSpec:
    """Architecture contract for the first plain GNN baseline.

    The first four fields deliberately match the legacy dense ``ModelSpec``
    constructor so the existing online-training driver can be reused.
    """

    activation: str = "relu"
    depth: int = 2
    width: int = DEFAULT_GNN_WIDTH
    use_bias: bool = False
    message_passing_steps: int = DEFAULT_MESSAGE_PASSING_STEPS

    def __post_init__(self) -> None:
        if self.activation != "relu":
            raise ValueError("The baseline GNN uses ReLU only")
        if self.depth != 2:
            raise ValueError("The baseline GNN uses exactly two linear layers per MLP")
        if self.width <= 0:
            raise ValueError("GNN width must be positive")
        if self.use_bias:
            raise ValueError("The baseline GNN does not use bias")
        if self.message_passing_steps <= 0:
            raise ValueError("message_passing_steps must be positive")

    @property
    def experiment_name(self) -> str:
        return (
            f"gnn_raw_residual_mp{self.message_passing_steps:02d}_"
            f"width_{self.width:04d}_depth_{self.depth:02d}_no_bias"
        )


class _TwoLayerReLU(nn.Module):
    """Two linear layers, with ReLU after both layers."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, width, bias=False, dtype=dtype, device=device)
        self.linear2 = nn.Linear(width, width, bias=False, dtype=dtype, device=device)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear2(torch.relu(self.linear1(value))))


class _TwoLayerResidualBranch(nn.Module):
    """Two linear layers with a linear output suitable for a zero-init branch.

    Keeping the final layer linear is important: a zero-initialized linear
    layer followed by ReLU has zero derivative and can never start learning.
    """

    def __init__(
        self,
        input_dim: int,
        width: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, width, bias=False, dtype=dtype, device=device)
        self.linear2 = nn.Linear(width, width, bias=False, dtype=dtype, device=device)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(value)))


class _TwoLayerDecoder(nn.Module):
    """128 -> 128 -> 3 decoder; the final output has no activation."""

    def __init__(self, width: int, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.linear1 = nn.Linear(width, width, bias=False, dtype=dtype, device=device)
        self.linear2 = nn.Linear(width, 3, bias=False, dtype=dtype, device=device)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(value)))


class LearnedOptimizerGNN(nn.Module):
    """Plain graph-network learned optimizer with shared processor weights.

    Per-vertex input is ``[residual, previous residual, previous delta]`` (9D).
    Residuals are raw stationarity residuals: no mass preconditioning is used.
    There are no fixed-point indicators, edge attributes, or input/output scale
    factors in this first baseline.
    """

    def __init__(
        self,
        *,
        physics: Any,
        residual_length_scale: float = 1.0,
        model_spec: GNNModelSpec = GNNModelSpec(),
        initialize: bool = True,
    ) -> None:
        super().__init__()
        if residual_length_scale <= 0.0:
            raise ValueError("residual_length_scale compatibility value must be positive")
        self.physics = physics
        self.full_state_dim = 3 * physics.num_vertices
        self.model_spec = model_spec
        self.message_passing_steps = model_spec.message_passing_steps
        self.width = model_spec.width

        edges = torch.as_tensor(physics.edges, dtype=torch.long, device=physics.device)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"physics.edges must have shape [E, 2], got {tuple(edges.shape)}")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= physics.num_vertices):
            raise ValueError("physics.edges contains an out-of-range vertex index")
        self.register_buffer("edge_sources", edges[:, 0].contiguous())
        self.register_buffer("edge_targets", edges[:, 1].contiguous())

        self.encoder = _TwoLayerReLU(9, self.width, dtype=physics.dtype, device=physics.device)
        self.edge_mlp = _TwoLayerReLU(
            2 * self.width, self.width, dtype=physics.dtype, device=physics.device
        )
        self.node_mlp = _TwoLayerResidualBranch(
            2 * self.width, self.width, dtype=physics.dtype, device=physics.device
        )
        self.decoder = _TwoLayerDecoder(self.width, dtype=physics.dtype, device=physics.device)

        # Compatibility only: legacy checkpoint code reads this field.  It is
        # intentionally not used anywhere in forward().
        self.register_buffer(
            "residual_length_scale",
            torch.tensor(1.0, dtype=physics.dtype, device=physics.device),
        )

        if initialize:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = sqrt(2.0)
        for layer in (
            self.encoder.linear1,
            self.encoder.linear2,
            self.edge_mlp.linear1,
            self.edge_mlp.linear2,
            self.node_mlp.linear1,
            self.decoder.linear1,
        ):
            nn.init.orthogonal_(layer.weight, gain=gain)
        # The processor starts as an identity map, and the whole learned update
        # starts at exactly zero.
        nn.init.zeros_(self.node_mlp.linear2.weight)
        nn.init.zeros_(self.decoder.linear2.weight)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def current_residual(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        fixed_targets: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.physics.stationarity_residual(y, q, fixed_targets)
        return residual.reshape(residual.shape[0], -1).detach()

    def current_preconditioned_residual(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        fixed_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility alias; this baseline deliberately returns raw residual."""
        return self.current_residual(y, q, fixed_targets)

    def _directed_messages(self, hidden: torch.Tensor) -> torch.Tensor:
        source_hidden = hidden.index_select(1, self.edge_sources)
        target_hidden = hidden.index_select(1, self.edge_targets)

        # Receiver feature comes first.  Every undirected mesh edge produces two
        # directed messages while both directions share the same edge MLP.
        source_to_target = self.edge_mlp(torch.cat((target_hidden, source_hidden), dim=-1))
        target_to_source = self.edge_mlp(torch.cat((source_hidden, target_hidden), dim=-1))

        aggregated = torch.zeros_like(hidden)
        aggregated.index_add_(1, self.edge_targets, source_to_target)
        aggregated.index_add_(1, self.edge_sources, target_to_source)
        return aggregated

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        fixed_targets: torch.Tensor,
        *,
        previous_residual: torch.Tensor | None = None,
        previous_update: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.physics.check_state(y, "y")
        batch_size = y.shape[0]
        current = self.current_residual(y, q, fixed_targets)
        if previous_residual is None:
            previous_residual = torch.zeros_like(current)
        if previous_update is None:
            previous_update = torch.zeros_like(current)

        expected = (batch_size, self.full_state_dim)
        for name, value in (
            ("current_residual", current),
            ("previous_residual", previous_residual),
            ("previous_update", previous_update),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")

        node_input = torch.cat(
            (
                current.reshape(batch_size, self.physics.num_vertices, 3),
                previous_residual.reshape(batch_size, self.physics.num_vertices, 3),
                previous_update.reshape(batch_size, self.physics.num_vertices, 3),
            ),
            dim=-1,
        )
        hidden = self.encoder(node_input)
        for _ in range(self.message_passing_steps):
            messages = self._directed_messages(hidden)
            hidden = hidden + self.node_mlp(torch.cat((hidden, messages), dim=-1))

        # No output scaling.  Fixed vertices are still hard-gated, then the
        # caller re-projects them exactly as in the existing optimizer pipeline.
        raw_delta = self.decoder(hidden)
        gate = self.physics.free_update_gate(batch_size, dtype=raw_delta.dtype)
        return (raw_delta * gate).reshape(batch_size, -1), current


def load_gnn_checkpoint(
    path: Path,
    *,
    physics: Any,
    load_optimizer: bool = False,
) -> tuple[LearnedOptimizerGNN, torch.optim.Optimizer | None, dict[str, Any]]:
    payload = torch.load(path, map_location=physics.device, weights_only=False)
    if payload.get("mesh_sha256") != physics.model.mesh_sha256:
        raise ValueError("checkpoint mesh hash does not match the fixed T-shirt model")
    model_type = payload.get("model_type")
    if model_type != "shared_message_passing_gnn":
        raise ValueError(f"checkpoint is not a supported GNN model: {model_type!r}")
    model = LearnedOptimizerGNN(
        physics=physics,
        model_spec=GNNModelSpec(**payload["model_spec"]),
        initialize=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    optimizer = None
    if load_optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return model, optimizer, payload
