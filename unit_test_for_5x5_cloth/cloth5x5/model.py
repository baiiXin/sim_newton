from __future__ import annotations

import torch
import torch.nn as nn

from .config import PhysicalConfig
from .constants import FREE_STATE_DIM, HIDDEN_DIM, TORCH_DTYPE
from .physics import stationarity_residual


def mass_preconditioned_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    residual = stationarity_residual(y, q, masses, physical)
    mass_per_coordinate = masses.repeat_interleave(3, dim=-1)
    return physical.dt**2 * residual / mass_per_coordinate


class MLPOptimizer(nn.Module):
    def __init__(self, residual_length_scale: float) -> None:
        super().__init__()
        if residual_length_scale <= 0.0:
            raise ValueError("residual_length_scale must be positive")
        self.linear1 = nn.Linear(FREE_STATE_DIM, HIDDEN_DIM, bias=False)
        self.activation = nn.Identity()
        self.linear2 = nn.Linear(HIDDEN_DIM, FREE_STATE_DIM, bias=False)
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.zeros_(self.linear2.weight)
        self.register_buffer(
            "residual_length_scale",
            torch.tensor(float(residual_length_scale), dtype=TORCH_DTYPE),
        )

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        *,
        physical: PhysicalConfig,
    ) -> torch.Tensor:
        u = mass_preconditioned_residual(y, q, masses, physical)
        u = u / self.residual_length_scale
        return self.residual_length_scale * self.linear2(
            self.activation(self.linear1(u))
        )


def apply_model_update(
    model: MLPOptimizer,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = model(y, q, masses, physical=physical)
    return y + delta, delta


def physical_energy_scale(
    masses: torch.Tensor,
    physical: PhysicalConfig,
    residual_length_scale: float,
) -> float:
    return float(masses.mean().item()) * residual_length_scale**2 / physical.dt**2
