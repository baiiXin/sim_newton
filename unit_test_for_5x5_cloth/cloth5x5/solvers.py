from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .physics import apply_gradient_descent_update, apply_newton_update

if TYPE_CHECKING:
    from .config import PhysicalConfig
    from .model import MLPOptimizer

SolverName = str


def apply_solver_step(
    solver: SolverName,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: "PhysicalConfig",
    *,
    model: "MLPOptimizer | None" = None,
    gd_step_size: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if solver == "learned":
        if model is None:
            raise ValueError("model is required for learned solver")
        from .model import apply_model_update

        return apply_model_update(model, y, q, masses, physical)
    if solver == "gradient_descent":
        if gd_step_size is None:
            raise ValueError("gd_step_size is required for gradient_descent solver")
        return apply_gradient_descent_update(y, q, masses, physical, gd_step_size)
    if solver == "full_newton":
        return apply_newton_update(y, q, masses, physical)
    raise ValueError(f"Unknown solver: {solver}")


def run_solver_steps(
    solver: SolverName,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: "PhysicalConfig",
    steps: int,
    *,
    model: "MLPOptimizer | None" = None,
    gd_step_size: float | None = None,
    require_finite: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    last_delta = torch.zeros_like(y)
    for step in range(steps):
        y, last_delta = apply_solver_step(
            solver,
            y,
            q,
            masses,
            physical,
            model=model,
            gd_step_size=gd_step_size,
        )
        if require_finite and not bool(torch.isfinite(y).all()):
            raise RuntimeError(f"{solver} produced non-finite state at inner iteration {step + 1}")
    return y, last_delta
