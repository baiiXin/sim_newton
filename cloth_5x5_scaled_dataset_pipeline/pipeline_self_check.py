#!/usr/bin/env python3
"""Fast physics/model invariants check; it does not build the formal datasets."""

from __future__ import annotations

import torch

from cloth_scale_common import (
    LearnedOptimizerState,
    MODEL_INPUT_DIM,
    MODEL_INPUT_SIGNATURE,
    MLPOptimizer,
    ModelSpec,
    apply_model_update,
    boundary_mask,
    boundary_target,
    build_boundary_catalogue,
    build_motion_catalogue,
    default_physical_config,
    make_q,
    masked_stationarity_residual,
    reshape_state,
    solve_reference_solution,
)


def main() -> None:
    physical = default_physical_config()
    boundary = build_boundary_catalogue()[0]
    motion = build_motion_catalogue(physical)[0]
    fixed_mask = boundary_mask(boundary)
    fixed_target = boundary_target(boundary, physical)
    p = torch.tensor(motion.positions, dtype=torch.float64)
    v = torch.tensor(motion.velocities, dtype=torch.float64)
    p[fixed_mask] = reshape_state(fixed_target)[fixed_mask]
    v[fixed_mask] = 0.0
    q = make_q(p, v, physical)
    masses = torch.tensor(physical.masses, dtype=torch.float64)
    current = p.reshape(-1)

    exact, info = solve_reference_solution(
        q=q,
        masses=masses,
        initial_y=current,
        fixed_mask=fixed_mask,
        fixed_target=fixed_target,
        physical=physical,
    )
    assert exact.shape == (75,)
    assert torch.equal(reshape_state(exact)[fixed_mask], reshape_state(fixed_target)[fixed_mask])
    residual = masked_stationarity_residual(
        exact.unsqueeze(0), q.unsqueeze(0), masses.unsqueeze(0), fixed_mask.unsqueeze(0), physical
    )
    assert torch.all(residual.reshape(1, 25, 3)[:, fixed_mask, :] == 0)

    assert MODEL_INPUT_DIM == 225
    assert "no_fixed_onehot" in MODEL_INPUT_SIGNATURE
    model = MLPOptimizer(0.05, ModelSpec("relu", 2, 32, False))
    state = LearnedOptimizerState.zeros_like(current.unsqueeze(0))
    next_y, delta, _ = apply_model_update(
        model,
        current.unsqueeze(0),
        q.unsqueeze(0),
        masses.unsqueeze(0),
        fixed_mask.unsqueeze(0),
        fixed_target.unsqueeze(0),
        physical,
        state,
    )
    assert next_y.shape == (1, 75)
    assert delta.shape == (1, 75)
    assert torch.all(delta.reshape(1, 25, 3)[:, fixed_mask, :] == 0)
    print("Self-check passed.")
    print(f"Reference residual: {info['residual_norm']:.6e}")
    print(f"Model input dimension: {MODEL_INPUT_DIM}")
    print(f"Model parameters: {model.parameter_count:,}")


if __name__ == "__main__":
    main()
