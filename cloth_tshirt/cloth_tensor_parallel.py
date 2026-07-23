"""Shared two-rank tensor-parallel primitives for the full-state T-shirt MLP."""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from cloth03_training_pool import (
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    LearnedOptimizerMLP,
    ModelSpec,
    _activation_gain,
    apply_model_update,
    normalized_one_step_energy_loss,
)


TENSOR_PARALLEL_SIZE = 2
RECOMMENDED_WIDTH = 39_936


def network_dimensions(
    *, num_vertices: int, width: int, tensor_parallel_size: int = TENSOR_PARALLEL_SIZE
) -> dict[str, int | float]:
    """Return global and per-rank sizes for a depth-one, bias-free MLP."""
    full_state_dim = 3 * int(num_vertices)
    input_dim = 3 * full_state_dim
    global_parameters = input_dim * int(width) + int(width) * full_state_dim
    if global_parameters % int(tensor_parallel_size):
        raise ValueError("global parameter count does not split evenly across the mesh")
    return {
        "num_vertices": int(num_vertices),
        "full_state_dim": full_state_dim,
        "input_dim": input_dim,
        "width": int(width),
        "width_to_input_ratio": float(width / input_dim),
        "global_parameter_count": global_parameters,
        "local_parameter_count": global_parameters // int(tensor_parallel_size),
    }


def build_tensor_parallel_model(
    *,
    physics,
    activation: str,
    width: int,
    device_mesh,
    rank: int,
    seed: int,
    residual_length_scale: float = DEFAULT_RESIDUAL_LENGTH_SCALE,
):
    """Materialize only the local shards of the large depth-one MLP.

    A meta-device source prevents either rank from ever allocating the complete
    unsharded model. The hidden dimension is column sharded and the output layer
    consumes that shard with row parallelism.
    """
    if int(width) <= 0 or int(width) % TENSOR_PARALLEL_SIZE:
        raise ValueError(f"width must be positive and divisible by {TENSOR_PARALLEL_SIZE}")
    model_spec = ModelSpec(
        activation=str(activation),
        depth=1,
        width=int(width),
        use_bias=False,
    )
    meta_physics = SimpleNamespace(
        num_vertices=physics.num_vertices,
        dtype=physics.dtype,
        device=torch.device("meta"),
    )
    model = LearnedOptimizerMLP(physics=meta_physics, model_spec=model_spec)
    model = parallelize_module(
        model,
        device_mesh,
        {
            "hidden_layers.0": ColwiseParallel(use_local_output=True),
            "output_layer": RowwiseParallel(use_local_output=True),
        },
        src_data_rank=None,
    )
    model.to_empty(device=physics.device)
    model.physics = physics

    input_dim = 3 * model.full_state_dim
    hidden_std = _activation_gain(activation) / (input_dim ** 0.5)
    torch.manual_seed(int(seed) + int(rank))
    with torch.no_grad():
        torch.nn.init.normal_(model.hidden_layers[0].weight.to_local(), std=hidden_std)
        torch.nn.init.zeros_(model.output_layer.weight.to_local())
        model.residual_length_scale.fill_(float(residual_length_scale))
    return model


def local_parameter_count(model) -> int:
    """Count only locally materialized parameter shards."""
    return sum(
        int(parameter.to_local().numel())
        if isinstance(parameter, DTensor)
        else int(parameter.numel())
        for parameter in model.parameters()
    )


class SynchronizedOnlineTrainingPool:
    """Keep duplicate online pools identical by broadcasting rank-0 batches."""

    def __init__(self, pool) -> None:
        self._pool = pool

    def __getattr__(self, name: str):
        return getattr(self._pool, name)

    def ask(self):
        batch = self._pool.ask()
        for value in vars(batch).values():
            if isinstance(value, torch.Tensor):
                dist.broadcast(value, src=0)
        return batch


def local_gradient_norm_and_clip(
    model, *, max_norm: float, device
) -> tuple[float, float]:
    """Compute and clip the true global norm without gathering large weights."""
    local_squared = torch.zeros((), dtype=torch.float64, device=device)
    local_gradients: list[torch.Tensor] = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        local = gradient.to_local() if isinstance(gradient, DTensor) else gradient
        local = local.detach()
        local_gradients.append(local)
        # Never cast a multi-GiB shard to float64: reduce its float32 norm to a
        # scalar first, then accumulate that scalar accurately.
        local_norm = torch.linalg.vector_norm(local)
        local_squared.add_(local_norm.to(torch.float64).square())
    dist.all_reduce(local_squared, op=dist.ReduceOp.SUM)
    before = float(torch.sqrt(local_squared).item())
    scale = 1.0
    if max_norm > 0.0 and before > max_norm:
        scale = float(max_norm) / (before + 1e-12)
        with torch.no_grad():
            for gradient in local_gradients:
                gradient.mul_(scale)
    return before, before * scale


def tensor_parallel_training_step(
    *,
    model,
    optimizer,
    pool: SynchronizedOnlineTrainingPool,
    gradient_clip_norm: float,
    step_regularization_weight: float = 0.0,
) -> dict[str, Any]:
    """Run one complete synchronized online-training update on both ranks."""
    model.train()
    batch = pool.ask()
    optimizer.zero_grad(set_to_none=True)
    residual_before = pool.physics.stationarity_residual_norm(
        batch.y, batch.q, batch.fixed_targets
    )
    y_next, delta, current = apply_model_update(
        model,
        batch.y,
        batch.q,
        batch.fixed_targets,
        previous_residual=batch.previous_residual,
        previous_update=batch.previous_update,
    )
    loss_result = normalized_one_step_energy_loss(
        physics=pool.physics,
        y_before=batch.y,
        y_after=y_next,
        q=batch.q,
        fixed_targets=batch.fixed_targets,
        delta=delta,
        step_regularization_weight=step_regularization_weight,
    )
    if not bool(torch.isfinite(loss_result.loss).item()):
        raise FloatingPointError("non-finite tensor-parallel training loss")
    loss_result.loss.backward()
    grad_before, grad_after = local_gradient_norm_and_clip(
        model,
        max_norm=gradient_clip_norm,
        device=pool.physics.device,
    )
    if not math.isfinite(grad_before) or not math.isfinite(grad_after):
        raise FloatingPointError("non-finite tensor-parallel gradient norm")
    optimizer.step()
    residual_after = pool.physics.stationarity_residual_norm(
        y_next.detach(), batch.q, batch.fixed_targets
    )
    pool_stats = pool.tell(
        batch,
        y_next=y_next,
        delta=delta,
        current_residual=current,
        energy_after=loss_result.energy_after,
        residual_after=residual_after,
    )
    eps = torch.finfo(residual_after.dtype).eps
    ratio = residual_after / (residual_before + eps)
    update_norm = torch.linalg.vector_norm(delta.detach(), dim=-1)
    metrics: dict[str, Any] = {
        "loss": float(loss_result.loss.detach().cpu()),
        "normalized_energy_change_mean": float(
            loss_result.normalized_change.mean().detach().cpu()
        ),
        "normalized_energy_change_p95": float(
            torch.quantile(loss_result.normalized_change.detach(), 0.95).cpu()
        ),
        "energy_increase_fraction": float(
            (loss_result.normalized_change.detach() > 0).double().mean().cpu()
        ),
        "residual_before_mean": float(residual_before.mean().detach().cpu()),
        "residual_after_mean": float(residual_after.mean().detach().cpu()),
        "residual_ratio_p50": float(torch.quantile(ratio.detach(), 0.50).cpu()),
        "residual_ratio_p95": float(torch.quantile(ratio.detach(), 0.95).cpu()),
        "update_norm_mean": float(update_norm.mean().cpu()),
        "update_norm_p95": float(torch.quantile(update_norm, 0.95).cpu()),
        "gradient_norm_before_clip": grad_before,
        "gradient_norm_after_clip": grad_after,
        "batch_size": int(pool.batch_size),
    }
    metrics.update(pool_stats)
    return metrics


def assert_replicated_scalars(
    values: Mapping[str, float | int], *, relative_tolerance: float = 1e-6
) -> None:
    """Fail early if duplicate-rank state has silently diverged."""
    gathered: list[dict[str, float | int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(values))
    reference = gathered[0]
    if reference is None:
        raise RuntimeError("rank 0 did not contribute synchronization diagnostics")
    for rank, candidate in enumerate(gathered[1:], start=1):
        if candidate is None or candidate.keys() != reference.keys():
            raise RuntimeError(f"rank {rank} produced different diagnostic fields")
        for key, expected in reference.items():
            actual = candidate[key]
            if isinstance(expected, int) and isinstance(actual, int):
                equal = actual == expected
            else:
                expected_float = float(expected)
                actual_float = float(actual)
                scale = max(abs(expected_float), abs(actual_float), 1.0)
                equal = abs(expected_float - actual_float) <= relative_tolerance * scale
            if not equal:
                raise RuntimeError(
                    f"tensor-parallel ranks diverged for {key}: "
                    f"rank0={expected!r}, rank{rank}={actual!r}"
                )
