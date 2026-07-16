"""Frozen validation contracts for online-randomized T-shirt training."""
from __future__ import annotations

from dataclasses import dataclass

from tshirt_config import DEFAULT_EVALUATION


@dataclass(frozen=True)
class ValidationProtocol:
    id: str
    motion_count: int
    rollout_frames: int
    inner_steps: int
    interval_updates: int
    selects_checkpoint: bool
    residual_ratio_tolerance: float = DEFAULT_EVALUATION.convergence_residual_ratio
    absolute_residual_tolerance: float = 1e-10
    single_step_ratio_threshold: float = DEFAULT_EVALUATION.two_order_single_step_ratio
    early_stop: bool = False


# Fast monitoring is intentionally short, but uses all 32 frozen validation
# initial states.  It never decides which checkpoint is best.
FAST_MONITOR = ValidationProtocol(
    id="fast_monitor_k15",
    motion_count=DEFAULT_EVALUATION.validation_count,
    rollout_frames=32,
    inner_steps=DEFAULT_EVALUATION.quick_inner_steps,
    interval_updates=10_000,
    selects_checkpoint=False,
    early_stop=False,
)


# The checkpoint protocol uses the same initial states and longer free rollouts.
# Its 50-step cap is also the default used by test and single-motion evaluation.
CHECKPOINT_VALIDATION = ValidationProtocol(
    id="checkpoint_validation_k50",
    motion_count=DEFAULT_EVALUATION.validation_count,
    rollout_frames=100,
    inner_steps=DEFAULT_EVALUATION.full_inner_steps,
    interval_updates=50_000,
    selects_checkpoint=True,
    early_stop=False,
)


VALIDATION_PROTOCOLS = (FAST_MONITOR, CHECKPOINT_VALIDATION)

# Stability-first lexicographic checkpoint ordering.  Lower tuples are better.
CHECKPOINT_SELECTION_FIELDS = (
    ("failed_motion_count", "min"),
    ("survival_frame_p05", "max"),
    ("residual_ratio_p95", "min"),
    ("single_step_le_two_orders_frame_count", "min"),
    ("energy_increase_fraction", "min"),
)


def checkpoint_rank(summary: dict[str, float | int]) -> tuple[float, ...]:
    return (
        float(summary["failed_motion_count"]),
        -float(summary["survival_frame_p05"]),
        float(summary["residual_ratio_p95"]),
        float(summary["single_step_le_two_orders_frame_count"]),
        float(summary["energy_increase_fraction"]),
    )


if sum(protocol.selects_checkpoint for protocol in VALIDATION_PROTOCOLS) != 1:
    raise AssertionError("Exactly one validation protocol must select checkpoints")
