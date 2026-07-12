"""Frozen validation contracts for the scale-up training-pool project.

Both validation modes are continuous free rollouts and both must save raw
per-motion records, aggregate curves, and plots.  Only the checkpoint protocol
is allowed to select ``best_validation_model.pt``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationProtocol:
    id: str
    motion_count: int
    rollout_frames: int
    inner_steps: int
    interval_updates: int
    selects_checkpoint: bool
    save_per_motion: bool = True
    save_aggregate_curves: bool = True
    render_plots: bool = True


FAST_MONITOR = ValidationProtocol(
    id="fast_monitor",
    motion_count=32,
    rollout_frames=32,
    inner_steps=10,
    interval_updates=2_000,
    selects_checkpoint=False,
)

CHECKPOINT_VALIDATION = ValidationProtocol(
    id="checkpoint_validation",
    motion_count=128,
    rollout_frames=100,
    inner_steps=10,
    interval_updates=10_000,
    selects_checkpoint=True,
)

VALIDATION_PROTOCOLS = (FAST_MONITOR, CHECKPOINT_VALIDATION)

# Stability-first lexicographic checkpoint ordering.
CHECKPOINT_SELECTION_FIELDS = (
    ("failed_motion_count", "min"),
    ("survival_frame_p05", "max"),
    ("residual_ratio_p95", "min"),
    ("energy_increase_fraction", "min"),
)

# Post-training diagnostic evaluation.  It does not select checkpoints.
FINAL_VALIDATION_ROLLOUT_FRAMES = 500
FINAL_VALIDATION_INNER_STEPS = (1, 3, 10, 30)

VALIDATION_OUTPUTS = {
    protocol.id: {
        "history": f"validation/{protocol.id}/history.csv",
        "per_motion": f"validation/{protocol.id}/per_motion.csv",
        "curves": f"validation/{protocol.id}/curves.pt",
        "figures": f"validation/{protocol.id}/figures",
    }
    for protocol in VALIDATION_PROTOCOLS
}

if sum(protocol.selects_checkpoint for protocol in VALIDATION_PROTOCOLS) != 1:
    raise AssertionError("Exactly one validation protocol must select checkpoints")
if not all(
    protocol.save_per_motion and protocol.save_aggregate_curves and protocol.render_plots
    for protocol in VALIDATION_PROTOCOLS
):
    raise AssertionError("Both validation modes must retain data and plots")
