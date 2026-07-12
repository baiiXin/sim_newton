"""Official training entry with non-finite-safe checkpoint ordering.

The main trainer stores failed trajectories with an infinite selection residual. Linear
quantiles can become NaN when finite and infinite values are interpolated. This wrapper
keeps the frozen lexicographic policy, but maps non-finite tie-break metrics to their
worst values before the trainer compares checkpoint ranks.
"""
from __future__ import annotations

import math
from typing import Any

import cloth05_train_scale_up as trainer


def robust_checkpoint_rank(
    summary: dict[str, Any],
) -> tuple[float, float, float, float]:
    failed = float(summary["failed_motion_count"])
    survival = float(summary["survival_frame_p05"])
    residual = float(summary["residual_ratio_p95"])
    energy = float(summary["energy_increase_fraction"])
    if not math.isfinite(failed):
        failed = float("inf")
    if not math.isfinite(survival):
        survival = 0.0
    if not math.isfinite(residual):
        residual = float("inf")
    if not math.isfinite(energy):
        energy = float("inf")
    return failed, -survival, residual, energy


def main() -> None:
    trainer.checkpoint_rank = robust_checkpoint_rank
    trainer.main()


if __name__ == "__main__":
    main()
