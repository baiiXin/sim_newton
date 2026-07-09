"""
Fixed-left-edge 5x5 triangular-cloth learned optimizer:
independent multi-motion and multi-time-step generalization experiment.

Confirmed experiment
--------------------
1. Keep the 5x5 triangular spring mesh with the left-top and left-bottom
   corner vertices fixed. The 23 free vertices form a 69-dimensional state.
2. Build 32 complete motions and split by COMPLETE MOTION: 16 train,
   4 validation, 4 in-domain test, and 8 out-of-domain test motions.
3. Generate 100 high-accuracy physical steps for every motion, while treating
   each physical time step as an independent optimization problem during
   learned-optimizer training and fixed-state evaluation.
4. Train on 16 stratified time steps from each training motion, with 32
   optimization starts per time step (8192 full-batch training states).
5. Train an equal-sample-budget single-motion baseline.
6. Use float64, Adam(lr=1e-3), K=1->5 over 500 epochs, validation-selected
   checkpoints, validation-selected raw-gradient-descent step size, and full
   Newton as a numerical baseline.
7. Report both p95 and sample maximum metrics. Also aggregate by complete
   motion and report the worst-motion p95 and worst-motion maximum, so boundary
   capability is visible rather than hidden by pooled averages.
8. The continuous-rollout script uses the hardest OOD case selected by initial
   residual maximum. Reference non-convergence is recorded as a warning and
   does not stop the rollout as long as a finite fallback state exists.

Exact/reference solutions are never network inputs or supervised training
labels. Training uses only the physical variational-energy objective.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from cloth5x5 import *  # noqa: F403
from cloth5x5.constants import *  # noqa: F403
from cloth5x5.experiment_main import main, parse_args, validate_args

if __name__ == "__main__":
    main(script_file=Path(__file__))
