"""
500-frame continuous rollout for the fixed-left-edge 5x5 triangular cloth
trained on multiple complete motions.

The script loads the validation-selected multi-problem MLP checkpoint produced
by ``fixed_left_edge_5x5_cloth_multi_motion_train_compare.py``. Starting from the
solver-independent hard extrapolation case selected by the training script, it
propagates four independent physical trajectories:

    1. high-accuracy damped-Newton reference,
    2. MLP learned optimizer,
    3. validation-selected fixed-step gradient descent,
    4. undamped full Newton.

MLP, gradient descent, and Newton each execute exactly 50 inner iterations per
physical frame. There is no convergence-based early stopping. Each method uses
its own predicted position and velocity to construct the next frame, so errors
are allowed to accumulate naturally.

The visualization uses cloth grid rendering. The MLP-reference and
GD-reference comparison panels render the predicted triangular cloth with a
per-triangle temperature map of vertex position error, while a translucent
reference wireframe is overlaid for context.
"""

from __future__ import annotations

from pathlib import Path

import fixed_left_edge_5x5_cloth_multi_motion_train_compare as core
from cloth5x5.rollout_main import main

__all__ = ["core", "main"]

if __name__ == "__main__":
    main(script_file=Path(__file__))
