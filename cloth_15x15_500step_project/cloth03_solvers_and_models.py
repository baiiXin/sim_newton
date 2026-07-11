"""15x15 compatibility layer for the 5x5 learned-optimizer physics module.

The original project already keeps almost all topology and state dimensions in
module globals. This module loads that implementation, replaces grid-dependent
globals before experiment code runs, extends the activation set, and re-exports
the resulting API.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch.nn as nn

_BASE_PATH = (
    Path(__file__).resolve().parent.parent
    / "cloth_5x5_500step_project"
    / "cloth03_solvers_and_models.py"
)
if not _BASE_PATH.exists():
    raise FileNotFoundError(f"Missing shared 5x5 physics module: {_BASE_PATH}")

_MODULE_NAME = "_cloth_5x5_shared_solvers_and_models"
_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _BASE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_BASE_PATH}")
_base = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = _base
_spec.loader.exec_module(_base)

_base.GRID_ROWS = 15
_base.GRID_COLS = 15
_base.SPATIAL_DIM = 3
_base.NUM_PARTICLES = _base.GRID_ROWS * _base.GRID_COLS
_base.FIXED_VERTEX_INDICES = (0, (_base.GRID_ROWS - 1) * _base.GRID_COLS)
_fixed = set(_base.FIXED_VERTEX_INDICES)
_base.FREE_VERTEX_INDICES = tuple(
    index for index in range(_base.NUM_PARTICLES) if index not in _fixed
)
_base.NUM_FREE_PARTICLES = len(_base.FREE_VERTEX_INDICES)
_base.FREE_STATE_DIM = _base.NUM_FREE_PARTICLES * _base.SPATIAL_DIM
_base.FULL_STATE_DIM = _base.NUM_PARTICLES * _base.SPATIAL_DIM
_base.SPRING_EDGES, _base.TRIANGLE_FACES = _base.build_triangular_cloth_topology()
_base.NUM_SPRINGS = len(_base.SPRING_EDGES)
_base.NUM_TRIANGLES = len(_base.TRIANGLE_FACES)
_base.GLOBAL_TO_FREE_INDEX = tuple(
    _base.FREE_VERTEX_INDICES.index(index)
    if index in _base.FREE_VERTEX_INDICES
    else -1
    for index in range(_base.NUM_PARTICLES)
)
_base.MODEL_INPUT_DIM = (3 if _base.USE_HISTORY_INPUT else 1) * _base.FULL_STATE_DIM
_base.DEFAULT_EVALUATION_STEPS = 50
_base.DEFAULT_EVALUATION_BATCH_SIZE = 512
_base.ACTIVATION_NAMES = ("identity", "relu", "gelu", "silu", "tanh")


def _make_activation(name: str) -> nn.Module:
    if name == "identity":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _activation_gain(name: str) -> float:
    if name == "identity":
        return 1.0
    if name in {"relu", "gelu", "silu"}:
        return math.sqrt(2.0)
    if name == "tanh":
        return 5.0 / 3.0
    raise ValueError(f"Unsupported activation: {name}")


_base.make_activation = _make_activation
_base.activation_gain = _activation_gain

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

PROJECT_GRID = (15, 15)
PROJECT_STATE_DESCRIPTION = (
    "225 vertices; 675D full state; 223 free vertices; 669D reduced state"
)
