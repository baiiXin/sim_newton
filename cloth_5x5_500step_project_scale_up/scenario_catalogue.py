"""Public facade for deterministic cloth scenario catalogues.

Project policy: every scenario must contain at least one Dirichlet/fixed vertex.
The low-level template module keeps construction primitives, while this facade
removes the experimental ``no_fixed`` OOD template before catalogue builders,
geometry helpers, and audits are imported.
"""
from __future__ import annotations

import scenario_templates as _templates

_templates.OOD_BOUNDARIES = tuple(
    item for item in _templates.OOD_BOUNDARIES if item.id != "no_fixed"
)
_templates.ALL_BOUNDARIES = _templates.TRAIN_BOUNDARIES + _templates.OOD_BOUNDARIES
_templates.BOUNDARY_BY_ID = _templates._index_by_id(_templates.ALL_BOUNDARIES)

if any(item.family == "none" or item.selector == "none" for item in _templates.ALL_BOUNDARIES):
    raise AssertionError("Unpinned cloth scenarios are disabled in the scale-up project")

from scenario_templates import *
from scenario_geometry import *
from scenario_builder import *
from scenario_audit import *
