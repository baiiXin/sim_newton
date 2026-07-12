from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json
import numpy as np
from scenario_templates import *
from scenario_geometry import build_initial_state, build_triangular_edges, dirichlet_targets, is_compatible
from scenario_builder import _pair_set, legal_pair_universe


def _count_by(scenarios: Sequence[ScenarioSpec], field_name: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for scenario in scenarios:
        key = str(getattr(scenario, field_name))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def pairwise_coverage(scenarios: Sequence[ScenarioSpec]) -> dict[str, Any]:
    universe = legal_pair_universe()
    covered: set[tuple[str, str, str, str]] = set()
    for scenario in scenarios:
        if all(getattr(scenario, axis) in TRAIN_DOMAINS[axis] for axis in CORE_AXES):
            covered.update(_pair_set(scenario))
    covered_legal = covered & set(universe)
    missing = sorted(universe - covered_legal)
    return {
        'covered': len(covered_legal),
        'total': len(universe),
        'ratio': len(covered_legal) / max(len(universe), 1),
        'missing': [list(item) for item in missing],
    }


def _geometry_audit(scenarios: Sequence[ScenarioSpec], rows: int=15, cols: int=15) -> dict[str, Any]:
    edges = np.asarray(build_triangular_edges(rows, cols), dtype=np.int64)
    min_length = float('inf')
    max_length = 0.0
    nonfinite = 0
    bad_t0_targets = 0
    bad_fixed_indices = 0
    for scenario in scenarios:
        state = build_initial_state(scenario, rows=rows, cols=cols)
        positions = state['positions']
        velocities = state['velocities']
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            nonfinite += 1
        lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=-1)
        min_length = min(min_length, float(np.min(lengths)))
        max_length = max(max_length, float(np.max(lengths)))
        fixed_indices = state['fixed_indices']
        if len(set(fixed_indices)) != len(fixed_indices) or any(index < 0 or index >= rows * cols for index in fixed_indices):
            bad_fixed_indices += 1
        if fixed_indices:
            targets, _ = dirichlet_targets(scenario, positions, t=0.0, fixed_indices=fixed_indices)
            if not np.allclose(targets, positions[np.asarray(fixed_indices)], atol=1e-12, rtol=0.0):
                bad_t0_targets += 1
    return {
        'grid': [rows, cols],
        'minimum_initial_edge_length': min_length,
        'maximum_initial_edge_length': max_length,
        'nonfinite_scenarios': nonfinite,
        'bad_t0_dirichlet_targets': bad_t0_targets,
        'bad_fixed_indices': bad_fixed_indices,
    }


def audit_catalogues(catalogues: Mapping[str, Sequence[ScenarioSpec]]) -> dict[str, Any]:
    expected_counts = {
        'train_c1_1024': 1024,
        'train_c2_2048': 2048,
        'train_c3_3072': 3072,
        'validation_128': 128,
        'test_256': 256,
    }
    audit: dict[str, Any] = {
        'expected_counts': expected_counts,
        'catalogues': {},
        'nested_training_prefixes': False,
        'cross_split_duplicate_signatures': [],
    }
    all_signature_owner: dict[tuple[str, ...], str] = {}
    duplicates: list[dict[str, Any]] = []
    for name, scenarios in catalogues.items():
        signatures = [scenario.signature() for scenario in scenarios]
        local_duplicate_count = len(signatures) - len(set(signatures))
        incompatible = [
            scenario.scenario_id
            for scenario in scenarios
            if not is_compatible(scenario.boundary_id, scenario.dirichlet_id)
        ]
        for scenario, signature in zip(scenarios, signatures):
            owner = all_signature_owner.get(signature)
            if owner is not None and owner != name and not (owner.startswith('train_c') and name.startswith('train_c')):
                duplicates.append({'signature': list(signature), 'first': owner, 'second': name})
            all_signature_owner.setdefault(signature, name)
        audit['catalogues'][name] = {
            'count': len(scenarios),
            'expected_count': expected_counts[name],
            'local_duplicate_count': local_duplicate_count,
            'incompatible_count': len(incompatible),
            'incompatible_scenario_ids': incompatible,
            'group_counts': _count_by(scenarios, 'group'),
            'difficulty_counts': _count_by(scenarios, 'difficulty'),
            'shape_counts': _count_by(scenarios, 'shape_id'),
            'strain_counts': _count_by(scenarios, 'strain_id'),
            'velocity_counts': _count_by(scenarios, 'velocity_id'),
            'boundary_counts': _count_by(scenarios, 'boundary_id'),
            'dirichlet_counts': _count_by(scenarios, 'dirichlet_id'),
            'material_counts': _count_by(scenarios, 'material_id'),
            'orientation_counts': _count_by(scenarios, 'orientation_id'),
            'pairwise_coverage': pairwise_coverage(scenarios),
        }
    c1 = [scenario.signature() for scenario in catalogues['train_c1_1024']]
    c2 = [scenario.signature() for scenario in catalogues['train_c2_2048']]
    c3 = [scenario.signature() for scenario in catalogues['train_c3_3072']]
    audit['nested_training_prefixes'] = c2[:len(c1)] == c1 and c3[:len(c2)] == c2
    audit['cross_split_duplicate_signatures'] = duplicates
    geometry_scenarios = list(catalogues['train_c3_3072']) + list(catalogues['validation_128']) + list(catalogues['test_256'])
    audit['geometry'] = _geometry_audit(geometry_scenarios)
    tests = {
        'counts_match': all(audit['catalogues'][name]['count'] == expected for name, expected in expected_counts.items()),
        'training_prefixes_nested': bool(audit['nested_training_prefixes']),
        'no_local_duplicates': all(item['local_duplicate_count'] == 0 for item in audit['catalogues'].values()),
        'no_cross_split_duplicates': not duplicates,
        'all_boundary_motion_pairs_compatible': all(item['incompatible_count'] == 0 for item in audit['catalogues'].values()),
        'c1_full_pairwise_coverage': audit['catalogues']['train_c1_1024']['pairwise_coverage']['ratio'] == 1.0,
        'geometry_finite': audit['geometry']['nonfinite_scenarios'] == 0,
        'geometry_non_degenerate': audit['geometry']['minimum_initial_edge_length'] > 1e-8,
        'dirichlet_starts_without_position_jump': audit['geometry']['bad_t0_dirichlet_targets'] == 0,
    }
    audit['tests'] = tests
    audit['passed'] = all(tests.values())
    return audit


def template_manifest() -> dict[str, Any]:
    return {
        'shapes': [asdict(item) for item in ALL_SHAPES],
        'strains': [asdict(item) for item in ALL_STRAINS],
        'velocities': [asdict(item) for item in ALL_VELOCITIES],
        'boundaries': [asdict(item) for item in ALL_BOUNDARIES],
        'dirichlet': [asdict(item) for item in ALL_DIRICHLET],
        'materials': [asdict(item) for item in ALL_MATERIALS],
        'orientations': [asdict(item) for item in ALL_ORIENTATIONS],
    }


def save_catalogues(catalogues: Mapping[str, Sequence[ScenarioSpec]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        'schema_version': 1,
        'construction': 'hand-authored templates + deterministic enumeration; no random sampling',
        'training_prefixes': {'C1': 'train_c1_1024', 'C2': 'train_c2_2048', 'C3': 'train_c3_3072'},
        'templates': template_manifest(),
        'files': {},
    }
    fields = tuple(ScenarioSpec.__dataclass_fields__.keys())
    for name, scenarios in catalogues.items():
        json_path = output_dir / f'{name}.json'
        csv_path = output_dir / f'{name}.csv'
        json_path.write_text(json.dumps([asdict(item) for item in scenarios], indent=2, ensure_ascii=False), encoding='utf-8')
        with csv_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for scenario in scenarios:
                row = asdict(scenario)
                row['notes'] = '|'.join(scenario.notes)
                writer.writerow(row)
        manifest['files'][name] = {'count': len(scenarios), 'json': json_path.name, 'csv': csv_path.name}
    audit = audit_catalogues(catalogues)
    (output_dir / 'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    manifest['audit'] = {'file': 'audit.json', 'passed': audit['passed']}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest
