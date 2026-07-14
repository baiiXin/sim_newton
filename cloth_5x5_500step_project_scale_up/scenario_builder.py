from __future__ import annotations
from dataclasses import asdict
from itertools import combinations, product
from math import gcd
from typing import Iterator, Mapping, Sequence
from scenario_templates import *
from scenario_geometry import is_compatible


def _scenario_from_values(values: Mapping[str, str], *, scenario_id: int, split: str, group: str, difficulty: str, orientation_id: str='horizontal', notes: Sequence[str]=()) -> ScenarioSpec | None:
    if not is_compatible(values['boundary_id'], values['dirichlet_id']):
        return None
    return ScenarioSpec(scenario_id=scenario_id, split=split, group=group, difficulty=difficulty, shape_id=values['shape_id'], strain_id=values['strain_id'], velocity_id=values['velocity_id'], boundary_id=values['boundary_id'], dirichlet_id=values['dirichlet_id'], material_id=values['material_id'], orientation_id=orientation_id, notes=tuple(notes))


def _decode_rank(rank: int, domains: Mapping[str, Sequence[str]]) -> dict[str, str]:
    values: dict[str, str] = {}
    remainder = int(rank)
    for axis in reversed(CORE_AXES):
        domain = domains[axis]
        remainder, position = divmod(remainder, len(domain))
        values[axis] = domain[position]
    return {axis: values[axis] for axis in CORE_AXES}


def _coprime_step(total: int, preferred: int=65537) -> int:
    step = preferred % total
    if step == 0:
        step = 1
    while gcd(step, total) != 1:
        step += 1
    return step


def deterministic_value_stream(domains: Mapping[str, Sequence[str]]=TRAIN_DOMAINS, *, offset: int=0) -> Iterator[dict[str, str]]:
    total = 1
    for axis in CORE_AXES:
        total *= len(domains[axis])
    step = _coprime_step(total)
    for n in range(total):
        rank = (int(offset) + n * step) % total
        values = _decode_rank(rank, domains)
        if is_compatible(values['boundary_id'], values['dirichlet_id']):
            yield values


def _pair_set(scenario: ScenarioSpec) -> frozenset[tuple[str, str, str, str]]:
    values = {axis: getattr(scenario, axis) for axis in CORE_AXES}
    return frozenset((axis_a, values[axis_a], axis_b, values[axis_b]) for axis_a, axis_b in combinations(CORE_AXES, 2))


def legal_pair_universe() -> frozenset[tuple[str, str, str, str]]:
    pairs: set[tuple[str, str, str, str]] = set()
    for axis_a, axis_b in combinations(CORE_AXES, 2):
        for value_a, value_b in product(TRAIN_DOMAINS[axis_a], TRAIN_DOMAINS[axis_b]):
            if {axis_a, axis_b} == {'boundary_id', 'dirichlet_id'}:
                boundary_id = value_a if axis_a == 'boundary_id' else value_b
                dirichlet_id = value_a if axis_a == 'dirichlet_id' else value_b
                if not is_compatible(boundary_id, dirichlet_id):
                    continue
            pairs.add((axis_a, value_a, axis_b, value_b))
    return frozenset(pairs)


def _stress_score(values: Mapping[str, str]) -> int:
    score = 0
    if values['shape_id'] != BASELINE_VALUES['shape_id']:
        score += 1
    if values['strain_id'] != BASELINE_VALUES['strain_id']:
        score += 1
    velocity = VELOCITY_BY_ID[values['velocity_id']]
    if velocity.id != 'velocity_zero':
        score += 1
        if velocity.magnitude_level in {'high', 'rotation', 'extreme', 'extreme_rotation'}:
            score += 1
    if values['boundary_id'] != BASELINE_VALUES['boundary_id']:
        score += 1
    if values['dirichlet_id'] != 'static':
        score += 2
    if values['material_id'] != 'material_baseline':
        score += 1
    return score


def _append_unique(output: list[ScenarioSpec], signatures: set[tuple[str, ...]], scenario: ScenarioSpec | None) -> bool:
    if scenario is None or scenario.signature() in signatures:
        return False
    output.append(scenario)
    signatures.add(scenario.signature())
    return True


def _build_anchor_block() -> list[ScenarioSpec]:
    output: list[ScenarioSpec] = []
    signatures: set[tuple[str, ...]] = set()

    def add(values: Mapping[str, str], note: str) -> None:
        scenario = _scenario_from_values(values, scenario_id=len(output), split='train', group='anchors', difficulty='basic', notes=(note,))
        _append_unique(output, signatures, scenario)

    add(dict(BASELINE_VALUES), 'baseline')
    for axis in CORE_AXES:
        for value in TRAIN_DOMAINS[axis]:
            values = dict(BASELINE_VALUES)
            values[axis] = value
            if not is_compatible(values['boundary_id'], values['dirichlet_id']):
                if axis == 'dirichlet_id' and DIRICHLET_BY_ID[value].kind == 'twist':
                    values['boundary_id'] = 'pair_diagonal_main'
                else:
                    continue
            add(values, f'single_axis:{axis}')
    for shape_id, strain_id in product(TRAIN_DOMAINS['shape_id'], TRAIN_DOMAINS['strain_id']):
        values = dict(BASELINE_VALUES)
        values['shape_id'] = shape_id
        values['strain_id'] = strain_id
        add(values, 'shape_x_strain_anchor')
    for boundary_id, dirichlet_id in product(TRAIN_DOMAINS['boundary_id'], TRAIN_DOMAINS['dirichlet_id']):
        if not is_compatible(boundary_id, dirichlet_id):
            continue
        values = dict(BASELINE_VALUES)
        values['boundary_id'] = boundary_id
        values['dirichlet_id'] = dirichlet_id
        add(values, 'boundary_x_dirichlet_anchor')
        if len(output) >= 224:
            break
    stream = deterministic_value_stream(offset=17)
    while len(output) < 256:
        values = next(stream)
        scenario = _scenario_from_values(values, scenario_id=len(output), split='train', group='anchors', difficulty='basic', notes=('deterministic_anchor_fill',))
        _append_unique(output, signatures, scenario)
    return output


def _pair_seed_candidates(start_id: int=0) -> list[ScenarioSpec]:
    candidates: list[ScenarioSpec] = []
    signatures: set[tuple[str, ...]] = set()
    counter = 0
    for axis_a, axis_b in combinations(CORE_AXES, 2):
        for value_a, value_b in product(TRAIN_DOMAINS[axis_a], TRAIN_DOMAINS[axis_b]):
            if {axis_a, axis_b} == {'boundary_id', 'dirichlet_id'}:
                boundary_id = value_a if axis_a == 'boundary_id' else value_b
                dirichlet_id = value_a if axis_a == 'dirichlet_id' else value_b
                if not is_compatible(boundary_id, dirichlet_id):
                    continue
            stream = deterministic_value_stream(offset=counter * 97 + 31)
            counter += 1
            for _ in range(64):
                values = next(stream)
                values[axis_a] = value_a
                values[axis_b] = value_b
                scenario = _scenario_from_values(values, scenario_id=start_id + len(candidates), split='train', group='pairwise', difficulty='coverage', notes=(f'pair_seed:{axis_a}x{axis_b}',))
                if _append_unique(candidates, signatures, scenario):
                    break
            else:
                raise RuntimeError(f'Could not complete legal pair {axis_a}={value_a}, {axis_b}={value_b}')
    stream = deterministic_value_stream(offset=7919)
    while len(candidates) < 12000:
        values = next(stream)
        scenario = _scenario_from_values(values, scenario_id=start_id + len(candidates), split='train', group='pairwise', difficulty='coverage', notes=('pairwise_candidate_fill',))
        _append_unique(candidates, signatures, scenario)
    return candidates


def _build_pairwise_block(anchors: Sequence[ScenarioSpec], count: int=768) -> list[ScenarioSpec]:
    universe = legal_pair_universe()
    covered: set[tuple[str, str, str, str]] = set()
    for scenario in anchors:
        covered.update(_pair_set(scenario))
    uncovered = set(universe - covered)
    candidates = _pair_seed_candidates(start_id=len(anchors))
    candidate_pairs = [_pair_set(candidate) for candidate in candidates]
    pair_to_candidates: dict[tuple[str, str, str, str], list[int]] = {}
    for candidate_index, pairs in enumerate(candidate_pairs):
        for pair in pairs:
            pair_to_candidates.setdefault(pair, []).append(candidate_index)
    selected_indices: set[int] = set()
    selected: list[ScenarioSpec] = []
    while uncovered and len(selected) < count:
        target_pair = min(uncovered, key=lambda pair: (len(pair_to_candidates.get(pair, ())), pair))
        options = pair_to_candidates.get(target_pair, ())
        best_index: int | None = None
        best_gain = -1
        for candidate_index in options:
            if candidate_index in selected_indices:
                continue
            gain = sum(pair in uncovered for pair in candidate_pairs[candidate_index])
            if gain > best_gain:
                best_gain = gain
                best_index = candidate_index
        if best_index is None:
            raise RuntimeError(f'Pairwise coverage stalled at {target_pair}')
        selected_indices.add(best_index)
        source = candidates[best_index]
        selected.append(ScenarioSpec(**{**asdict(source), 'scenario_id': len(anchors) + len(selected)}))
        uncovered.difference_update(candidate_pairs[best_index])
    if uncovered:
        raise RuntimeError(f'Pairwise block failed to cover {len(uncovered)} legal pairs')
    existing = {scenario.signature() for scenario in anchors}
    existing.update(scenario.signature() for scenario in selected)
    for candidate_index, source in enumerate(candidates):
        if len(selected) >= count:
            break
        if candidate_index in selected_indices or source.signature() in existing:
            continue
        selected.append(ScenarioSpec(**{**asdict(source), 'scenario_id': len(anchors) + len(selected)}))
        existing.add(source.signature())
    if len(selected) != count:
        raise RuntimeError(f'Expected {count} pairwise scenarios, got {len(selected)}')
    return selected


def _fill_training_block(existing: Sequence[ScenarioSpec], *, count: int, group: str, difficulty: str, offset: int, min_stress_score: int) -> list[ScenarioSpec]:
    signatures = {scenario.signature() for scenario in existing}
    output: list[ScenarioSpec] = []
    for values in deterministic_value_stream(offset=offset):
        if _stress_score(values) < min_stress_score:
            continue
        scenario = _scenario_from_values(values, scenario_id=len(existing) + len(output), split='train', group=group, difficulty=difficulty, notes=('deterministic_mixed_radix_enumeration',))
        if _append_unique(output, signatures, scenario) and len(output) == count:
            return output
    raise RuntimeError(f'Could not fill {group} block to {count}')


def build_train_catalogue() -> list[ScenarioSpec]:
    anchors = _build_anchor_block()
    pairwise = _build_pairwise_block(anchors)
    first_1024 = anchors + pairwise
    common = _fill_training_block(first_1024, count=1024, group='common_multifactor', difficulty='common', offset=104729, min_stress_score=3)
    first_2048 = first_1024 + common
    hard = _fill_training_block(first_2048, count=1024, group='hard_in_domain', difficulty='hard', offset=524287, min_stress_score=6)
    train = first_2048 + hard
    if len(train) != 3072:
        raise AssertionError(len(train))
    return train


def _build_holdout_from_train_domains(*, count: int, split: str, group: str, start_id: int, excluded_signatures: set[tuple[str, ...]], offset: int, min_stress_score: int) -> list[ScenarioSpec]:
    output: list[ScenarioSpec] = []
    signatures = set(excluded_signatures)
    for values in deterministic_value_stream(offset=offset):
        if _stress_score(values) < min_stress_score:
            continue
        scenario = _scenario_from_values(values, scenario_id=start_id + len(output), split=split, group=group, difficulty='holdout', notes=('unseen_combination',))
        if _append_unique(output, signatures, scenario) and len(output) == count:
            return output
    raise RuntimeError(f'Could not fill {split}/{group} to {count}')


def _cycled_values(domains: Mapping[str, Sequence[str]], n: int, offset: int) -> dict[str, str]:
    values: dict[str, str] = {}
    strides = (1, 3, 5, 7, 11, 13)
    for axis, stride in zip(CORE_AXES, strides):
        domain = domains[axis]
        values[axis] = domain[(offset + stride * n + n * n) % len(domain)]
    return values


def _append_test_scenarios(output: list[ScenarioSpec], signatures: set[tuple[str, ...]], *, count: int, group: str, difficulty: str, start_id: int, domains: Mapping[str, Sequence[str]], orientation_ids: Sequence[str]=('horizontal',), require_ood: bool=False, offset: int=0) -> None:
    attempts = 0
    while sum(s.group == group for s in output) < count:
        values = _cycled_values(domains, attempts, offset)
        orientation_id = orientation_ids[(attempts * 3 + offset) % len(orientation_ids)]
        attempts += 1
        if not is_compatible(values['boundary_id'], values['dirichlet_id']):
            continue
        if require_ood:
            ids = tuple(values.values()) + (orientation_id,)
            if not any(item in SHAPE_BY_ID and SHAPE_BY_ID[item].ood or item in STRAIN_BY_ID and STRAIN_BY_ID[item].ood or item in VELOCITY_BY_ID and VELOCITY_BY_ID[item].ood or item in BOUNDARY_BY_ID and BOUNDARY_BY_ID[item].ood or item in MATERIAL_BY_ID and MATERIAL_BY_ID[item].ood or item in ORIENTATION_BY_ID and ORIENTATION_BY_ID[item].ood for item in ids):
                continue
        scenario = _scenario_from_values(values, scenario_id=start_id + len(output), split='test', group=group, difficulty=difficulty, orientation_id=orientation_id, notes=(group,))
        if not _append_unique(output, signatures, scenario):
            if attempts > 200000:
                raise RuntimeError(f'Could not fill test group {group}')
            continue
        if attempts > 200000:
            raise RuntimeError(f'Could not fill test group {group}')


def build_test_catalogue(excluded_signatures: set[tuple[str, ...]], *, start_id: int=200000) -> list[ScenarioSpec]:
    output: list[ScenarioSpec] = []
    signatures = set(excluded_signatures)
    id_holdout = _build_holdout_from_train_domains(count=96, split='test', group='id_combination', start_id=start_id, excluded_signatures=signatures, offset=99991, min_stress_score=4)
    output.extend(id_holdout)
    signatures.update(s.signature() for s in id_holdout)
    magnitude_domains = dict(TRAIN_DOMAINS)
    magnitude_domains['shape_id'] = tuple(item.id for item in OOD_SHAPES)
    magnitude_domains['strain_id'] = tuple(item.id for item in OOD_STRAINS)
    magnitude_domains['velocity_id'] = tuple(item.id for item in OOD_VELOCITIES)
    _append_test_scenarios(output, signatures, count=48, group='magnitude_ood', difficulty='ood', start_id=start_id, domains=magnitude_domains, require_ood=True, offset=17)
    boundary_material_domains = dict(TRAIN_DOMAINS)
    boundary_material_domains['boundary_id'] = tuple(item.id for item in OOD_BOUNDARIES)
    boundary_material_domains['material_id'] = tuple(item.id for item in OOD_MATERIALS)
    _append_test_scenarios(output, signatures, count=48, group='boundary_material_ood', difficulty='ood', start_id=start_id, domains=boundary_material_domains, require_ood=True, offset=29)
    _append_test_scenarios(output, signatures, count=32, group='orientation_ood', difficulty='ood', start_id=start_id, domains=TRAIN_DOMAINS, orientation_ids=tuple(item.id for item in OOD_ORIENTATIONS), require_ood=True, offset=43)
    hard_domains = {
        'shape_id': tuple(item.id for item in OOD_SHAPES),
        'strain_id': tuple(item.id for item in OOD_STRAINS),
        'velocity_id': tuple(item.id for item in OOD_VELOCITIES),
        'boundary_id': tuple(item.id for item in OOD_BOUNDARIES),
        'dirichlet_id': tuple(item.id for item in TRAIN_DIRICHLET),
        'material_id': tuple(item.id for item in OOD_MATERIALS),
    }
    _append_test_scenarios(output, signatures, count=32, group='hard_combined_ood', difficulty='hard_ood', start_id=start_id, domains=hard_domains, orientation_ids=tuple(item.id for item in OOD_ORIENTATIONS), require_ood=True, offset=71)
    if len(output) != 256:
        raise AssertionError(len(output))
    return [ScenarioSpec(**{**asdict(s), 'scenario_id': start_id + i}) for i, s in enumerate(output)]


def build_catalogues() -> dict[str, list[ScenarioSpec]]:
    train = build_train_catalogue()
    train_signatures = {scenario.signature() for scenario in train}
    validation = _build_holdout_from_train_domains(count=128, split='validation', group='combination_holdout', start_id=100000, excluded_signatures=train_signatures, offset=65521, min_stress_score=4)
    excluded = train_signatures | {scenario.signature() for scenario in validation}
    test = build_test_catalogue(excluded, start_id=200000)
    return {
        'train_c1_1024': train[:1024],
        'train_c2_2048': train[:2048],
        'train_c3_3072': train,
        'validation_128': validation,
        'test_256': test,
    }
