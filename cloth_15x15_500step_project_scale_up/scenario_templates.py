"""Hand-authored deterministic scenario templates for cloth scale-up."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class ShapeTemplate:
    id: str
    name: str
    family: str
    amplitude_fraction: float
    ood: bool = False


@dataclass(frozen=True)
class StrainTemplate:
    id: str
    name: str
    scale_x: float
    scale_y: float
    ood: bool = False


@dataclass(frozen=True)
class VelocityTemplate:
    id: str
    name: str
    kind: str
    vector: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    angular_speed: float = 0.0
    magnitude_level: str = "zero"
    ood: bool = False


@dataclass(frozen=True)
class BoundaryTemplate:
    id: str
    name: str
    family: str
    selector: str
    normalized_points: tuple[tuple[float, float], ...] = ()
    edge: str | None = None
    ood: bool = False


@dataclass(frozen=True)
class DirichletTemplate:
    id: str
    name: str
    kind: str
    amplitude: float
    frequency_hz: float


@dataclass(frozen=True)
class MaterialTemplate:
    id: str
    name: str
    areal_density: float
    stretch_stiffness: float
    shear_stiffness: float
    bending_stiffness: float | None
    damping: float | None
    ood: bool = False


@dataclass(frozen=True)
class OrientationTemplate:
    id: str
    name: str
    euler_xyz_degrees: tuple[float, float, float]
    ood: bool = False


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: int
    split: str
    group: str
    difficulty: str
    shape_id: str
    strain_id: str
    velocity_id: str
    boundary_id: str
    dirichlet_id: str
    material_id: str
    orientation_id: str = "horizontal"
    fixed_target_mode: str = "pin_to_initial_pose"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def signature(self) -> tuple[str, ...]:
        return (
            self.shape_id,
            self.strain_id,
            self.velocity_id,
            self.boundary_id,
            self.dirichlet_id,
            self.material_id,
            self.orientation_id,
            self.fixed_target_mode,
        )

    def core_signature(self) -> tuple[str, ...]:
        return (
            self.shape_id,
            self.strain_id,
            self.velocity_id,
            self.boundary_id,
            self.dirichlet_id,
            self.material_id,
        )


CORE_AXES = (
    "shape_id",
    "strain_id",
    "velocity_id",
    "boundary_id",
    "dirichlet_id",
    "material_id",
)

TRAIN_SHAPES: tuple[ShapeTemplate, ...] = (
    ShapeTemplate("plane", "平面", "plane", 0.00),
    ShapeTemplate("saddle_uv_pos", "鞍形 uv 正", "saddle_uv", 0.18),
    ShapeTemplate("saddle_uv_neg", "鞍形 uv 负", "saddle_uv", -0.18),
    ShapeTemplate("saddle_quad_pos", "鞍形 u²-v² 正", "saddle_quad", 0.18),
    ShapeTemplate("saddle_quad_neg", "鞍形 u²-v² 负", "saddle_quad", -0.18),
    ShapeTemplate("dome_up", "向上凸包", "dome", 0.18),
    ShapeTemplate("dome_down", "向下凸包", "dome", -0.18),
)

OOD_SHAPES: tuple[ShapeTemplate, ...] = (
    ShapeTemplate("saddle_uv_pos_strong", "强鞍形 uv 正", "saddle_uv", 0.32, True),
    ShapeTemplate("saddle_uv_neg_strong", "强鞍形 uv 负", "saddle_uv", -0.32, True),
    ShapeTemplate("saddle_quad_pos_strong", "强鞍形 u²-v² 正", "saddle_quad", 0.32, True),
    ShapeTemplate("saddle_quad_neg_strong", "强鞍形 u²-v² 负", "saddle_quad", -0.32, True),
    ShapeTemplate("dome_up_strong", "强向上凸包", "dome", 0.32, True),
    ShapeTemplate("dome_down_strong", "强向下凸包", "dome", -0.32, True),
)

TRAIN_STRAINS: tuple[StrainTemplate, ...] = (
    StrainTemplate("strain_none", "无拉伸压缩", 1.00, 1.00),
    StrainTemplate("stretch_x", "x 方向拉伸", 1.15, 1.00),
    StrainTemplate("compress_x", "x 方向压缩", 0.85, 1.00),
    StrainTemplate("stretch_y", "y 方向拉伸", 1.00, 1.15),
    StrainTemplate("compress_y", "y 方向压缩", 1.00, 0.85),
    StrainTemplate("stretch_xy", "双向拉伸", 1.15, 1.15),
    StrainTemplate("compress_xy", "双向压缩", 0.85, 0.85),
)

OOD_STRAINS: tuple[StrainTemplate, ...] = (
    StrainTemplate("stretch_x_strong", "x 方向强拉伸", 1.30, 1.00, True),
    StrainTemplate("compress_x_strong", "x 方向强压缩", 0.70, 1.00, True),
    StrainTemplate("stretch_y_strong", "y 方向强拉伸", 1.00, 1.30, True),
    StrainTemplate("compress_y_strong", "y 方向强压缩", 1.00, 0.70, True),
    StrainTemplate("stretch_xy_strong", "双向强拉伸", 1.30, 1.30, True),
    StrainTemplate("compress_xy_strong", "双向强压缩", 0.70, 0.70, True),
)


def _translation_velocity_templates() -> list[VelocityTemplate]:
    out = [VelocityTemplate("velocity_zero", "静止", "zero")]
    axes = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    for axis_name, axis in axes.items():
        for sign_name, sign_value in (("pos", 1.0), ("neg", -1.0)):
            for level_name, speed in (("medium", 2.5), ("high", 5.0)):
                out.append(VelocityTemplate(
                    id=f"translate_{sign_name}_{axis_name}_{level_name}",
                    name=f"{axis_name} 轴{sign_name}平移 {level_name}",
                    kind="translation",
                    vector=tuple(sign_value * speed * value for value in axis),
                    magnitude_level=level_name,
                ))
    return out


def _rotation_velocity_templates() -> list[VelocityTemplate]:
    out: list[VelocityTemplate] = []
    axes = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    for axis_name, axis in axes.items():
        for sign_name, sign_value in (("pos", 1.0), ("neg", -1.0)):
            out.append(VelocityTemplate(
                id=f"rotate_{sign_name}_{axis_name}",
                name=f"绕 {axis_name} 轴{sign_name}旋转",
                kind="rotation",
                axis=axis,
                angular_speed=sign_value * 1.5,
                magnitude_level="rotation",
            ))
    return out


TRAIN_VELOCITIES: tuple[VelocityTemplate, ...] = tuple(_translation_velocity_templates() + _rotation_velocity_templates())
assert len(TRAIN_VELOCITIES) == 19

OOD_VELOCITIES: tuple[VelocityTemplate, ...] = tuple(
    VelocityTemplate(
        id=f"translate_{sign_name}_{axis_name}_extreme",
        name=f"{axis_name} 轴{sign_name}极高速平移",
        kind="translation",
        vector=tuple(sign_value * 7.5 * value for value in axis),
        magnitude_level="extreme",
        ood=True,
    )
    for axis_name, axis in {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}.items()
    for sign_name, sign_value in (("pos", 1.0), ("neg", -1.0))
) + tuple(
    VelocityTemplate(
        id=f"rotate_{sign_name}_{axis_name}_extreme",
        name=f"绕 {axis_name} 轴{sign_name}高速旋转",
        kind="rotation",
        axis=axis,
        angular_speed=sign_value * 3.0,
        magnitude_level="extreme_rotation",
        ood=True,
    )
    for axis_name, axis in {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}.items()
    for sign_name, sign_value in (("pos", 1.0), ("neg", -1.0))
)

TRAIN_BOUNDARIES: tuple[BoundaryTemplate, ...] = (
    BoundaryTemplate("single_corner_tl", "单点：左上角", "single", "points", ((0.0, 0.0),)),
    BoundaryTemplate("single_corner_tr", "单点：右上角", "single", "points", ((1.0, 0.0),)),
    BoundaryTemplate("single_corner_bl", "单点：左下角", "single", "points", ((0.0, 1.0),)),
    BoundaryTemplate("single_corner_br", "单点：右下角", "single", "points", ((1.0, 1.0),)),
    BoundaryTemplate("single_edge_top", "单点：上边中点", "single", "points", ((0.5, 0.0),)),
    BoundaryTemplate("single_edge_bottom", "单点：下边中点", "single", "points", ((0.5, 1.0),)),
    BoundaryTemplate("single_edge_left", "单点：左边中点", "single", "points", ((0.0, 0.5),)),
    BoundaryTemplate("single_edge_right", "单点：右边中点", "single", "points", ((1.0, 0.5),)),
    BoundaryTemplate("single_center", "单点：中心点", "single", "points", ((0.5, 0.5),)),
    BoundaryTemplate("pair_diagonal_main", "两点：主对角", "pair", "points", ((0.0, 0.0), (1.0, 1.0))),
    BoundaryTemplate("pair_diagonal_anti", "两点：副对角", "pair", "points", ((1.0, 0.0), (0.0, 1.0))),
    BoundaryTemplate("pair_top_corners", "两点：上边两角", "pair", "points", ((0.0, 0.0), (1.0, 0.0))),
    BoundaryTemplate("pair_bottom_corners", "两点：下边两角", "pair", "points", ((0.0, 1.0), (1.0, 1.0))),
    BoundaryTemplate("pair_left_corners", "两点：左边两角", "pair", "points", ((0.0, 0.0), (0.0, 1.0))),
    BoundaryTemplate("pair_right_corners", "两点：右边两角", "pair", "points", ((1.0, 0.0), (1.0, 1.0))),
    BoundaryTemplate("pair_mid_top_bottom", "两点：上下边中点", "pair", "points", ((0.5, 0.0), (0.5, 1.0))),
    BoundaryTemplate("pair_mid_left_right", "两点：左右边中点", "pair", "points", ((0.0, 0.5), (1.0, 0.5))),
    BoundaryTemplate("pair_center_top", "两点：中心+上边中点", "pair", "points", ((0.5, 0.5), (0.5, 0.0))),
    BoundaryTemplate("pair_center_bottom", "两点：中心+下边中点", "pair", "points", ((0.5, 0.5), (0.5, 1.0))),
    BoundaryTemplate("pair_center_left", "两点：中心+左边中点", "pair", "points", ((0.5, 0.5), (0.0, 0.5))),
    BoundaryTemplate("pair_center_right", "两点：中心+右边中点", "pair", "points", ((0.5, 0.5), (1.0, 0.5))),
    BoundaryTemplate("four_corners", "四点：四个角", "four", "points", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))),
)
assert len(TRAIN_BOUNDARIES) == 22

OOD_BOUNDARIES: tuple[BoundaryTemplate, ...] = (
    BoundaryTemplate("no_fixed", "无固定点", "none", "none", ood=True),
    BoundaryTemplate("three_corners_no_tl", "三角点：缺左上", "three", "points", ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), ood=True),
    BoundaryTemplate("three_corners_no_tr", "三角点：缺右上", "three", "points", ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0)), ood=True),
    BoundaryTemplate("three_corners_no_bl", "三角点：缺左下", "three", "points", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), ood=True),
    BoundaryTemplate("three_corners_no_br", "三角点：缺右下", "three", "points", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ood=True),
    BoundaryTemplate("full_edge_top", "整条上边", "edge", "edge", edge="top", ood=True),
    BoundaryTemplate("full_edge_bottom", "整条下边", "edge", "edge", edge="bottom", ood=True),
    BoundaryTemplate("full_edge_left", "整条左边", "edge", "edge", edge="left", ood=True),
    BoundaryTemplate("full_edge_right", "整条右边", "edge", "edge", edge="right", ood=True),
    BoundaryTemplate("center_plus_edge_mids", "中心+四边中点", "five", "points", ((0.5, 0.5), (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5)), ood=True),
)

TRAIN_DIRICHLET: tuple[DirichletTemplate, ...] = (
    DirichletTemplate("static", "静态固定", "static", 0.0, 0.0),
    DirichletTemplate("circle_horizontal_pos", "水平圆周正向", "circle_horizontal", 0.35, 0.50),
    DirichletTemplate("circle_horizontal_neg", "水平圆周反向", "circle_horizontal", 0.35, -0.50),
    DirichletTemplate("circle_vertical_pos", "竖直圆周正向", "circle_vertical", 0.35, 0.50),
    DirichletTemplate("circle_vertical_neg", "竖直圆周反向", "circle_vertical", 0.35, -0.50),
    DirichletTemplate("twist_pos", "正向扭布", "twist", 0.40, 0.40),
    DirichletTemplate("twist_neg", "反向扭布", "twist", 0.40, -0.40),
)

TRAIN_MATERIALS: tuple[MaterialTemplate, ...] = (
    MaterialTemplate("material_baseline", "基准材料", 1.0, 1.0, 1.0, None, None),
    MaterialTemplate("material_light", "轻质", 0.5, 1.0, 1.0, None, None),
    MaterialTemplate("material_heavy", "重质", 2.0, 1.0, 1.0, None, None),
    MaterialTemplate("material_soft", "整体偏软", 1.0, 0.5, 0.5, None, None),
    MaterialTemplate("material_stiff", "整体偏硬", 1.0, 2.0, 2.0, None, None),
    MaterialTemplate("material_shear_soft", "剪切偏软", 1.0, 1.0, 0.5, None, None),
    MaterialTemplate("material_shear_stiff", "剪切偏硬", 1.0, 1.0, 2.0, None, None),
    MaterialTemplate("material_heavy_soft", "重且偏软", 2.0, 0.5, 0.5, None, None),
)

OOD_MATERIALS: tuple[MaterialTemplate, ...] = (
    MaterialTemplate("material_very_light", "极轻", 0.25, 1.0, 1.0, None, None, True),
    MaterialTemplate("material_very_heavy", "极重", 4.0, 1.0, 1.0, None, None, True),
    MaterialTemplate("material_very_soft", "极软", 1.0, 0.25, 0.25, None, None, True),
    MaterialTemplate("material_very_stiff", "极硬", 1.0, 4.0, 4.0, None, None, True),
    MaterialTemplate("material_shear_very_soft", "剪切极软", 1.0, 1.0, 0.25, None, None, True),
    MaterialTemplate("material_shear_very_stiff", "剪切极硬", 1.0, 1.0, 4.0, None, None, True),
)

TRAIN_ORIENTATIONS: tuple[OrientationTemplate, ...] = (OrientationTemplate("horizontal", "水平", (0.0, 0.0, 0.0)),)
OOD_ORIENTATIONS: tuple[OrientationTemplate, ...] = (
    OrientationTemplate("vertical_x", "绕 x 轴竖直", (90.0, 0.0, 0.0), True),
    OrientationTemplate("vertical_y", "绕 y 轴竖直", (0.0, 90.0, 0.0), True),
    OrientationTemplate("tilt_x_45", "绕 x 轴倾斜 45°", (45.0, 0.0, 0.0), True),
    OrientationTemplate("tilt_y_45", "绕 y 轴倾斜 45°", (0.0, 45.0, 0.0), True),
)

ALL_SHAPES = TRAIN_SHAPES + OOD_SHAPES
ALL_STRAINS = TRAIN_STRAINS + OOD_STRAINS
ALL_VELOCITIES = TRAIN_VELOCITIES + OOD_VELOCITIES
ALL_BOUNDARIES = TRAIN_BOUNDARIES + OOD_BOUNDARIES
ALL_DIRICHLET = TRAIN_DIRICHLET
ALL_MATERIALS = TRAIN_MATERIALS + OOD_MATERIALS
ALL_ORIENTATIONS = TRAIN_ORIENTATIONS + OOD_ORIENTATIONS


def _index_by_id(items: Sequence[Any]) -> dict[str, Any]:
    result = {item.id: item for item in items}
    if len(result) != len(items):
        raise AssertionError("Template IDs must be unique")
    return result


SHAPE_BY_ID = _index_by_id(ALL_SHAPES)
STRAIN_BY_ID = _index_by_id(ALL_STRAINS)
VELOCITY_BY_ID = _index_by_id(ALL_VELOCITIES)
BOUNDARY_BY_ID = _index_by_id(ALL_BOUNDARIES)
DIRICHLET_BY_ID = _index_by_id(ALL_DIRICHLET)
MATERIAL_BY_ID = _index_by_id(ALL_MATERIALS)
ORIENTATION_BY_ID = _index_by_id(ALL_ORIENTATIONS)

TRAIN_DOMAINS: dict[str, tuple[str, ...]] = {
    "shape_id": tuple(item.id for item in TRAIN_SHAPES),
    "strain_id": tuple(item.id for item in TRAIN_STRAINS),
    "velocity_id": tuple(item.id for item in TRAIN_VELOCITIES),
    "boundary_id": tuple(item.id for item in TRAIN_BOUNDARIES),
    "dirichlet_id": tuple(item.id for item in TRAIN_DIRICHLET),
    "material_id": tuple(item.id for item in TRAIN_MATERIALS),
}

BASELINE_VALUES = {
    "shape_id": "plane",
    "strain_id": "strain_none",
    "velocity_id": "velocity_zero",
    "boundary_id": "pair_left_corners",
    "dirichlet_id": "static",
    "material_id": "material_baseline",
}
