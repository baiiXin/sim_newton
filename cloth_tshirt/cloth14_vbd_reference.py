"""Run Newton VBD on frozen T-shirt typical motion 0.

The default run advances 6000 physical steps and writes a sparse trajectory,
diagnostics, exact configuration, and a comparison note to ``vbd_reference``.
Run this file in the ``cloth_opter`` conda environment.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
# The managed runner may have a read-only home directory.  Setting this before
# importing Warp also keeps compiler artifacts out of the result directory.
os.environ.setdefault(
    "WARP_CACHE_PATH", str(Path(tempfile.gettempdir()) / "cloth_tshirt_warp_cache")
)

import newton  # noqa: E402
import warp as wp  # noqa: E402

from tshirt_config import DEFAULT_FIXED_DATA_DIR, load_model_spec  # noqa: E402
from tshirt_mesh import load_tshirt_mesh  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_DIR / "vbd_reference"
DEFAULT_STEPS = 6000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate frozen typical motion 0 with Newton's VBD solver."
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--self-contact-radius",
        type=float,
        default=None,
        help="Defaults to the cloth thickness from model_spec.json.",
    )
    parser.add_argument(
        "--self-contact-margin-factor",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--disable-self-contact",
        action="store_true",
        help="Off by default only for debugging; reference runs keep self-contact enabled.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an already completed result with a fresh run.",
    )
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.sample_stride <= 0 or args.checkpoint_every <= 0 or args.progress_every <= 0:
        parser.error("stride/progress/checkpoint values must be positive")
    if args.self_contact_margin_factor < 1.0:
        parser.error("--self-contact-margin-factor must be at least 1")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def select_device(requested: str) -> str:
    devices = [str(device) for device in wp.get_devices()]
    if requested == "auto":
        cuda_devices = [device for device in devices if device.startswith("cuda")]
        return cuda_devices[0] if cuda_devices else "cpu"
    if requested not in devices and requested != "cuda":
        raise RuntimeError(f"Requested device {requested!r} is unavailable; found {devices}")
    return requested


def load_inputs(fixed_data_dir: Path):
    fixed_data_dir = fixed_data_dir.resolve()
    spec_path = fixed_data_dir / "model_spec.json"
    typical_path = fixed_data_dir / "typical_single_motions_4.npz"
    model_spec = load_model_spec(spec_path)
    mesh_path = Path(model_spec.mesh_path)
    if not mesh_path.is_absolute():
        mesh_path = PROJECT_DIR / mesh_path
    mesh = load_tshirt_mesh(mesh_path)
    if mesh.sha256 != model_spec.mesh_sha256:
        raise RuntimeError("T-shirt OBJ hash does not match fixed_data/model_spec.json")
    with np.load(typical_path, allow_pickle=False) as data:
        motion_ids = data["motion_ids"].astype(str)
        positions = np.asarray(data["positions"][0], dtype=np.float64)
        velocities = np.asarray(data["velocities"][0], dtype=np.float64)
    if motion_ids[0] != "typical_00_horizontal_gravity_release":
        raise RuntimeError(f"Unexpected typical motion 0: {motion_ids[0]!r}")
    if positions.shape != mesh.vertices.shape or velocities.shape != mesh.vertices.shape:
        raise RuntimeError("Typical motion 0 has an incompatible particle shape")
    return model_spec, mesh, positions, velocities, spec_path, typical_path


def build_vbd(
    model_spec,
    mesh,
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    *,
    device: str,
    iterations: int,
    self_contact_enabled: bool,
    self_contact_radius: float,
    self_contact_margin: float,
):
    material = model_spec.material
    # Newton integrates the membrane energy over rest area.  The project Lamé
    # parameters are volumetric, so multiplying by thickness gives the 2-D
    # coefficients expected by add_cloth_mesh.
    membrane_mu_2d = material.lame_mu * material.thickness
    membrane_lambda_2d = material.lame_lambda * material.thickness

    builder = newton.ModelBuilder()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=initial_positions.tolist(),
        indices=mesh.faces.reshape(-1).tolist(),
        density=material.areal_density,
        tri_ke=membrane_mu_2d,
        tri_ka=membrane_lambda_2d,
        tri_kd=0.0,
        edge_ke=material.bending_stiffness,
        edge_kd=0.0,
        particle_radius=0.5 * material.thickness,
        validate_mesh=True,
        label="typical_00_tshirt",
    )

    active_flag = int(newton.ParticleFlags.ACTIVE)
    for particle, velocity in enumerate(initial_velocities):
        builder.particle_qd[particle] = wp.vec3(*velocity)
    for particle in model_spec.fixed_indices:
        builder.particle_mass[particle] = 0.0
        builder.particle_flags[particle] = int(builder.particle_flags[particle]) & ~active_flag
        builder.particle_qd[particle] = wp.vec3(0.0, 0.0, 0.0)

    builder.color(include_bending=True)
    model = builder.finalize(device=device)
    model.set_gravity(model_spec.gravity)
    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=iterations,
        particle_enable_self_contact=self_contact_enabled,
        particle_self_contact_radius=self_contact_radius,
        particle_self_contact_margin=self_contact_margin,
        # Keep Newton's collision cadence and filtering defaults explicit.
        particle_collision_detection_interval=0,
        particle_topological_contact_filter_threshold=2,
        particle_rest_shape_contact_exclusion_radius=0.0,
    )
    return model, solver, membrane_mu_2d, membrane_lambda_2d


def frame_diagnostics(
    step: int,
    dt: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    initial_positions: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    fixed_indices: tuple[int, ...],
    rest_double_areas: np.ndarray,
    rest_edge_lengths: np.ndarray,
) -> dict[str, float | int | bool]:
    speed = np.linalg.norm(velocities, axis=1)
    triangles = positions[faces]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    edge_lengths = np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
    area_ratio = double_areas / rest_double_areas
    edge_ratio = edge_lengths / rest_edge_lengths
    fixed_error = np.linalg.norm(
        positions[np.asarray(fixed_indices)] - initial_positions[np.asarray(fixed_indices)], axis=1
    )
    return {
        "step": int(step),
        "time": float(step * dt),
        "finite": bool(np.isfinite(positions).all() and np.isfinite(velocities).all()),
        "min_y": float(np.min(positions[:, 1])),
        "max_y": float(np.max(positions[:, 1])),
        "mean_speed": float(np.mean(speed)),
        "max_speed": float(np.max(speed)),
        "max_displacement": float(np.max(np.linalg.norm(positions - initial_positions, axis=1))),
        "max_fixed_error": float(np.max(fixed_error)),
        "min_area_ratio": float(np.min(area_ratio)),
        "max_area_ratio": float(np.max(area_ratio)),
        "min_edge_length_ratio": float(np.min(edge_ratio)),
        "max_edge_length_ratio": float(np.max(edge_ratio)),
    }


def write_diagnostics_csv(path: Path, diagnostics: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    os.replace(temporary, path)


def save_checkpoint(
    path: Path,
    step: int,
    state,
    sample_steps: list[int],
    sampled_positions: list[np.ndarray],
    sampled_velocities: list[np.ndarray],
) -> None:
    write_npz(
        path,
        completed_step=np.asarray(step, dtype=np.int64),
        current_positions=state.particle_q.numpy().astype(np.float32),
        current_velocities=state.particle_qd.numpy().astype(np.float32),
        sample_steps=np.asarray(sample_steps, dtype=np.int64),
        sampled_positions=np.asarray(sampled_positions, dtype=np.float32),
        sampled_velocities=np.asarray(sampled_velocities, dtype=np.float32),
    )


def write_readme(path: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    config = manifest["simulation"]
    material = manifest["material_mapping"]
    text = f"""# Newton VBD reference: typical 0

本目录由 `cloth14_vbd_reference.py` 生成。参考轨迹使用冻结数据中的
`typical_00_horizontal_gravity_release`：衣服面水平、初速度为零、4 个肩部顶点固定，
在重力下释放。

## 实际运行配置

- 环境：conda `cloth_opter`
- Newton `{manifest['software']['newton']}`，Warp `{manifest['software']['warp']}`
- 设备：`{manifest['software']['device']}`
- 物理步数：{config['steps']}，`dt={config['dt']}` 秒，总仿真时间 {config['steps'] * config['dt']:.6g} 秒
- 每步 VBD 迭代：{config['iterations']}
- 自碰撞：{config['self_contact_enabled']}，半径 {config['self_contact_radius']:.8g} m，margin {config['self_contact_margin']:.8g} m
- 稀疏轨迹间隔：每 {config['sample_stride']} 步保存一次，共 {metrics['saved_frame_count']} 帧

## 参数映射

- `density = areal_density = {material['areal_density']:.12g}` kg/m²。
- Newton 1.4.0 VBD 使用与项目 README 相同形式的 stable Neo-Hookean 膜能。
  Newton 的面积积分参数采用二维系数，因此设置
  `tri_ke = thickness * lame_mu = {material['tri_ke']:.12g}` N/m、
  `tri_ka = thickness * lame_lambda = {material['tri_ka']:.12g}` N/m。
- `edge_ke = bending_stiffness = {material['edge_ke']:.12g}`；VBD 的二面角项同样乘静止边长。
- 项目基线没有材料阻尼，故 `tri_kd=edge_kd=0`。

## 与当前项目模型不完全一致的内容

1. Newton/Warp 内部使用 `float32`；当前学习优化器基线默认 `float64`。
2. 本结果保留 Newton VBD 的 Hessian 投影、顶点着色/Gauss-Seidel 更新和默认碰撞检测节奏，
   因而不是项目里三种 GD 基线的逐迭代复刻。
3. 项目基线用 `wrap(theta-theta_rest)` 计算弹性二面角差；Newton 1.4.0 的 VBD 弹性弯曲核
   直接使用 `theta-theta_rest`（只有阻尼的逐步角度差会 wrap）。跨过 `-pi/pi` 分支时两者会不同。
4. VBD 自碰撞已开启；当前项目能量基线明确没有碰撞项。自碰撞半径取布厚，margin 为其
   {config['self_contact_margin_factor']:.6g} 倍。没有添加 README 未定义的人体、地面或其他刚体碰撞体。
5. 四个固定点在 Newton 中通过零质量和清除 `ParticleFlags.ACTIVE` 实现；其目标位置恒定。
6. `trajectory.npz` 是每 {config['sample_stride']} 步采样一次的稀疏轨迹，不含全部 {config['steps'] + 1} 个状态。

## 结果健康检查

- 全部保存状态有限：{metrics['all_saved_states_finite']}。
- 终态最大固定点误差：{metrics['final']['max_fixed_error']:.8g} m。
- 终态速度：mean={metrics['final']['mean_speed']:.8g} m/s，max={metrics['final']['max_speed']:.8g} m/s。
- 终态三角形面积比范围：[{metrics['final']['min_area_ratio']:.8g}, {metrics['final']['max_area_ratio']:.8g}]。
- 终态边长比范围：[{metrics['final']['min_edge_length_ratio']:.8g}, {metrics['final']['max_edge_length_ratio']:.8g}]。

状态没有 NaN/Inf，但边长比极值表明局部变形明显；加之上面的碰撞和弯曲角分支差异，
该结果适合作为 Newton VBD 行为参考，不应视为当前无碰撞 `float64` 优化目标的高精度真值。

## 文件

- `trajectory.npz`：`steps`、`times`、`positions`、`velocities`、`faces`、`fixed_indices`。
- `diagnostics.csv`：每个保存帧的速度、位移、面积/边长比和固定点误差。
- `metrics.json`：完成状态、运行耗时和终态摘要。
- `manifest.json`：输入哈希、软件版本及完整参数映射。
- `resume_checkpoint.npz`：断点续跑状态；从断点恢复会重建 VBD 碰撞检测器的瞬态内部状态。

运行命令：

```bash
conda run --no-capture-output -n cloth_opter python cloth14_vbd_reference.py
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_metrics = output_dir / "metrics.json"
    if completed_metrics.exists() and not (args.overwrite or args.resume):
        previous = json.loads(completed_metrics.read_text(encoding="utf-8"))
        if previous.get("completed", False):
            raise RuntimeError(
                f"{completed_metrics} already contains a completed run; use --overwrite to replace it"
            )

    model_spec, mesh, initial_q, initial_qd, spec_path, typical_path = load_inputs(
        args.fixed_data_dir
    )
    device = select_device(args.device)
    contact_radius = (
        float(args.self_contact_radius)
        if args.self_contact_radius is not None
        else float(model_spec.material.thickness)
    )
    if contact_radius <= 0.0:
        raise ValueError("self-contact radius must be positive")
    contact_margin = contact_radius * args.self_contact_margin_factor
    self_contact_enabled = not args.disable_self_contact

    print(
        f"Building typical 0: {mesh.num_vertices} vertices, {mesh.num_faces} faces, "
        f"device={device}, steps={args.steps}, VBD iterations={args.iterations}",
        flush=True,
    )
    model, solver, membrane_mu_2d, membrane_lambda_2d = build_vbd(
        model_spec,
        mesh,
        initial_q,
        initial_qd,
        device=device,
        iterations=args.iterations,
        self_contact_enabled=self_contact_enabled,
        self_contact_radius=contact_radius,
        self_contact_margin=contact_margin,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()

    start_step = 0
    sample_steps: list[int] = []
    sampled_positions: list[np.ndarray] = []
    sampled_velocities: list[np.ndarray] = []
    checkpoint_path = output_dir / "resume_checkpoint.npz"
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume: {checkpoint_path} does not exist")
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            start_step = int(checkpoint["completed_step"])
            state_in.particle_q.assign(checkpoint["current_positions"])
            state_in.particle_qd.assign(checkpoint["current_velocities"])
            sample_steps = checkpoint["sample_steps"].astype(int).tolist()
            sampled_positions = [frame.copy() for frame in checkpoint["sampled_positions"]]
            sampled_velocities = [frame.copy() for frame in checkpoint["sampled_velocities"]]
        if start_step > args.steps:
            raise RuntimeError(f"Checkpoint step {start_step} exceeds requested {args.steps}")
        print(f"Resuming from step {start_step}", flush=True)
    else:
        sampled_positions.append(state_in.particle_q.numpy().astype(np.float32))
        sampled_velocities.append(state_in.particle_qd.numpy().astype(np.float32))
        sample_steps.append(0)

    cuda_graph = None
    cuda_graph_steps = 0
    # Capturing an even number of steps makes the ping-pong state buffers land
    # back on the same Python objects.  The default sample stride (10) is an
    # ideal graph chunk: one launch per saved frame instead of hundreds of
    # individual kernel launches.
    if (
        device.startswith("cuda")
        and args.sample_stride % 2 == 0
        and start_step % args.sample_stride == 0
        and args.steps - start_step >= args.sample_stride
    ):
        cuda_graph_steps = args.sample_stride
        graph_state_in = state_in
        graph_state_out = state_out
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(cuda_graph_steps):
                graph_state_in.clear_forces()
                model.collide(graph_state_in, contacts)
                solver.step(
                    graph_state_in, graph_state_out, control, contacts, model_spec.dt
                )
                graph_state_in, graph_state_out = graph_state_out, graph_state_in
        cuda_graph = capture.graph
        if graph_state_in is not state_in:
            raise AssertionError("CUDA graph chunk must contain an even number of steps")
        print(f"Captured a {cuda_graph_steps}-step CUDA graph", flush=True)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "motion_id": "typical_00_horizontal_gravity_release",
        "software": {
            "python": platform.python_version(),
            "newton": str(newton.__version__),
            "warp": str(wp.__version__),
            "device": device,
            "warp_devices": [str(item) for item in wp.get_devices()],
        },
        "inputs": {
            "model_spec": str(spec_path),
            "model_spec_sha256": sha256(spec_path),
            "typical_states": str(typical_path),
            "typical_states_sha256": sha256(typical_path),
            "mesh": str(mesh.path),
            "mesh_sha256": mesh.sha256,
            "num_vertices": mesh.num_vertices,
            "num_faces": mesh.num_faces,
            "fixed_indices": list(model_spec.fixed_indices),
        },
        "simulation": {
            "steps": args.steps,
            "dt": model_spec.dt,
            "iterations": args.iterations,
            "sample_stride": args.sample_stride,
            "gravity": list(model_spec.gravity),
            "self_contact_enabled": self_contact_enabled,
            "self_contact_radius": contact_radius,
            "self_contact_margin": contact_margin,
            "self_contact_margin_factor": args.self_contact_margin_factor,
            "particle_collision_detection_interval": 0,
            "particle_topological_contact_filter_threshold": 2,
            "ground_or_body_colliders": False,
            "cuda_graph_chunk_steps": cuda_graph_steps,
        },
        "material_mapping": {
            **asdict(model_spec.material),
            "tri_ke": membrane_mu_2d,
            "tri_ka": membrane_lambda_2d,
            "tri_kd": 0.0,
            "edge_ke": model_spec.material.bending_stiffness,
            "edge_kd": 0.0,
            "particle_radius": 0.5 * model_spec.material.thickness,
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    rest_triangles = initial_q[mesh.faces]
    rest_double_areas = np.linalg.norm(
        np.cross(
            rest_triangles[:, 1] - rest_triangles[:, 0],
            rest_triangles[:, 2] - rest_triangles[:, 0],
        ),
        axis=1,
    )
    rest_edge_lengths = np.linalg.norm(
        initial_q[mesh.edges[:, 1]] - initial_q[mesh.edges[:, 0]], axis=1
    )

    wall_start = time.perf_counter()
    step = start_step
    last_checkpoint_step = start_step
    last_progress_step = start_step
    while step < args.steps:
        if cuda_graph is not None and step + cuda_graph_steps <= args.steps:
            wp.capture_launch(cuda_graph)
            step += cuda_graph_steps
        else:
            state_in.clear_forces()
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, model_spec.dt)
            state_in, state_out = state_out, state_in
            step += 1

        if step % args.sample_stride == 0 or step == args.steps:
            sampled_positions.append(state_in.particle_q.numpy().astype(np.float32))
            sampled_velocities.append(state_in.particle_qd.numpy().astype(np.float32))
            sample_steps.append(step)
            if not (
                np.isfinite(sampled_positions[-1]).all()
                and np.isfinite(sampled_velocities[-1]).all()
            ):
                save_checkpoint(
                    checkpoint_path,
                    step,
                    state_in,
                    sample_steps,
                    sampled_positions,
                    sampled_velocities,
                )
                raise FloatingPointError(f"Non-finite VBD state at step {step}")

        if step - last_checkpoint_step >= args.checkpoint_every or step == args.steps:
            save_checkpoint(
                checkpoint_path,
                step,
                state_in,
                sample_steps,
                sampled_positions,
                sampled_velocities,
            )
            last_checkpoint_step = step
        if step - last_progress_step >= args.progress_every or step == args.steps:
            elapsed = time.perf_counter() - wall_start
            rate = (step - start_step) / max(elapsed, 1.0e-12)
            remaining = (args.steps - step) / max(rate, 1.0e-12)
            print(
                f"step {step}/{args.steps} | {rate:.2f} steps/s | ETA {remaining / 60.0:.1f} min",
                flush=True,
            )
            last_progress_step = step

    wp.synchronize_device(device)
    wall_seconds = time.perf_counter() - wall_start
    positions_array = np.asarray(sampled_positions, dtype=np.float32)
    velocities_array = np.asarray(sampled_velocities, dtype=np.float32)
    steps_array = np.asarray(sample_steps, dtype=np.int64)
    write_npz(
        output_dir / "trajectory.npz",
        steps=steps_array,
        times=steps_array.astype(np.float64) * model_spec.dt,
        positions=positions_array,
        velocities=velocities_array,
        faces=mesh.faces.astype(np.int32),
        fixed_indices=np.asarray(model_spec.fixed_indices, dtype=np.int32),
    )

    diagnostics = [
        frame_diagnostics(
            int(step),
            model_spec.dt,
            positions.astype(np.float64),
            velocities.astype(np.float64),
            initial_q,
            mesh.faces,
            mesh.edges,
            model_spec.fixed_indices,
            rest_double_areas,
            rest_edge_lengths,
        )
        for step, positions, velocities in zip(
            steps_array, positions_array, velocities_array, strict=True
        )
    ]
    write_diagnostics_csv(output_dir / "diagnostics.csv", diagnostics)
    metrics = {
        "completed": True,
        "completed_steps": args.steps,
        "simulated_seconds": args.steps * model_spec.dt,
        "wall_seconds_this_invocation": wall_seconds,
        "steps_per_wall_second_this_invocation": (args.steps - start_step)
        / max(wall_seconds, 1.0e-12),
        "saved_frame_count": len(sample_steps),
        "final": diagnostics[-1],
        "all_saved_states_finite": all(bool(item["finite"]) for item in diagnostics),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_readme(output_dir / "README.md", manifest, metrics)
    print(f"Completed. Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
