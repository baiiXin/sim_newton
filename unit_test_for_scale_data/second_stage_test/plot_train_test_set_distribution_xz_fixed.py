from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Fixed note: x-z training plots use unique projected (x_i, z_k) lattice points.
# Do not subsample the flattened 3D grid for x-z visualization, because that creates aliasing stripes.

GRID_MATCH_TOL = 1e-10


# ============================================================
# 1. 文件与配置读取
# ============================================================


@dataclass(frozen=True)
class ExperimentFile:
    experiment_dir: Path
    report_path: Path
    model_path: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class RegularGridSpec:
    target_dataset_size: int
    actual_dataset_size: int
    points_per_axis: int
    sampling_radius: float
    axis_spacing: float


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_results_dir(results_dir_arg: str | None) -> Path:
    if results_dir_arg is not None:
        path = Path(results_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"results_dir does not exist: {path}")
        return path

    candidates: list[Path] = []
    search_roots = [Path.cwd(), Path(__file__).resolve().parent]
    seen_roots: set[Path] = set()

    for root in search_roots:
        root = root.resolve()
        if root in seen_roots or not root.exists():
            continue
        seen_roots.add(root)
        candidates.extend(root.glob("dataset_scale_ablation_summary.json"))
        candidates.extend(root.glob("*/dataset_scale_ablation_summary.json"))
        candidates.extend(root.glob("*/*/dataset_scale_ablation_summary.json"))

    if not candidates:
        raise FileNotFoundError(
            "Cannot find dataset_scale_ablation_summary.json automatically. "
            "Please pass --results-dir <training_output_dir>."
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].parent.resolve()


def find_experiment_files(results_dir: Path) -> list[ExperimentFile]:
    experiment_files: list[ExperimentFile] = []
    for report_path in sorted(results_dir.glob("*/optimization_report.json")):
        experiment_dir = report_path.parent
        model_path = experiment_dir / "mlp_optimizer_state_dict.pt"
        if not model_path.exists():
            continue
        report = load_json(report_path)
        experiment_files.append(
            ExperimentFile(
                experiment_dir=experiment_dir,
                report_path=report_path,
                model_path=model_path,
                report=report,
            )
        )

    if not experiment_files:
        raise FileNotFoundError(
            f"No experiment subdirectories found under {results_dir}. "
            "Expected */optimization_report.json and */mlp_optimizer_state_dict.pt."
        )
    return experiment_files


def parse_dataset_size_from_name(name: str) -> int | None:
    match = re.search(r"num_samples_(\d+)", name)
    return int(match.group(1)) if match else None


def extract_unique_grid_specs(experiment_files: Sequence[ExperimentFile]) -> list[RegularGridSpec]:
    specs: dict[tuple[int, int], RegularGridSpec] = {}
    for exp in experiment_files:
        cfg = exp.report.get("config", {})
        points_per_axis = int(cfg.get("points_per_axis", -1))
        actual_dataset_size = int(
            cfg.get("actual_dataset_size", parse_dataset_size_from_name(exp.experiment_dir.name) or -1)
        )
        target_dataset_size = int(cfg.get("target_dataset_size", actual_dataset_size))
        radius = float(cfg.get("sampling_radius_per_axis", cfg.get("sampling_radius", 0.01)))
        axis_spacing = float(cfg.get("axis_spacing", (2.0 * radius) / max(points_per_axis - 1, 1)))
        if points_per_axis <= 0 or actual_dataset_size <= 0:
            continue
        specs[(points_per_axis, actual_dataset_size)] = RegularGridSpec(
            target_dataset_size=target_dataset_size,
            actual_dataset_size=actual_dataset_size,
            points_per_axis=points_per_axis,
            sampling_radius=radius,
            axis_spacing=axis_spacing,
        )
    return sorted(specs.values(), key=lambda s: s.actual_dataset_size)


# ============================================================
# 2. 训练集重建与测试集构造
# ============================================================


def sample_regular_grid_points_for_plot(
    *,
    grid_spec: RegularGridSpec,
    y_star: np.ndarray,
    max_points: int,
) -> np.ndarray:
    """
    从规则 3D 网格中抽样一部分点用于 3D 作图。

    注意：这个函数保留给可能的 3D 可视化使用。
    对 x-z 投影图，不应该从扁平化 3D index 中等距抽样，否则会产生 aliasing 斜纹。
    x-z 投影图应该使用 sample_regular_grid_xz_projection_for_plot。
    """
    n = int(grid_spec.points_per_axis)
    total = int(grid_spec.actual_dataset_size)
    radius = float(grid_spec.sampling_radius)
    spacing = float(grid_spec.axis_spacing)
    lower = y_star - radius

    if total <= max_points:
        flat_indices = np.arange(total, dtype=np.int64)
    else:
        rng = np.random.default_rng(20260617 + n)
        flat_indices = np.sort(rng.choice(total, size=max_points, replace=False))

    n2 = n * n
    ix = flat_indices // n2
    rem = flat_indices % n2
    iy = rem // n
    iz = rem % n

    points = np.empty((flat_indices.shape[0], 3), dtype=float)
    points[:, 0] = lower[0] + ix * spacing
    points[:, 1] = lower[1] + iy * spacing
    points[:, 2] = lower[2] + iz * spacing
    return points


def sample_regular_grid_xz_projection_for_plot(
    *,
    grid_spec: RegularGridSpec,
    y_star: np.ndarray,
    max_points: int,
) -> np.ndarray:
    """
    正确绘制规则训练集在 x-z 平面的投影。

    训练集是三维规则网格：
        (x_i, y_j, z_k), i,j,k=0,...,n-1

    投影到 x-z 平面后，j 维被消去，因此唯一投影点应该是：
        (x_i, z_k), i,k=0,...,n-1

    对于 N=n^3=1,000,000, n=100，x-z 投影应有 100*100=10,000 个唯一点，
    而不是沿扁平化 index 采样形成的斜纹。
    """
    n = int(grid_spec.points_per_axis)
    radius = float(grid_spec.sampling_radius)
    spacing = float(grid_spec.axis_spacing)
    lower = y_star - radius

    total_projected = n * n
    if total_projected <= max_points:
        flat_indices = np.arange(total_projected, dtype=np.int64)
    else:
        # 如果未来 n 很大，则在 2D 投影网格上随机抽样，而不是在 3D 扁平索引上抽样。
        rng = np.random.default_rng(20260617 + n)
        flat_indices = np.sort(rng.choice(total_projected, size=max_points, replace=False))

    ix = flat_indices // n
    iz = flat_indices % n

    points_xz = np.empty((flat_indices.shape[0], 2), dtype=float)
    points_xz[:, 0] = lower[0] + ix * spacing
    points_xz[:, 1] = lower[2] + iz * spacing
    return points_xz


def points_in_grid_mask(
    points: torch.Tensor,
    *,
    grid_spec: RegularGridSpec,
    y_star: torch.Tensor,
    tol: float = GRID_MATCH_TOL,
) -> torch.Tensor:
    n = int(grid_spec.points_per_axis)
    radius = float(grid_spec.sampling_radius)
    spacing = float(grid_spec.axis_spacing)

    lower = y_star.to(dtype=points.dtype).unsqueeze(0) - radius
    coords = (points - lower) / spacing
    rounded = torch.round(coords)
    close_to_integer = torch.abs(coords - rounded) <= tol
    in_range = (rounded >= 0) & (rounded <= (n - 1))
    return torch.all(close_to_integer & in_range, dim=1)


def build_fixed_heldout_test_set_excluding_training_grids(
    *,
    y_star: torch.Tensor,
    radius: float,
    num_random: int,
    seed: int,
    radius_scale: float,
    dtype: torch.dtype,
    grid_specs: Sequence[RegularGridSpec],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if num_random <= 0:
        raise ValueError("num_random must be positive.")
    if radius <= 0.0:
        raise ValueError("sampling radius must be positive.")
    if radius_scale <= 0.0:
        raise ValueError("radius_scale must be positive.")

    effective_radius = float(radius) * float(radius_scale)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    y_star_cpu = y_star.detach().cpu().to(dtype=dtype)

    collected_chunks: list[torch.Tensor] = []
    num_collected = 0
    num_candidates_generated = 0
    num_rejected_overlap = 0
    batch_candidates = max(4096, min(65536, num_random * 2))

    for _ in range(10000):
        if num_collected >= num_random:
            break

        remaining = num_random - num_collected
        current_batch = max(batch_candidates, remaining * 2)
        offsets = (
            2.0 * torch.rand((current_batch, 3), generator=generator, dtype=dtype) - 1.0
        ) * effective_radius
        candidates = y_star_cpu.unsqueeze(0) + offsets
        keep_mask = torch.ones(current_batch, dtype=torch.bool)

        for grid_spec in grid_specs:
            keep_mask &= ~points_in_grid_mask(candidates, grid_spec=grid_spec, y_star=y_star_cpu)

        kept = candidates[keep_mask]
        num_candidates_generated += int(current_batch)
        num_rejected_overlap += int((~keep_mask).sum().item())

        if kept.shape[0] > remaining:
            kept = kept[:remaining]
        if kept.numel() > 0:
            collected_chunks.append(kept)
            num_collected += int(kept.shape[0])

    if num_collected < num_random:
        raise RuntimeError(
            f"Failed to construct enough held-out test points. requested={num_random}, collected={num_collected}."
        )

    test_points = torch.cat(collected_chunks, dim=0)
    metadata = {
        "num_random_points": int(num_random),
        "effective_sampling_radius": float(effective_radius),
        "strictly_excludes_all_training_points": True,
        "num_candidate_points_generated": int(num_candidates_generated),
        "num_candidate_points_rejected_due_to_training_overlap": int(num_rejected_overlap),
    }
    return test_points, metadata


# ============================================================
# 3. 绘图：全部改成 x-z 平面
# ============================================================


def set_equal_2d_axes(ax, points_xz: np.ndarray) -> None:
    points_xz = np.asarray(points_xz, dtype=float).reshape(-1, 2)
    points_xz = points_xz[np.isfinite(points_xz).all(axis=1)]
    if points_xz.shape[0] == 0:
        points_xz = np.zeros((1, 2), dtype=float)
    center = points_xz.mean(axis=0)
    radius = max(float(np.ptp(points_xz, axis=0).max()) / 2.0, 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def plot_training_distribution_xz_overview(
    *,
    grid_specs: Sequence[RegularGridSpec],
    y_star: Sequence[float],
    p_n: Sequence[float],
    save_path: Path,
    max_training_points_per_subplot: int = 12000,
) -> None:
    y_star_np = np.asarray(y_star, dtype=float)
    p_n_np = np.asarray(p_n, dtype=float)

    num_plots = len(grid_specs)
    num_cols = min(4, max(1, num_plots))
    num_rows = math.ceil(num_plots / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5.2 * num_cols, 4.8 * num_rows))
    axes = np.asarray(axes).reshape(-1)

    for i, spec in enumerate(grid_specs):
        ax = axes[i]
        train_xz = sample_regular_grid_xz_projection_for_plot(
            grid_spec=spec,
            y_star=y_star_np,
            max_points=max_training_points_per_subplot,
        )
        total_projected = spec.points_per_axis * spec.points_per_axis

        ax.scatter(
            train_xz[:, 0],
            train_xz[:, 1],
            s=4,
            alpha=0.35,
            color="C0",
            label=f"x-z projection ({train_xz.shape[0]}/{total_projected}); each point has {spec.points_per_axis} y-values",
            rasterized=True,
        )
        ax.scatter(p_n_np[0], p_n_np[2], marker="x", s=90, linewidths=2.0, color="C3", label=r"$p_n$")
        ax.scatter(y_star_np[0], y_star_np[2], marker="*", s=180, color="C2", label=r"$y^*$")

        set_equal_2d_axes(ax, np.vstack([train_xz, p_n_np[[0, 2]].reshape(1, 2), y_star_np[[0, 2]].reshape(1, 2)]))
        ax.set_title(f"Training set x-z distribution\nN={spec.actual_dataset_size:,}, axis={spec.points_per_axis}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for ax in axes[num_plots:]:
        ax.axis("off")

    fig.suptitle("Training set distributions on x-z plane", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_test_distribution_xz(
    *,
    test_points_cpu: torch.Tensor,
    y_star: Sequence[float],
    p_n: Sequence[float],
    save_path: Path,
    max_test_points: int = 12000,
) -> None:
    y_star_np = np.asarray(y_star, dtype=float)
    p_n_np = np.asarray(p_n, dtype=float)
    test_points = test_points_cpu.detach().cpu().numpy()
    if test_points.shape[0] > max_test_points:
        idx = np.linspace(0, test_points.shape[0] - 1, max_test_points).round().astype(int)
        test_points = test_points[idx]
    test_xz = test_points[:, [0, 2]]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(
        test_xz[:, 0],
        test_xz[:, 1],
        s=5,
        alpha=0.35,
        color="C1",
        label=f"held-out test ({test_points.shape[0]}/{test_points_cpu.shape[0]})",
        rasterized=True,
    )
    ax.scatter(p_n_np[0], p_n_np[2], marker="x", s=90, linewidths=2.0, color="C3", label=r"$p_n$")
    ax.scatter(y_star_np[0], y_star_np[2], marker="*", s=180, color="C2", label=r"$y^*$")

    set_equal_2d_axes(ax, np.vstack([test_xz, p_n_np[[0, 2]].reshape(1, 2), y_star_np[[0, 2]].reshape(1, 2)]))
    ax.set_title("Held-out test set x-z distribution\n(excludes all training-grid points)")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_and_test_distribution_xz_overview(
    *,
    grid_specs: Sequence[RegularGridSpec],
    test_points_cpu: torch.Tensor,
    y_star: Sequence[float],
    p_n: Sequence[float],
    save_path: Path,
    max_training_points_per_subplot: int = 12000,
    max_test_points: int = 12000,
) -> None:
    y_star_np = np.asarray(y_star, dtype=float)
    p_n_np = np.asarray(p_n, dtype=float)
    test_points = test_points_cpu.detach().cpu().numpy()
    if test_points.shape[0] > max_test_points:
        idx = np.linspace(0, test_points.shape[0] - 1, max_test_points).round().astype(int)
        test_points = test_points[idx]
    test_xz = test_points[:, [0, 2]]

    num_plots = len(grid_specs) + 1
    num_cols = min(4, num_plots)
    num_rows = math.ceil(num_plots / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5.2 * num_cols, 4.8 * num_rows))
    axes = np.asarray(axes).reshape(-1)

    for i, spec in enumerate(grid_specs):
        ax = axes[i]
        train_xz = sample_regular_grid_xz_projection_for_plot(
            grid_spec=spec,
            y_star=y_star_np,
            max_points=max_training_points_per_subplot,
        )
        total_projected = spec.points_per_axis * spec.points_per_axis
        ax.scatter(
            train_xz[:, 0],
            train_xz[:, 1],
            s=4,
            alpha=0.35,
            color="C0",
            label=f"x-z projection ({train_xz.shape[0]}/{total_projected}); each has {spec.points_per_axis} y-values",
            rasterized=True,
        )
        ax.scatter(p_n_np[0], p_n_np[2], marker="x", s=90, linewidths=2.0, color="C3", label=r"$p_n$")
        ax.scatter(y_star_np[0], y_star_np[2], marker="*", s=180, color="C2", label=r"$y^*$")
        set_equal_2d_axes(ax, np.vstack([train_xz, p_n_np[[0, 2]].reshape(1, 2), y_star_np[[0, 2]].reshape(1, 2)]))
        ax.set_title(f"Training x-z\nN={spec.actual_dataset_size:,}, axis={spec.points_per_axis}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    ax = axes[len(grid_specs)]
    ax.scatter(
        test_xz[:, 0],
        test_xz[:, 1],
        s=5,
        alpha=0.35,
        color="C1",
        label=f"held-out test ({test_points.shape[0]}/{test_points_cpu.shape[0]})",
        rasterized=True,
    )
    ax.scatter(p_n_np[0], p_n_np[2], marker="x", s=90, linewidths=2.0, color="C3", label=r"$p_n$")
    ax.scatter(y_star_np[0], y_star_np[2], marker="*", s=180, color="C2", label=r"$y^*$")
    set_equal_2d_axes(ax, np.vstack([test_xz, p_n_np[[0, 2]].reshape(1, 2), y_star_np[[0, 2]].reshape(1, 2)]))
    ax.set_title("Held-out test x-z\n(excludes all training-grid points)")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    for ax in axes[num_plots:]:
        ax.axis("off")

    fig.suptitle(
        "Training and held-out test set distributions on x-z plane\n"
        "Color code: training=C0, test=C1, p_n=C3, y*=C2",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 4. 主程序
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Only plot training/test set distributions, using x-z plane projection."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Training output directory containing dataset_scale_ablation_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <results-dir>/set_distribution_xz_only.",
    )
    parser.add_argument("--num-test", type=int, default=4096, help="Number of held-out random test points.")
    parser.add_argument("--test-seed", type=int, default=20260617)
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Test cube radius = training sampling_radius * radius_scale.",
    )
    parser.add_argument(
        "--max-training-points-per-subplot",
        type=int,
        default=12000,
        help="Maximum number of unique projected x-z training points shown in each subplot.",
    )
    parser.add_argument(
        "--max-test-points",
        type=int,
        default=12000,
        help="Maximum number of shown test points in plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = discover_results_dir(args.results_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_dir / "set_distribution_xz_only"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_files = find_experiment_files(results_dir)
    grid_specs = extract_unique_grid_specs(experiment_files)

    first_cfg = experiment_files[0].report.get("config", {})
    p_n_cpu = torch.tensor(first_cfg.get("p_n", [3.0, 4.0, 5.0]), dtype=torch.float64)
    v_n_cpu = torch.tensor(first_cfg.get("v_n", [0.5, -0.5, 0.0]), dtype=torch.float64)
    g = float(first_cfg.get("g", 9.8))
    dt = float(first_cfg.get("dt", 0.01))
    y_star_cpu = torch.tensor(
        first_cfg.get(
            "y_star",
            (p_n_cpu + dt * v_n_cpu - dt**2 * torch.tensor([0.0, 0.0, g], dtype=torch.float64)).tolist(),
        ),
        dtype=torch.float64,
    )
    sampling_radius = float(first_cfg.get("sampling_radius_per_axis", first_cfg.get("sampling_radius", 0.01)))

    test_points_cpu, test_metadata = build_fixed_heldout_test_set_excluding_training_grids(
        y_star=y_star_cpu,
        radius=sampling_radius,
        num_random=int(args.num_test),
        seed=int(args.test_seed),
        radius_scale=float(args.radius_scale),
        dtype=torch.float64,
        grid_specs=grid_specs,
    )

    plot_training_distribution_xz_overview(
        grid_specs=grid_specs,
        y_star=y_star_cpu.tolist(),
        p_n=p_n_cpu.tolist(),
        save_path=output_dir / "training_set_distribution_xz_overview.png",
        max_training_points_per_subplot=int(args.max_training_points_per_subplot),
    )
    plot_test_distribution_xz(
        test_points_cpu=test_points_cpu,
        y_star=y_star_cpu.tolist(),
        p_n=p_n_cpu.tolist(),
        save_path=output_dir / "test_set_distribution_xz.png",
        max_test_points=int(args.max_test_points),
    )
    plot_training_and_test_distribution_xz_overview(
        grid_specs=grid_specs,
        test_points_cpu=test_points_cpu,
        y_star=y_star_cpu.tolist(),
        p_n=p_n_cpu.tolist(),
        save_path=output_dir / "training_and_test_set_distribution_xz_overview.png",
        max_training_points_per_subplot=int(args.max_training_points_per_subplot),
        max_test_points=int(args.max_test_points),
    )

    meta = {
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "num_unique_training_grids": len(grid_specs),
        "grid_specs": [spec.__dict__ for spec in grid_specs],
        "test_metadata": test_metadata,
        "physics": {
            "p_n": p_n_cpu.tolist(),
            "v_n": v_n_cpu.tolist(),
            "y_star": y_star_cpu.tolist(),
            "dt": dt,
            "g": g,
            "sampling_radius": sampling_radius,
        },
    }
    with (output_dir / "set_distribution_xz_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("✅ 完成：仅绘制训练集和测试集分布图（x-z 平面）")
    print(f"results_dir = {results_dir}")
    print(f"output_dir  = {output_dir}")
    print(f"训练集总览图: {output_dir / 'training_set_distribution_xz_overview.png'}")
    print(f"测试集图: {output_dir / 'test_set_distribution_xz.png'}")
    print(f"训练+测试总览图: {output_dir / 'training_and_test_set_distribution_xz_overview.png'}")


if __name__ == "__main__":
    main()
