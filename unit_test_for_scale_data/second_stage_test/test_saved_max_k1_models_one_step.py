
"""
已训练 max_K=1 模型的独立一步测试脚本
======================================

用途
----
加载已经训练完成的模型参数，对每个实验只执行一次 MLP 更新：

    y^(1) = p_n + MLP(p_n, history, params)

并计算：
1. 初始 p_n 的 energy gap 与 stationarity residual；
2. MLP 更新 1 次后的 energy gap 与 residual；
3. Newton 更新 1 次后的 energy gap 与 residual；
4. MLP 一步结果与精确解 y_star 的位置误差；
5. MLP 一步更新与 Newton 一步更新的差异。

本脚本不会训练模型，也不会执行 50 步测试，不会覆盖原训练结果。

预期目录结构
------------
<results_dir>/
    dataset_scale_ablation_summary.json
    <experiment_name>/
        optimization_report.json
        mlp_optimizer_state_dict.pt

输出
----
默认写入：

    <results_dir>/one_step_test_only/

包括：
    one_step_test_summary.json
    one_step_dataset_scale_summary.png

以及每个实验的：
    <experiment_name>/
        one_step_test_report.json
        one_step_residual_comparison.png
        one_step_trajectory_3d.png

运行示例
--------
python test_saved_max_k1_models_one_step.py \
    --results-dir /path/to/training/output \
    --device cuda:0
"""

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
import torch.nn as nn


PLOT_FLOOR = 1e-14


# ============================================================
# 1. 网络与物理问题
# ============================================================


class MLPOptimizer(nn.Module):
    """与训练脚本一致的 12 -> 32 -> 32 -> 3 MLP optimizer。"""

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        use_input_normalization: bool,
        use_output_dt_scaling: bool,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
    ) -> None:
        super().__init__()

        self.use_input_normalization = bool(use_input_normalization)
        self.use_output_dt_scaling = bool(use_output_dt_scaling)

        self.net = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        self.register_buffer(
            "input_mean",
            input_mean.clone().detach().to(dtype=dtype),
        )
        self.register_buffer(
            "input_std",
            input_std.clone().detach().to(dtype=dtype),
        )
        self.to(dtype=dtype)

    def forward(
        self,
        y: torch.Tensor,
        history: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        inp = torch.cat([y, history, params], dim=-1)

        if self.use_input_normalization:
            inp = (inp - self.input_mean) / self.input_std

        delta = self.net(inp)

        if self.use_output_dt_scaling:
            delta = params[2] * delta

        return delta


def variational_energy(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> torch.Tensor:
    residual = y - p_n - dt * v_n
    kinetic_term = (m / (2.0 * dt**2)) * torch.sum(residual**2, dim=-1)
    potential_term = m * g * y[..., 2]
    return kinetic_term + potential_term


def stationarity_residual(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> torch.Tensor:
    residual = (m / dt**2) * (y - p_n - dt * v_n)
    gravity = torch.zeros_like(residual)
    gravity[..., 2] = m * g
    return residual + gravity


def stationarity_residual_norm(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> torch.Tensor:
    return torch.linalg.vector_norm(
        stationarity_residual(y, p_n, v_n, m, g, dt),
        dim=-1,
    )


def newton_direction(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> torch.Tensor:
    grad = stationarity_residual(y, p_n, v_n, m, g, dt)
    return -(dt**2 / m) * grad


# ============================================================
# 2. 文件与模型加载
# ============================================================


@dataclass(frozen=True)
class ExperimentFile:
    experiment_dir: Path
    report_path: Path
    model_path: Path
    report: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def safe_torch_load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")

    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a state_dict.")
    return state


def infer_state_dtype(state_dict: dict[str, torch.Tensor]) -> torch.dtype:
    for value in state_dict.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.dtype
    raise TypeError("Cannot infer floating dtype from state_dict.")


def parse_dataset_size_from_name(name: str) -> int | None:
    match = re.search(r"num_samples_(\d+)", name)
    return int(match.group(1)) if match else None


def find_experiment_files(results_dir: Path) -> list[ExperimentFile]:
    experiment_files: list[ExperimentFile] = []

    for report_path in sorted(results_dir.glob("*/optimization_report.json")):
        experiment_dir = report_path.parent
        model_path = experiment_dir / "mlp_optimizer_state_dict.pt"

        if not model_path.exists():
            print(f"[skip] missing model: {model_path}")
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
            f"No trained experiment found under {results_dir}. "
            "Expected */optimization_report.json and "
            "*/mlp_optimizer_state_dict.pt."
        )

    return experiment_files


def instantiate_model(
    experiment: ExperimentFile,
    device: torch.device,
) -> tuple[MLPOptimizer, torch.dtype, dict[str, Any]]:
    state_dict = safe_torch_load_state_dict(experiment.model_path)
    dtype = infer_state_dtype(state_dict)
    config = experiment.report.get("config", {})

    input_mean = torch.tensor(
        config.get("input_mean", [0.0] * 12),
        dtype=dtype,
    )
    input_std = torch.tensor(
        config.get("input_std", [1.0] * 12),
        dtype=dtype,
    )

    model = MLPOptimizer(
        dtype=dtype,
        use_input_normalization=config.get(
            "use_input_normalization",
            True,
        ),
        use_output_dt_scaling=config.get(
            "use_output_dt_scaling",
            True,
        ),
        input_mean=input_mean,
        input_std=input_std,
    ).to(device=device, dtype=dtype)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, dtype, config


# ============================================================
# 3. 严格的一步测试
# ============================================================


def point_metrics(
    y: torch.Tensor,
    *,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    y_star: torch.Tensor,
    e_star: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> dict[str, Any]:
    energy = variational_energy(y, p_n, v_n, m, g, dt)
    residual = stationarity_residual_norm(y, p_n, v_n, m, g, dt)

    return {
        "y": [float(x) for x in y.detach().cpu().tolist()],
        "energy": float(energy.detach().cpu().item()),
        "energy_gap": float((energy - e_star).detach().cpu().item()),
        "residual_norm": float(residual.detach().cpu().item()),
        "position_error_to_y_star": float(
            torch.linalg.vector_norm(y - y_star).detach().cpu().item()
        ),
    }


@torch.no_grad()
def evaluate_exactly_one_step(
    *,
    model: MLPOptimizer,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    p_n = torch.tensor(
        config.get("p_n", [3.0, 4.0, 5.0]),
        device=device,
        dtype=dtype,
    )
    v_n = torch.tensor(
        config.get("v_n", [0.5, -0.5, 0.0]),
        device=device,
        dtype=dtype,
    )

    m = float(config.get("m", 1.0))
    g = float(config.get("g", 9.8))
    dt = float(config.get("dt", 0.01))

    default_y_star = (
        p_n
        + dt * v_n
        - dt**2
        * torch.tensor([0.0, 0.0, g], device=device, dtype=dtype)
    )
    y_star = torch.tensor(
        config.get(
            "y_star",
            [float(x) for x in default_y_star.detach().cpu().tolist()],
        ),
        device=device,
        dtype=dtype,
    )

    history = torch.cat([p_n, v_n], dim=0)
    params = torch.tensor([m, g, dt], device=device, dtype=dtype)
    e_star = variational_energy(y_star, p_n, v_n, m, g, dt)

    # 初始点。
    y0 = p_n.clone()

    # 严格只调用一次模型。
    mlp_delta = model(y0, history, params)
    y1_mlp = y0 + mlp_delta

    # Newton 也只更新一次。
    newton_delta = newton_direction(y0, p_n, v_n, m, g, dt)
    y1_newton = y0 + newton_delta

    initial = point_metrics(
        y0,
        p_n=p_n,
        v_n=v_n,
        y_star=y_star,
        e_star=e_star,
        m=m,
        g=g,
        dt=dt,
    )
    mlp_step_1 = point_metrics(
        y1_mlp,
        p_n=p_n,
        v_n=v_n,
        y_star=y_star,
        e_star=e_star,
        m=m,
        g=g,
        dt=dt,
    )
    newton_step_1 = point_metrics(
        y1_newton,
        p_n=p_n,
        v_n=v_n,
        y_star=y_star,
        e_star=e_star,
        m=m,
        g=g,
        dt=dt,
    )

    return {
        "test_definition": {
            "initial_state": "p_n",
            "mlp_forward_calls": 1,
            "mlp_update_steps": 1,
            "newton_update_steps": 1,
            "formula": "y_1 = p_n + model(p_n, history, params)",
        },
        "physics": {
            "p_n": [float(x) for x in p_n.detach().cpu().tolist()],
            "v_n": [float(x) for x in v_n.detach().cpu().tolist()],
            "m": m,
            "g": g,
            "dt": dt,
            "y_star": [float(x) for x in y_star.detach().cpu().tolist()],
            "E_star": float(e_star.detach().cpu().item()),
        },
        "initial": initial,
        "mlp_step_1": mlp_step_1,
        "newton_step_1": newton_step_1,
        "updates": {
            "mlp_delta": [
                float(x) for x in mlp_delta.detach().cpu().tolist()
            ],
            "newton_delta": [
                float(x) for x in newton_delta.detach().cpu().tolist()
            ],
            "mlp_delta_norm": float(
                torch.linalg.vector_norm(mlp_delta).detach().cpu().item()
            ),
            "newton_delta_norm": float(
                torch.linalg.vector_norm(newton_delta).detach().cpu().item()
            ),
            "mlp_vs_newton_delta_error": float(
                torch.linalg.vector_norm(
                    mlp_delta - newton_delta
                ).detach().cpu().item()
            ),
            "mlp_vs_newton_final_position_error": float(
                torch.linalg.vector_norm(
                    y1_mlp - y1_newton
                ).detach().cpu().item()
            ),
        },
    }


# ============================================================
# 4. 绘图
# ============================================================


def finite_plot_value(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return float("nan")
    return max(value, PLOT_FLOOR)


def plot_one_step_residual_comparison(
    *,
    result: dict[str, Any],
    save_path: Path,
    title: str,
) -> None:
    initial_residual = finite_plot_value(
        result["initial"]["residual_norm"]
    )
    mlp_residual = finite_plot_value(
        result["mlp_step_1"]["residual_norm"]
    )
    newton_residual = finite_plot_value(
        result["newton_step_1"]["residual_norm"]
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    ax.plot(
        [0, 1],
        [initial_residual, mlp_residual],
        marker="o",
        label="MLP: exactly 1 update",
    )
    ax.plot(
        [0, 1],
        [initial_residual, newton_residual],
        marker="s",
        linestyle="--",
        label="Newton: exactly 1 update",
    )

    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Stationarity residual $\|\nabla E(y)\|_2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]

    center = points.mean(axis=0)
    radius = max(float(np.ptp(points, axis=0).max()) / 2.0, 1e-7)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_one_step_trajectory_3d(
    *,
    result: dict[str, Any],
    save_path: Path,
    title: str,
) -> None:
    y0 = np.asarray(result["initial"]["y"], dtype=float)
    y1_mlp = np.asarray(result["mlp_step_1"]["y"], dtype=float)
    y1_newton = np.asarray(result["newton_step_1"]["y"], dtype=float)
    y_star = np.asarray(result["physics"]["y_star"], dtype=float)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        [y0[0], y1_mlp[0]],
        [y0[1], y1_mlp[1]],
        [y0[2], y1_mlp[2]],
        "-o",
        label="MLP one-step",
    )
    ax.plot(
        [y0[0], y1_newton[0]],
        [y0[1], y1_newton[1]],
        [y0[2], y1_newton[2]],
        "--s",
        label="Newton one-step",
    )

    ax.scatter(
        y0[0],
        y0[1],
        y0[2],
        marker="x",
        s=130,
        linewidths=2.0,
        label=r"Initial $p_n$",
    )
    ax.scatter(
        y_star[0],
        y_star[1],
        y_star[2],
        marker="*",
        s=280,
        label=r"Exact solution $y^*$",
    )

    set_equal_3d_axes(
        ax,
        np.vstack([y0, y1_mlp, y1_newton, y_star]),
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def optimizer_key(record: dict[str, Any]) -> tuple[str, float]:
    return (
        str(record["optimizer_name"]).lower(),
        float(record["learning_rate"]),
    )


def optimizer_label(key: tuple[str, float]) -> str:
    name, lr = key
    return f"{name.upper()} lr={lr:.0e}"


def plot_dataset_scale_summary(
    records: Sequence[dict[str, Any]],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    keys = sorted(
        {optimizer_key(record) for record in records},
        key=lambda item: (item[0], -item[1]),
    )

    for key in keys:
        selected = sorted(
            [
                record
                for record in records
                if optimizer_key(record) == key
            ],
            key=lambda record: int(record["dataset_size"]),
        )

        sizes = [int(record["dataset_size"]) for record in selected]
        residuals = [
            finite_plot_value(record["one_step_residual"])
            for record in selected
        ]
        gaps = [
            finite_plot_value(record["one_step_energy_gap"])
            for record in selected
        ]
        position_errors = [
            finite_plot_value(record["one_step_position_error"])
            for record in selected
        ]

        label = optimizer_label(key)
        axes[0].plot(sizes, residuals, marker="o", label=label)
        axes[1].plot(sizes, gaps, marker="s", label=label)
        axes[2].plot(
            sizes,
            position_errors,
            marker="^",
            label=label,
        )

    axes[0].set_title(r"One-step residual from $p_n$")
    axes[0].set_ylabel(r"$\|\nabla E(y^{(1)})\|_2$")

    axes[1].set_title(r"One-step energy gap from $p_n$")
    axes[1].set_ylabel(r"$E(y^{(1)})-E(y^*)$")

    axes[2].set_title(r"One-step position error from $p_n$")
    axes[2].set_ylabel(r"$\|y^{(1)}-y^*\|_2$")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of training initial states")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Saved max_K=1 models: exactly one MLP update during testing",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. 主程序
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load already trained max_K=1 models and test exactly one "
            "MLP update from p_n."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help=(
            "Training output directory containing experiment "
            "subdirectories."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Default: "
            "<results-dir>/one_step_test_only."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--allow-non-max-k-1",
        action="store_true",
        help=(
            "Also test models whose saved config does not declare max_K=1."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(
            f"results_dir does not exist: {results_dir}"
        )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_dir / "one_step_test_only"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    experiments = find_experiment_files(results_dir)

    print(f"results_dir = {results_dir}")
    print(f"output_dir  = {output_dir}")
    print(f"device      = {device}")
    print(f"found experiments = {len(experiments)}")

    records: list[dict[str, Any]] = []

    for experiment in experiments:
        saved_config = experiment.report.get("config", {})
        saved_max_k = int(
            saved_config.get(
                "max_K",
                saved_config.get("max_k", -1),
            )
        )

        if saved_max_k != 1 and not args.allow_non_max_k_1:
            print(
                f"[skip] {experiment.experiment_dir.name}: "
                f"saved max_K={saved_max_k}, not 1"
            )
            continue

        print(f"\n[test one step] {experiment.experiment_dir.name}")

        model, dtype, config = instantiate_model(
            experiment,
            device,
        )

        result = evaluate_exactly_one_step(
            model=model,
            config=config,
            device=device,
            dtype=dtype,
        )

        dataset_size = int(
            config.get(
                "actual_dataset_size",
                parse_dataset_size_from_name(
                    experiment.experiment_dir.name
                )
                or -1,
            )
        )
        optimizer_name = str(
            config.get("optimizer_name", "unknown")
        )
        learning_rate = float(
            config.get("learning_rate", float("nan"))
        )

        experiment_output_dir = (
            output_dir / experiment.experiment_dir.name
        )
        experiment_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = {
            "experiment_name": config.get(
                "experiment_name",
                experiment.experiment_dir.name,
            ),
            "experiment_dir": str(experiment.experiment_dir),
            "model_path": str(experiment.model_path),
            "saved_max_K": saved_max_k,
            "optimizer_name": optimizer_name,
            "learning_rate": learning_rate,
            "dataset_size": dataset_size,
            "dtype": str(dtype),
            "device": str(device),
            "one_step_test": result,
        }

        report_path = (
            experiment_output_dir / "one_step_test_report.json"
        )
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(
                make_json_safe(report),
                f,
                indent=2,
                ensure_ascii=False,
            )

        title_prefix = (
            f"{optimizer_name.upper()} lr={learning_rate:.0e}, "
            f"N={dataset_size:,}"
        )
        plot_one_step_residual_comparison(
            result=result,
            save_path=(
                experiment_output_dir
                / "one_step_residual_comparison.png"
            ),
            title=title_prefix + "\nExactly one test iteration",
        )
        plot_one_step_trajectory_3d(
            result=result,
            save_path=(
                experiment_output_dir
                / "one_step_trajectory_3d.png"
            ),
            title=title_prefix + "\nExactly one test iteration",
        )

        record = {
            "experiment_name": report["experiment_name"],
            "experiment_dir": report["experiment_dir"],
            "optimizer_name": optimizer_name,
            "learning_rate": learning_rate,
            "dataset_size": dataset_size,
            "saved_max_K": saved_max_k,
            "initial_residual": result["initial"]["residual_norm"],
            "initial_energy_gap": result["initial"]["energy_gap"],
            "one_step_residual": (
                result["mlp_step_1"]["residual_norm"]
            ),
            "one_step_energy_gap": (
                result["mlp_step_1"]["energy_gap"]
            ),
            "one_step_position_error": (
                result["mlp_step_1"][
                    "position_error_to_y_star"
                ]
            ),
            "newton_one_step_residual": (
                result["newton_step_1"]["residual_norm"]
            ),
            "newton_one_step_energy_gap": (
                result["newton_step_1"]["energy_gap"]
            ),
            "mlp_vs_newton_delta_error": (
                result["updates"]["mlp_vs_newton_delta_error"]
            ),
            "mlp_vs_newton_final_position_error": (
                result["updates"][
                    "mlp_vs_newton_final_position_error"
                ]
            ),
            "report_path": str(report_path),
        }
        records.append(record)

        print(
            "  one-step gap={:.6e}, residual={:.6e}, "
            "position error={:.6e}".format(
                record["one_step_energy_gap"],
                record["one_step_residual"],
                record["one_step_position_error"],
            )
        )

    if not records:
        raise RuntimeError(
            "No max_K=1 experiments were tested. "
            "Use --allow-non-max-k-1 only when intentionally testing "
            "other saved models."
        )

    records = sorted(
        records,
        key=lambda record: (
            optimizer_key(record),
            int(record["dataset_size"]),
        ),
    )

    summary = {
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "test_definition": {
            "initial_state": "p_n",
            "mlp_forward_calls_per_model": 1,
            "mlp_update_steps_per_model": 1,
            "newton_update_steps_per_model": 1,
            "no_training_performed": True,
            "original_training_results_are_not_overwritten": True,
        },
        "experiments": records,
    }

    summary_path = output_dir / "one_step_test_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            make_json_safe(summary),
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_plot_path = (
        output_dir / "one_step_dataset_scale_summary.png"
    )
    plot_dataset_scale_summary(records, summary_plot_path)

    print("\n✅ 一步测试完成")
    print(f"汇总 JSON: {summary_path}")
    print(f"汇总图: {summary_plot_path}")


if __name__ == "__main__":
    main()
