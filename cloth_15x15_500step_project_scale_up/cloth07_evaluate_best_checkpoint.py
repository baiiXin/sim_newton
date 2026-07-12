"""Evaluate the best scale-up checkpoint on long validation and grouped test rollouts."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from cloth03_training_pool import LearnedOptimizerMLP, ModelSpec
from cloth04_reference_free_validation import (
    FailureThresholds,
    ValidationResult,
    run_reference_free_validation,
    save_validation_result,
)
from scenario_catalogue import build_catalogues
from validation_protocol import ValidationProtocol


DEFAULT_ROOT = Path("cloth_15x15_scale_up_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--catalogue", choices=("c1", "c2", "c3"), default="c2")
    parser.add_argument("--activation", default="identity")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-update", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float64", "float32"), default="auto")
    parser.add_argument("--validation-frames", type=int, default=500)
    parser.add_argument("--test-frames", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, nargs="+", default=[1, 3, 10, 30])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--render-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-residual", type=float, default=1e12)
    parser.add_argument("--max-abs-position", type=float, default=1e4)
    parser.add_argument("--min-edge-ratio", type=float, default=1e-5)
    parser.add_argument("--max-edge-ratio", type=float, default=1e4)
    parser.add_argument("--max-constraint-error", type=float, default=1e-9)
    return parser.parse_args()


def run_directory(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir
    spec = ModelSpec(
        activation=args.activation,
        depth=args.depth,
        width=args.width,
        use_bias=args.use_bias,
    )
    return (
        args.root
        / "experiments"
        / f"train_{args.catalogue}"
        / spec.experiment_name
        / f"seed_{args.seed}"
    )


def checkpoint_path(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.checkpoint is not None and args.checkpoint_update is not None:
        raise ValueError("--checkpoint and --checkpoint-update cannot be used together")
    if args.checkpoint is not None:
        return args.checkpoint
    if args.checkpoint_update is not None:
        if args.checkpoint_update <= 0:
            raise ValueError("--checkpoint-update must be positive")
        return run_dir / "periodic" / f"checkpoint_update_{args.checkpoint_update:09d}.pt"
    return run_dir / "best_validation_model.pt"


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _safe(value) for key, value in row.items()} for row in rows])


def resolve_dtype(name: str, checkpoint: dict[str, Any]) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    configured = str(checkpoint.get("config", {}).get("dtype", "float64"))
    return torch.float32 if configured == "float32" else torch.float64


def finite_quantile(values: Sequence[float], q: float, default: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float(default)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(default)
    return float(np.quantile(finite, q))


def summarize_motion_rows(
    rows: Sequence[dict[str, Any]],
    *,
    rollout_frames: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    failed = [bool(row["failed"]) for row in rows]
    survival = [int(row["survival_frames"]) for row in rows]
    selection = [float(row["residual_ratio_selection"]) for row in rows]
    energy_fraction = [float(row["energy_increase_fraction"]) for row in rows]
    final_residual = [float(row["final_residual"]) for row in rows]
    minimum_edge = [float(row["minimum_edge_ratio"]) for row in rows]
    maximum_edge = [float(row["maximum_edge_ratio"]) for row in rows]
    constraint = [float(row["maximum_constraint_error"]) for row in rows]
    return {
        "motion_count": len(rows),
        "rollout_frames": int(rollout_frames),
        "failed_motion_count": int(sum(failed)),
        "survival_rate": float(sum(not item for item in failed) / len(rows)),
        "survival_frame_p05": float(np.quantile(np.asarray(survival), 0.05)),
        "survival_frame_median": float(np.quantile(np.asarray(survival), 0.50)),
        "residual_ratio_p95": finite_quantile(selection, 0.95, float("inf")),
        "energy_increase_fraction": float(np.mean(energy_fraction)),
        "final_residual_p95": finite_quantile(final_residual, 0.95, float("inf")),
        "final_residual_max": finite_quantile(final_residual, 1.00, float("inf")),
        "minimum_edge_ratio": finite_quantile(minimum_edge, 0.00, 0.0),
        "maximum_edge_ratio": finite_quantile(maximum_edge, 1.00, float("inf")),
        "maximum_constraint_error": finite_quantile(constraint, 1.00, float("inf")),
    }


def grouped_test_summaries(
    result: ValidationResult,
    *,
    inner_steps: int,
) -> list[dict[str, Any]]:
    groups = sorted({str(row["scenario_group"]) for row in result.per_motion})
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "split": "test",
            "group": "all",
            "inner_steps": int(inner_steps),
            **summarize_motion_rows(
                result.per_motion,
                rollout_frames=int(result.summary["rollout_frames"]),
            ),
        }
    )
    for group in groups:
        selected = [
            row for row in result.per_motion if str(row["scenario_group"]) == group
        ]
        rows.append(
            {
                "split": "test",
                "group": group,
                "inner_steps": int(inner_steps),
                **summarize_motion_rows(
                    selected,
                    rollout_frames=int(result.summary["rollout_frames"]),
                ),
            }
        )
    return rows


def plot_group_summary(rows: Sequence[dict[str, Any]], figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted({str(row["group"]) for row in rows if row["group"] != "all"})
    for metric, ylabel, log_scale in (
        ("failed_motion_count", "失败 motion 数", False),
        ("residual_ratio_p95", "Residual ratio p95", True),
        ("energy_increase_fraction", "能量上升比例", False),
    ):
        plt.figure(figsize=(8.0, 5.0))
        for group in groups:
            selected = sorted(
                (row for row in rows if row["group"] == group),
                key=lambda row: int(row["inner_steps"]),
            )
            plt.plot(
                [int(row["inner_steps"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                label=group,
            )
        if log_scale:
            plt.yscale("log")
        plt.xlabel("每个物理帧的网络迭代次数 K")
        plt.ylabel(ylabel)
        plt.title(f"测试集分组对比：{ylabel}")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figure_dir / f"test_group_{metric}.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    if args.validation_frames <= 0 or args.test_frames <= 0:
        raise ValueError("validation/test frames must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not args.inner_steps or any(value <= 0 for value in args.inner_steps):
        raise ValueError("inner-steps must contain positive integers")

    run_dir = run_directory(args)
    selected_checkpoint = checkpoint_path(args, run_dir)
    if not selected_checkpoint.exists():
        raise FileNotFoundError(selected_checkpoint)
    checkpoint = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    dtype = resolve_dtype(args.dtype, checkpoint)
    spec = ModelSpec(**checkpoint["model_spec"])
    model = LearnedOptimizerMLP(
        full_state_dim=15 * 15 * 3,
        model_spec=spec,
        dtype=dtype,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    catalogues = build_catalogues()
    validation = tuple(catalogues["validation_128"])
    test = tuple(catalogues["test_256"])
    output_dir = args.output_dir or run_dir / "final_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    update_count = int(checkpoint.get("update_count", 0))
    thresholds = FailureThresholds(
        max_residual=args.max_residual,
        max_abs_position=args.max_abs_position,
        min_edge_ratio=args.min_edge_ratio,
        max_edge_ratio=args.max_edge_ratio,
        max_constraint_error=args.max_constraint_error,
    )

    summary_rows: list[dict[str, Any]] = []
    test_group_rows: list[dict[str, Any]] = []
    for inner_steps in args.inner_steps:
        validation_protocol = ValidationProtocol(
            id=f"final_validation_k{inner_steps}",
            motion_count=len(validation),
            rollout_frames=args.validation_frames,
            inner_steps=int(inner_steps),
            interval_updates=0,
            selects_checkpoint=False,
        )
        validation_result = run_reference_free_validation(
            model=model,
            scenarios=validation,
            protocol=validation_protocol,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            thresholds=thresholds,
        )
        save_validation_result(
            result=validation_result,
            output_root=output_dir,
            update_count=update_count,
            wall_clock_seconds=0.0,
            render_plots=args.render_plots,
        )
        summary_rows.append(
            {
                "split": "validation",
                "group": "all",
                "inner_steps": int(inner_steps),
                **validation_result.summary,
            }
        )

        test_protocol = ValidationProtocol(
            id=f"test_all_k{inner_steps}",
            motion_count=len(test),
            rollout_frames=args.test_frames,
            inner_steps=int(inner_steps),
            interval_updates=0,
            selects_checkpoint=False,
        )
        test_result = run_reference_free_validation(
            model=model,
            scenarios=test,
            protocol=test_protocol,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            thresholds=thresholds,
        )
        save_validation_result(
            result=test_result,
            output_root=output_dir,
            update_count=update_count,
            wall_clock_seconds=0.0,
            render_plots=args.render_plots,
        )
        grouped = grouped_test_summaries(test_result, inner_steps=int(inner_steps))
        test_group_rows.extend(grouped)
        summary_rows.extend(row for row in grouped if row["group"] == "all")

    write_csv(output_dir / "summary.csv", summary_rows)
    write_json(output_dir / "summary.json", summary_rows)
    write_csv(output_dir / "test_group_summary.csv", test_group_rows)
    write_json(output_dir / "test_group_summary.json", test_group_rows)
    if args.render_plots:
        plot_group_summary(test_group_rows, output_dir / "figures")
    write_json(
        output_dir / "evaluation_manifest.json",
        {
            "checkpoint": str(selected_checkpoint),
            "checkpoint_update": update_count,
            "model_spec": asdict(spec),
            "requested_model_spec": asdict(
                ModelSpec(
                    activation=args.activation,
                    depth=args.depth,
                    width=args.width,
                    use_bias=args.use_bias,
                )
            ),
            "run_directory": str(run_dir),
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "validation_scenarios": len(validation),
            "test_scenarios": len(test),
            "validation_frames": args.validation_frames,
            "test_frames": args.test_frames,
            "inner_steps": list(args.inner_steps),
            "batch_size": args.batch_size,
            "reference_free": True,
            "checkpoint_selection_changed": False,
        },
    )
    print(f"最终评估完成：{output_dir}")


if __name__ == "__main__":
    main()
