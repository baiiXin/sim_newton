"""Evaluate initial-point-count ablation models with the dataset-style test evaluator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_POOL_DIR = REPO_ROOT / "unit_test_for_spring" / "multi_motion_cloth" / "training_pool"
if str(TRAINING_POOL_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_POOL_DIR))

from evaluate_pool_vs_full_dataset_tests import (  # noqa: E402
    DEFAULT_DATASETS,
    DEFAULT_REPORT_STEPS,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    DEFAULT_DEVICE,
    ModelSpec,
    evaluate_model_on_dataset,
    load_dataset,
    load_json,
    load_model_from_dir,
    load_physical,
    plot_residual_curves,
    save_json,
    summary_row,
    write_csv,
)


DEFAULT_POINT_GROUPS = ("points_0001", "points_0008", "points_0032", "points_0064", "points_0128", "points_1024")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cloth_5x5_initial_sample_ablation models on the original dataset-style tests."
    )
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "cloth_5x5_500step_project" / "cloth_5x5_500step_pipeline")
    parser.add_argument("--ablation-root", type=Path, default=REPO_ROOT / "cloth_5x5_500step_project" / "cloth_5x5_initial_sample_ablation")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "unit_test_for_spring" / "multi_motion_cloth" / "scale_down_de_initial_points")
    parser.add_argument("--point-groups", nargs="+", default=list(DEFAULT_POINT_GROUPS))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--activation", type=str, default="identity")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--report-steps", type=int, nargs="+", default=list(DEFAULT_REPORT_STEPS))
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-curves", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    source_root = args.source_root.resolve()
    ablation_root = args.ablation_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    spec = ModelSpec(
        activation=str(args.activation),
        depth=int(args.depth),
        width=int(args.width),
        use_bias=bool(args.use_bias),
    )
    physical = load_physical(source_root)
    datasets = {name: load_dataset(source_root, name) for name in args.datasets}

    save_json(
        {
            "source_root": str(source_root),
            "ablation_root": str(ablation_root),
            "output_root": str(output_root),
            "point_groups": list(args.point_groups),
            "datasets": list(args.datasets),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "report_steps": list(args.report_steps),
            "model_spec": asdict(spec),
            "dataset_source_confirmation": (
                "Reuses cloth_5x5_500step_pipeline/data/datasets. "
                "Default test datasets are validation, seen_extrap, unseen_id, and ood."
            ),
        },
        output_root / "run_config.json",
    )

    rows: list[dict] = []
    plot_records: list[dict] = []
    all_metrics: dict[str, dict] = {}

    for point_group in args.point_groups:
        model_dir = ablation_root / point_group / "models" / spec.experiment_name
        try:
            model, model_info = load_model_from_dir(model_dir, device, args.residual_length_scale)
        except FileNotFoundError as exc:
            print(f"skip {point_group} {spec.experiment_name}: {exc}")
            continue

        solver_name = f"{point_group}_{spec.activation}_d{spec.depth:02d}_w{spec.width:03d}"
        for dataset_name, dataset in datasets.items():
            output_dir = output_root / point_group / spec.experiment_name / dataset_name
            metrics_path = output_dir / "metrics.json"
            curves_path = output_dir / "curves.pt"
            if metrics_path.exists() and (not args.overwrite or args.plot_only):
                metrics = load_json(metrics_path)
                print(f"reuse {solver_name} on {dataset_name}")
                rows.append(summary_row(point_group, spec, dataset_name, metrics))
                plot_records.append(
                    {
                        "group": point_group,
                        "activation": spec.activation,
                        "dataset": dataset_name,
                        "residual_mean_by_step": metrics["residual_mean_by_step"],
                    }
                )
                all_metrics[f"{point_group}/{spec.experiment_name}/{dataset_name}"] = metrics
                continue
            if args.plot_only:
                print(f"skip missing metrics for {solver_name} on {dataset_name}: {metrics_path}")
                continue

            print(f"evaluating {solver_name} on {dataset_name} ({len(dataset['initial_y'])} points)")
            metrics, curves = evaluate_model_on_dataset(
                model=model,
                model_info=model_info,
                solver_name=solver_name,
                dataset_name=dataset_name,
                dataset_cpu=dataset,
                physical=physical,
                steps=args.steps,
                batch_size=args.batch_size,
                report_steps=args.report_steps,
                device=device,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            save_json(metrics, metrics_path)
            if not args.skip_curves:
                torch.save(
                    {
                        "solver": solver_name,
                        "dataset": dataset_name,
                        "metrics": {name: torch.from_numpy(values) for name, values in curves.items()},
                    },
                    curves_path,
                )
            rows.append(summary_row(point_group, spec, dataset_name, metrics))
            plot_records.append(
                {
                    "group": point_group,
                    "activation": spec.activation,
                    "dataset": dataset_name,
                    "residual_mean_by_step": metrics["residual_mean_by_step"],
                }
            )
            all_metrics[f"{point_group}/{spec.experiment_name}/{dataset_name}"] = metrics

    write_csv(rows, output_root / "summary_metrics.csv")
    save_json({"records": all_metrics}, output_root / "all_metrics.json")
    plot_residual_curves(plot_records, output_root)
    print(f"wrote summary to {output_root / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
