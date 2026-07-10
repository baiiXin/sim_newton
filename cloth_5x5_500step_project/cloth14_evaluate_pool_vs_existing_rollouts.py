"""Script 14: compare existing 500-step-trained models with pool-trained models.

This script reuses the rollout semantics from cloth12:
- every frame starts from the solver's own propagated physical state,
- y^(0)=x_n,
- each frame runs 50 learned iterations by default,
- residual_by_frame_and_iteration has shape [rollout_length, inner_steps + 1].

Default comparison lines:
- full_500step_<activation>: models trained by cloth05
- pool_<activation>: models trained by cloth13
Optional:
- points_0032_<activation>: initial-point ablation model trained by cloth11
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from cloth03_solvers_and_models import (
    ACTIVATION_NAMES,
    DEFAULT_DEVICE,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    HIDDEN_DEPTHS,
    HIDDEN_WIDTHS,
    MLPOptimizer,
    ModelSpec,
)

from cloth12_evaluate_initial_point_ablation_rollouts import (
    compute_reference_endpoint_record,
    load_baseline_params,
    load_physical,
    load_reference_states,
    plot_motion,
    reference_for_motion,
    run_solver_rollout,
    solve_baseline_frame,
    solve_model_frame,
    summarize_record,
    write_summary_csv,
)

DEFAULT_MOTIONS = tuple(range(20, 32))
DEFAULT_ROLLOUT_LENGTH = 500
DEFAULT_INNER_STEPS = 50


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_generic_model(
    model_dir: Path,
    device: torch.device,
    residual_length_scale: float,
) -> tuple[MLPOptimizer, dict[str, Any]]:
    checkpoint_path = model_dir / "best_validation_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    spec_data = checkpoint["model_spec"]
    model_spec = ModelSpec(
        activation=str(spec_data["activation"]),
        depth=int(spec_data["depth"]),
        width=int(spec_data["width"]),
        use_bias=bool(spec_data["use_bias"]),
    )
    config = checkpoint.get("config", {})
    scale = float(config.get("residual_length_scale", residual_length_scale))
    model = MLPOptimizer(scale, model_spec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "update_count": int(checkpoint.get("update_count", -1)),
        "best_validation_max": float(checkpoint.get("best_validation_max", float("inf"))),
        "model_spec": asdict(model_spec),
        "residual_length_scale": scale,
    }


def make_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs = [
        ModelSpec(activation=a, depth=int(d), width=int(w), use_bias=bool(args.use_bias))
        for a in args.activations
        for d in args.depths
        for w in args.widths
    ]
    return [specs[int(args.config_index)]] if args.config_index is not None else specs


def solver_name(prefix: str, model_spec: ModelSpec) -> str:
    return f"{prefix}_{model_spec.activation}"


def run_model_line(
    *,
    name: str,
    model_dir: Path,
    device: torch.device,
    residual_length_scale: float,
    reference: dict[str, torch.Tensor],
    physical,
    motion_index: int,
    rollout_length: int,
    inner_steps: int,
    output_path: Path,
    overwrite: bool,
    checkpoint_every: int,
) -> dict[str, Any]:
    model, info = load_generic_model(model_dir, device, residual_length_scale)

    def solve_frame(**kwargs):
        return solve_model_frame(model=model, **kwargs)

    return run_solver_rollout(
        solver_name=name,
        solver_info=info,
        solve_frame=solve_frame,
        reference=reference,
        physical=physical,
        device=device,
        motion_index=motion_index,
        rollout_length=rollout_length,
        inner_steps=inner_steps,
        output_path=output_path,
        overwrite=overwrite,
        checkpoint_every=checkpoint_every,
    )


def run_baseline_line(
    *,
    method: str,
    source_root: Path,
    device: torch.device,
    reference: dict[str, torch.Tensor],
    physical,
    motion_index: int,
    rollout_length: int,
    inner_steps: int,
    output_path: Path,
    overwrite: bool,
    checkpoint_every: int,
) -> dict[str, Any]:
    params = load_baseline_params(source_root, method)

    def solve_frame(**kwargs):
        return solve_baseline_frame(method=method, params=params, **kwargs)

    return run_solver_rollout(
        solver_name=f"baseline_{method}",
        solver_info={"method": method, "params": params},
        solve_frame=solve_frame,
        reference=reference,
        physical=physical,
        device=device,
        motion_index=motion_index,
        rollout_length=rollout_length,
        inner_steps=inner_steps,
        output_path=output_path,
        overwrite=overwrite,
        checkpoint_every=checkpoint_every,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full-dataset and Metamizer-pool models by continuous rollout.")
    parser.add_argument("--source-root", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--pool-root", type=Path, default=Path("cloth_5x5_metamizer_pool_training"))
    parser.add_argument("--ablation-root", type=Path, default=Path("cloth_5x5_initial_sample_ablation"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--motion-indices", type=int, nargs="+", default=list(DEFAULT_MOTIONS))
    parser.add_argument("--rollout-length", type=int, default=DEFAULT_ROLLOUT_LENGTH)
    parser.add_argument("--inner-steps", type=int, default=DEFAULT_INNER_STEPS)
    parser.add_argument("--activations", nargs="+", default=list(ACTIVATION_NAMES))
    parser.add_argument("--depths", type=int, nargs="+", default=list(HIDDEN_DEPTHS))
    parser.add_argument("--widths", type=int, nargs="+", default=list(HIDDEN_WIDTHS))
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--config-index", type=int, default=None)
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--include-full", action="store_true", default=True)
    parser.add_argument("--no-full", action="store_false", dest="include_full")
    parser.add_argument("--include-pool", action="store_true", default=True)
    parser.add_argument("--no-pool", action="store_false", dest="include_pool")
    parser.add_argument("--include-points-0032", action="store_true")
    parser.add_argument("--baselines", nargs="*", default=[])
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    output_root = args.output_root or (args.pool_root / "rollout_evaluation")
    output_root.mkdir(parents=True, exist_ok=True)

    physical = load_physical(args.source_root)
    reference_states = load_reference_states(args.source_root)
    specs = make_model_specs(args)

    all_rows: list[dict[str, Any]] = []
    for motion_index in args.motion_indices:
        motion_dir = output_root / f"motion_{motion_index:03d}"
        motion_dir.mkdir(parents=True, exist_ok=True)
        reference = reference_for_motion(reference_states, int(motion_index), int(args.rollout_length))
        reference_record = compute_reference_endpoint_record(
            reference=reference,
            physical=physical,
            device=device,
            rollout_length=args.rollout_length,
            inner_steps=args.inner_steps,
        )
        torch.save(reference_record, motion_dir / "reference_endpoints.pt")

        records: list[dict[str, Any]] = []
        if not args.plot_only:
            for spec in specs:
                if args.include_full:
                    name = solver_name("full_500step", spec)
                    model_dir = args.source_root / "models" / spec.experiment_name
                    try:
                        records.append(run_model_line(
                            name=name,
                            model_dir=model_dir,
                            device=device,
                            residual_length_scale=args.residual_length_scale,
                            reference=reference,
                            physical=physical,
                            motion_index=int(motion_index),
                            rollout_length=args.rollout_length,
                            inner_steps=args.inner_steps,
                            output_path=motion_dir / name / "curve.pt",
                            overwrite=args.overwrite,
                            checkpoint_every=args.checkpoint_every,
                        ))
                    except FileNotFoundError as exc:
                        print(f"skip {name}: {exc}")

                if args.include_pool:
                    name = solver_name("pool", spec)
                    model_dir = args.pool_root / "models" / spec.experiment_name
                    try:
                        records.append(run_model_line(
                            name=name,
                            model_dir=model_dir,
                            device=device,
                            residual_length_scale=args.residual_length_scale,
                            reference=reference,
                            physical=physical,
                            motion_index=int(motion_index),
                            rollout_length=args.rollout_length,
                            inner_steps=args.inner_steps,
                            output_path=motion_dir / name / "curve.pt",
                            overwrite=args.overwrite,
                            checkpoint_every=args.checkpoint_every,
                        ))
                    except FileNotFoundError as exc:
                        print(f"skip {name}: {exc}")

                if args.include_points_0032:
                    name = solver_name("points_0032", spec)
                    model_dir = args.ablation_root / "points_0032" / "models" / spec.experiment_name
                    try:
                        records.append(run_model_line(
                            name=name,
                            model_dir=model_dir,
                            device=device,
                            residual_length_scale=args.residual_length_scale,
                            reference=reference,
                            physical=physical,
                            motion_index=int(motion_index),
                            rollout_length=args.rollout_length,
                            inner_steps=args.inner_steps,
                            output_path=motion_dir / name / "curve.pt",
                            overwrite=args.overwrite,
                            checkpoint_every=args.checkpoint_every,
                        ))
                    except FileNotFoundError as exc:
                        print(f"skip {name}: {exc}")

            for method in args.baselines:
                records.append(run_baseline_line(
                    method=method,
                    source_root=args.source_root,
                    device=device,
                    reference=reference,
                    physical=physical,
                    motion_index=int(motion_index),
                    rollout_length=args.rollout_length,
                    inner_steps=args.inner_steps,
                    output_path=motion_dir / f"baseline_{method}" / "curve.pt",
                    overwrite=args.overwrite,
                    checkpoint_every=args.checkpoint_every,
                ))
        else:
            for child in sorted(motion_dir.iterdir()):
                curve_path = child / "curve.pt"
                if curve_path.exists():
                    records.append(torch.load(curve_path, map_location="cpu"))

        plot_motion(
            motion_dir=motion_dir,
            records=records,
            reference_record=reference_record,
            rollout_length=args.rollout_length,
            inner_steps=args.inner_steps,
        )

        summary_rows = [summarize_record(record, args.rollout_length, args.inner_steps) for record in records]
        write_summary_csv(summary_rows, motion_dir / "summary_metrics.csv")
        all_rows.extend(summary_rows)
        torch.save({"reference": reference_record, "records": records}, motion_dir / "all_curves.pt")

    write_summary_csv(all_rows, output_root / "all_motion_summary.csv")
    save_json({
        "source_root": str(args.source_root),
        "pool_root": str(args.pool_root),
        "ablation_root": str(args.ablation_root),
        "motion_indices": list(args.motion_indices),
        "rollout_length": args.rollout_length,
        "inner_steps": args.inner_steps,
        "activations": list(args.activations),
        "include_full": args.include_full,
        "include_pool": args.include_pool,
        "include_points_0032": args.include_points_0032,
        "baselines": list(args.baselines),
    }, output_root / "run_config.json")


if __name__ == "__main__":
    main()
