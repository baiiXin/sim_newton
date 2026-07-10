"""Train the initial-point-count ablation with fixed optimizer-update semantics.

For each original time-window minibatch, sample slots are processed as microbatches.
Their losses are divided by the selected sample count and backpropagated one by one.
Gradient clipping and optimizer.step() happen once after all sample slots have been
visited. Therefore every epoch visits every selected state, while the number of
optimizer updates and the CUDA peak batch shape are independent of sample count.

Checkpoint selection uses continuous validation rollout:
- validation motions: 16, 17, 18, 19
- rollout length: 300 physical frames
- learned iterations per frame: 15
- selection metric: maximum of all 4 * 300 final residuals
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cloth03_solvers_and_models import (
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_K_VALUES,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    DEFAULT_VALIDATION_INTERVAL,
    FIXED_VERTEX_INDICES,
    LEARNING_RATE,
    MLPOptimizer,
    ModelSpec,
    NUM_FREE_PARTICLES,
    NUM_PARTICLES,
    SPATIAL_DIM,
    TORCH_DTYPE,
    apply_model_update,
    full_state_from_free_state,
    full_state_from_positions,
    make_q_free,
    physical_config_from_dict,
    physical_energy_scale,
    project_fixed_vertices,
    stationarity_residual_norm_full,
    variational_energy_full,
)

TRAIN_MOTIONS = tuple(range(0, 16))
VALIDATION_MOTIONS = tuple(range(16, 20))
DEFAULT_SAMPLE_COUNTS = (1, 8, 32, 64, 128, 1024)
DEFAULT_TRAIN_TIME_STOP = 400
DEFAULT_TIME_STEPS_PER_MOTION_BATCH = 32
DEFAULT_VALIDATION_ROLLOUT_LENGTH = 300
DEFAULT_VALIDATION_INNER_STEPS = 15


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return make_json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(data), handle, indent=2, ensure_ascii=False, allow_nan=False)


def safe_float(value: float | torch.Tensor) -> float:
    number = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
    return number if math.isfinite(number) else float("inf")


def torch_load_cpu(path: Path, *, mmap: bool = False) -> dict[str, Any]:
    if mmap:
        try:
            return torch.load(path, map_location="cpu", mmap=True)
        except (TypeError, RuntimeError):
            pass
    return torch.load(path, map_location="cpu")


def load_physical(source_root: Path):
    runtime = load_json(source_root / "data" / "reference" / "runtime_config.json")
    return physical_config_from_dict(runtime["physical_config"])


def load_reference_motion_states(source_root: Path) -> dict[str, Any]:
    path = source_root / "data" / "reference" / "reference_motion_states.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch_load_cpu(path)


def load_motion_sample_files(ablation_root: Path) -> dict[int, dict[str, Any]]:
    sample_dir = ablation_root / "shared_samples_1024"
    records: dict[int, dict[str, Any]] = {}
    for motion_index in TRAIN_MOTIONS:
        path = sample_dir / f"motion_{motion_index:03d}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run cloth10_prepare_initial_point_ablation.py first"
            )
        records[motion_index] = torch_load_cpu(path, mmap=True)
    return records


def build_time_windows(train_time_stop: int, width: int) -> list[tuple[int, int]]:
    if train_time_stop <= 0 or width <= 0:
        raise ValueError("train_time_stop and width must be positive")
    return [(start, min(start + width, train_time_stop)) for start in range(0, train_time_stop, width)]


def make_microbatch(
    motion_records: dict[int, dict[str, Any]],
    *,
    time_start: int,
    time_stop: int,
    sample_slot: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    initial_y = []
    q = []
    masses = []
    exact_y = []
    for motion_index in TRAIN_MOTIONS:
        record = motion_records[motion_index]
        initial_y.append(record["initial_y"][time_start:time_stop, sample_slot, :])
        q.append(record["q"][time_start:time_stop])
        masses.append(record["masses"][time_start:time_stop])
        exact_y.append(record["exact_y"][time_start:time_stop])
    return {
        "initial_y": torch.cat(initial_y, dim=0).to(device=device, dtype=TORCH_DTYPE),
        "q": torch.cat(q, dim=0).to(device=device, dtype=TORCH_DTYPE),
        "masses": torch.cat(masses, dim=0).to(device=device, dtype=TORCH_DTYPE),
        "exact_y": torch.cat(exact_y, dim=0).to(device=device, dtype=TORCH_DTYPE),
    }


def k_for_epoch(epoch: int, k_values: tuple[int, ...], epochs_per_k: int) -> int:
    if epoch <= 0 or epochs_per_k <= 0:
        raise ValueError("epoch and epochs_per_k must be positive")
    return int(k_values[min((epoch - 1) // epochs_per_k, len(k_values) - 1)])


def rollout_model_steps(
    model: MLPOptimizer,
    batch: dict[str, torch.Tensor],
    physical,
    steps: int,
) -> torch.Tensor:
    y = project_fixed_vertices(batch["initial_y"].clone(), physical)
    previous_residual = torch.zeros_like(y)
    previous_update = torch.zeros_like(y)
    for _ in range(int(steps)):
        y, delta, current_residual = apply_model_update(
            model,
            y,
            batch["q"],
            batch["masses"],
            physical,
            previous_residual=previous_residual,
            previous_update=previous_update,
        )
        previous_residual = current_residual.detach()
        previous_update = delta.detach()
    return y


def microbatch_loss(
    model: MLPOptimizer,
    batch: dict[str, torch.Tensor],
    physical,
    *,
    steps: int,
    residual_length_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    y = rollout_model_steps(model, batch, physical, steps)
    exact_energy = variational_energy_full(
        batch["exact_y"], batch["q"], batch["masses"], physical
    ).detach()
    energy = variational_energy_full(y, batch["q"], batch["masses"], physical)
    energy_gap = torch.clamp(energy - exact_energy, min=0.0)
    scale = physical_energy_scale(batch["masses"].detach(), physical, residual_length_scale)
    loss = energy_gap.mean() / max(scale, 1e-30)
    with torch.no_grad():
        residual = stationarity_residual_norm_full(y, batch["q"], batch["masses"], physical)
    return loss, {
        "residual_mean": safe_float(residual.mean()),
        "residual_max": safe_float(residual.max()),
        "energy_gap_mean": safe_float(energy_gap.mean()),
    }


def free_masses_tensor(physical, device: torch.device) -> torch.Tensor:
    fixed = set(FIXED_VERTEX_INDICES)
    values = [physical.masses[i] for i in range(NUM_PARTICLES) if i not in fixed]
    return torch.tensor(values, dtype=TORCH_DTYPE, device=device).reshape(1, NUM_FREE_PARTICLES)


def velocity_from_positions(p_prev: torch.Tensor, p_next: torch.Tensor, physical) -> torch.Tensor:
    velocity = (p_next - p_prev) / physical.dt
    velocity[list(FIXED_VERTEX_INDICES), :] = 0.0
    return velocity


@torch.no_grad()
def validate_rollout_worst_residual(
    *,
    model: MLPOptimizer,
    reference_states: dict[str, Any],
    physical,
    device: torch.device,
    rollout_length: int,
    inner_steps: int,
) -> dict[str, Any]:
    model.eval()
    motion_ids = [int(value) for value in reference_states["motion_index"].tolist()]
    masses = free_masses_tensor(physical, device)
    all_final_residuals: list[float] = []
    per_motion: dict[str, Any] = {}

    for motion_index in VALIDATION_MOTIONS:
        row = motion_ids.index(motion_index)
        positions_ref = reference_states["positions"][row]
        velocities_ref = reference_states["velocities"][row]
        if positions_ref.shape[0] < rollout_length + 1:
            raise RuntimeError(f"motion {motion_index} reference is shorter than {rollout_length}")

        p_n = positions_ref[0].to(device=device, dtype=TORCH_DTYPE).clone()
        v_n = velocities_ref[0].to(device=device, dtype=TORCH_DTYPE).clone()
        residuals: list[float] = []
        reference_errors: list[float] = []

        for frame_index in range(rollout_length):
            q_free = make_q_free(p_n, v_n, physical).reshape(1, -1)
            q_full = project_fixed_vertices(full_state_from_free_state(q_free, physical), physical)
            y = project_fixed_vertices(full_state_from_positions(p_n).reshape(1, -1), physical)
            previous_residual = torch.zeros_like(y)
            previous_update = torch.zeros_like(y)

            for _ in range(inner_steps):
                y, delta, current_residual = apply_model_update(
                    model,
                    y,
                    q_full,
                    masses,
                    physical,
                    previous_residual=previous_residual,
                    previous_update=previous_update,
                )
                previous_residual = current_residual
                previous_update = delta

            residual = stationarity_residual_norm_full(y, q_full, masses, physical)
            value = safe_float(residual)
            residuals.append(value)
            all_final_residuals.append(value)

            if not math.isfinite(value) or not bool(torch.isfinite(y).all()):
                residuals.extend([float("inf")] * (rollout_length - frame_index - 1))
                all_final_residuals.extend([float("inf")] * (rollout_length - frame_index - 1))
                reference_errors.extend([float("inf")] * (rollout_length - frame_index))
                break

            p_next = y.reshape(NUM_PARTICLES, SPATIAL_DIM)
            ref_next = positions_ref[frame_index + 1].to(device=device, dtype=TORCH_DTYPE)
            reference_errors.append(safe_float(torch.linalg.vector_norm(p_next - ref_next)))
            v_next = velocity_from_positions(p_n, p_next, physical)
            p_n = p_next
            v_n = v_next

        residual_array = np.asarray(residuals, dtype=float)
        finite_residual = residual_array[np.isfinite(residual_array)]
        per_motion[str(motion_index)] = {
            "final_residual_by_frame": residuals,
            "reference_error_by_frame": reference_errors,
            "max_final_residual": float(np.max(residual_array)) if residual_array.size else float("inf"),
            "p95_final_residual": (
                float(np.percentile(finite_residual, 95)) if finite_residual.size else float("inf")
            ),
            "worst_frame": int(np.argmax(residual_array)) if residual_array.size else None,
        }

    all_array = np.asarray(all_final_residuals, dtype=float)
    finite_all = all_array[np.isfinite(all_array)]
    return {
        "selection_metric_name": "global_max_of_4x300_final_residuals",
        "selection_metric": float(np.max(all_array)) if all_array.size else float("inf"),
        "global_p95": float(np.percentile(finite_all, 95)) if finite_all.size else float("inf"),
        "num_final_residuals": int(all_array.size),
        "validation_motion_indices": list(VALIDATION_MOTIONS),
        "rollout_length": rollout_length,
        "inner_steps": inner_steps,
        "per_motion": per_motion,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_payload(
    *,
    model: MLPOptimizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_spec: ModelSpec,
    sample_count: int,
    best_validation_max: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_spec": asdict(model_spec),
        "sample_count": int(sample_count),
        "best_validation_max": float(best_validation_max),
        "config": config,
    }


def train_one_sample_count(
    *,
    sample_count: int,
    args: argparse.Namespace,
    source_root: Path,
    ablation_root: Path,
    physical,
    reference_states: dict[str, Any],
    motion_records: dict[int, dict[str, Any]],
    device: torch.device,
) -> None:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    available = int(motion_records[0]["initial_y"].shape[1])
    if sample_count > available:
        raise ValueError(f"sample_count={sample_count} exceeds available={available}")

    model_spec = ModelSpec(
        activation=args.activation,
        depth=int(args.depth),
        width=int(args.width),
        use_bias=bool(args.use_bias),
    )
    point_root = ablation_root / f"points_{sample_count:04d}"
    output_dir = point_root / "models" / model_spec.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    latest_path = output_dir / "latest_checkpoint.pt"
    best_path = output_dir / "best_validation_model.pt"

    windows = build_time_windows(args.train_time_stop, args.time_steps_per_motion_batch)
    config = {
        "source_root": str(source_root),
        "ablation_root": str(ablation_root),
        "sample_count": sample_count,
        "physical_initial_included": True,
        "nested_prefix": [0, sample_count - 1],
        "model_spec": asdict(model_spec),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_clip_norm": args.gradient_clip_norm,
        "residual_length_scale": args.residual_length_scale,
        "k_values": list(args.k_values),
        "epochs_per_k": args.epochs_per_k,
        "train_motion_indices": list(TRAIN_MOTIONS),
        "train_time_range": [0, args.train_time_stop - 1],
        "time_steps_per_motion_batch": args.time_steps_per_motion_batch,
        "optimizer_updates_per_epoch": len(windows),
        "epoch_semantics": "every selected state of every train problem is visited once",
        "gradient_accumulation": (
            "for each time-window batch, backward(loss/sample_count) once per sample slot; "
            "clip and optimizer.step once after all slots"
        ),
        "validation_motion_indices": list(VALIDATION_MOTIONS),
        "validation_rollout_length": args.validation_rollout_length,
        "validation_inner_steps": args.validation_inner_steps,
        "checkpoint_selection": "maximum of all 4x300 final residuals",
        "validation_interval": args.validation_interval,
        "device": str(device),
        "dtype": str(TORCH_DTYPE),
        "seed": args.seed,
    }
    save_json(config, config_path)

    torch.manual_seed(args.seed)
    model = MLPOptimizer(args.residual_length_scale, model_spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 1
    best_validation_max = float("inf")
    train_rows: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []

    if latest_path.exists() and args.resume and not args.overwrite:
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_max = float(checkpoint.get("best_validation_max", float("inf")))
        train_log_path = output_dir / "train_log.csv"
        if train_log_path.exists():
            with train_log_path.open("r", encoding="utf-8", newline="") as handle:
                train_rows = list(csv.DictReader(handle))
        history_path = output_dir / "validation_history.json"
        if history_path.exists():
            validation_history = load_json(history_path).get("history", [])
        print(f"resume points={sample_count}: epoch {start_epoch}/{args.epochs}")
    elif args.overwrite:
        for path in (
            latest_path,
            best_path,
            output_dir / "train_log.csv",
            output_dir / "validation_history.json",
        ):
            if path.exists():
                path.unlink()

    if start_epoch > args.epochs:
        print(f"skip points={sample_count}: already completed {start_epoch - 1} epochs")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        k = k_for_epoch(epoch, tuple(args.k_values), args.epochs_per_k)
        epoch_start = time.perf_counter()
        window_losses: list[float] = []
        residual_means: list[float] = []
        residual_maxes: list[float] = []

        for window_index, (time_start, time_stop) in enumerate(windows):
            optimizer.zero_grad(set_to_none=True)
            slot_losses: list[float] = []
            slot_residual_means: list[float] = []
            slot_residual_maxes: list[float] = []

            for sample_slot in range(sample_count):
                batch = make_microbatch(
                    motion_records,
                    time_start=time_start,
                    time_stop=time_stop,
                    sample_slot=sample_slot,
                    device=device,
                )
                loss, metrics = microbatch_loss(
                    model,
                    batch,
                    physical,
                    steps=k,
                    residual_length_scale=args.residual_length_scale,
                )
                (loss / float(sample_count)).backward()
                slot_losses.append(safe_float(loss))
                slot_residual_means.append(metrics["residual_mean"])
                slot_residual_maxes.append(metrics["residual_max"])
                del batch, loss

            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            window_losses.append(float(np.mean(slot_losses)))
            residual_means.append(float(np.mean(slot_residual_means)))
            residual_maxes.append(float(np.max(slot_residual_maxes)))
            print(
                f"points={sample_count:04d} epoch={epoch:04d}/{args.epochs} K={k:02d} "
                f"batch={window_index + 1:02d}/{len(windows):02d} "
                f"loss={window_losses[-1]:.3e} residual={residual_means[-1]:.3e}"
            )

        row: dict[str, Any] = {
            "epoch": epoch,
            "k": k,
            "sample_count": sample_count,
            "optimizer_updates": len(windows),
            "states_visited": len(TRAIN_MOTIONS) * args.train_time_stop * sample_count,
            "loss_mean": float(np.mean(window_losses)),
            "train_residual_mean": float(np.mean(residual_means)),
            "train_residual_max": float(np.max(residual_maxes)),
            "elapsed_seconds": time.perf_counter() - epoch_start,
            "validation_global_max": "",
            "validation_global_p95": "",
        }

        should_validate = (
            epoch == 1
            or epoch % args.validation_interval == 0
            or epoch == args.epochs
        )
        if should_validate:
            validation = validate_rollout_worst_residual(
                model=model,
                reference_states=reference_states,
                physical=physical,
                device=device,
                rollout_length=args.validation_rollout_length,
                inner_steps=args.validation_inner_steps,
            )
            validation_record = {"epoch": epoch, **validation}
            validation_history.append(validation_record)
            save_json({"history": validation_history}, output_dir / "validation_history.json")
            row["validation_global_max"] = validation["selection_metric"]
            row["validation_global_p95"] = validation["global_p95"]

            if float(validation["selection_metric"]) < best_validation_max:
                best_validation_max = float(validation["selection_metric"])
                torch.save(
                    checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        model_spec=model_spec,
                        sample_count=sample_count,
                        best_validation_max=best_validation_max,
                        config=config,
                    ),
                    best_path,
                )
            print(
                f"  validation global_max={validation['selection_metric']:.3e} "
                f"p95={validation['global_p95']:.3e} best={best_validation_max:.3e}"
            )

        train_rows.append(row)
        write_csv(train_rows, output_dir / "train_log.csv")
        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                model_spec=model_spec,
                sample_count=sample_count,
                best_validation_max=best_validation_max,
                config=config,
            ),
            latest_path,
        )

    print(f"completed points={sample_count}; best validation max={best_validation_max:.6e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train initial-point-count ablation models.")
    parser.add_argument("--source-root", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--ablation-root", type=Path, default=Path("cloth_5x5_initial_sample_ablation"))
    parser.add_argument("--sample-counts", type=int, nargs="+", default=list(DEFAULT_SAMPLE_COUNTS))
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--validation-rollout-length", type=int, default=DEFAULT_VALIDATION_ROLLOUT_LENGTH)
    parser.add_argument("--validation-inner-steps", type=int, default=DEFAULT_VALIDATION_INNER_STEPS)
    parser.add_argument("--train-time-stop", type=int, default=DEFAULT_TRAIN_TIME_STOP)
    parser.add_argument(
        "--time-steps-per-motion-batch",
        type=int,
        default=DEFAULT_TIME_STEPS_PER_MOTION_BATCH,
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--epochs-per-k", type=int, default=100)
    parser.add_argument("--activation", choices=("identity", "relu", "tanh"), default="identity")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.validation_interval <= 0:
        raise ValueError("epochs and validation-interval must be positive")
    if args.validation_rollout_length <= 0 or args.validation_inner_steps <= 0:
        raise ValueError("validation rollout settings must be positive")
    sample_counts = tuple(int(value) for value in args.sample_counts)
    if tuple(sorted(set(sample_counts))) != sample_counts or any(value <= 0 for value in sample_counts):
        raise ValueError("sample-counts must be sorted, unique, and positive")

    source_root = args.source_root.resolve()
    ablation_root = args.ablation_root.resolve()
    device = torch.device(args.device)
    physical = load_physical(source_root)
    reference_states = load_reference_motion_states(source_root)
    motion_records = load_motion_sample_files(ablation_root)

    for sample_count in sample_counts:
        train_one_sample_count(
            sample_count=sample_count,
            args=args,
            source_root=source_root,
            ablation_root=ablation_root,
            physical=physical,
            reference_states=reference_states,
            motion_records=motion_records,
            device=device,
        )


if __name__ == "__main__":
    main()
