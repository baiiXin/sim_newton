"""Script 13: Metamizer-style training-pool experiment for 5x5 cloth.

The training pool is initialized only from motion initial states.  For each train
motion, five live environments are created with iterations_per_timestep in
{1, 3, 5, 10, 30}.  One optimizer.step applies exactly one learned update to
each live environment.  An environment advances its physical state only after
its K inner updates are complete.  The next frame starts from y^(0)=x_n.

Loss is intentionally simple:
    mean(variational_energy_full(y_after_one_update, q, masses)) / energy_scale

No exact_y, no K-step unroll, and no K-step averaged loss are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from cloth03_solvers_and_models import (
    ACTIVATION_NAMES,
    DEFAULT_DEVICE,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    FIXED_VERTEX_INDICES,
    HIDDEN_DEPTHS,
    HIDDEN_WIDTHS,
    LEARNING_RATE,
    MLPOptimizer,
    ModelSpec,
    NUM_FREE_PARTICLES,
    NUM_PARTICLES,
    SPATIAL_DIM,
    SPRING_EDGES,
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
DEFAULT_K_BUCKETS = (1, 3, 5, 10, 30)
DEFAULT_EPOCHS = 50
DEFAULT_UPDATES_PER_EPOCH = 1000
DEFAULT_VALIDATION_INTERVAL = 10
DEFAULT_VALIDATION_ROLLOUT_LENGTH = 100
DEFAULT_VALIDATION_INNER_STEPS = 50
PLOT_FLOOR = 1e-16


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    def safe(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [safe(v) for v in x]
        if isinstance(x, torch.Tensor):
            return safe(x.detach().cpu().tolist())
        if isinstance(x, np.ndarray):
            return safe(x.tolist())
        if isinstance(x, np.generic):
            return safe(x.item())
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(safe(data), f, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(x: float | torch.Tensor) -> float:
    value = float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)
    return value if math.isfinite(value) else float("inf")


def load_physical(source_root: Path):
    runtime = load_json(source_root / "data" / "reference" / "runtime_config.json")
    return physical_config_from_dict(runtime["physical_config"])


def load_motions(source_root: Path) -> list[dict[str, Any]]:
    path = source_root / "data" / "reference" / "motion_catalogue.json"
    if path.exists():
        return list(load_json(path)["motions"])
    runtime = load_json(source_root / "data" / "reference" / "runtime_config.json")
    return list(runtime["motions"])


def load_reference_states(source_root: Path) -> dict[str, Any]:
    return torch.load(source_root / "data" / "reference" / "reference_motion_states.pt", map_location="cpu")


def free_masses(physical, device: torch.device, n: int = 1) -> torch.Tensor:
    fixed = set(FIXED_VERTEX_INDICES)
    values = [physical.masses[i] for i in range(NUM_PARTICLES) if i not in fixed]
    return torch.tensor(values, dtype=TORCH_DTYPE, device=device).reshape(1, NUM_FREE_PARTICLES).expand(n, -1).clone()


def flatten_full(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(*x.shape[:-2], NUM_PARTICLES * SPATIAL_DIM)


def q_full_from_p_v(p: torch.Tensor, v: torch.Tensor, physical) -> torch.Tensor:
    q_free = make_q_free(p, v, physical).reshape(1, -1)
    return project_fixed_vertices(full_state_from_free_state(q_free, physical), physical)


def velocity_from_positions(p_old: torch.Tensor, p_new: torch.Tensor, physical) -> torch.Tensor:
    v = (p_new - p_old) / physical.dt
    v[..., list(FIXED_VERTEX_INDICES), :] = 0.0
    return v


def reference_for_motion(reference_states: dict[str, Any], motion_index: int, rollout_length: int) -> dict[str, torch.Tensor]:
    ids = [int(v) for v in reference_states["motion_index"].tolist()]
    row = ids.index(int(motion_index))
    return {
        "positions": reference_states["positions"][row, : rollout_length + 1].contiguous(),
        "velocities": reference_states["velocities"][row, : rollout_length + 1].contiguous(),
    }


class ClothPool:
    def __init__(self, *, motions: Sequence[dict[str, Any]], motion_indices: Sequence[int], k_buckets: Sequence[int], physical, device: torch.device, args: argparse.Namespace) -> None:
        self.physical = physical
        self.device = device
        self.motion_by_index = {int(m["index"]): m for m in motions}
        self.motion_indices: list[int] = []
        self.k_values: list[int] = []
        for motion_index in motion_indices:
            for k in k_buckets:
                self.motion_indices.append(int(motion_index))
                self.k_values.append(int(k))
        self.n = len(self.motion_indices)
        self.k = torch.tensor(self.k_values, dtype=torch.long, device=device)
        self.masses = free_masses(physical, device, self.n)
        self.initial_p = torch.zeros(self.n, NUM_PARTICLES * SPATIAL_DIM, dtype=TORCH_DTYPE, device=device)
        self.initial_v = torch.zeros_like(self.initial_p)
        self.p = torch.zeros_like(self.initial_p)
        self.v = torch.zeros_like(self.initial_p)
        self.q = torch.zeros_like(self.initial_p)
        self.y = torch.zeros_like(self.initial_p)
        self.prev_residual = torch.zeros_like(self.initial_p)
        self.prev_update = torch.zeros_like(self.initial_p)
        self.inner_iteration = torch.zeros(self.n, dtype=torch.long, device=device)
        self.physical_step = torch.zeros(self.n, dtype=torch.long, device=device)
        self.age_physical_step = torch.zeros(self.n, dtype=torch.long, device=device)
        self.edge_i = torch.tensor([e[0] for e in SPRING_EDGES], dtype=torch.long, device=device)
        self.edge_j = torch.tensor([e[1] for e in SPRING_EDGES], dtype=torch.long, device=device)
        self.args = args
        for row, motion_index in enumerate(self.motion_indices):
            motion = self.motion_by_index[int(motion_index)]
            p = torch.tensor(motion["p0"], dtype=TORCH_DTYPE, device=device)
            v = torch.tensor(motion["v0"], dtype=TORCH_DTYPE, device=device)
            p[list(FIXED_VERTEX_INDICES), :] = torch.tensor(physical.fixed_positions, dtype=TORCH_DTYPE, device=device)
            v[list(FIXED_VERTEX_INDICES), :] = 0.0
            self.initial_p[row] = project_fixed_vertices(full_state_from_positions(p).reshape(1, -1), physical).squeeze(0)
            self.initial_v[row] = flatten_full(v)
            self.reset(row)

    def reset(self, row: int) -> None:
        self.p[row] = self.initial_p[row]
        self.v[row] = self.initial_v[row]
        p_pos = self.p[row].reshape(NUM_PARTICLES, SPATIAL_DIM)
        v_pos = self.v[row].reshape(NUM_PARTICLES, SPATIAL_DIM)
        self.q[row] = q_full_from_p_v(p_pos, v_pos, self.physical).squeeze(0)
        self.y[row] = self.p[row]  # y^(0)=x_n
        self.prev_residual[row].zero_()
        self.prev_update[row].zero_()
        self.inner_iteration[row] = 0
        self.physical_step[row] = 0
        self.age_physical_step[row] = 0

    def ask(self) -> dict[str, torch.Tensor]:
        return {
            "y": self.y.detach().clone(),
            "q": self.q.detach().clone(),
            "masses": self.masses.detach().clone(),
            "prev_residual": self.prev_residual.detach().clone(),
            "prev_update": self.prev_update.detach().clone(),
        }

    def spring_bad(self, y: torch.Tensor) -> torch.Tensor:
        pos = y.detach().reshape(-1, NUM_PARTICLES, SPATIAL_DIM)
        lengths = torch.linalg.vector_norm(pos[:, self.edge_i, :] - pos[:, self.edge_j, :], dim=-1)
        return (lengths.min(dim=-1).values < self.args.min_spring_length) | (lengths.max(dim=-1).values > self.args.max_spring_length)

    @torch.no_grad()
    def tell(self, *, y_next: torch.Tensor, delta: torch.Tensor, current_residual: torch.Tensor, energy: torch.Tensor, residual_norm: torch.Tensor) -> dict[str, int | float]:
        y_next = y_next.detach()
        delta = delta.detach()
        current_residual = current_residual.detach()
        energy = energy.detach()
        residual_norm = residual_norm.detach()

        nonfinite = (~torch.isfinite(y_next).all(dim=-1)) | (~torch.isfinite(energy)) | (~torch.isfinite(residual_norm))
        energy_bad = torch.isfinite(energy) & (energy.abs() > self.args.max_energy)
        residual_bad = torch.isfinite(residual_norm) & (residual_norm > self.args.max_residual)
        position_bad = torch.isfinite(y_next).all(dim=-1) & (y_next.abs().amax(dim=-1) > self.args.max_abs_position)
        spring_bad = self.spring_bad(y_next)
        bad = nonfinite | energy_bad | residual_bad | position_bad | spring_bad

        self.y.copy_(y_next)
        self.prev_update.copy_(delta)
        self.prev_residual.copy_(current_residual)
        self.inner_iteration += 1

        completed = (self.inner_iteration % self.k == 0) & (~bad)
        for row in torch.nonzero(completed, as_tuple=False).flatten().tolist():
            p_old = self.p[row].reshape(NUM_PARTICLES, SPATIAL_DIM)
            p_new = self.y[row].reshape(NUM_PARTICLES, SPATIAL_DIM)
            v_new = velocity_from_positions(p_old, p_new, self.physical)
            self.p[row] = self.y[row]
            self.v[row] = flatten_full(v_new)
            self.q[row] = q_full_from_p_v(p_new, v_new, self.physical).squeeze(0)
            self.y[row] = self.p[row]  # next frame starts from x_n
            self.prev_residual[row].zero_()
            self.prev_update[row].zero_()
            self.physical_step[row] += 1
            self.age_physical_step[row] += 1

        lifetime_bad = self.age_physical_step >= self.args.max_lifetime_physical_steps
        reset_any = bad | lifetime_bad
        for row in torch.nonzero(reset_any, as_tuple=False).flatten().tolist():
            self.reset(int(row))

        def count(mask: torch.Tensor) -> int:
            return int(mask.sum().detach().cpu().item())

        out: dict[str, int | float] = {
            "resets_total": count(reset_any),
            "resets_nonfinite": count(nonfinite),
            "resets_energy": count(energy_bad),
            "resets_residual": count(residual_bad),
            "resets_position": count(position_bad),
            "resets_spring": count(spring_bad),
            "resets_lifetime": count(lifetime_bad),
        }
        for k in DEFAULT_K_BUCKETS:
            select = self.k == int(k)
            out[f"physical_step_mean_k{k}"] = float(self.physical_step[select].double().mean().item()) if bool(select.any()) else 0.0
        return out

    def manifest(self) -> dict[str, Any]:
        return {
            "num_envs": self.n,
            "motion_indices": self.motion_indices,
            "k_values": self.k_values,
            "semantics": {
                "parameter_update": "one learned optimizer update",
                "physical_update": "after iterations_per_timestep learned updates",
                "new_frame_initial_guess": "y^(0)=x_n",
            },
        }


@torch.no_grad()
def solve_model_frame(*, model: MLPOptimizer, y0: torch.Tensor, q: torch.Tensor, masses: torch.Tensor, physical, inner_steps: int) -> tuple[torch.Tensor, list[float]]:
    y = project_fixed_vertices(y0.clone(), physical)
    residuals = [safe_float(stationarity_residual_norm_full(y, q, masses, physical))]
    prev_residual = torch.zeros_like(y)
    prev_update = torch.zeros_like(y)
    for _ in range(int(inner_steps)):
        y, delta, current_residual = apply_model_update(
            model, y, q, masses, physical, previous_residual=prev_residual, previous_update=prev_update
        )
        prev_residual = current_residual.detach()
        prev_update = delta.detach()
        residuals.append(safe_float(stationarity_residual_norm_full(y, q, masses, physical)))
    return y, residuals


@torch.no_grad()
def validation_rollout(*, model: MLPOptimizer, reference_states: dict[str, Any], motion_indices: Sequence[int], physical, device: torch.device, rollout_length: int, inner_steps: int) -> dict[str, Any]:
    model.eval()
    masses = free_masses(physical, device, 1)
    final_residuals: list[float] = []
    reference_errors: list[float] = []
    failures = 0
    for motion_index in motion_indices:
        ref = reference_for_motion(reference_states, int(motion_index), int(rollout_length))
        positions = [ref["positions"][0].to(device=device, dtype=TORCH_DTYPE)]
        velocities = [ref["velocities"][0].to(device=device, dtype=TORCH_DTYPE)]
        for frame in range(int(rollout_length)):
            p_n = positions[-1]
            v_n = velocities[-1]
            q = q_full_from_p_v(p_n, v_n, physical)
            y0 = project_fixed_vertices(full_state_from_positions(p_n).reshape(1, -1), physical)
            y_next, residuals = solve_model_frame(model=model, y0=y0, q=q, masses=masses, physical=physical, inner_steps=inner_steps)
            if not bool(torch.isfinite(y_next).all()) or not all(math.isfinite(v) for v in residuals):
                failures += 1
                final_residuals.append(float("inf"))
                break
            p_next = y_next.reshape(NUM_PARTICLES, SPATIAL_DIM)
            v_next = velocity_from_positions(p_n, p_next, physical)
            ref_next = ref["positions"][frame + 1].to(device=device, dtype=TORCH_DTYPE)
            final_residuals.append(float(residuals[-1]))
            reference_errors.append(safe_float(torch.linalg.vector_norm(p_next - ref_next)))
            positions.append(p_next.detach())
            velocities.append(v_next.detach())
    finite_res = np.asarray([v for v in final_residuals if math.isfinite(v)], dtype=float)
    finite_err = np.asarray([v for v in reference_errors if math.isfinite(v)], dtype=float)
    return {
        "max_final_residual": float(np.max(finite_res)) if finite_res.size else float("inf"),
        "p95_final_residual": float(np.percentile(finite_res, 95)) if finite_res.size else float("inf"),
        "mean_final_residual": float(np.mean(finite_res)) if finite_res.size else float("inf"),
        "max_reference_error": float(np.max(finite_err)) if finite_err.size else float("inf"),
        "p95_reference_error": float(np.percentile(finite_err, 95)) if finite_err.size else float("inf"),
        "num_failures": failures,
    }


def make_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs = [
        ModelSpec(activation=a, depth=int(d), width=int(w), use_bias=bool(args.use_bias))
        for a in args.activations for d in args.depths for w in args.widths
    ]
    return [specs[int(args.config_index)]] if args.config_index is not None else specs


def save_checkpoint(path: Path, *, model: MLPOptimizer, optimizer: torch.optim.Optimizer, epoch: int, update_count: int, model_spec: ModelSpec, config: dict[str, Any], best_validation_max: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "update_count": int(update_count),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_spec": asdict(model_spec),
        "config": config,
        "best_validation_max": float(best_validation_max),
    }, path)


def train_one_model(*, model_spec: ModelSpec, args: argparse.Namespace, physical, motions: list[dict[str, Any]], reference_states: dict[str, Any], device: torch.device) -> None:
    output_dir = args.pool_root / "models" / model_spec.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "best_validation_model.pt").exists() and not args.overwrite and not args.resume:
        print(f"skip existing {model_spec.experiment_name}; use --overwrite or --resume")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MLPOptimizer(args.residual_length_scale, model_spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 1
    update_count = 0
    best_validation_max = math.inf

    config = {
        "experiment": "metamizer_style_pool_training",
        "model_spec": asdict(model_spec),
        "architecture": model.architecture_description,
        "parameter_count": model.parameter_count,
        "loss": "mean energy after one learned update divided by physical_energy_scale",
        "no_unroll": True,
        "new_frame_initial_guess": "x_n",
        "epochs": args.epochs,
        "updates_per_epoch": args.updates_per_epoch,
        "k_buckets": list(args.k_buckets),
        "train_motions": list(args.train_motions),
        "learning_rate": args.learning_rate,
        "gradient_clip_norm": args.gradient_clip_norm,
        "residual_length_scale": args.residual_length_scale,
        "validation_motions": list(args.validation_motions),
        "validation_rollout_length": args.validation_rollout_length,
        "validation_inner_steps": args.validation_inner_steps,
        "device": str(device),
        "seed": args.seed,
    }
    save_json(config, output_dir / "config.json")

    if args.resume and (output_dir / "latest_checkpoint.pt").exists() and not args.overwrite:
        ckpt = torch.load(output_dir / "latest_checkpoint.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        update_count = int(ckpt.get("update_count", 0))
        best_validation_max = float(ckpt.get("best_validation_max", math.inf))
        print(f"resume {model_spec.experiment_name} from epoch {start_epoch}")

    pool = ClothPool(motions=motions, motion_indices=args.train_motions, k_buckets=args.k_buckets, physical=physical, device=device, args=args)
    save_json(pool.manifest(), output_dir / "pool_manifest.json")
    energy_scale = physical_energy_scale(pool.masses.detach(), physical, args.residual_length_scale)

    rows: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        losses: list[float] = []
        energies: list[float] = []
        residuals: list[float] = []
        reset_totals = {k: 0 for k in ["resets_total", "resets_nonfinite", "resets_energy", "resets_residual", "resets_position", "resets_spring", "resets_lifetime"]}
        last_stats: dict[str, Any] = {}

        for step in range(1, args.updates_per_epoch + 1):
            batch = pool.ask()
            optimizer.zero_grad(set_to_none=True)
            y_next, delta, current_residual = apply_model_update(
                model, batch["y"], batch["q"], batch["masses"], physical,
                previous_residual=batch["prev_residual"], previous_update=batch["prev_update"]
            )
            energy = variational_energy_full(y_next, batch["q"], batch["masses"], physical)
            loss = energy.mean() / max(float(energy_scale), 1e-30)
            loss.backward()
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()

            with torch.no_grad():
                residual_norm = stationarity_residual_norm_full(y_next, batch["q"], batch["masses"], physical)
                stats = pool.tell(y_next=y_next, delta=delta, current_residual=current_residual, energy=energy, residual_norm=residual_norm)
            last_stats = stats
            for key in reset_totals:
                reset_totals[key] += int(stats.get(key, 0))
            losses.append(safe_float(loss))
            energies.append(safe_float(energy.mean()))
            residuals.append(safe_float(residual_norm.mean()))
            update_count += 1

            if step == 1 or step % args.log_interval == 0 or step == args.updates_per_epoch:
                print(
                    f"{model_spec.experiment_name} epoch={epoch:03d}/{args.epochs} "
                    f"update={step:04d}/{args.updates_per_epoch} loss={losses[-1]:.3e} "
                    f"energy={energies[-1]:.3e} residual={residuals[-1]:.3e} resets={stats['resets_total']}"
                )

        row = {
            "epoch": epoch,
            "update_count": update_count,
            "loss_mean": float(np.mean(losses)),
            "energy_mean": float(np.mean(energies)),
            "residual_mean": float(np.mean(residuals)),
            "elapsed_seconds": time.perf_counter() - t0,
            **reset_totals,
            **{k: last_stats.get(k, 0.0) for k in ["physical_step_mean_k1", "physical_step_mean_k3", "physical_step_mean_k5", "physical_step_mean_k10", "physical_step_mean_k30"]},
        }
        rows.append(row)
        write_csv(rows, output_dir / "train_log.csv")
        save_checkpoint(output_dir / "latest_checkpoint.pt", model=model, optimizer=optimizer, epoch=epoch, update_count=update_count, model_spec=model_spec, config=config, best_validation_max=best_validation_max)

        if epoch == 1 or epoch % args.validation_interval == 0 or epoch == args.epochs:
            vt0 = time.perf_counter()
            val = validation_rollout(model=model, reference_states=reference_states, motion_indices=args.validation_motions, physical=physical, device=device, rollout_length=args.validation_rollout_length, inner_steps=args.validation_inner_steps)
            val_record = {"epoch": epoch, "update_count": update_count, "elapsed_seconds": time.perf_counter() - vt0, **val}
            validation_history.append(val_record)
            save_json({"history": validation_history}, output_dir / "validation_metrics.json")
            score = float(val["max_final_residual"])
            if score < best_validation_max:
                best_validation_max = score
                save_checkpoint(output_dir / "best_validation_model.pt", model=model, optimizer=optimizer, epoch=epoch, update_count=update_count, model_spec=model_spec, config=config, best_validation_max=best_validation_max)
            print(f"  validation max={val['max_final_residual']:.3e} p95={val['p95_final_residual']:.3e} best={best_validation_max:.3e}")

    if not (output_dir / "best_validation_model.pt").exists():
        save_checkpoint(output_dir / "best_validation_model.pt", model=model, optimizer=optimizer, epoch=args.epochs, update_count=update_count, model_spec=model_spec, config=config, best_validation_max=best_validation_max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Metamizer-style cloth pool learned optimizers.")
    parser.add_argument("--source-root", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--pool-root", type=Path, default=Path("cloth_5x5_metamizer_pool_training"))
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--updates-per-epoch", type=int, default=DEFAULT_UPDATES_PER_EPOCH)
    parser.add_argument("--k-buckets", type=int, nargs="+", default=list(DEFAULT_K_BUCKETS))
    parser.add_argument("--train-motions", type=int, nargs="+", default=list(TRAIN_MOTIONS))
    parser.add_argument("--validation-motions", type=int, nargs="+", default=list(VALIDATION_MOTIONS))
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--validation-rollout-length", type=int, default=DEFAULT_VALIDATION_ROLLOUT_LENGTH)
    parser.add_argument("--validation-inner-steps", type=int, default=DEFAULT_VALIDATION_INNER_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--activations", nargs="+", default=list(ACTIVATION_NAMES))
    parser.add_argument("--depths", type=int, nargs="+", default=list(HIDDEN_DEPTHS))
    parser.add_argument("--widths", type=int, nargs="+", default=list(HIDDEN_WIDTHS))
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--config-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-energy", type=float, default=1e8)
    parser.add_argument("--max-residual", type=float, default=1e8)
    parser.add_argument("--max-abs-position", type=float, default=1e3)
    parser.add_argument("--min-spring-length", type=float, default=1e-8)
    parser.add_argument("--max-spring-length", type=float, default=1e3)
    parser.add_argument("--max-lifetime-physical-steps", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    physical = load_physical(args.source_root)
    motions = load_motions(args.source_root)
    reference_states = load_reference_states(args.source_root)
    args.pool_root.mkdir(parents=True, exist_ok=True)
    for spec in make_model_specs(args):
        train_one_model(model_spec=spec, args=args, physical=physical, motions=motions, reference_states=reference_states, device=device)


if __name__ == "__main__":
    main()
