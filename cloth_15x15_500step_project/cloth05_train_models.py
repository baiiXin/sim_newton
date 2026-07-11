"""Train 15x15 MLP learned optimizers with one-step xn validation/test.

The trainer supports all architecture stages and the initial-perturbation-count
stage. Training samples are read from compact motion shards (main 32-point data)
or window shards produced by cloth10. One optimizer update corresponds to one
physical-time window; sample chunks only accumulate gradients and do not change
optimizer-update count.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cloth02_dataset_catalog import load_dataset
from cloth03_solvers_and_models import (
    ACTIVATION_NAMES,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_K_VALUES,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    LEARNING_RATE,
    MLPOptimizer,
    ModelSpec,
    TORCH_DTYPE,
    apply_model_update,
    physical_energy_scale,
    project_fixed_vertices,
    variational_energy_full,
)
from cloth_common import evaluate_one_step, load_json, load_physical, save_json, write_csv


def k_for_epoch(epoch: int, values: list[int], epochs_per_k: int) -> int:
    return int(values[min((epoch - 1) // epochs_per_k, len(values) - 1)])


class SampleSource:
    def __init__(self, root: Path, train_manifest: dict[str, Any]) -> None:
        self.root = root
        self.train_manifest = train_manifest
        self.source_manifest = load_json(root / "manifest.json")
        self.format = str(self.source_manifest["format"])
        self.cache: dict[int, dict[str, Any]] = {}
        self.window_index: dict[tuple[int, int, int], Path] = {}
        if self.format == "window_shards_v1":
            for item in self.source_manifest["records"]:
                shard_path = Path(item["path"])
                if not shard_path.is_absolute():
                    shard_path = root / shard_path
                self.window_index[(int(item["motion_index"]), int(item["time_start"]), int(item["time_stop"]))] = shard_path

    @property
    def available_points(self) -> int:
        return int(self.source_manifest.get("points_per_problem", self.source_manifest.get("max_points", 0)))

    def _motion(self, motion_index: int) -> dict[str, Any]:
        if motion_index not in self.cache:
            path = self.root / f"motion_{motion_index:03d}.pt"
            try:
                self.cache[motion_index] = torch.load(path, map_location="cpu", mmap=True)
            except (TypeError, RuntimeError):
                self.cache[motion_index] = torch.load(path, map_location="cpu")
        return self.cache[motion_index]

    def load_window(
        self,
        *,
        motion_index: int,
        time_start: int,
        time_stop: int,
        sample_start: int,
        sample_stop: int,
        reference: dict[str, Any] | None,
    ) -> dict[str, torch.Tensor]:
        if self.format == "motion_shards_v1":
            record = self._motion(motion_index)
            return {
                "initial_y": record["initial_y"][time_start:time_stop, sample_start:sample_stop],
                "q": record["q"][time_start:time_stop],
                "masses": record["masses"][time_start:time_stop],
                "exact_y": record["exact_y"][time_start:time_stop],
            }
        if self.format == "window_shards_v1":
            path = self.window_index[(motion_index, time_start, time_stop)]
            try:
                record = torch.load(path, map_location="cpu", mmap=True)
            except (TypeError, RuntimeError):
                record = torch.load(path, map_location="cpu")
            if reference is None:
                raise RuntimeError("window-sharded samples require reference data")
            mask = (
                (reference["motion_index"] == motion_index)
                & (reference["time_index"] >= time_start)
                & (reference["time_index"] < time_stop)
            )
            rows = torch.nonzero(mask, as_tuple=False).flatten()
            rows = rows[torch.argsort(reference["time_index"].index_select(0, rows))]
            return {
                "initial_y": record["initial_y"][:, sample_start:sample_stop],
                "q": reference["q"].index_select(0, rows),
                "masses": reference["masses"].index_select(0, rows),
                "exact_y": reference["exact_y"].index_select(0, rows),
            }
        raise ValueError(f"unsupported sample format {self.format}")


def make_train_chunk(
    *, source: SampleSource, reference: dict[str, Any] | None,
    motions: list[int], time_start: int, time_stop: int,
    sample_start: int, sample_stop: int, device: torch.device,
) -> dict[str, torch.Tensor]:
    initial: list[torch.Tensor] = []
    q: list[torch.Tensor] = []
    masses: list[torch.Tensor] = []
    exact: list[torch.Tensor] = []
    slots = sample_stop - sample_start
    for motion in motions:
        record = source.load_window(
            motion_index=motion, time_start=time_start, time_stop=time_stop,
            sample_start=sample_start, sample_stop=sample_stop, reference=reference,
        )
        times = time_stop - time_start
        initial.append(record["initial_y"].reshape(times * slots, -1))
        q.append(record["q"].unsqueeze(1).expand(-1, slots, -1).reshape(times * slots, -1))
        masses.append(record["masses"].unsqueeze(1).expand(-1, slots, -1).reshape(times * slots, -1))
        exact.append(record["exact_y"].unsqueeze(1).expand(-1, slots, -1).reshape(times * slots, -1))
    return {
        "initial_y": torch.cat(initial).to(device=device, dtype=TORCH_DTYPE),
        "q": torch.cat(q).to(device=device, dtype=TORCH_DTYPE),
        "masses": torch.cat(masses).to(device=device, dtype=TORCH_DTYPE),
        "exact_y": torch.cat(exact).to(device=device, dtype=TORCH_DTYPE),
    }


def rollout_loss(model: MLPOptimizer, batch: dict[str, torch.Tensor], physical, steps: int, scale_length: float):
    y = project_fixed_vertices(batch["initial_y"].clone(), physical)
    prev_r = torch.zeros_like(y)
    prev_u = torch.zeros_like(y)
    for _ in range(steps):
        y, delta, current = apply_model_update(
            model, y, batch["q"], batch["masses"], physical,
            previous_residual=prev_r, previous_update=prev_u,
        )
        prev_r = current.detach()
        prev_u = delta.detach()
    exact_energy = variational_energy_full(batch["exact_y"], batch["q"], batch["masses"], physical).detach()
    energy = variational_energy_full(y, batch["q"], batch["masses"], physical)
    gap = torch.clamp(energy - exact_energy, min=0.0)
    scale = physical_energy_scale(batch["masses"].detach(), physical, scale_length)
    return gap.mean() / max(scale, 1e-30), float(gap.mean().detach().cpu().item())


def model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    biases = [False, True] if args.bias_mode == "both" else [args.bias_mode == "with-bias"]
    specs = [ModelSpec(a, int(d), int(w), b) for a in args.activations for d in args.depths for w in args.widths for b in biases]
    return [specs[args.config_index]] if args.config_index is not None else specs


def checkpoint(path: Path, model, optimizer, epoch, spec, best, config):
    torch.save({
        "epoch": int(epoch), "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "model_spec": asdict(spec),
        "best_validation": float(best), "config": config,
    }, path)


def train_one(args, spec, physical, device, train_manifest, source, validation, tests, reference):
    out = args.root / "experiments" / args.stage / f"samples_{args.sample_count:04d}" / spec.experiment_name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completed.json").exists() and args.skip_completed and not args.overwrite:
        print(f"skip completed {out}")
        return
    torch.manual_seed(args.seed)
    model = MLPOptimizer(args.residual_length_scale, spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch, best = 1, math.inf
    history: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    latest = out / "latest_checkpoint.pt"
    if args.resume and latest.exists() and not args.overwrite:
        saved = torch.load(latest, map_location=device)
        model.load_state_dict(saved["model_state_dict"]); optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"]) + 1; best = float(saved.get("best_validation", math.inf))
    config = {
        "stage": args.stage, "sample_count": args.sample_count, "sample_source": str(args.sample_source_root),
        "model_spec": asdict(spec), "architecture": model.architecture_description,
        "parameter_count": model.parameter_count, "epochs": args.epochs,
        "k_values": args.k_values, "epochs_per_k": args.epochs_per_k,
        "validation": "all validation motions/all 500 frames, y0=x_n, one learned update",
        "checkpoint_metric": "p95(log10(r1/r0))", "sample_chunk_size_per_problem": args.sample_chunk_size,
        "residual_length_scale": args.residual_length_scale,
    }
    save_json(config, out / "config.json")
    motions = [int(v) for v in train_manifest["train_motion_indices"]]
    windows = train_manifest["windows"]
    chunk_size = args.sample_chunk_size or args.sample_count
    if args.sample_count > source.available_points:
        raise ValueError(f"sample_count={args.sample_count} exceeds available={source.available_points}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); k = k_for_epoch(epoch, args.k_values, args.epochs_per_k)
        epoch_start = time.perf_counter(); epoch_losses = []
        for window in windows:
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            for s0 in range(0, args.sample_count, chunk_size):
                s1 = min(s0 + chunk_size, args.sample_count)
                batch = make_train_chunk(
                    source=source, reference=reference, motions=motions,
                    time_start=int(window["time_start"]), time_stop=int(window["time_stop"]),
                    sample_start=s0, sample_stop=s1, device=device,
                )
                loss, _ = rollout_loss(model, batch, physical, k, args.residual_length_scale)
                weight = (s1 - s0) / args.sample_count
                (loss * weight).backward()
                weighted_loss += float(loss.detach().cpu().item()) * weight
                del batch, loss
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step(); epoch_losses.append(weighted_loss)
        row = {"epoch": epoch, "k": k, "loss_mean": float(np.mean(epoch_losses)), "elapsed_seconds": time.perf_counter()-epoch_start}
        logs.append(row); write_csv(logs, out / "train_log.csv")
        checkpoint(latest, model, optimizer, epoch, spec, best, config)
        if epoch == 1 or epoch % args.validation_interval == 0 or epoch == args.epochs:
            val = evaluate_one_step(model=model, dataset=validation, physical=physical, device=device, batch_size=args.evaluation_batch_size)
            record = {"epoch": epoch, **val["summary"]}; history.append(record)
            save_json({"history": history}, out / "validation_metrics.json")
            torch.save({"residual_before": val["residual_before"], "residual_after": val["residual_after"]}, out / f"validation_epoch_{epoch:04d}.pt")
            score = float(val["summary"]["selection_metric"])
            best_path = out / "best_validation_model.pt"
            if (not best_path.exists()) or score < best:
                best = score; checkpoint(best_path, model, optimizer, epoch, spec, best, config)
            print(f"{spec.experiment_name} epoch={epoch:04d} K={k} loss={row['loss_mean']:.3e} val_p95_log_ratio={score:.3e} best={best:.3e}")
    saved = torch.load(out / "best_validation_model.pt", map_location=device); model.load_state_dict(saved["model_state_dict"])
    test_summary = {}; test_curves = {}
    for name, dataset in tests.items():
        result = evaluate_one_step(model=model, dataset=dataset, physical=physical, device=device, batch_size=args.evaluation_batch_size)
        test_summary[name] = result["summary"]; test_curves[name] = {"r0": result["residual_before"], "r1": result["residual_after"]}
    save_json(test_summary, out / "test_metrics.json"); torch.save(test_curves, out / "test_curves.pt")
    save_json({"completed": True, "best_validation": best}, out / "completed.json")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train 15x15 learned optimizer architecture/data ablations.")
    p.add_argument("--root", type=Path, default=Path("cloth_15x15_500step_pipeline"))
    p.add_argument("--stage", default="custom")
    p.add_argument("--sample-source-root", type=Path, default=None)
    p.add_argument("--sample-count", type=int, default=32)
    p.add_argument("--sample-chunk-size", type=int, default=0, help="samples per problem per gradient chunk; 0 means all")
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--validation-interval", type=int, default=50)
    p.add_argument("--evaluation-batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    p.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    p.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    p.add_argument("--epochs-per-k", type=int, default=100)
    p.add_argument("--activations", nargs="+", default=["relu"], choices=ACTIVATION_NAMES)
    p.add_argument("--depths", type=int, nargs="+", default=[1])
    p.add_argument("--widths", type=int, nargs="+", default=[256])
    p.add_argument("--bias-mode", choices=("no-bias", "with-bias", "both"), default="no-bias")
    p.add_argument("--config-index", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--list-configs", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv); args.sample_source_root = args.sample_source_root or (args.root / "data" / "samples")
    specs = model_specs(args)
    if args.list_configs:
        for i, spec in enumerate(specs): print(i, spec.experiment_name)
        return
    device = torch.device(args.device); physical = load_physical(args.root)
    train_manifest = load_json(args.root / "data" / "datasets" / "train_manifest.json")
    source = SampleSource(args.sample_source_root, train_manifest)
    reference = None
    if source.format == "window_shards_v1":
        reference = torch.load(args.root / "data" / "reference" / "reference_problems.pt", map_location="cpu")
    validation = load_dataset("validation_xn", args.root)
    tests = {name: load_dataset(name, args.root) for name in ("test_id_xn", "test_ood_xn", "test_all_xn")}
    for spec in specs:
        train_one(args, spec, physical, device, train_manifest, source, validation, tests, reference)


if __name__ == "__main__": main()
