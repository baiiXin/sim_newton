"""Evaluate missing checkpoints on one motion, reuse baselines, then plot four comparisons."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np

from tshirt_config import DEFAULT_FIXED_DATA_DIR, write_json


DEFAULT_ROOT = Path("cloth_tshirt_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument(
        "--checkpoint-kind", choices=("best", "latest", "periodic", "all"), default="best"
    )
    parser.add_argument("--periodic-update", type=int, action="append", default=None)
    parser.add_argument("--include", default=None, help="regex applied to experiment/seed/checkpoint")
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--require-completed", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--split", choices=("typical", "validation", "test"), default="typical")
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--rollout-frames", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--residual-ratio-tolerance", type=float, default=1e-3)
    parser.add_argument("--fixed-gd-step-size", type=float, default=5e-5)
    parser.add_argument("--mass-ls-step-size", type=float, default=1.0)
    parser.add_argument("--block-ls-step-size", type=float, default=1.0)
    parser.add_argument("--line-search-max-trials", type=int, default=12)
    parser.add_argument("--trajectory-stride", type=int, default=5)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _seed_dir(checkpoint: Path) -> Path:
    return checkpoint.parent.parent if checkpoint.parent.name == "periodic" else checkpoint.parent


def _checkpoint_label(checkpoint: Path) -> str:
    if checkpoint.parent.name == "periodic":
        return checkpoint.stem.replace("checkpoint_update_", "periodic_")
    if checkpoint.name == "best_validation_model.pt":
        return "best"
    if checkpoint.name == "latest_checkpoint.pt":
        return "latest"
    return checkpoint.stem


def _completed(seed_dir: Path) -> bool:
    path = seed_dir / "completed.json"
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("completed"))
    except (OSError, json.JSONDecodeError):
        return False


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    root = args.root.resolve()
    candidates: list[Path] = []
    if args.checkpoint_kind in ("best", "all"):
        candidates.extend(root.glob("**/best_validation_model.pt"))
    if args.checkpoint_kind in ("latest", "all"):
        candidates.extend(root.glob("**/latest_checkpoint.pt"))
    if args.checkpoint_kind in ("periodic", "all"):
        if args.periodic_update:
            for update in args.periodic_update:
                candidates.extend(root.glob(f"**/periodic/checkpoint_update_{update:09d}.pt"))
        else:
            candidates.extend(root.glob("**/periodic/checkpoint_update_*.pt"))
    filtered: list[Path] = []
    for checkpoint in sorted(set(path.resolve() for path in candidates)):
        seed_dir = _seed_dir(checkpoint)
        label = f"{seed_dir.parent.name}/{seed_dir.name}/{_checkpoint_label(checkpoint)}"
        if args.include and re.search(args.include, label) is None:
            continue
        if args.exclude and re.search(args.exclude, label) is not None:
            continue
        if args.require_completed and not _completed(seed_dir):
            continue
        filtered.append(checkpoint)
    if args.limit is not None:
        filtered = filtered[:args.limit]
    return filtered


def _motion_id(args: argparse.Namespace) -> str:
    names = {
        "typical": "typical_single_motions_4.npz",
        "validation": "validation_32.npz",
        "test": "test_64.npz",
    }
    with np.load(Path(args.fixed_data_dir) / names[args.split]) as data:
        ids = tuple(str(item) for item in data["motion_ids"].tolist())
    if args.motion_index < 0 or args.motion_index >= len(ids):
        raise ValueError(f"motion index must be in [0, {len(ids) - 1}]")
    return ids[args.motion_index]


def _common_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        str(Path(__file__).resolve().parent / "cloth09_rollout_single_motion.py"),
        "--root", str(args.root.resolve()),
        "--fixed-data-dir", str(Path(args.fixed_data_dir).resolve()),
        "--device", args.device,
        "--dtype", args.dtype,
        "--split", args.split,
        "--motion-index", str(args.motion_index),
        "--rollout-frames", str(args.rollout_frames),
        "--inner-steps", str(args.inner_steps),
        "--residual-ratio-tolerance", str(args.residual_ratio_tolerance),
        "--fixed-gd-step-size", str(args.fixed_gd_step_size),
        "--mass-ls-step-size", str(args.mass_ls_step_size),
        "--block-ls-step-size", str(args.block_ls_step_size),
        "--line-search-max-trials", str(args.line_search_max_trials),
        "--trajectory-stride", str(args.trajectory_stride),
    ]


def _run(command: list[str], dry_run: bool) -> tuple[str, str]:
    print(" ".join(command))
    if dry_run:
        return "planned", ""
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        return "failed", completed.stderr
    status = "skipped" if "skipped:" in completed.stdout else "completed"
    return status, completed.stdout


def _read_result(directory: Path, solver: str, label: str, kind: str) -> dict[str, Any] | None:
    metrics_path = directory / solver / "metrics.json"
    curves_path = directory / solver / "curves.npz"
    if not metrics_path.exists() or not curves_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    curves_file = np.load(curves_path)
    curves = {key: np.asarray(curves_file[key]) for key in curves_file.files}
    return {
        "label": label,
        "kind": kind,
        "solver": solver,
        "directory": str(directory),
        "metrics": metrics,
        "curves": curves,
    }


def _plot_comparisons(results: list[dict[str, Any]], output: Path, threshold: float) -> list[Path]:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    for index, item in enumerate(results):
        values = item["curves"]["residual_ratio"]
        frames = np.arange(values.size)
        valid = np.isfinite(values) & (values > 0)
        axis.semilogy(
            frames[valid], values[valid], label=item["label"],
            color=colors(index % 10), ls="--" if item["kind"] == "baseline" else "-",
        )
    axis.axhline(1e-3, color="black", ls=":", lw=1, label="target 1e-3")
    axis.set(xlabel="physical frame", ylabel="final / initial residual", title="Per-frame solver convergence")
    axis.grid(True, which="both", alpha=0.25); axis.legend(fontsize=7, ncol=2)
    path = output / "01_residual_ratio_by_frame.png"
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure); paths.append(path)

    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    for index, item in enumerate(results):
        values = item["curves"]["first_step_ratio"]
        valid = np.isfinite(values) & (values > 0)
        axis.semilogy(
            np.arange(values.size)[valid], values[valid], label=item["label"],
            color=colors(index % 10), ls="--" if item["kind"] == "baseline" else "-",
        )
    axis.axhline(threshold, color="black", ls=":", lw=1, label=f"two orders ({threshold:g})")
    axis.set(xlabel="physical frame", ylabel="residual after first step / initial", title="Single-step decrease")
    axis.grid(True, which="both", alpha=0.25); axis.legend(fontsize=7, ncol=2)
    path = output / "02_first_step_ratio_by_frame.png"
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure); paths.append(path)

    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    for index, item in enumerate(results):
        values = np.cumsum(item["curves"]["objective_evaluations"])
        axis.plot(
            np.arange(values.size), values, label=item["label"],
            color=colors(index % 10), ls="--" if item["kind"] == "baseline" else "-",
        )
    axis.set(xlabel="physical frame", ylabel="cumulative objective/residual evaluations", title="Solver work")
    axis.grid(True, alpha=0.25); axis.legend(fontsize=7, ncol=2)
    path = output / "03_cumulative_objective_evaluations.png"
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure); paths.append(path)

    labels = [item["label"] for item in results]
    metrics = (
        ("residual_ratio_p95", "residual ratio p95", True),
        ("converged_frame_fraction", "converged frame fraction", False),
        ("single_step_le_two_orders_frame_count", "slow first-step frames", False),
        ("survival_frames", "survival frames", False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (key, title, log_y) in zip(axes.flat, metrics):
        values = [float(item["metrics"][key]) for item in results]
        axis.bar(np.arange(len(values)), values, color=[colors(i % 10) for i in range(len(values))])
        if log_y and all(value > 0 for value in values):
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right", fontsize=7)
        axis.grid(True, axis="y", which="both", alpha=0.25)
    path = output / "04_terminal_metrics.png"
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure); paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    if not 1 <= args.line_search_max_trials <= 12:
        raise ValueError("line-search trials must be in [1, 12]")
    checkpoints = discover_checkpoints(args)
    if not checkpoints:
        raise SystemExit("No trained checkpoints matched the scan options.")
    motion_id = _motion_id(args)
    summary_dir = (
        Path(args.summary_dir)
        if args.summary_dir is not None
        else Path(args.root) / "single_motion_scans" / f"{args.split}_{args.motion_index:04d}_{motion_id}"
    ).resolve()
    baseline_dir = (
        Path(args.root) / "single_motion_baselines" /
        f"{args.split}_{args.motion_index:04d}_{motion_id}"
    ).resolve()
    common = _common_command(args)
    baseline_command = [*common, "--mode", "baseline"]
    if args.overwrite:
        baseline_command.append("--overwrite")
    statuses: list[dict[str, Any]] = []
    status, message = _run(baseline_command, args.dry_run)
    statuses.append({"kind": "baseline", "status": status, "directory": str(baseline_dir), "message": message})
    if status == "failed" and args.stop_on_error:
        raise SystemExit(1)

    network_dirs: list[tuple[Path, str]] = []
    for checkpoint in checkpoints:
        seed_dir = _seed_dir(checkpoint)
        label = f"{seed_dir.parent.name}/{seed_dir.name}/{_checkpoint_label(checkpoint)}"
        output = (
            summary_dir / "network_results" / _safe(seed_dir.parent.name) /
            _safe(seed_dir.name) / _safe(_checkpoint_label(checkpoint))
        )
        command = [
            *common,
            "--mode", "network",
            "--checkpoint", str(checkpoint),
            "--output-dir", str(output),
        ]
        if args.overwrite:
            command.append("--overwrite")
        status, message = _run(command, args.dry_run)
        statuses.append(
            {
                "kind": "network", "label": label, "checkpoint": str(checkpoint),
                "status": status, "directory": str(output), "message": message,
            }
        )
        if status != "failed":
            network_dirs.append((output, label))
        elif args.stop_on_error:
            break

    results: list[dict[str, Any]] = []
    if not args.dry_run and statuses[0]["status"] != "failed":
        baseline_labels = {
            "gd_fixed": "GD fixed",
            "gd_mass_ls": "GD mass+LS",
            "gd_block3x3_ls": "GD 3x3-H+LS",
        }
        for solver, label in baseline_labels.items():
            item = _read_result(baseline_dir, solver, label, "baseline")
            if item is not None:
                results.append(item)
        for directory, label in network_dirs:
            item = _read_result(directory, "network", label, "network")
            if item is not None:
                results.append(item)
    plots: list[Path] = []
    if results:
        plots = _plot_comparisons(results, summary_dir / "figures", 1e-2)
    write_json(
        summary_dir / "scan_manifest.json",
        {
            "completed": not any(item["status"] == "failed" for item in statuses),
            "motion_id": motion_id,
            "configuration": vars(args),
            "baseline_directory": str(baseline_dir),
            "checkpoint_count": len(checkpoints),
            "statuses": statuses,
            "result_count": len(results),
            "comparison_plots": [str(path) for path in plots],
        },
    )
    print(
        f"scan finished: completed={sum(item['status'] == 'completed' for item in statuses)} "
        f"skipped={sum(item['status'] == 'skipped' for item in statuses)} "
        f"failed={sum(item['status'] == 'failed' for item in statuses)}; "
        f"plots={len(plots)}"
    )


if __name__ == "__main__":
    main()
