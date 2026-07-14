"""Run one selected motion rollout for every trained scale-up model."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Sequence


DEFAULT_ROOT = Path("cloth_15x15_scale_up_pipeline")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
CHECKPOINT_FILENAMES = {
    "best": "best_validation_model.pt",
    "latest": "latest_checkpoint.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan trained model directories and run cloth09_rollout_single_motion_compare.py "
            "on the same selected motion for each checkpoint."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--catalogue", choices=("c1", "c2", "c3"), default="c2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="scan every seed_* directory instead of only --seed",
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=("best", "latest", "periodic", "all"),
        default="best",
        help="which checkpoint type to run from each model directory",
    )
    parser.add_argument(
        "--periodic-update",
        type=int,
        action="append",
        default=None,
        help=(
            "specific periodic update to run; may be passed multiple times. "
            "Without this, --checkpoint-kind periodic/all uses every periodic checkpoint."
        ),
    )
    parser.add_argument(
        "--include",
        default=None,
        help="regex filter applied to the experiment directory name",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="regex filter applied to the experiment directory name",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refresh-baseline", action="store_true")
    parser.add_argument(
        "--aggregate-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write aggregate comparison plots into the scan summary directory",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch cloth09; defaults to this interpreter",
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float64", "float32"), default="auto")
    parser.add_argument(
        "--split",
        choices=("validation", "test", "train"),
        default="test",
    )
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--rollout-frames", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--fixed-gd-step-size", type=float, default=5e-5)
    parser.add_argument("--line-search-gd-step-size", type=float, default=5e-5)
    parser.add_argument("--line-search-gd-reductions", type=int, default=30)
    parser.add_argument("--line-search-gd-growths", type=int, default=8)
    parser.add_argument("--mass-preconditioned-gd-step-size", type=float, default=1.0)
    parser.add_argument("--baseline-step-size", type=float, default=1.0)
    parser.add_argument("--baseline-line-search-reductions", type=int, default=12)
    parser.add_argument("--lbfgs-history-size", type=int, default=5)
    parser.add_argument("--lbfgs-step-size", type=float, default=1.0)
    parser.add_argument("--lbfgs-line-search-reductions", type=int, default=30)
    parser.add_argument("--disable-inner-early-stop", action="store_true")
    parser.add_argument("--render-format", choices=("mp4", "gif", "none"), default="mp4")
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-residual", type=float, default=1e12)
    parser.add_argument("--max-abs-position", type=float, default=1e4)
    parser.add_argument("--min-edge-ratio", type=float, default=1e-5)
    parser.add_argument("--max-edge-ratio", type=float, default=1e4)
    parser.add_argument("--max-constraint-error", type=float, default=1e-9)
    parser.add_argument("--summary-dir", type=Path, default=None)
    return parser.parse_args()


def script_directory() -> Path:
    return Path(__file__).resolve().parent


def resolve_root(root: Path) -> Path:
    if root.exists():
        return root.resolve()
    candidate = script_directory() / root
    if candidate.exists():
        return candidate.resolve()
    return root.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_name(seed_dir: Path) -> str:
    return seed_dir.parent.name


def seed_matches(seed_dir: Path, args: argparse.Namespace) -> bool:
    if args.all_seeds:
        return True
    return seed_dir.name == f"seed_{args.seed}"


def regex_matches(name: str, include: str | None, exclude: str | None) -> bool:
    if include is not None and re.search(include, name) is None:
        return False
    if exclude is not None and re.search(exclude, name) is not None:
        return False
    return True


def periodic_checkpoints(seed_dir: Path, updates: Sequence[int] | None) -> list[Path]:
    periodic_dir = seed_dir / "periodic"
    if updates:
        return [
            periodic_dir / f"checkpoint_update_{int(update):09d}.pt"
            for update in updates
        ]
    return sorted(periodic_dir.glob("checkpoint_update_*.pt"))


def checkpoint_label(checkpoint: Path, seed_dir: Path) -> str:
    if checkpoint.name == "best_validation_model.pt":
        return "best"
    if checkpoint.name == "latest_checkpoint.pt":
        return "latest"
    if checkpoint.parent == seed_dir / "periodic":
        stem = checkpoint.stem.replace("checkpoint_update_", "update_")
        return stem
    return checkpoint.stem


def fixed_gd_solver_name(step_size: float) -> str:
    return "gd_fixed_lr_5e-5" if step_size == 5e-5 else f"gd_fixed_lr_{step_size:g}"


def safe_float_label(value: float) -> str:
    return f"{float(value):g}".replace("+", "").replace("-", "m").replace(".", "p")


def discover_checkpoints(args: argparse.Namespace, root: Path) -> list[tuple[Path, Path]]:
    train_dir = root / "experiments" / f"train_{args.catalogue}"
    if not train_dir.exists():
        raise FileNotFoundError(train_dir)
    discovered: list[tuple[Path, Path]] = []
    seed_dirs = sorted(path for path in train_dir.glob("*/seed_*") if path.is_dir())
    for seed_dir in seed_dirs:
        name = experiment_name(seed_dir)
        if not seed_matches(seed_dir, args):
            continue
        if not regex_matches(name, args.include, args.exclude):
            continue
        if args.require_completed and not (seed_dir / "completed.json").exists():
            continue

        candidates: list[Path] = []
        if args.checkpoint_kind in {"best", "all"}:
            candidates.append(seed_dir / CHECKPOINT_FILENAMES["best"])
        if args.checkpoint_kind in {"latest", "all"}:
            candidates.append(seed_dir / CHECKPOINT_FILENAMES["latest"])
        if args.checkpoint_kind in {"periodic", "all"}:
            candidates.extend(periodic_checkpoints(seed_dir, args.periodic_update))

        for checkpoint in candidates:
            if checkpoint.exists():
                discovered.append((seed_dir, checkpoint))

    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be nonnegative")
        discovered = discovered[: args.limit]
    return discovered


def output_directory(
    *,
    seed_dir: Path,
    checkpoint: Path,
    args: argparse.Namespace,
) -> Path:
    label = checkpoint_label(checkpoint, seed_dir)
    name = (
        f"{args.split}_motion_{args.motion_index:04d}_"
        f"f{args.rollout_frames:03d}_k{args.inner_steps:03d}_"
        f"gd{safe_float_label(args.fixed_gd_step_size)}_{label}"
    )
    return seed_dir / "single_motion_rollout_scan" / name


def default_summary_dir(args: argparse.Namespace, root: Path) -> Path:
    name = (
        f"{args.catalogue}_{args.split}_motion_{args.motion_index:04d}_"
        f"f{args.rollout_frames:03d}_k{args.inner_steps:03d}_"
        f"gd{safe_float_label(args.fixed_gd_step_size)}_"
        f"{args.checkpoint_kind}"
    )
    if not args.all_seeds:
        name += f"_seed_{args.seed}"
    return root / "single_motion_rollout_scans" / name


def command_for(
    *,
    root: Path,
    checkpoint: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        args.python,
        str(script_directory() / "cloth09_rollout_single_motion_compare.py"),
        "--mode",
        "mlp",
        "--root",
        str(root),
        "--catalogue",
        args.catalogue,
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(out_dir),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--split",
        args.split,
        "--motion-index",
        str(args.motion_index),
        "--rollout-frames",
        str(args.rollout_frames),
        "--inner-steps",
        str(args.inner_steps),
        "--fixed-gd-step-size",
        str(args.fixed_gd_step_size),
        "--line-search-gd-step-size",
        str(args.line_search_gd_step_size),
        "--line-search-gd-reductions",
        str(args.line_search_gd_reductions),
        "--line-search-gd-growths",
        str(args.line_search_gd_growths),
        "--mass-preconditioned-gd-step-size",
        str(args.mass_preconditioned_gd_step_size),
        "--baseline-step-size",
        str(args.baseline_step_size),
        "--baseline-line-search-reductions",
        str(args.baseline_line_search_reductions),
        "--lbfgs-history-size",
        str(args.lbfgs_history_size),
        "--lbfgs-step-size",
        str(args.lbfgs_step_size),
        "--lbfgs-line-search-reductions",
        str(args.lbfgs_line_search_reductions),
        "--render-format",
        args.render_format,
        "--render-stride",
        str(args.render_stride),
        "--fps",
        str(args.fps),
        "--max-residual",
        str(args.max_residual),
        "--max-abs-position",
        str(args.max_abs_position),
        "--min-edge-ratio",
        str(args.min_edge_ratio),
        "--max-edge-ratio",
        str(args.max_edge_ratio),
        "--max-constraint-error",
        str(args.max_constraint_error),
    ]
    if args.disable_inner_early_stop:
        command.append("--disable-inner-early-stop")
    if args.refresh_baseline:
        command.append("--refresh-baseline")
    if args.overwrite:
        command.append("--overwrite")
    return command


def rollout_output_matches(out_dir: Path, args: argparse.Namespace) -> bool:
    metrics_path = out_dir / "metrics.json"
    payload_path = out_dir / "rollout_compare.pt"
    per_frame_path = out_dir / "per_frame.csv"
    if not metrics_path.exists() or not payload_path.exists() or not per_frame_path.exists():
        return False
    try:
        metrics = load_json(metrics_path)
    except Exception:
        return False
    if metrics.get("mode") != "mlp":
        return False
    checks = {
        "split": args.split,
        "motion_index": int(args.motion_index),
        "rollout_frames": int(args.rollout_frames),
        "inner_steps": int(args.inner_steps),
    }
    for key, expected in checks.items():
        if metrics.get(key) != expected:
            return False
    expected_solver = fixed_gd_solver_name(args.fixed_gd_step_size)
    for baseline in metrics.get("baselines", []):
        if not isinstance(baseline, dict):
            continue
        if baseline.get("solver") != expected_solver:
            continue
        return float(baseline.get("step_size", float("nan"))) == float(args.fixed_gd_step_size)
    return False


def should_skip(out_dir: Path, args: argparse.Namespace) -> bool:
    if args.overwrite:
        return False
    return rollout_output_matches(out_dir, args)


def write_summary(summary_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(list(rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    commands = [
        row["command"]
        for row in rows
        if row.get("command") is not None
    ]
    (summary_dir / "commands.txt").write_text(
        "\n".join(commands) + ("\n" if commands else ""),
        encoding="utf-8",
    )


def finite_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def read_per_frame(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def model_plot_label(row: dict[str, Any]) -> str:
    label = str(row["experiment"])
    checkpoint = str(row.get("checkpoint_label", ""))
    if checkpoint and checkpoint != "best":
        label = f"{label} {checkpoint}"
    return label


def collect_aggregate_series(
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    baseline_added: set[str] = set()
    baseline_solvers = {
        fixed_gd_solver_name(args.fixed_gd_step_size): f"fixed GD lr={args.fixed_gd_step_size:g}",
        "newton": "Newton",
    }
    for row in rows:
        if row.get("status") not in {"completed", "skipped_existing"}:
            continue
        out_dir = Path(str(row["output_dir"]))
        per_frame = out_dir / "per_frame.csv"
        if not per_frame.exists():
            continue
        frame_rows = read_per_frame(per_frame)
        mlp_rows = [item for item in frame_rows if item.get("solver") == "mlp"]
        if mlp_rows:
            series.append(
                {
                    "label": model_plot_label(row),
                    "kind": "model",
                    "solver": "mlp",
                    "rows": mlp_rows,
                    "source": str(per_frame),
                    "payload": str(out_dir / "rollout_compare.pt"),
                }
            )
        for solver, label in baseline_solvers.items():
            if solver in baseline_added:
                continue
            solver_rows = [item for item in frame_rows if item.get("solver") == solver]
            if not solver_rows:
                continue
            baseline_added.add(solver)
            series.append(
                {
                    "label": label,
                    "kind": "baseline",
                    "solver": solver,
                    "rows": solver_rows,
                    "source": str(per_frame),
                    "payload": str(out_dir / "rollout_compare.pt"),
                }
            )
    return series


def plot_aggregate_metric(
    *,
    output: Path,
    series: Sequence[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    log_y: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    model_index = 0
    plotted = False
    for item in series:
        points: list[tuple[int, float]] = []
        for frame_row in item["rows"]:
            frame_value = finite_float(frame_row.get("frame"))
            if not math.isfinite(frame_value):
                continue
            frame = int(frame_value)
            value = finite_float(frame_row.get(metric))
            if not math.isfinite(value):
                continue
            if log_y and value <= 0:
                continue
            points.append((frame, value))
        if not points:
            continue
        points.sort(key=lambda pair: pair[0])
        frames = [point[0] for point in points]
        values = [point[1] for point in points]
        if item["kind"] == "baseline":
            style = "--" if item["solver"] == "newton" else ":"
            color = "black" if item["solver"] == "newton" else "0.35"
            linewidth = 2.2
        else:
            style = "-"
            color = f"C{model_index % 10}"
            linewidth = 1.4
            model_index += 1
        ax.plot(
            frames,
            values,
            linestyle=style,
            color=color,
            linewidth=linewidth,
            label=str(item["label"]),
            alpha=0.95,
        )
        plotted = True
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("physical frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    if plotted:
        ax.legend(fontsize=7, ncol=1, loc="best")
    else:
        ax.text(0.5, 0.5, "no matching data", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def residual_frame_selection(series: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for item in series:
        for frame_row in item["rows"]:
            frame = finite_float(frame_row.get("frame"))
            residual = finite_float(frame_row.get("final_residual"))
            if not math.isfinite(frame) or not math.isfinite(residual):
                continue
            values.append(
                {
                    "frame": int(frame),
                    "final_residual": float(residual),
                    "label": str(item["label"]),
                    "solver": str(item["solver"]),
                }
            )
    if not values:
        empty = {
            "frame": 0,
            "final_residual": float("nan"),
            "label": "",
            "solver": "",
        }
        return empty, empty
    values.sort(key=lambda row: float(row["final_residual"]))
    median = values[len(values) // 2]
    worst = values[-1]
    return worst, median


def load_rollout_results(payload: Path, cache: dict[Path, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if payload not in cache:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "aggregate residual-vs-iteration plots require torch to read rollout_compare.pt"
            ) from exc
        loaded = torch.load(payload, map_location="cpu", weights_only=False)
        cache[payload] = list(loaded["results"])
    return cache[payload]


def result_for_solver(results: Sequence[dict[str, Any]], solver: str) -> dict[str, Any] | None:
    for result in results:
        if str(result.get("solver", "")) == solver:
            return result
    return None


def iteration_curve_for_frame(
    *,
    item: dict[str, Any],
    frame: int,
    cache: dict[Path, list[dict[str, Any]]],
) -> list[float]:
    payload = Path(str(item["payload"]))
    if not payload.exists():
        return []
    result = result_for_solver(load_rollout_results(payload, cache), str(item["solver"]))
    if result is None:
        return []
    inner = result.get("residual_by_frame_and_iteration")
    if inner is None:
        return []
    if hasattr(inner, "ndim") and hasattr(inner, "shape"):
        if int(getattr(inner, "ndim")) != 2 or frame < 0 or frame >= int(inner.shape[0]):
            return []
        raw_values = inner[frame].detach().cpu().double().tolist()
    else:
        if frame < 0 or frame >= len(inner):
            return []
        raw_values = inner[frame]
    values = [finite_float(value) for value in raw_values]
    return [value for value in values if math.isfinite(value)]


def plot_iteration_residual(
    *,
    output: Path,
    series: Sequence[dict[str, Any]],
    frame_info: dict[str, Any],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = int(frame_info["frame"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    cache: dict[Path, list[dict[str, Any]]] = {}
    model_index = 0
    plotted = False
    for item in series:
        values = iteration_curve_for_frame(item=item, frame=frame, cache=cache)
        points = [
            (index, value)
            for index, value in enumerate(values)
            if math.isfinite(value) and value > 0
        ]
        if not points:
            continue
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        if item["kind"] == "baseline":
            style = "--" if item["solver"] == "newton" else ":"
            color = "black" if item["solver"] == "newton" else "0.35"
            linewidth = 2.2
        else:
            style = "-"
            color = f"C{model_index % 10}"
            linewidth = 1.4
            model_index += 1
        ax.plot(
            x,
            y,
            linestyle=style,
            color=color,
            linewidth=linewidth,
            label=str(item["label"]),
            alpha=0.95,
        )
        plotted = True
    ax.set_yscale("log")
    ax.set_xlabel("inner iteration")
    ax.set_ylabel("residual")
    ax.set_title(
        f"{title}: frame={frame}, selected={frame_info['label']} "
        f"{float(frame_info['final_residual']):.3e}"
    )
    ax.grid(True, which="both", alpha=0.25)
    if plotted:
        ax.legend(fontsize=7, ncol=1, loc="best")
    else:
        ax.text(0.5, 0.5, "no matching residual curves", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_aggregate_plots(
    *,
    summary_dir: Path,
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[Path]:
    series = collect_aggregate_series(rows, args)
    figure_dir = summary_dir / "aggregate_plots"
    frame_worst, frame_median = residual_frame_selection(series)
    frame_specs = [
        (
            "01_final_residual_vs_frame.png",
            "final_residual",
            "final residual",
            "Final residual vs frame",
            True,
        ),
        (
            "02_residual_ratio_vs_frame.png",
            "residual_ratio",
            "residual ratio",
            "Residual ratio vs frame",
            True,
        ),
    ]
    outputs: list[Path] = []
    for filename, metric, ylabel, title, log_y in frame_specs:
        output = figure_dir / filename
        plot_aggregate_metric(
            output=output,
            series=series,
            metric=metric,
            ylabel=ylabel,
            title=title,
            log_y=log_y,
        )
        outputs.append(output)
    iteration_outputs = [
        (
            figure_dir / "03_inner_residual_worst_frame.png",
            frame_worst,
            "Inner residual vs iteration at worst frame",
        ),
        (
            figure_dir / "04_inner_residual_median_frame.png",
            frame_median,
            "Inner residual vs iteration at median frame",
        ),
    ]
    for output, frame_info, title in iteration_outputs:
        plot_iteration_residual(
            output=output,
            series=series,
            frame_info=frame_info,
            title=title,
        )
        outputs.append(output)
    manifest = {
        "series": [
            {
                "label": item["label"],
                "kind": item["kind"],
                "solver": item["solver"],
                "source": item["source"],
                "payload": item["payload"],
                "frame_count": len(item["rows"]),
            }
            for item in series
        ],
        "baseline_solvers": [
            fixed_gd_solver_name(args.fixed_gd_step_size),
            "newton",
        ],
        "fixed_gd_step_size": float(args.fixed_gd_step_size),
        "selected_frames": {
            "worst": frame_worst,
            "median": frame_median,
        },
        "figures": [str(path) for path in outputs],
    }
    (figure_dir / "aggregate_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    args = parse_args()
    root = resolve_root(args.root)
    summary_dir = args.summary_dir or default_summary_dir(args, root)
    checkpoints = discover_checkpoints(args, root)
    print(f"root={root}")
    print(f"matched_checkpoints={len(checkpoints)}")
    if not checkpoints:
        raise SystemExit("No checkpoints matched the scan options.")

    rows: list[dict[str, Any]] = []
    for index, (seed_dir, checkpoint) in enumerate(checkpoints, start=1):
        out_dir = output_directory(seed_dir=seed_dir, checkpoint=checkpoint, args=args)
        command = command_for(root=root, checkpoint=checkpoint, out_dir=out_dir, args=args)
        command_text = shlex.join(command)
        row = {
            "index": index,
            "experiment": experiment_name(seed_dir),
            "seed_dir": str(seed_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_label": checkpoint_label(checkpoint, seed_dir),
            "output_dir": str(out_dir),
            "command": command_text,
            "status": "pending",
            "returncode": None,
        }

        print(f"[{index}/{len(checkpoints)}] {row['experiment']} {row['checkpoint_label']}")
        if args.dry_run:
            row["status"] = "dry_run"
            print(command_text)
            rows.append(row)
            continue
        if should_skip(out_dir, args):
            row["status"] = "skipped_existing"
            print(f"skip existing: {out_dir}")
            rows.append(row)
            write_summary(summary_dir, rows)
            continue

        result = subprocess.run(command, cwd=script_directory(), check=False)
        row["returncode"] = int(result.returncode)
        row["status"] = "completed" if result.returncode == 0 else "failed"
        rows.append(row)
        write_summary(summary_dir, rows)
        if result.returncode != 0 and args.stop_on_error:
            raise SystemExit(result.returncode)

    failures = sum(1 for row in rows if row["status"] == "failed")
    completed = sum(1 for row in rows if row["status"] == "completed")
    skipped = sum(1 for row in rows if row["status"] == "skipped_existing")
    dry = sum(1 for row in rows if row["status"] == "dry_run")
    print(
        f"scan finished: completed={completed} skipped={skipped} "
        f"dry_run={dry} failed={failures}"
    )
    if args.aggregate_plots and not args.dry_run:
        outputs = write_aggregate_plots(summary_dir=summary_dir, rows=rows, args=args)
        for output in outputs:
            print(f"aggregate plot: {output}")
    print(f"summary_dir={summary_dir}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
