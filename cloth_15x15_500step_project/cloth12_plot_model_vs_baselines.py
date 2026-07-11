"""Plot learned optimizer residual-vs-iteration curves against all baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cloth_common import save_json, summarize_residual_curve

DEFAULT_DATASETS = ("validation_xn", "test_id_xn", "test_ood_xn", "test_all_xn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("cloth_15x15_500step_pipeline")
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="directory containing evaluation_curves.pt from cloth05 or cloth13",
    )
    parser.add_argument("--model-label", default="learned_optimizer")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    baseline_path = args.root / "baselines" / "baseline_curves.pt"
    model_path = args.experiment_dir / "evaluation_curves.pt"
    if not baseline_path.exists():
        raise FileNotFoundError(f"run cloth08_evaluate_baselines.py first: {baseline_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"missing model curves: {model_path}")

    baseline_curves = torch.load(baseline_path, map_location="cpu")
    model_curves = torch.load(model_path, map_location="cpu")
    output_dir = args.output_dir or (args.experiment_dir / "figures" / "vs_baselines")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_output = {}

    fields = {
        "mean": "residual_mean_by_iter",
        "p95": "residual_p95_by_iter",
        "max": "residual_max_by_iter",
    }
    for dataset_name in args.datasets:
        if dataset_name not in baseline_curves or dataset_name not in model_curves:
            raise KeyError(f"dataset {dataset_name} missing from baseline or model curves")
        all_curves = dict(baseline_curves[dataset_name])
        all_curves[args.model_label] = model_curves[dataset_name]
        summaries = {
            label: summarize_residual_curve(curve.numpy())
            for label, curve in all_curves.items()
        }
        summary_output[dataset_name] = summaries

        for statistic, field in fields.items():
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for label, summary in summaries.items():
                values = np.asarray(summary[field], dtype=float)
                ax.plot(
                    np.arange(len(values)),
                    np.maximum(values, 1e-30),
                    label=label,
                    linewidth=2.0 if label == args.model_label else 1.2,
                )
            ax.set_yscale("log")
            ax.set_xlabel("inner iteration")
            ax.set_ylabel(f"{statistic} stationarity residual")
            ax.set_title(f"{dataset_name}: learned optimizer vs. baselines")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                output_dir / f"{dataset_name}_{statistic}_residual_vs_iteration.png",
                dpi=200,
            )
            plt.close(fig)

    save_json(summary_output, output_dir / "model_vs_baselines_metrics.json")
    print(output_dir)


if __name__ == "__main__":
    main()
