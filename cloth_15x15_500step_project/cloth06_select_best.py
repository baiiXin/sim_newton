"""Select the best completed configuration in one experiment stage."""
from __future__ import annotations

import argparse
from pathlib import Path

from cloth_common import load_json, save_json, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("cloth_15x15_500step_pipeline")
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--metric", default="selection_metric")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    stage_root = args.root / "experiments" / args.stage
    for path in stage_root.glob("samples_*/activation_*/validation_metrics.json"):
        data = load_json(path)
        history = data.get("history", [])
        if not history:
            continue
        best = min(history, key=lambda row: float(row.get(args.metric, float("inf"))))
        config = load_json(path.parent / "config.json")
        rows.append(
            {
                "experiment_dir": str(path.parent),
                "sample_count": config["sample_count"],
                **config["model_spec"],
                "parameter_count": config["parameter_count"],
                "best_epoch": best["epoch"],
                "selection_metric_name": best.get(
                    "selection_metric_name", config.get("checkpoint_metric", args.metric)
                ),
                args.metric: best[args.metric],
            }
        )
    if not rows:
        raise RuntimeError(f"no completed validation histories under {stage_root}")
    rows.sort(key=lambda row: (float(row[args.metric]), int(row["parameter_count"])))
    selection = {
        "stage": args.stage,
        "metric": args.metric,
        "metric_semantics": "default selection_metric is final validation residual p95 after 50 inner iterations",
        "best": rows[0],
        "top": rows[: args.top_k],
    }
    output = args.output or stage_root / "selection.json"
    save_json(selection, output)
    write_csv(rows, stage_root / "ranking.csv")
    print(
        f"best: {rows[0]['experiment_dir']} "
        f"{args.metric}={rows[0][args.metric]:.6e}"
    )
    print(
        f"width={rows[0]['width']} depth={rows[0]['depth']} "
        f"activation={rows[0]['activation']} bias={rows[0]['use_bias']}"
    )


if __name__ == "__main__":
    main()
