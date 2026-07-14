"""Build and audit deterministic scale-up scenario catalogues."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenario_catalogue import audit_catalogues, build_catalogues, save_catalogues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cloth_5x5_scale_up_pipeline/data/scenarios"),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Construct and audit in memory without writing catalogue files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalogues = build_catalogues()
    audit = audit_catalogues(catalogues)
    print(json.dumps({
        "counts": {name: len(items) for name, items in catalogues.items()},
        "c1_pairwise_coverage": audit["catalogues"]["train_c1_1024"]["pairwise_coverage"],
        "geometry": audit["geometry"],
        "tests": audit["tests"],
        "passed": audit["passed"],
    }, indent=2, ensure_ascii=False))
    if not audit["passed"]:
        raise SystemExit("Scenario catalogue audit failed")
    if not args.audit_only:
        save_catalogues(catalogues, args.output_dir)
        print(f"Wrote deterministic catalogues to {args.output_dir}")


if __name__ == "__main__":
    main()
