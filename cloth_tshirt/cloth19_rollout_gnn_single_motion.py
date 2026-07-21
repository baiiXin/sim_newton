"""Run the existing single-motion network protocol with a GNN checkpoint."""
from __future__ import annotations

import cloth09_rollout_single_motion as rollout
from cloth16_gnn_model import load_gnn_checkpoint


def main() -> None:
    rollout.load_model_checkpoint = load_gnn_checkpoint
    rollout.main()


if __name__ == "__main__":
    main()
