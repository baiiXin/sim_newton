"""Evaluate a GNN checkpoint with the existing frozen validation/test protocol."""
from __future__ import annotations

import cloth07_evaluate_checkpoint as evaluation
from cloth16_gnn_model import load_gnn_checkpoint


def main() -> None:
    evaluation.load_model_checkpoint = load_gnn_checkpoint
    evaluation.main()


if __name__ == "__main__":
    main()
