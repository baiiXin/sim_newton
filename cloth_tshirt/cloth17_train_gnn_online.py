"""Train the shared-weight raw-residual GNN on online-randomized T-shirt motions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cloth05_train_online as training
from cloth16_gnn_model import GNNModelSpec, LearnedOptimizerGNN, load_gnn_checkpoint


_ORIGINAL_MODEL_CHECKPOINT_PAYLOAD = training.model_checkpoint_payload


def _gnn_model_checkpoint_payload(**kwargs: Any) -> dict[str, Any]:
    payload = _ORIGINAL_MODEL_CHECKPOINT_PAYLOAD(**kwargs)
    payload.pop("residual_length_scale", None)
    payload.update(
        {
            "project": "cloth_tshirt_gnn_baseline",
            "model_type": "shared_message_passing_gnn",
            "node_input": ["raw_residual", "previous_raw_residual", "previous_delta"],
            "edge_attributes": False,
            "fixed_indicator_input": False,
            "input_scaling": False,
            "output_scaling": False,
            "message_passing_weights_shared": True,
        }
    )
    return payload


def main() -> None:
    # Reuse the mature online pool, validation, checkpoint, logging, and resume
    # machinery while replacing only the learned-optimizer architecture.
    training.DEFAULT_OUTPUT_ROOT = Path("cloth_tshirt_gnn_pipeline")
    training.ModelSpec = GNNModelSpec
    training.LearnedOptimizerMLP = LearnedOptimizerGNN
    training.model_checkpoint_payload = _gnn_model_checkpoint_payload
    training.load_model_checkpoint = load_gnn_checkpoint
    training.main()


if __name__ == "__main__":
    main()
