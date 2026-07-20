# T-shirt shared-weight GNN baseline

This adds a parallel GNN baseline without deleting the existing dense MLP path.
It reuses the current online randomized training pool, energy loss, frozen
validation/test sets, failure checks, and single-motion evaluation protocol.

## Architecture

For every vertex, concatenate three world-space 3-vectors:

- current **raw** stationarity residual;
- previous raw residual;
- previous predicted displacement.

The resulting 9D feature is encoded by a two-layer `9 -> 128 -> 128` ReLU MLP.
The model then performs 15 message-passing rounds with one shared edge MLP and
one shared node-update MLP across all rounds:

1. each undirected mesh edge creates two directed messages;
2. a message uses `[receiver_hidden, sender_hidden]` and a shared
   `256 -> 128 -> 128` ReLU MLP;
3. incoming messages are summed at each vertex;
4. the node update uses `[node_hidden, summed_message]` and a shared
   `256 -> 128 -> 128` ReLU MLP;
5. the node update is added through a residual connection.

The decoder is `128 -> 128 -> 3`, with ReLU only after the first linear layer.
All linear layers have no bias. The node-update final layer and decoder final
layer are zero initialized.

This first baseline deliberately has:

- no mass preconditioning of the residual;
- no fixed-point indicator in the input;
- no edge attributes;
- no input normalization or output scale factor;
- shared processor parameters for all 15 rounds.

Fixed vertices are still hard-gated to zero at the output and projected by the
existing pipeline.

## Unit test

```bash
python -m unittest -v test_tshirt_gnn.py
```

## Train

Start with a small batch because 15 rounds retain edge and node activations for
backpropagation:

```bash
python cloth17_train_gnn_online.py \
  --device cuda:0 \
  --dtype float64 \
  --pool-size 512 \
  --batch-size 4 \
  --max-wall-hours 10
```

The default output root is `cloth_tshirt_gnn_pipeline/`, and the default run
name records raw residual input, 15 shared message-passing rounds, width 128,
two-layer MLPs, and no bias.

## Full frozen validation/test

```bash
python cloth18_evaluate_gnn_checkpoint.py \
  --checkpoint cloth_tshirt_gnn_pipeline/gnn_raw_residual_mp15_width_0128_depth_02_no_bias/seed_42/best_validation_model.pt \
  --rollout-frames 500 \
  --inner-steps 50
```

## Single-motion network rollout

```bash
python cloth19_rollout_gnn_single_motion.py \
  --mode network \
  --split typical \
  --motion-index 0 \
  --checkpoint /path/to/best_validation_model.pt \
  --rollout-frames 500 \
  --inner-steps 50
```
