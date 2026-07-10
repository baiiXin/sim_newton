# Metamizer-Style Pool Training for Cloth 5×5

This experiment compares the existing 500-step dataset training pipeline with a
Metamizer-style live training pool.

The key change is that training no longer uses the full 500-step training
dataset. The pool is initialized only from training motion initial states.

## Files

```text
cloth13_train_metamizer_pool_models.py
cloth14_evaluate_pool_vs_existing_rollouts.py
```

- `cloth13` trains learned optimizers from a live pool.
- `cloth14` compares the pool-trained models with existing models by continuous
  500-frame rollout with 50 inner iterations per frame.

The existing `cloth10`–`cloth12` initial-point sampling ablation is left
unchanged.

## Training semantics

For every training motion, the pool creates five environments:

```text
iterations_per_timestep = 1, 3, 5, 10, 30
```

With the default 16 training motions this gives:

```text
16 motions × 5 K-buckets = 80 live environments
```

One parameter update means exactly one learned optimizer update for every live
environment:

```text
optimizer.step() == one neural update, not one physical step
```

A K-bucket advances the physical environment only after K learned updates:

```text
K = 1  -> 1000 physical steps per epoch
K = 3  -> 333 physical steps per epoch
K = 5  -> 200 physical steps per epoch
K = 10 -> 100 physical steps per epoch
K = 30 -> 33 physical steps per epoch
```

The default schedule is:

```text
epochs            = 50
updates_per_epoch = 1000
total updates     = 50,000
```

## State update rule

The learned optimizer update is unchanged from the existing 75D full-state model:

```text
input  = [current residual, previous residual, previous update]
output = 75D full-state displacement update
fixed vertices are gated/projected after the update
```

When an environment completes its K inner updates, it advances:

```text
x_{n+1} = y
v_{n+1} = (x_{n+1} - x_n) / dt
```

The next physical frame uses:

```text
y^(0) = x_n
```

This matches the continuous rollout evaluator in `cloth12`.

## Loss

The pool training loss is intentionally simple:

```text
loss = mean(variational_energy_full(y_after_one_update, q, masses)) / physical_energy_scale
```

There is:

```text
no exact_y
no energy - exact_energy
no K-step unroll
no K-step average loss
```

## Pool reset checks

Every pool update checks for bad states and resets the corresponding environment
to the motion initial state when needed.

Default reset triggers:

```text
non-finite y / energy / residual
abs(energy) > 1e8
residual > 1e8
max abs position > 1e3
min spring length < 1e-8
max spring length > 1e3
physical age >= 500 steps
```

The training log records reset counts by reason.

## Commands

Run from `cloth_5x5_500step_project/`.

### Smoke test

```bash
python cloth13_train_metamizer_pool_models.py \
  --source-root cloth_5x5_500step_pipeline \
  --pool-root cloth_5x5_metamizer_pool_training \
  --activations identity \
  --epochs 2 \
  --updates-per-epoch 20 \
  --validation-interval 1 \
  --validation-rollout-length 5 \
  --validation-inner-steps 3 \
  --device cuda:0 \
  --overwrite
```

### Formal pool training

```bash
python cloth13_train_metamizer_pool_models.py \
  --source-root cloth_5x5_500step_pipeline \
  --pool-root cloth_5x5_metamizer_pool_training \
  --activations identity relu tanh \
  --depths 1 \
  --widths 256 \
  --epochs 50 \
  --updates-per-epoch 1000 \
  --validation-interval 10 \
  --validation-rollout-length 100 \
  --validation-inner-steps 50 \
  --device cuda:0 \
  --resume
```

### Compare pool models against existing 500-step models

Default evaluation uses motion 20–31, rollout length 500, and inner steps 50:

```bash
python cloth14_evaluate_pool_vs_existing_rollouts.py \
  --source-root cloth_5x5_500step_pipeline \
  --pool-root cloth_5x5_metamizer_pool_training \
  --motion-indices 20 21 22 23 24 25 26 27 28 29 30 31 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0
```

To include the `points_0032` initial-point ablation model and baselines:

```bash
python cloth14_evaluate_pool_vs_existing_rollouts.py \
  --source-root cloth_5x5_500step_pipeline \
  --pool-root cloth_5x5_metamizer_pool_training \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --include-points-0032 \
  --baselines gd adam lbfgs newton \
  --motion-indices 20 21 22 23 24 25 26 27 28 29 30 31 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0
```

To compare all named motions:

```bash
python cloth14_evaluate_pool_vs_existing_rollouts.py \
  --motion-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
                   16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0
```

## Outputs

Training outputs:

```text
cloth_5x5_metamizer_pool_training/
└── models/
    └── activation_<activation>_depth_01_width_256_no_bias/
        ├── config.json
        ├── pool_manifest.json
        ├── train_log.csv
        ├── validation_metrics.json
        ├── latest_checkpoint.pt
        └── best_validation_model.pt
```

Rollout comparison outputs:

```text
cloth_5x5_metamizer_pool_training/
└── rollout_evaluation/
    ├── all_motion_summary.csv
    ├── run_config.json
    └── motion_020/
        ├── full_500step_identity/curve.pt
        ├── pool_identity/curve.pt
        ├── ...
        ├── summary_metrics.csv
        ├── all_curves.pt
        └── figures/
            ├── rollout_x_iteration_vs_residual.png
            └── rollout_frame_vs_final_residual.png
```

Each curve follows the `cloth12` format:

```text
residual_by_frame_and_iteration     # [completed_frames, inner_steps + 1]
final_residual_by_frame
global_residual
positions
velocities
reference_error_by_frame
```

## Main interpretation

This experiment tests whether a learned optimizer trained only from initial
states and its own live residual distribution can match or exceed the continuous
rollout stability of the model trained on the full 500-step time-step dataset.
