# Initial-Point Sampling Ablation

This experiment studies only the number of initial states sampled for each
motion/time-step problem:

```text
points_per_problem = {1, 8, 32, 64, 128, 1024}
```

All other settings are fixed. Every sample-count experiment includes the true
physical initial state. The remaining states are scrambled Sobol samples around
the stored reference solution.

## 1. Files

```text
cloth10_prepare_initial_point_ablation.py
cloth11_train_initial_point_ablation.py
cloth12_evaluate_initial_point_ablation_rollouts.py
```

- `cloth10`: reuses the existing reference trajectories and creates one shared,
  nested 1024-state sequence for every training problem.
- `cloth11`: trains one model for each selected prefix length with fixed epoch
  count, fixed minibatches, and fixed optimizer-update count.
- `cloth12`: evaluates the selected checkpoints on motions 20-31 with 500-frame
  continuous rollout and 50 inner iterations per frame.

## 2. Data semantics

For every training problem, the shared sample axis is:

```text
slot 0      : true physical initial state p_n
slots 1..   : scrambled Sobol states around exact_y
```

The ablation sets are nested prefixes:

```text
points_0001 = slots [0:1]
points_0008 = slots [0:8]
points_0032 = slots [0:32]
points_0064 = slots [0:64]
points_0128 = slots [0:128]
points_1024 = slots [0:1024]
```

Therefore every experiment contains the physical initial state, and increasing
sample count only expands the covered initial-state region.

The existing files below are reused and are not regenerated:

```text
cloth_5x5_500step_pipeline/data/reference/reference_problems.pt
cloth_5x5_500step_pipeline/data/reference/reference_motion_states.pt
cloth_5x5_500step_pipeline/data/reference/runtime_config.json
```

## 3. Training semantics

The original time-problem minibatch is unchanged:

```text
16 training motions x 32 time steps per motion
```

There are 13 optimizer updates per epoch:

```text
12 full windows: 16 motions x 32 time steps
1 tail window : 16 motions x 16 time steps
```

For one time-window minibatch, training performs:

```python
optimizer.zero_grad()
for sample_slot in range(points_per_problem):
    loss = loss_for_this_sample_slot()
    (loss / points_per_problem).backward()
clip_grad_norm_()
optimizer.step()
```

Consequences:

- every epoch visits every selected state exactly once;
- every sample-count experiment has the same number of optimizer updates;
- the GPU microbatch shape is independent of sample count;
- CUDA peak memory should stay approximately fixed;
- total runtime still grows approximately linearly with sample count.

Default model and training settings:

```text
activation        = identity
depth             = 1
width             = 256
bias              = false
epochs            = 500
learning_rate     = 1e-3
gradient_clip     = 10
K curriculum      = 1, 3, 5, 10, 30
epochs_per_K      = 100
validation every  = 50 epochs
```

## 4. Validation checkpoint selection

Validation uses the original validation motions:

```text
motion 16, 17, 18, 19
```

For each validation event:

```text
4 motions x 300 rollout frames x 15 learned iterations per frame
```

Only the residual after the 15th iteration of each frame is used. This gives
1200 final residuals. The checkpoint selection metric is exactly:

```text
global maximum of the 1200 final residuals
```

The p95 value, each motion's maximum, and the worst frame are saved only for
diagnostics and do not participate in checkpoint selection.

## 5. Test rollout

The default test motions are all existing motions outside training and
validation:

```text
ID test : motion 20-23
OOD test: motion 24-31
```

Each solver is evaluated with:

```text
rollout length       = 500 frames
inner iterations     = 50 per frame
frame initial state  = the solver's own propagated physical state
```

For every frame and every model/baseline, the saved curve contains:

```text
initial_y_by_frame                  # y^(0)
solution_y_by_frame                 # y^(50)
residual_by_frame_and_iteration     # shape [completed_frames, 51]
final_residual_by_frame
positions                           # propagated frames, including frame 0
velocities
reference_error_by_frame
global_iteration
global_residual                     # iterations 1..50 flattened, no separators
```

The existing reference trajectory is not rerun. Its stored solution and
residual are plotted only at each frame's iteration-50 endpoint.

Default comparison lines:

```text
model_points_0001
model_points_0008
model_points_0032
model_points_0064
model_points_0128
model_points_1024
baseline_gd
baseline_adam
baseline_lbfgs
baseline_newton
reference endpoints
```

## 6. Output structure

```text
cloth_5x5_initial_sample_ablation/
├── shared_reference/
├── shared_samples_1024/
│   ├── motion_000.pt
│   ├── ...
│   ├── motion_015.pt
│   └── manifest.json
├── points_0001/
│   ├── experiment.json
│   └── models/
├── points_0008/
├── points_0032/
├── points_0064/
├── points_0128/
├── points_1024/
└── rollout_evaluation/
    ├── all_motion_summary.csv
    ├── motion_020/
    │   ├── reference_len_500.pt
    │   ├── reference_endpoints.pt
    │   ├── model_points_0001/curve.pt
    │   ├── ...
    │   ├── baseline_newton/curve.pt
    │   ├── all_curves.pt
    │   ├── summary_metrics.csv
    │   └── figures/
    │       ├── rollout_x_iteration_vs_residual.png
    │       └── rollout_frame_vs_final_residual.png
    └── ...
```

## 7. Commands

Run from `cloth_5x5_500step_project/`.

### Prepare the shared nested samples

```bash
python cloth10_prepare_initial_point_ablation.py \
  --source-root cloth_5x5_500step_pipeline \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --sample-counts 1 8 32 64 128 1024 \
  --max-points 1024
```

### Smoke-test one small training run

```bash
python cloth11_train_initial_point_ablation.py \
  --source-root cloth_5x5_500step_pipeline \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --sample-counts 1 \
  --epochs 2 \
  --validation-interval 1 \
  --validation-rollout-length 5 \
  --validation-inner-steps 2 \
  --device cuda:0 \
  --overwrite
```

### Train the formal ablation

```bash
python cloth11_train_initial_point_ablation.py \
  --source-root cloth_5x5_500step_pipeline \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --sample-counts 1 8 32 64 128 1024 \
  --epochs 500 \
  --validation-interval 50 \
  --validation-rollout-length 300 \
  --validation-inner-steps 15 \
  --device cuda:0 \
  --resume
```

Because runtime grows with sample count, the experiments may also be launched
separately, for example:

```bash
python cloth11_train_initial_point_ablation.py \
  --sample-counts 1024 \
  --device cuda:0 \
  --resume
```

### Smoke-test rollout

```bash
python cloth12_evaluate_initial_point_ablation_rollouts.py \
  --source-root cloth_5x5_500step_pipeline \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --sample-counts 1 \
  --motion-indices 20 \
  --baselines gd newton \
  --rollout-length 5 \
  --inner-steps 3 \
  --device cuda:0 \
  --overwrite
```

### Run the formal 12-motion rollout evaluation

```bash
python cloth12_evaluate_initial_point_ablation_rollouts.py \
  --source-root cloth_5x5_500step_pipeline \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --sample-counts 1 8 32 64 128 1024 \
  --motion-indices 20 21 22 23 24 25 26 27 28 29 30 31 \
  --baselines gd adam lbfgs newton \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0
```

### Rebuild plots from saved line data only

```bash
python cloth12_evaluate_initial_point_ablation_rollouts.py \
  --source-root cloth_5x5_500step_pipeline \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --motion-indices 20 21 22 23 24 25 26 27 28 29 30 31 \
  --rollout-length 500 \
  --inner-steps 50 \
  --plot-only
```

## 8. Interpretation warning

This design fixes epochs, minibatch definitions, and optimizer-update count. It
isolates initial-state coverage more cleanly than feeding all states in one huge
batch. However, it does not fix total floating-point work: the 1024-point model
performs roughly 32 times as many sample-slot forward/backward passes per epoch
as the 32-point model. Report both performance and wall-clock cost.
