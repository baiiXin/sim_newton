# Cloth 15×15 500-Step Learned Optimizer Project

This directory extends `cloth_5x5_500step_project` to a **15×15 triangular cloth** while changing the data/evaluation protocol for architecture, perturbation-count, and live-training-pool experiments.

## 1. Design decisions

### 1.1 Physics and network state

- Grid: `15×15 = 225` vertices.
- Fixed vertices: left-top and left-bottom, identical semantics to the 5×5 project.
- Full state: `225×3 = 675D`.
- Reduced physical state: `223×3 = 669D`.
- Network input: `[current residual, previous residual, previous update] = 3×675 = 2025D`.
- Network output: `675D`; fixed entries are gated to zero and positions are hard-projected.
- Precision: `torch.float64`.
- `cloth03_solvers_and_models.py` is a compatibility layer over the tested 5×5 physics implementation, with the grid globals patched before use and GELU/SiLU added.

### 1.2 Raw reference first; filtering later

Do **not** delete failed frames while constructing the raw reference. A failed frame changes every later propagated state, so convergence filtering is performed at the **whole-motion level**.

The recommended sequence is:

1. Generate all 32 motions and save the residual at every physical frame.
2. Inspect per-motion curves and render suspicious/worst motions.
3. Decide a threshold based on both residual and visible behavior.
4. Pass complete motion indices through `--exclude-motion-indices` when generating samples and catalogues.

The raw reference remains untouched, so the exclusion decision is auditable and reversible.

### 1.3 Validation and test protocol

Validation and test are **offline single-time-step evaluations**, not continuous rollout:

- original validation motions `16–19`, every frame `0–499`;
- original test motions `20–31`, every frame `0–499`;
- exactly one initial state per problem: `y^(0)=x_n`;
- exactly one learned update;
- history inputs are zeros at the first update.

For each point, the code stores residuals before and after the update, `r0` and `r1`. Checkpoint selection uses the 95th percentile of `log10(r1/r0)`; lower is better. Absolute `r1` mean/p50/p95/p99/max, improvement fraction, and non-finite counts are also retained.

Continuous rollout is separate: the hardest finite reference motion among original test motions is selected by reference-residual p95 and rolled out for 500 physical frames.

### 1.4 Compact storage

A naïve 15×15 copy of the old flat dataset would repeat `q`, `exact_y`, and masses for every perturbation. This project instead uses motion/time-window shards:

- main samples: one shard per training motion;
- perturbation ablation: one shard per motion and 32-frame time window;
- only `initial_y` owns the perturbation axis;
- physical problem data are stored once per time step;
- perturbation sets are deterministic nested Sobol prefixes;
- the physical state `x_n` is **not** injected into training perturbations.

The 16-motion, 400-frame, 1024-point, 675D `initial_y` tensor alone is about **33 GiB** in float64, so window sharding is mandatory.

## 2. Experiment sequence

The staged search is intentionally greedy and computationally tractable:

1. **Width**: ReLU, depth 1, no bias; widths `128 256 512 1024 2048 4096`.
2. **Depth**: selected width, ReLU, no bias; depths `1 2 3 5 7 10`.
3. **Activation × bias**: selected width/depth; activations `relu gelu silu tanh identity`; both bias settings.
4. **Initial perturbation count**: selected architecture; `1 8 32 128 512 1024` nested samples/problem.
5. **Training pool**: selected architecture with Metamizer-style live pool updates.

A greedy sequence can miss width–depth interactions. After stages 1–2, run the top two widths × top two depths as a small interaction check before freezing the architecture.

## 3. Files

```text
cloth_common.py                         shared I/O and one-step evaluation
cloth01_generate_reference_and_samples.py
cloth02_dataset_catalog.py
cloth03_solvers_and_models.py           15×15 compatibility/activation layer
cloth04_probe_memory.py                 full forward/backward CUDA probe
cloth05_train_models.py                 common staged trainer
cloth06_select_best.py                  rank/select a stage
cloth07_rollout_hardest_motion.py       500-frame continuous rollout
cloth09_render_reference_motion.py      render reference by motion index
cloth10_prepare_initial_point_ablation.py
cloth11_train_initial_point_ablation.py
cloth13_train_pool.py                    Metamizer-style live pool trainer
```

## 4. End-to-end commands

Run from this directory.

### A. Generate raw reference and audit residuals

```bash
python cloth01_generate_reference_and_samples.py \
  --output-dir cloth_15x15_500step_pipeline \
  --reference-only
```

Outputs include every motion's framewise residual curve and `reference_audit.json`.

Render any suspicious motion:

```bash
python cloth09_render_reference_motion.py \
  --root cloth_15x15_500step_pipeline \
  --motion-index 27 --format mp4
```

### B. Choose exclusions and generate 32-point training samples

The indices below are examples only; choose them after inspecting your generated reference.

```bash
python cloth01_generate_reference_and_samples.py \
  --output-dir cloth_15x15_500step_pipeline \
  --samples-only \
  --exclude-motion-indices 27 31 \
  --points-per-problem 32

python cloth02_dataset_catalog.py \
  --root cloth_15x15_500step_pipeline \
  --exclude-motion-indices 27 31
```

### C. Probe the largest configuration

```bash
python cloth04_probe_memory.py \
  --root cloth_15x15_500step_pipeline \
  --activation relu --depth 10 --width 4096 \
  --sample-count 32 --k 30 --device cuda:0
```

If it OOMs, reduce `--sample-chunk-size` in training. Chunking accumulates gradients and still performs one optimizer update per physical-time window.

### Stage 1 — width

```bash
python cloth05_train_models.py \
  --root cloth_15x15_500step_pipeline \
  --stage width \
  --activations relu --depths 1 \
  --widths 128 256 512 1024 2048 4096 \
  --bias-mode no-bias --sample-count 32 \
  --device cuda:0 --skip-completed

python cloth06_select_best.py \
  --root cloth_15x15_500step_pipeline --stage width
```

### Stage 2 — depth

Replace `BEST_WIDTH` from `experiments/width/selection.json`:

```bash
python cloth05_train_models.py \
  --root cloth_15x15_500step_pipeline \
  --stage depth \
  --activations relu --depths 1 2 3 5 7 10 \
  --widths BEST_WIDTH --bias-mode no-bias \
  --sample-count 32 --device cuda:0 --skip-completed

python cloth06_select_best.py \
  --root cloth_15x15_500step_pipeline --stage depth
```

### Stage 3 — activation × bias

```bash
python cloth05_train_models.py \
  --root cloth_15x15_500step_pipeline \
  --stage activation_bias \
  --activations relu gelu silu tanh identity \
  --depths BEST_DEPTH --widths BEST_WIDTH \
  --bias-mode both --sample-count 32 \
  --device cuda:0 --skip-completed

python cloth06_select_best.py \
  --root cloth_15x15_500step_pipeline --stage activation_bias
```

### Stage 4 — initial perturbation count

```bash
python cloth10_prepare_initial_point_ablation.py \
  --root cloth_15x15_500step_pipeline \
  --sample-counts 1 8 32 128 512 1024 \
  --max-points 1024

python cloth11_train_initial_point_ablation.py \
  --root cloth_15x15_500step_pipeline \
  --activation BEST_ACTIVATION \
  --depth BEST_DEPTH --width BEST_WIDTH \
  --bias-mode no-bias \
  --sample-counts 1 8 32 128 512 1024 \
  --sample-chunk-size 8 \
  --device cuda:0 --skip-completed

python cloth06_select_best.py \
  --root cloth_15x15_500step_pipeline --stage initial_points
```

### Stage 5 — live training pool

```bash
python cloth13_train_pool.py \
  --root cloth_15x15_500step_pipeline \
  --activation BEST_ACTIVATION \
  --depth BEST_DEPTH --width BEST_WIDTH \
  --device cuda:0
```

Add `--use-bias` when the selected model uses bias.

### Final 500-frame rollout

```bash
python cloth07_rollout_hardest_motion.py \
  --root cloth_15x15_500step_pipeline \
  --checkpoint PATH/TO/best_validation_model.pt \
  --rollout-length 500 --inner-steps 50 \
  --device cuda:0
```

By default the script chooses the highest reference-residual-p95 motion among finite, non-excluded original test motions `20–31`. Pass `--motion-index` to override it.

## 5. Interpretation cautions

- Reference residual measures numerical solve quality, not visual difficulty alone. Inspect both curves and renderings before exclusion.
- Fix the reference-convergence rule before using model test results.
- A one-step validation winner is not guaranteed to be the best 50-inner-step rollout model. Report both separately.
- Width/depth comparisons must use identical sample prefixes, epoch/K schedule, seed, optimizer, and validation set.
- Bias breaks the structural property “zero residual → zero update”; treat it as a structural ablation, not only a parameter-count increase.
