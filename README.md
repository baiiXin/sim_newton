# Environment Setup

This file only documents the software environment used by this repository.

## 1. Validated configuration

The environment below has been tested successfully with the repository's 15×15 cloth scale-up project and Newton's VBD solver.

| Component | Validated version |
|---|---:|
| Operating system | Ubuntu 20.04, x86_64 |
| glibc | 2.31 |
| GPU | 2 × NVIDIA GeForce RTX 3090 24 GB |
| NVIDIA driver | 580.82.07 |
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| Matplotlib | 3.9.1 |
| Pillow | 12.3.0 |
| FFmpeg | 8.0.1 |
| PyTorch | 2.13.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| Polyscope | 2.6.1 |
| imageio-ffmpeg | 0.6.0 |
| Warp | 1.15.0 |
| Newton | 1.4.0 |

The project normally uses `torch.float64` and CUDA devices such as `cuda:0` or `cuda:1`.

## 2. Create the Conda environment

From the repository root, run:

```bash
conda env create -f environment.yml
conda activate cloth_opter
```

The YAML file installs Python and the Conda-managed numerical, plotting, image, and FFmpeg dependencies.

## 3. Install CUDA PyTorch

PyTorch is installed separately because the CUDA 13.0 wheel is hosted on the PyTorch package index:

```bash
python -m pip install "torch==2.13.0" \
  --index-url https://download.pytorch.org/whl/cu130
```

Verify that CUDA and double precision are available:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

x = torch.randn(1024, 1024, device="cuda:0", dtype=torch.float64)
y = x @ x.T
print("float64 CUDA test:", y.device, y.dtype, torch.isfinite(y).all().item())
PY
```

## 4. Install Warp, Newton, and visualization packages

```bash
python -m pip install \
  "warp-lang==1.15.0" \
  "newton==1.4.0" \
  "polyscope==2.6.1" \
  "imageio-ffmpeg==0.6.0"
```

Newton's VBD solver is available through:

```python
newton.solvers.SolverVBD
```

## 5. Configure the Conda C++ runtime

On Ubuntu 20.04, NumPy may otherwise load the older system `libstdc++.so.6` and report an error such as:

```text
GLIBCXX_3.4.29 not found
```

Store the Conda environment library path as an environment-specific variable:

```bash
conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
conda deactivate
conda activate cloth_opter
```

Confirm that it is active:

```bash
echo "$LD_LIBRARY_PATH"
```

The output should point to the active environment, for example:

```text
/data/zhoucy/anaconda3/envs/cloth_opter/lib
```

This setting only applies while `cloth_opter` is active. It does not modify the system libraries or other Conda environments.

## 6. Verify Warp and Newton VBD

```bash
python - <<'PY'
import warp as wp
import newton

wp.init()

print("Warp:", wp.__version__)
print("Newton:", newton.__version__)
print("Warp devices:")
for device in wp.get_devices():
    print(" ", device)

print("SolverVBD available:", hasattr(newton.solvers, "SolverVBD"))
PY
```

A correct GPU installation should list the available CUDA devices and print:

```text
SolverVBD available: True
```

## 7. Run the project tests

```bash
cd cloth_15x15_500step_project_scale_up
python -m unittest -v
```

The validated environment currently passes all 26 tests:

```text
Ran 26 tests
OK
```

## 8. Useful environment checks

```bash
conda activate cloth_opter
which python
python --version
python -m pip --version
ffmpeg -version | head -n 1
nvidia-smi
```

To remove and recreate the environment:

```bash
conda deactivate
conda env remove -n cloth_opter
conda env create -f environment.yml
```
