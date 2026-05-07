#!/usr/bin/env python3
import subprocess

PYTHON_BIN = "/data/zhoucy/anaconda3/envs/sim/bin/python"
SCRIPT_PATH = "/data/zhoucy/sim/cloth_simulation_newton/examples/assets/cloth_fall_sphere/test_polyscope_cloth_fall_sphere.py"

cmd = [
    "xvfb-run",
    "-a",
    "-s", "-screen 0 3840x2160x24",   # ← 必须分开写
    PYTHON_BIN,
    SCRIPT_PATH
]

def main():
    print("Running render_three_triangles.py with xvfb-run ...\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
