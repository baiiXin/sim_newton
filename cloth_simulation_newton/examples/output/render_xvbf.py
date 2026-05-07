#!/usr/bin/env python3
import subprocess
import os

def select_file(folder, suffix=None):
    files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and (suffix is None or f.endswith(suffix))
    ]
    files.sort()

    for i, f in enumerate(files):
        print(f"[{i}] {f}")

    idx = int(input("请选择文件编号: "))
    return os.path.join(folder, files[idx])

# 取自身目录并拼接
EXAMPLES =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(EXAMPLES, "output")


PYTHON_BIN = "/data/zhoucy/anaconda3/envs/sim/bin/python"
SCRIPT_PATH = select_file(OUTPUT, suffix=".py")

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
