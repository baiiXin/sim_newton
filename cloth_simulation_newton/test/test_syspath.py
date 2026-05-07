# tests/test_sys_path.py

import sys
from pathlib import Path

# 1️⃣ 找到你想加入的目录（绝对路径）
ROOT = Path(__file__).resolve().parents[2]  # 父目录，或者父的父目录
# 2️⃣ 转成字符串并插入 sys.path
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("Current file:")
print(Path(__file__).resolve())
print("=" * 60)

print("sys.path:")
for i, p in enumerate(sys.path):
    print(f"[{i}] {p}")

print("=" * 60)
