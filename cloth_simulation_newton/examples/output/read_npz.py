import json
import numpy as np
import os

def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "text")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 列出 npy / npz
files = sorted(
    f for f in os.listdir(INPUT_DIR)
    if f.endswith((".npy", ".npz"))
)

for i, f in enumerate(files):
    print(f"[{i}] {f}")

idx = int(input("请输入要读取的文件索引: "))
file_name = files[idx]

input_path = os.path.join(INPUT_DIR, file_name)
output_path = os.path.join(
    OUTPUT_DIR,
    file_name.replace(".npz", ".json").replace(".npy", ".json")
)

data = np.load(input_path, allow_pickle=True)
print('npzfile_key:',data.files)

json_data = {}

# ===== npz 情况 =====
if isinstance(data, np.lib.npyio.NpzFile):
    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            v = v.item()
        json_data[k] = to_jsonable(v)

# ===== npy 情况 =====
else:  # ndarray
    v = data
    if v.dtype == object and v.shape == ():
        v = v.item()
    json_data["data"] = to_jsonable(v)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"已导出: {output_path}")
