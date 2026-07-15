from pathlib import Path
import pickle
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_OBJ = ROOT / "data/aux_data/garment_meshes/hood/tshirt.obj"
SRC_PKL = ROOT / "data/aux_data/garments_dict.pkl"
OUT_DIR = ROOT / "tool/t-shirt"


def write_obj(path, vertices, faces):
    with path.open("w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        for face in faces:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def obj_stats(path):
    v_count = 0
    f_count = 0
    verts = []
    with path.open() as f:
        for line in f:
            if line.startswith("v "):
                v_count += 1
                if len(verts) < 5:
                    verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f_count += 1
    return v_count, f_count, np.asarray(verts, dtype=np.float64)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    obj_copy = OUT_DIR / "tshirt_from_garment_meshes.obj"
    pkl_out = OUT_DIR / "tshirt_from_garments_dict.pkl"
    obj_from_pkl = OUT_DIR / "tshirt_from_garments_dict.obj"
    summary_out = OUT_DIR / "README.md"

    shutil.copy2(SRC_OBJ, obj_copy)

    with SRC_PKL.open("rb") as f:
        garments = pickle.load(f)

    tshirt = garments["tshirt"]
    with pkl_out.open("wb") as f:
        pickle.dump({"tshirt": tshirt}, f, protocol=pickle.HIGHEST_PROTOCOL)

    vertices = np.asarray(tshirt["rest_pos"])
    faces = np.asarray(tshirt["faces"])
    write_obj(obj_from_pkl, vertices, faces)

    obj_v, obj_f, obj_first = obj_stats(obj_copy)
    pkl_v, pkl_f, pkl_first = obj_stats(obj_from_pkl)
    first_vertices_max_abs_diff = None
    if obj_first.shape == pkl_first.shape and obj_first.size:
        first_vertices_max_abs_diff = float(np.max(np.abs(obj_first - pkl_first)))

    summary = [
        "# T-shirt Mesh Exports",
        "",
        "Files in this directory:",
        "",
        "- `tshirt_from_garment_meshes.obj`: direct copy of `data/aux_data/garment_meshes/hood/tshirt.obj`.",
        "- `tshirt_from_garments_dict.pkl`: standalone pickle containing only `{\"tshirt\": ...}` from `data/aux_data/garments_dict.pkl`.",
        "- `tshirt_from_garments_dict.obj`: OBJ generated from `garments_dict[\"tshirt\"][\"rest_pos\"]` and `faces`.",
        "- `export_tshirt.py`: reproducible export script; run it inside the `hood` conda environment.",
        "",
        "Stats:",
        "",
        f"- OBJ source: {obj_v} vertices, {obj_f} faces.",
        f"- garments_dict tshirt: {pkl_v} vertices, {pkl_f} faces.",
        f"- garments_dict keys: {sorted(tshirt.keys())}.",
    ]
    if first_vertices_max_abs_diff is not None:
        summary.append(
            "- Max absolute difference across first five vertices between "
            f"the two OBJ exports: {first_vertices_max_abs_diff:.9g}."
        )
    summary_out.write_text("\n".join(summary) + "\n")

    print("wrote", obj_copy.relative_to(ROOT), obj_copy.stat().st_size)
    print("wrote", pkl_out.relative_to(ROOT), pkl_out.stat().st_size)
    print("wrote", obj_from_pkl.relative_to(ROOT), obj_from_pkl.stat().st_size)
    print("obj source vertices/faces", obj_v, obj_f)
    print("garments_dict tshirt vertices/faces", vertices.shape, faces.shape)


if __name__ == "__main__":
    main()
