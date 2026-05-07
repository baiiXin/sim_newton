目前：
dict
├── cloth0 → dict
│    ├── vertices → ndarray
│    ├── triangles → ndarray
│    └── index info
├── cloth1 → ...
├── sphere → ...
├── ground → ...
├── vertices_all → ndarray
├── triangles_all → ndarray
└── fixed_index → ndarray

推荐：
output/
├── mesh_data.npz
└── mesh_layout.json


np.savez(
    "mesh_data.npz",
    vertices_all=vertices,              # (N, 3)
    triangles_all=triangles,            # (M, 3)
    fixed_index=fixed_index,             # (K,)
)


{
  "cloth0": {
    "start_pos_index": 0,
    "pos_num": 4900,
    "start_tri_index": 0,
    "triangle_num": 9602
  },
  "sphere": {
    "start_pos_index": 4900,
    "pos_num": 642,
    "start_tri_index": 9602,
    "triangle_num": 1280
  },
  "ground": {
    "start_pos_index": 5542,
    "pos_num": 121,
    "start_tri_index": 10882,
    "triangle_num": 200
  }
}
