# T-shirt Mesh Exports

Files in this directory:

- `tshirt_from_garment_meshes.obj`: direct copy of `data/aux_data/garment_meshes/hood/tshirt.obj`.
- `tshirt_from_garments_dict.pkl`: standalone pickle containing only `{"tshirt": ...}` from `data/aux_data/garments_dict.pkl`.
- `tshirt_from_garments_dict.obj`: OBJ generated from `garments_dict["tshirt"]["rest_pos"]` and `faces`.
- `export_tshirt.py`: reproducible export script; run it inside the `hood` conda environment.

Stats:

- OBJ source: 4424 vertices, 8710 faces.
- garments_dict tshirt: 4424 vertices, 8710 faces.
- garments_dict keys: ['center', 'coarse_edges', 'faces', 'gender', 'lbs', 'model_type', 'node_type', 'rest_pos'].
- Max absolute difference across first five vertices between the two OBJ exports: 1.4e-08.
