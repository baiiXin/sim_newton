"""OBJ loading and reusable topology/rest-state preprocessing."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TShirtMesh:
    path: Path
    sha256: str
    vertices: np.ndarray  # [N, 3], float64
    faces: np.ndarray  # [F, 3], int64
    edges: np.ndarray  # [E, 2], int64
    face_areas: np.ndarray  # [F]
    inv_dm: np.ndarray  # [F, 2, 2]
    vertex_areas: np.ndarray  # [N]
    vertex_normals: np.ndarray  # [N, 3]
    hinge_indices: np.ndarray  # [H, 4]: opposite0, opposite1, edge0, edge1
    hinge_rest_angles: np.ndarray  # [H]
    hinge_rest_lengths: np.ndarray  # [H]
    boundary_edges: np.ndarray  # [Eb, 2]
    median_edge_length: float

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_triangle_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError(f"Malformed vertex at {path}:{line_number}")
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise ValueError(
                        f"Only triangular faces are supported; got {len(fields)-1} "
                        f"vertices at {path}:{line_number}"
                    )
                face = [int(token.split("/")[0]) - 1 for token in fields[1:]]
                faces.append(face)
    vertices_array = np.asarray(vertices, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise ValueError(f"OBJ has invalid vertex array: {vertices_array.shape}")
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise ValueError(f"OBJ has invalid face array: {faces_array.shape}")
    if faces_array.size and (faces_array.min() < 0 or faces_array.max() >= len(vertices_array)):
        raise ValueError("OBJ face index is out of range")
    return vertices_array, faces_array


def _intrinsic_rest_data(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    e1_norm = np.linalg.norm(e1, axis=-1)
    cross = np.cross(e1, e2)
    double_area = np.linalg.norm(cross, axis=-1)
    if float(e1_norm.min(initial=np.inf)) <= 1e-12:
        raise ValueError("OBJ contains a zero-length triangle edge")
    if float(double_area.min(initial=np.inf)) <= 1e-12:
        raise ValueError("OBJ contains a degenerate triangle")
    basis_u = e1 / e1_norm[:, None]
    normal = cross / double_area[:, None]
    basis_v = np.cross(normal, basis_u)
    dm = np.zeros((len(faces), 2, 2), dtype=np.float64)
    dm[:, 0, 0] = e1_norm
    dm[:, 0, 1] = np.einsum("ij,ij->i", e2, basis_u)
    dm[:, 1, 1] = np.einsum("ij,ij->i", e2, basis_v)
    inv_dm = np.linalg.inv(dm)
    return 0.5 * double_area, inv_dm, normal


def _edge_records(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    records: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for face_id, (a, b, c) in enumerate(faces.tolist()):
        for edge_a, edge_b, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            records[tuple(sorted((edge_a, edge_b)))].append((face_id, opposite))
    return records


def _signed_dihedral(
    vertices: np.ndarray,
    hinge_indices: np.ndarray,
) -> np.ndarray:
    if len(hinge_indices) == 0:
        return np.empty((0,), dtype=np.float64)
    x0 = vertices[hinge_indices[:, 0]]
    x1 = vertices[hinge_indices[:, 1]]
    x2 = vertices[hinge_indices[:, 2]]
    x3 = vertices[hinge_indices[:, 3]]
    edge = x3 - x2
    n0 = np.cross(x2 - x0, x3 - x0)
    n1 = np.cross(x3 - x1, x2 - x1)
    edge_hat = edge / np.linalg.norm(edge, axis=-1, keepdims=True)
    n0_hat = n0 / np.linalg.norm(n0, axis=-1, keepdims=True)
    n1_hat = n1 / np.linalg.norm(n1, axis=-1, keepdims=True)
    sine = np.einsum("ij,ij->i", np.cross(n0_hat, n1_hat), edge_hat)
    cosine = np.einsum("ij,ij->i", n0_hat, n1_hat)
    return np.arctan2(sine, cosine)


def load_tshirt_mesh(path: Path) -> TShirtMesh:
    path = path.resolve()
    vertices, faces = load_triangle_obj(path)
    face_areas, inv_dm, face_normals = _intrinsic_rest_data(vertices, faces)
    records = _edge_records(faces)
    incidence = Counter({edge: len(items) for edge, items in records.items()})
    if any(count > 2 for count in incidence.values()):
        raise ValueError("OBJ is non-manifold: at least one edge has more than two faces")
    edges = np.asarray(sorted(records), dtype=np.int64)
    boundary_edges = np.asarray(
        [edge for edge in sorted(records) if len(records[edge]) == 1],
        dtype=np.int64,
    ).reshape(-1, 2)
    hinges: list[tuple[int, int, int, int]] = []
    for edge in sorted(records):
        adjacent = records[edge]
        if len(adjacent) != 2:
            continue
        adjacent = sorted(adjacent, key=lambda item: item[0])
        hinges.append((adjacent[0][1], adjacent[1][1], edge[0], edge[1]))
    hinge_indices = np.asarray(hinges, dtype=np.int64).reshape(-1, 4)
    hinge_rest_angles = _signed_dihedral(vertices, hinge_indices)
    hinge_rest_lengths = np.linalg.norm(
        vertices[hinge_indices[:, 3]] - vertices[hinge_indices[:, 2]], axis=-1
    )
    vertex_areas = np.zeros(len(vertices), dtype=np.float64)
    for local in range(3):
        np.add.at(vertex_areas, faces[:, local], face_areas / 3.0)
    vertex_normals = np.zeros_like(vertices)
    weighted_normals = face_normals * (2.0 * face_areas[:, None])
    for local in range(3):
        np.add.at(vertex_normals, faces[:, local], weighted_normals)
    normal_lengths = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    vertex_normals /= np.maximum(normal_lengths, 1e-15)
    edge_lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=-1)
    return TShirtMesh(
        path=path,
        sha256=_sha256(path),
        vertices=vertices,
        faces=faces,
        edges=edges,
        face_areas=face_areas,
        inv_dm=inv_dm,
        vertex_areas=vertex_areas,
        vertex_normals=vertex_normals,
        hinge_indices=hinge_indices,
        hinge_rest_angles=hinge_rest_angles,
        hinge_rest_lengths=hinge_rest_lengths,
        boundary_edges=boundary_edges,
        median_edge_length=float(np.median(edge_lengths)),
    )


def select_four_shoulder_vertices(mesh: TShirtMesh) -> tuple[int, int, int, int]:
    """Pick front/back anchors on the left and right shoulders.

    The HOOD T-shirt uses x=left/right, y=vertical, and z=front/back.  Selection
    is geometry based so it remains auditable rather than relying on hidden OBJ
    indices.
    """

    vertices = mesh.vertices
    top_cut = float(np.quantile(vertices[:, 1], 0.94))
    top = vertices[:, 1] >= top_cut
    chosen: list[int] = []
    x_extent = float(np.ptp(vertices[:, 0]))
    tolerance = max(3.0 * mesh.median_edge_length, 0.045 * x_extent)
    for side in (-1, 1):
        side_ids = np.flatnonzero(top & ((vertices[:, 0] * side) > 0.0))
        if side_ids.size < 2:
            raise ValueError("Not enough vertices in a shoulder candidate region")
        x_values = vertices[side_ids, 0]
        target_x = float(np.quantile(x_values, 0.10 if side < 0 else 0.90))
        shoulder = side_ids[np.abs(x_values - target_x) <= tolerance]
        if shoulder.size < 2:
            shoulder = side_ids
        front = int(shoulder[np.argmin(vertices[shoulder, 2])])
        back = int(shoulder[np.argmax(vertices[shoulder, 2])])
        if front == back:
            raise ValueError("Could not separate front/back shoulder vertices")
        chosen.extend((front, back))
    if len(set(chosen)) != 4:
        raise ValueError(f"Shoulder selection produced duplicates: {chosen}")
    return tuple(chosen)  # left front/back, right front/back


def adjacency_degrees(num_vertices: int, edges: np.ndarray) -> np.ndarray:
    degree = np.zeros(num_vertices, dtype=np.int64)
    np.add.at(degree, edges[:, 0], 1)
    np.add.at(degree, edges[:, 1], 1)
    return degree


def connected_component_sizes(num_vertices: int, edges: np.ndarray) -> tuple[int, ...]:
    adjacency: list[list[int]] = [[] for _ in range(num_vertices)]
    for i, j in edges.tolist():
        adjacency[i].append(j)
        adjacency[j].append(i)
    seen = np.zeros(num_vertices, dtype=bool)
    sizes: list[int] = []
    for start in range(num_vertices):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for other in adjacency[current]:
                if not seen[other]:
                    seen[other] = True
                    stack.append(other)
        sizes.append(size)
    return tuple(sorted(sizes, reverse=True))
