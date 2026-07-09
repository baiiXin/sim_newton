from __future__ import annotations

import torch

GRID_ROWS = 5
GRID_COLS = 5
SPATIAL_DIM = 3
NUM_PARTICLES = GRID_ROWS * GRID_COLS
FIXED_VERTEX_INDICES = (0, (GRID_ROWS - 1) * GRID_COLS)  # left-top and left-bottom
FREE_VERTEX_INDICES = tuple(
    index for index in range(NUM_PARTICLES) if index not in set(FIXED_VERTEX_INDICES)
)
NUM_FREE_PARTICLES = len(FREE_VERTEX_INDICES)
FREE_STATE_DIM = NUM_FREE_PARTICLES * SPATIAL_DIM
HIDDEN_DIM = FREE_STATE_DIM


def grid_index(row: int, col: int) -> int:
    return row * GRID_COLS + col


def build_triangular_cloth_topology() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Return unique spring edges and triangle faces for an alternating 5x5 mesh."""
    edge_set: set[tuple[int, int]] = set()
    faces: list[tuple[int, int, int]] = []

    def add_edge(a: int, b: int) -> None:
        if a == b:
            raise ValueError("Degenerate edge")
        edge_set.add((min(a, b), max(a, b)))

    # Horizontal and vertical structural mesh edges.
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS - 1):
            add_edge(grid_index(row, col), grid_index(row, col + 1))
    for row in range(GRID_ROWS - 1):
        for col in range(GRID_COLS):
            add_edge(grid_index(row, col), grid_index(row + 1, col))

    # One alternating diagonal per cell: this is the triangle-mesh spring network.
    for row in range(GRID_ROWS - 1):
        for col in range(GRID_COLS - 1):
            tl = grid_index(row, col)
            tr = grid_index(row, col + 1)
            bl = grid_index(row + 1, col)
            br = grid_index(row + 1, col + 1)
            if (row + col) % 2 == 0:
                add_edge(tl, br)
                faces.append((tl, tr, br))
                faces.append((tl, br, bl))
            else:
                add_edge(bl, tr)
                faces.append((tl, tr, bl))
                faces.append((tr, br, bl))

    edges = tuple(sorted(edge_set))
    faces_tuple = tuple(faces)
    expected_edges = GRID_ROWS * (GRID_COLS - 1) + (GRID_ROWS - 1) * GRID_COLS + (GRID_ROWS - 1) * (GRID_COLS - 1)
    expected_faces = 2 * (GRID_ROWS - 1) * (GRID_COLS - 1)
    if len(edges) != expected_edges:
        raise RuntimeError(f"Expected {expected_edges} spring edges, got {len(edges)}")
    if len(faces_tuple) != expected_faces:
        raise RuntimeError(f"Expected {expected_faces} triangle faces, got {len(faces_tuple)}")
    return edges, faces_tuple


SPRING_EDGES, TRIANGLE_FACES = build_triangular_cloth_topology()
NUM_SPRINGS = len(SPRING_EDGES)
NUM_TRIANGLES = len(TRIANGLE_FACES)
GLOBAL_TO_FREE_INDEX = tuple(
    FREE_VERTEX_INDICES.index(index) if index in FREE_VERTEX_INDICES else -1
    for index in range(NUM_PARTICLES)
)


TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

ACTIVATION_NAME = "identity"
OPTIMIZER_NAME = "adam"
LEARNING_RATE = 1e-3
DEFAULT_DEVICE = "cuda:0"

DEFAULT_TOTAL_TIME_STEPS = 100
DEFAULT_TRAIN_POINTS_PER_PROBLEM = 32
DEFAULT_EVAL_POINTS_PER_PROBLEM = 128
DEFAULT_EPOCHS = 500
DEFAULT_VALIDATION_INTERVAL = 50
DEFAULT_DIAGNOSTIC_INTERVAL = 50
DEFAULT_EVALUATION_STEPS = 50
DEFAULT_EVALUATION_BATCH_SIZE = 8192
DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 100
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 5
DEFAULT_REPORT_STEPS = (1, 5, 10, 50)
DEFAULT_RESIDUAL_LENGTH_SCALE = 5e-2
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_SAMPLING_RADIUS_MIN = 1e-2
DEFAULT_SAMPLING_RADIUS_MAX = 1e-1

TRAIN_TIME_INDICES = (0, 5, 11, 16, 21, 26, 32, 37, 42, 47, 53, 58, 63, 68, 74, 79)
SEEN_INTERPOLATION_TIME_INDICES = (2, 8, 13, 18, 24, 29, 34, 39, 45, 50, 55, 61, 66, 71, 76, 78)
SEEN_EXTRAPOLATION_TIME_INDICES = tuple(range(80, 100))
VALIDATION_TIME_INDICES = (4, 14, 24, 34, 44, 54, 64, 74, 84, 94)
UNSEEN_TEST_TIME_INDICES = tuple(range(0, 100, 5))

MODEL_RANDOM_SEED = 42
MOTION_SOBOL_SEED_TRAIN = 20260630
MOTION_SOBOL_SEED_VALIDATION = 20260701
MOTION_SOBOL_SEED_ID_TEST = 20260702
TRAIN_SOBOL_SEED = 20260620
VALIDATION_SOBOL_SEED = 20260621
SEEN_INTERPOLATION_TEST_SOBOL_SEED = 20260622
SEEN_EXTRAPOLATION_TEST_SOBOL_SEED = 20260623
UNSEEN_ID_TEST_SOBOL_SEED = 20260624
OOD_TEST_SOBOL_SEED = 20260625

GD_CANDIDATE_STEP_SIZES = (
    1e-6,
    2e-6,
    5e-6,
    1e-5,
    2e-5,
    5e-5,
    1e-4,
)

PLOT_FLOOR = 1e-16
DISTANCE_EPS = 1e-12
NEWTON_RESIDUAL_TOLERANCE = 1e-10
REFERENCE_RESIDUAL_TOLERANCE = 1e-11
REFERENCE_ACCEPTABLE_RESIDUAL = 1e-8
REFERENCE_MAX_ITERATIONS = 100
REFERENCE_LINE_SEARCH_MIN_ALPHA = 2.0**-30
