# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import warp as wp

from pypardiso import spsolve
from scipy.sparse import bsr_matrix
# from scipy.sparse.linalg import spsolve

from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase

from newton._src.geometry.kernels import triangle_closest_point
from newton._src.solvers.vbd.tri_mesh_collision import (
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
)

NUM_THREADS_PER_COLLISION_PRIMITIVE = 64

@wp.struct
class ForceElementAdjacencyInfo:
    r"""
    - vertex_adjacent_[element]: the flatten adjacency information. Its size is \sum_{i\inV} 2*N_i, where N_i is the
    number of vertex i's adjacent [element]. For each adjacent element it stores 2 information:
        - the id of the adjacent element
        - the order of the vertex in the element, which is essential to compute the force and hessian for the vertex
    - vertex_adjacent_[element]_offsets: stores where each vertex information starts in the  flatten adjacency array.
    Its size is |V|+1 such that the number of vertex i's adjacent [element] can be computed as
    vertex_adjacent_[element]_offsets[i+1]-vertex_adjacent_[element]_offsets[i].
    """

    v_adj_faces: wp.array(dtype=int)
    v_adj_faces_offsets: wp.array(dtype=int)

    v_adj_edges: wp.array(dtype=int)
    v_adj_edges_offsets: wp.array(dtype=int)

    v_adj_springs: wp.array(dtype=int)
    v_adj_springs_offsets: wp.array(dtype=int)

    def to(self, device):
        if device == self.v_adj_faces.device:
            return self
        else:
            adjacency_gpu = ForceElementAdjacencyInfo()
            adjacency_gpu.v_adj_faces = self.v_adj_faces.to(device)
            adjacency_gpu.v_adj_faces_offsets = self.v_adj_faces_offsets.to(device)

            adjacency_gpu.v_adj_edges = self.v_adj_edges.to(device)
            adjacency_gpu.v_adj_edges_offsets = self.v_adj_edges_offsets.to(device)

            adjacency_gpu.v_adj_springs = self.v_adj_springs.to(device)
            adjacency_gpu.v_adj_springs_offsets = self.v_adj_springs_offsets.to(device)

            return adjacency_gpu


@wp.func
def get_vertex_num_adjacent_edges(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_edges_offsets[vertex + 1] - adjacency.v_adj_edges_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_edge_id_order(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, edge: wp.int32):
    offset = adjacency.v_adj_edges_offsets[vertex]
    return adjacency.v_adj_edges[offset + edge * 2], adjacency.v_adj_edges[offset + edge * 2 + 1]


@wp.func
def get_vertex_num_adjacent_faces(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_faces_offsets[vertex + 1] - adjacency.v_adj_faces_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_face_id_order(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, face: wp.int32):
    offset = adjacency.v_adj_faces_offsets[vertex]
    return adjacency.v_adj_faces[offset + face * 2], adjacency.v_adj_faces[offset + face * 2 + 1]


@wp.func
def get_vertex_num_adjacent_springs(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return adjacency.v_adj_springs_offsets[vertex + 1] - adjacency.v_adj_springs_offsets[vertex]


@wp.func
def get_vertex_adjacent_spring_id(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, spring: wp.int32):
    offset = adjacency.v_adj_springs_offsets[vertex]
    return adjacency.v_adj_springs[offset + spring]


@wp.func
def evaluate_self_contact_force_norm(dis: float, collision_radius: float, k: float, barrier_threshold: float):
    # Adjust distance and calculate penetration depth

    penetration_depth = collision_radius - dis

    # Initialize outputs
    dEdD = wp.float32(0.0)
    d2E_dDdD = wp.float32(0.0)

    # C2 continuity calculation
    tau = collision_radius * 0.5
    if tau > dis > barrier_threshold:
        k2 = 0.5 * tau * tau * k
        dEdD = -k2 / dis
        d2E_dDdD = k2 / (dis * dis)
    else:
        dEdD = -k * penetration_depth
        d2E_dDdD = k

    return dEdD, d2E_dDdD


@wp.kernel
def compute_particle_conservative_bound(
    # inputs
    conservative_bound_relaxation: float,
    collision_query_radius: float,
    adjacency: ForceElementAdjacencyInfo,
    collision_info: TriMeshCollisionInfo,
    # outputs
    particle_conservative_bounds: wp.array(dtype=float),
):
    particle_index = wp.tid()
    min_dist = wp.min(collision_query_radius, collision_info.vertex_colliding_triangles_min_dist[particle_index])

    # bound from neighbor triangles
    for i_adj_tri in range(
        get_vertex_num_adjacent_faces(
            adjacency,
            particle_index,
        )
    ):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(
            adjacency,
            particle_index,
            i_adj_tri,
        )
        min_dist = wp.min(min_dist, collision_info.triangle_colliding_vertices_min_dist[tri_index])

    # bound from neighbor edges
    for i_adj_edge in range(
        get_vertex_num_adjacent_edges(
            adjacency,
            particle_index,
        )
    ):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency,
            particle_index,
            i_adj_edge,
        )
        # vertex is on the edge; otherwise it only effects the bending energy
        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            # collisions of neighbor edges
            min_dist = wp.min(min_dist, collision_info.edge_colliding_edges_min_dist[nei_edge_index])

    particle_conservative_bounds[particle_index] = conservative_bound_relaxation * min_dist



# region: zcy
# zcy
# svd
# ==============================================================================
# 1. 辅助函数 (6x6)
# ==============================================================================

@wp.func
def idx_6(r: int, c: int):
    return r * 6 + c

@wp.func
def load_block_3x3_6(
    row_block: int, col_block: int, 
    base_offset: int, 
    temp_buffer: wp.array(dtype=float), 
    m: wp.mat33
):
    """ 将 mat33 写入 flatten array """
    row_start = row_block * 3
    col_start = col_block * 3
    
    # Row 0
    temp_buffer[base_offset + idx_6(row_start+0, col_start+0)] = m[0, 0]
    temp_buffer[base_offset + idx_6(row_start+0, col_start+1)] = m[0, 1]
    temp_buffer[base_offset + idx_6(row_start+0, col_start+2)] = m[0, 2]
    # Row 1
    temp_buffer[base_offset + idx_6(row_start+1, col_start+0)] = m[1, 0]
    temp_buffer[base_offset + idx_6(row_start+1, col_start+1)] = m[1, 1]
    temp_buffer[base_offset + idx_6(row_start+1, col_start+2)] = m[1, 2]
    # Row 2
    temp_buffer[base_offset + idx_6(row_start+2, col_start+0)] = m[2, 0]
    temp_buffer[base_offset + idx_6(row_start+2, col_start+1)] = m[2, 1]
    temp_buffer[base_offset + idx_6(row_start+2, col_start+2)] = m[2, 2]

@wp.func
def reconstruct_block_3x3_6(
    row_block: int, col_block: int, 
    base_offset: int, offset_V: int, 
    temp_buffer: wp.array(dtype=float)
):
    res = wp.mat33(0.0)
    row_start = row_block * 3
    col_start = col_block * 3
    
    for i in range(3):
        for j in range(3):
            global_r = row_start + i
            global_c = col_start + j
            
            sum_val = float(0.0)
            
            # While loop
            k = int(0)
            while k < 6:
                lam = temp_buffer[base_offset + idx_6(k, k)]
                v_rk = temp_buffer[offset_V + idx_6(global_r, k)]
                v_ck = temp_buffer[offset_V + idx_6(global_c, k)]
                sum_val += v_rk * lam * v_ck
                k += 1
            
            res[i, j] = sum_val
    return res

# ==============================================================================
# 2. 核心设备函数 (6x6 Optimized)
# ==============================================================================

@wp.func
def filter_hessian_6x6_device(
    h_aa: wp.mat33, h_ab: wp.mat33,
    h_ba: wp.mat33, h_bb: wp.mat33,
    temp_buffer: wp.array(dtype=float),
    tid: int
):
    # 6x6 needs 36(Mat) + 36(Vec) = 72 floats
    base_offset = tid * 72
    offset_V = base_offset + 36
    
    # --- 1. Load Data ---
    load_block_3x3_6(0, 0, base_offset, temp_buffer, h_aa)
    load_block_3x3_6(0, 1, base_offset, temp_buffer, h_ab)
    load_block_3x3_6(1, 0, base_offset, temp_buffer, h_ba)
    load_block_3x3_6(1, 1, base_offset, temp_buffer, h_bb)

    # --- 2. Jacobi SVD ---
    
    # Init V
    i = int(0)
    while i < 6:
        j = int(0)
        while j < 6:
            temp_buffer[offset_V + idx_6(i, j)] = 1.0 if i == j else 0.0
            j += 1
        i += 1

    # 15 sweeps
    iter_count = int(0)
    while iter_count < 15:
        p = int(0)
        while p < 5:
            q = p + 1
            while q < 6:
                idx_pq = base_offset + idx_6(p, q)
                a_pq = temp_buffer[idx_pq]
                
                if wp.abs(a_pq) >= 1e-6:
                    idx_pp = base_offset + idx_6(p, p)
                    idx_qq = base_offset + idx_6(q, q)
                    a_pp = temp_buffer[idx_pp]
                    a_qq = temp_buffer[idx_qq]

                    tau = (a_qq - a_pp) / (2.0 * a_pq)
                    t = float(0.0)
                    if tau >= 0.0: t = 1.0 / (tau + wp.sqrt(1.0 + tau*tau))
                    else:          t = -1.0 / (-tau + wp.sqrt(1.0 + tau*tau))
                    c = 1.0 / wp.sqrt(1.0 + t*t)
                    s = t * c

                    # Rotate A
                    temp_buffer[idx_pp] = c*c*a_pp + s*s*a_qq - 2.0*c*s*a_pq
                    temp_buffer[idx_qq] = s*s*a_pp + c*c*a_qq + 2.0*c*s*a_pq
                    temp_buffer[idx_pq] = 0.0
                    temp_buffer[base_offset + idx_6(q, p)] = 0.0

                    # Rotate Rows/Cols
                    k = int(0)
                    while k < 6:
                        if k != p and k != q:
                            idx_ip = base_offset + idx_6(k, p)
                            idx_iq = base_offset + idx_6(k, q)
                            a_ip = temp_buffer[idx_ip]
                            a_iq = temp_buffer[idx_iq]
                            
                            a_ip_n = c * a_ip - s * a_iq
                            a_iq_n = s * a_ip + c * a_iq
                            
                            temp_buffer[idx_ip] = a_ip_n
                            temp_buffer[base_offset + idx_6(p, k)] = a_ip_n
                            temp_buffer[idx_iq] = a_iq_n
                            temp_buffer[base_offset + idx_6(q, k)] = a_iq_n
                        k += 1
                    
                    # Rotate V
                    k = int(0)
                    while k < 6:
                        idx_ip = offset_V + idx_6(k, p)
                        idx_iq = offset_V + idx_6(k, q)
                        v_ip = temp_buffer[idx_ip]
                        v_iq = temp_buffer[idx_iq]
                        
                        temp_buffer[idx_ip] = c * v_ip - s * v_iq
                        temp_buffer[idx_iq] = s * v_ip + c * v_iq
                        k += 1
                q += 1
            p += 1
        iter_count += 1

    # --- 3. Filter Eigenvalues ---
    k = int(0)
    while k < 6:
        idx = base_offset + idx_6(k, k)
        val = temp_buffer[idx]
        temp_buffer[idx] = wp.max(val, 0.0)
        k += 1

    # --- 4. Reconstruct ---
    out_aa = reconstruct_block_3x3_6(0, 0, base_offset, offset_V, temp_buffer)
    out_ab = reconstruct_block_3x3_6(0, 1, base_offset, offset_V, temp_buffer)
    out_ba = reconstruct_block_3x3_6(1, 0, base_offset, offset_V, temp_buffer)
    out_bb = reconstruct_block_3x3_6(1, 1, base_offset, offset_V, temp_buffer)

    return out_aa, out_ab, out_ba, out_bb

# ==============================================================================
# 1. 辅助函数 (9x9)
# ==============================================================================

@wp.func
def idx_9(r: int, c: int):
    return r * 9 + c

@wp.func
def load_block_3x3_9(
    row_block: int, col_block: int, 
    base_offset: int, 
    temp_buffer: wp.array(dtype=float), 
    m: wp.mat33
):
    """ 将 mat33 写入 flatten array """
    row_start = row_block * 3
    col_start = col_block * 3
    
    # Row 0
    temp_buffer[base_offset + idx_9(row_start+0, col_start+0)] = m[0, 0]
    temp_buffer[base_offset + idx_9(row_start+0, col_start+1)] = m[0, 1]
    temp_buffer[base_offset + idx_9(row_start+0, col_start+2)] = m[0, 2]
    # Row 1
    temp_buffer[base_offset + idx_9(row_start+1, col_start+0)] = m[1, 0]
    temp_buffer[base_offset + idx_9(row_start+1, col_start+1)] = m[1, 1]
    temp_buffer[base_offset + idx_9(row_start+1, col_start+2)] = m[1, 2]
    # Row 2
    temp_buffer[base_offset + idx_9(row_start+2, col_start+0)] = m[2, 0]
    temp_buffer[base_offset + idx_9(row_start+2, col_start+1)] = m[2, 1]
    temp_buffer[base_offset + idx_9(row_start+2, col_start+2)] = m[2, 2]

@wp.func
def reconstruct_block_3x3_9(
    row_block: int, col_block: int, 
    base_offset: int, offset_V: int, 
    temp_buffer: wp.array(dtype=float)
):
    res = wp.mat33(0.0)
    row_start = row_block * 3
    col_start = col_block * 3
    
    for i in range(3):
        for j in range(3):
            global_r = row_start + i
            global_c = col_start + j
            
            sum_val = float(0.0) # Explicit float
            
            # While loop to prevent unrolling
            k = int(0)
            while k < 9:
                lam = temp_buffer[base_offset + idx_9(k, k)]
                v_rk = temp_buffer[offset_V + idx_9(global_r, k)]
                v_ck = temp_buffer[offset_V + idx_9(global_c, k)]
                sum_val += v_rk * lam * v_ck
                k += 1
            
            res[i, j] = sum_val
    return res

# ==============================================================================
# 2. 核心设备函数 (9x9 Optimized)
# ==============================================================================

@wp.func
def filter_hessian_9x9_device(
    h_aa: wp.mat33, h_ab: wp.mat33, h_ac: wp.mat33,
    h_ba: wp.mat33, h_bb: wp.mat33, h_bc: wp.mat33,
    h_ca: wp.mat33, h_cb: wp.mat33, h_cc: wp.mat33,
    temp_buffer: wp.array(dtype=float),
    tid: int
):
    # 9x9 needs 81(Mat) + 81(Vec) = 162 floats
    base_offset = tid * 162
    offset_V = base_offset + 81
    
    # --- 1. Load Data ---
    load_block_3x3_9(0, 0, base_offset, temp_buffer, h_aa)
    load_block_3x3_9(0, 1, base_offset, temp_buffer, h_ab)
    load_block_3x3_9(0, 2, base_offset, temp_buffer, h_ac)
    
    load_block_3x3_9(1, 0, base_offset, temp_buffer, h_ba)
    load_block_3x3_9(1, 1, base_offset, temp_buffer, h_bb)
    load_block_3x3_9(1, 2, base_offset, temp_buffer, h_bc)
    
    load_block_3x3_9(2, 0, base_offset, temp_buffer, h_ca)
    load_block_3x3_9(2, 1, base_offset, temp_buffer, h_cb)
    load_block_3x3_9(2, 2, base_offset, temp_buffer, h_cc)

    # --- 2. Jacobi SVD ---
    
    # Init V
    i = int(0)
    while i < 9:
        j = int(0)
        while j < 9:
            temp_buffer[offset_V + idx_9(i, j)] = 1.0 if i == j else 0.0
            j += 1
        i += 1

    # 15 sweeps
    iter_count = int(0)
    while iter_count < 15:
        p = int(0)
        while p < 8:
            q = p + 1
            while q < 9:
                idx_pq = base_offset + idx_9(p, q)
                a_pq = temp_buffer[idx_pq]
                
                if wp.abs(a_pq) >= 1e-6:
                    idx_pp = base_offset + idx_9(p, p)
                    idx_qq = base_offset + idx_9(q, q)
                    a_pp = temp_buffer[idx_pp]
                    a_qq = temp_buffer[idx_qq]

                    tau = (a_qq - a_pp) / (2.0 * a_pq)
                    t = float(0.0)
                    if tau >= 0.0: t = 1.0 / (tau + wp.sqrt(1.0 + tau*tau))
                    else:          t = -1.0 / (-tau + wp.sqrt(1.0 + tau*tau))
                    c = 1.0 / wp.sqrt(1.0 + t*t)
                    s = t * c

                    # Rotate A
                    temp_buffer[idx_pp] = c*c*a_pp + s*s*a_qq - 2.0*c*s*a_pq
                    temp_buffer[idx_qq] = s*s*a_pp + c*c*a_qq + 2.0*c*s*a_pq
                    temp_buffer[idx_pq] = 0.0
                    temp_buffer[base_offset + idx_9(q, p)] = 0.0

                    # Rotate Rows/Cols
                    k = int(0)
                    while k < 9:
                        if k != p and k != q:
                            idx_ip = base_offset + idx_9(k, p)
                            idx_iq = base_offset + idx_9(k, q)
                            a_ip = temp_buffer[idx_ip]
                            a_iq = temp_buffer[idx_iq]
                            
                            a_ip_n = c * a_ip - s * a_iq
                            a_iq_n = s * a_ip + c * a_iq
                            
                            temp_buffer[idx_ip] = a_ip_n
                            temp_buffer[base_offset + idx_9(p, k)] = a_ip_n
                            temp_buffer[idx_iq] = a_iq_n
                            temp_buffer[base_offset + idx_9(q, k)] = a_iq_n
                        k += 1
                    
                    # Rotate V
                    k = int(0)
                    while k < 9:
                        idx_ip = offset_V + idx_9(k, p)
                        idx_iq = offset_V + idx_9(k, q)
                        v_ip = temp_buffer[idx_ip]
                        v_iq = temp_buffer[idx_iq]
                        
                        temp_buffer[idx_ip] = c * v_ip - s * v_iq
                        temp_buffer[idx_iq] = s * v_ip + c * v_iq
                        k += 1
                q += 1
            p += 1
        iter_count += 1

    # --- 3. Filter Eigenvalues ---
    k = int(0)
    while k < 9:
        idx = base_offset + idx_9(k, k)
        val = temp_buffer[idx]
        temp_buffer[idx] = wp.max(val, 0.0)
        k += 1

    # --- 4. Reconstruct ---
    out_aa = reconstruct_block_3x3_9(0, 0, base_offset, offset_V, temp_buffer)
    out_ab = reconstruct_block_3x3_9(0, 1, base_offset, offset_V, temp_buffer)
    out_ac = reconstruct_block_3x3_9(0, 2, base_offset, offset_V, temp_buffer)
    
    out_ba = reconstruct_block_3x3_9(1, 0, base_offset, offset_V, temp_buffer)
    out_bb = reconstruct_block_3x3_9(1, 1, base_offset, offset_V, temp_buffer)
    out_bc = reconstruct_block_3x3_9(1, 2, base_offset, offset_V, temp_buffer)
    
    out_ca = reconstruct_block_3x3_9(2, 0, base_offset, offset_V, temp_buffer)
    out_cb = reconstruct_block_3x3_9(2, 1, base_offset, offset_V, temp_buffer)
    out_cc = reconstruct_block_3x3_9(2, 2, base_offset, offset_V, temp_buffer)

    return out_aa, out_ab, out_ac, \
           out_ba, out_bb, out_bc, \
           out_ca, out_cb, out_cc

# ==============================================================================
# 1. 设备函数 (12x12 Jacobi Filtering)
# ==============================================================================

@wp.func
def idx_12(r: int, c: int):
    return r * 12 + c

@wp.func
def load_block_3x3_12(
    row_block: int, col_block: int, 
    base_offset: int, 
    temp_buffer: wp.array(dtype=float), 
    m: wp.mat33
):
    """ 将 mat33 写入 flatten array 的特定 block 位置 """
    row_start = row_block * 3
    col_start = col_block * 3
    
    # Row 0
    temp_buffer[base_offset + idx_12(row_start+0, col_start+0)] = m[0, 0]
    temp_buffer[base_offset + idx_12(row_start+0, col_start+1)] = m[0, 1]
    temp_buffer[base_offset + idx_12(row_start+0, col_start+2)] = m[0, 2]
    # Row 1
    temp_buffer[base_offset + idx_12(row_start+1, col_start+0)] = m[1, 0]
    temp_buffer[base_offset + idx_12(row_start+1, col_start+1)] = m[1, 1]
    temp_buffer[base_offset + idx_12(row_start+1, col_start+2)] = m[1, 2]
    # Row 2
    temp_buffer[base_offset + idx_12(row_start+2, col_start+0)] = m[2, 0]
    temp_buffer[base_offset + idx_12(row_start+2, col_start+1)] = m[2, 1]
    temp_buffer[base_offset + idx_12(row_start+2, col_start+2)] = m[2, 2]

@wp.func
def reconstruct_block_3x3_12(
    row_block: int, col_block: int, 
    base_offset: int, offset_V: int, 
    temp_buffer: wp.array(dtype=float)
):
    """ 重构 V * Lambda * V^T 并提取 3x3 """
    res = wp.mat33(0.0)
    row_start = row_block * 3
    col_start = col_block * 3
    
    for i in range(3):
        for j in range(3):
            global_r = row_start + i
            global_c = col_start + j
            
            # 【关键修复】使用 float(0.0) 声明这是一个可变的累加器
            sum_val = float(0.0)
            
            # 使用 while 循环防止编译器过度展开
            k = int(0)
            while k < 12:
                lam = temp_buffer[base_offset + idx_12(k, k)]
                v_rk = temp_buffer[offset_V + idx_12(global_r, k)]
                v_ck = temp_buffer[offset_V + idx_12(global_c, k)]
                sum_val += v_rk * lam * v_ck
                k += 1
            
            res[i, j] = sum_val
    return res

# ==============================================================================
# 2. 核心设备函数 (While 循环 + 类型声明优化版)
# ==============================================================================

@wp.func
def filter_hessian_12x12_device(
    h_aa: wp.mat33, h_ab: wp.mat33, h_ac: wp.mat33, h_ad: wp.mat33,
    h_ba: wp.mat33, h_bb: wp.mat33, h_bc: wp.mat33, h_bd: wp.mat33,
    h_ca: wp.mat33, h_cb: wp.mat33, h_cc: wp.mat33, h_cd: wp.mat33,
    h_da: wp.mat33, h_db: wp.mat33, h_dc: wp.mat33, h_dd: wp.mat33,
    temp_buffer: wp.array(dtype=float),
    tid: int
):
    base_offset = tid * 288
    offset_V = base_offset + 144
    
    # --- 1. 加载数据 ---
    load_block_3x3_12(0, 0, base_offset, temp_buffer, h_aa)
    load_block_3x3_12(0, 1, base_offset, temp_buffer, h_ab)
    load_block_3x3_12(0, 2, base_offset, temp_buffer, h_ac)
    load_block_3x3_12(0, 3, base_offset, temp_buffer, h_ad)
    
    load_block_3x3_12(1, 0, base_offset, temp_buffer, h_ba)
    load_block_3x3_12(1, 1, base_offset, temp_buffer, h_bb)
    load_block_3x3_12(1, 2, base_offset, temp_buffer, h_bc)
    load_block_3x3_12(1, 3, base_offset, temp_buffer, h_bd)
    
    load_block_3x3_12(2, 0, base_offset, temp_buffer, h_ca)
    load_block_3x3_12(2, 1, base_offset, temp_buffer, h_cb)
    load_block_3x3_12(2, 2, base_offset, temp_buffer, h_cc)
    load_block_3x3_12(2, 3, base_offset, temp_buffer, h_cd)
    
    load_block_3x3_12(3, 0, base_offset, temp_buffer, h_da)
    load_block_3x3_12(3, 1, base_offset, temp_buffer, h_db)
    load_block_3x3_12(3, 2, base_offset, temp_buffer, h_dc)
    load_block_3x3_12(3, 3, base_offset, temp_buffer, h_dd)

    # --- 2. Jacobi 分解 ---
    
    # Init V Identity
    i = int(0)
    while i < 12:
        j = int(0)
        while j < 12:
            temp_buffer[offset_V + idx_12(i, j)] = 1.0 if i == j else 0.0
            j += 1
        i += 1

    # Main Loop: 15 sweeps
    iter_count = int(0)
    while iter_count < 15:
        p = int(0)
        while p < 11:
            q = p + 1
            while q < 12:
                idx_pq = base_offset + idx_12(p, q)
                a_pq = temp_buffer[idx_pq]
                
                # Threshold check
                if wp.abs(a_pq) >= 1e-6:
                    idx_pp = base_offset + idx_12(p, p)
                    idx_qq = base_offset + idx_12(q, q)
                    a_pp = temp_buffer[idx_pp]
                    a_qq = temp_buffer[idx_qq]

                    tau = (a_qq - a_pp) / (2.0 * a_pq)
                    t = float(0.0) # Explicit float type
                    if tau >= 0.0: t = 1.0 / (tau + wp.sqrt(1.0 + tau*tau))
                    else:          t = -1.0 / (-tau + wp.sqrt(1.0 + tau*tau))
                    c = 1.0 / wp.sqrt(1.0 + t*t)
                    s = t * c

                    # Rotate A
                    temp_buffer[idx_pp] = c*c*a_pp + s*s*a_qq - 2.0*c*s*a_pq
                    temp_buffer[idx_qq] = s*s*a_pp + c*c*a_qq + 2.0*c*s*a_pq
                    temp_buffer[idx_pq] = 0.0
                    temp_buffer[base_offset + idx_12(q, p)] = 0.0

                    # Inner Loop: Rotate rows/cols
                    k = int(0)
                    while k < 12:
                        if k != p and k != q:
                            idx_ip = base_offset + idx_12(k, p)
                            idx_iq = base_offset + idx_12(k, q)
                            a_ip = temp_buffer[idx_ip]
                            a_iq = temp_buffer[idx_iq]
                            
                            a_ip_n = c * a_ip - s * a_iq
                            a_iq_n = s * a_ip + c * a_iq
                            
                            temp_buffer[idx_ip] = a_ip_n
                            temp_buffer[base_offset + idx_12(p, k)] = a_ip_n
                            temp_buffer[idx_iq] = a_iq_n
                            temp_buffer[base_offset + idx_12(q, k)] = a_iq_n
                        k += 1
                    
                    # Inner Loop: Rotate V
                    k = int(0)
                    while k < 12:
                        idx_ip = offset_V + idx_12(k, p)
                        idx_iq = offset_V + idx_12(k, q)
                        v_ip = temp_buffer[idx_ip]
                        v_iq = temp_buffer[idx_iq]
                        
                        temp_buffer[idx_ip] = c * v_ip - s * v_iq
                        temp_buffer[idx_iq] = s * v_ip + c * v_iq
                        k += 1

                q += 1 
            p += 1 
        iter_count += 1 

    # --- 3. 过滤特征值 ---
    k = int(0)
    while k < 12:
        idx = base_offset + idx_12(k, k)
        val = temp_buffer[idx]
        temp_buffer[idx] = wp.max(val, 0.0)
        k += 1

    # --- 4. 重构并返回 ---
    out_aa = reconstruct_block_3x3_12(0, 0, base_offset, offset_V, temp_buffer)
    out_ab = reconstruct_block_3x3_12(0, 1, base_offset, offset_V, temp_buffer)
    out_ac = reconstruct_block_3x3_12(0, 2, base_offset, offset_V, temp_buffer)
    out_ad = reconstruct_block_3x3_12(0, 3, base_offset, offset_V, temp_buffer)
    
    out_ba = reconstruct_block_3x3_12(1, 0, base_offset, offset_V, temp_buffer)
    out_bb = reconstruct_block_3x3_12(1, 1, base_offset, offset_V, temp_buffer)
    out_bc = reconstruct_block_3x3_12(1, 2, base_offset, offset_V, temp_buffer)
    out_bd = reconstruct_block_3x3_12(1, 3, base_offset, offset_V, temp_buffer)
    
    out_ca = reconstruct_block_3x3_12(2, 0, base_offset, offset_V, temp_buffer)
    out_cb = reconstruct_block_3x3_12(2, 1, base_offset, offset_V, temp_buffer)
    out_cc = reconstruct_block_3x3_12(2, 2, base_offset, offset_V, temp_buffer)
    out_cd = reconstruct_block_3x3_12(2, 3, base_offset, offset_V, temp_buffer)
    
    out_da = reconstruct_block_3x3_12(3, 0, base_offset, offset_V, temp_buffer)
    out_db = reconstruct_block_3x3_12(3, 1, base_offset, offset_V, temp_buffer)
    out_dc = reconstruct_block_3x3_12(3, 2, base_offset, offset_V, temp_buffer)
    out_dd = reconstruct_block_3x3_12(3, 3, base_offset, offset_V, temp_buffer)

    return out_aa, out_ab, out_ac, out_ad, \
           out_ba, out_bb, out_bc, out_bd, \
           out_ca, out_cb, out_cc, out_cd, \
           out_da, out_db, out_dc, out_dd

# svd



@wp.func
def zcy_evaluate_dihedral_angle_based_bending_force_hessian(
    bending_index: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),  # [新增] 需要上一帧位置算速度
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angle: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    stiffness: float,
    damping: float,   # [新增] 阻尼系数 (通常是一个小的比率，如 0.01~0.1)
    dt: float         # [新增] 时间步长
):
    eps = 1.0e-6

    vi0 = edge_indices[bending_index, 0]
    vi1 = edge_indices[bending_index, 1]
    vi2 = edge_indices[bending_index, 2]
    vi3 = edge_indices[bending_index, 3]

    x0 = pos[vi0]  # opposite 0
    x1 = pos[vi1]  # opposite 1
    x2 = pos[vi2]  # edge start
    x3 = pos[vi3]  # edge end

    # Compute edge vectors
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    e = x3 - x2

    # Compute normals
    n1 = wp.cross(x02, x03)
    n2 = wp.cross(x13, x12)

    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)
    e_norm = wp.length(e)

    # Early exit for degenerate cases
    if n1_norm < eps or n2_norm < eps or e_norm < eps:
        return wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)

    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm
    e_hat = e / e_norm

    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    cos_theta = wp.dot(n1_hat, n2_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    # 基础刚度 k
    k = stiffness * edge_rest_length[bending_index]
    
    # 弹性力的标量部分: k * (theta - theta_0)
    # 我们稍后会把阻尼加到这个变量里，这样后面计算 f0...f3 的代码不用变
    dE_dtheta = k * (theta - edge_rest_angle[bending_index])

    # Pre-compute skew matrices 
    skew_e = wp.skew(e)
    skew_x03 = wp.skew(x03)
    skew_x02 = wp.skew(x02)
    skew_x13 = wp.skew(x13)
    skew_x12 = wp.skew(x12)
    skew_n1 = wp.skew(n1_hat)
    skew_n2 = wp.skew(n2_hat)

    # Compute derivatives (省略中间未变代码，与原函数一致...)
    I3 = wp.identity(n=3, dtype=float)
    Pn1 = I3 - wp.outer(n1_hat, n1_hat)
    Pn2 = I3 - wp.outer(n2_hat, n2_hat)
    Pe = I3 - wp.outer(e_hat, e_hat)

    dn1hat_dx0 = (1.0 / n1_norm) * Pn1 * (-skew_x03 - skew_x02)
    dn2hat_dx0 = wp.mat33(0.0)

    dn1hat_dx1 = wp.mat33(0.0)
    dn2hat_dx1 = (1.0 / n2_norm) * Pn2 * (-skew_x12 - skew_x13)

    dn1hat_dx2 = (1.0 / n1_norm) * Pn1 * (-skew_x03)
    dn2hat_dx2 = (1.0 / n2_norm) * Pn2 * (skew_x13)

    dn1hat_dx3 = (1.0 / n1_norm) * Pn1 * (skew_x02)
    dn2hat_dx3 = (1.0 / n2_norm) * Pn2 * (-skew_x12)

    dehat_dx0 = wp.mat33(0.0)
    dehat_dx1 = wp.mat33(0.0)
    dehat_dx2 = (1.0 / e_norm) * Pe * (-I3)
    dehat_dx3 = (1.0 / e_norm) * Pe * (I3)

    c = cos_theta
    s = sin_theta
    denom = s * s + c * c

    cross_n1n2 = wp.cross(n1_hat, n2_hat)

    A0 = skew_n1 * dn2hat_dx0 - skew_n2 * dn1hat_dx0
    ds_dx0 = wp.transpose(A0) * e_hat + wp.transpose(dehat_dx0) * cross_n1n2
    dc_dx0 = wp.transpose(dn1hat_dx0) * n2_hat + wp.transpose(dn2hat_dx0) * n1_hat
    dtheta_dx0 = (c * ds_dx0 - s * dc_dx0) * (1.0 / denom)

    A1 = skew_n1 * dn2hat_dx1 - skew_n2 * dn1hat_dx1
    ds_dx1 = wp.transpose(A1) * e_hat + wp.transpose(dehat_dx1) * cross_n1n2
    dc_dx1 = wp.transpose(dn1hat_dx1) * n2_hat + wp.transpose(dn2hat_dx1) * n1_hat
    dtheta_dx1 = (c * ds_dx1 - s * dc_dx1) * (1.0 / denom)

    A2 = skew_n1 * dn2hat_dx2 - skew_n2 * dn1hat_dx2
    ds_dx2 = wp.transpose(A2) * e_hat + wp.transpose(dehat_dx2) * cross_n1n2
    dc_dx2 = wp.transpose(dn1hat_dx2) * n2_hat + wp.transpose(dn2hat_dx2) * n1_hat
    dtheta_dx2 = (c * ds_dx2 - s * dc_dx2) * (1.0 / denom)

    A3 = skew_n1 * dn2hat_dx3 - skew_n2 * dn1hat_dx3
    ds_dx3 = wp.transpose(A3) * e_hat + wp.transpose(dehat_dx3) * cross_n1n2
    dc_dx3 = wp.transpose(dn1hat_dx3) * n2_hat + wp.transpose(dn2hat_dx3) * n1_hat
    dtheta_dx3 = (c * ds_dx3 - s * dc_dx3) * (1.0 / denom)

    # 四顶点的角度梯度向量 (g)
    g0 = dtheta_dx0
    g1 = dtheta_dx1
    g2 = dtheta_dx2
    g3 = dtheta_dx3
    
    # ----------------------------------------------
    # [Damping 处理核心部分]
    # ----------------------------------------------
    if damping > 0.0:
        inv_dt = 1.0 / dt
        
        # 1. 计算四个点的速度 v = (x - x_prev) / dt
        # 你的参考代码里直接用了 dx，但我这里为了物理单位统一，还原为速度
        v0 = (x0 - pos_prev[vi0]) * inv_dt
        v1 = (x1 - pos_prev[vi1]) * inv_dt
        v2 = (x2 - pos_prev[vi2]) * inv_dt
        v3 = (x3 - pos_prev[vi3]) * inv_dt
        
        # 2. 计算角度随时间的变化率 dtheta / dt
        # Chain rule: dtheta/dt = sum( dtheta/dx_i * v_i )
        dtheta_dt = wp.dot(g0, v0) + wp.dot(g1, v1) + wp.dot(g2, v2) + wp.dot(g3, v3)
        
        # 3. 计算阻尼力标量
        # Force_damping = - (damping * k) * dtheta_dt * Gradient
        # 我们把标量部分加到 dE_dtheta 上，下面计算 force 的代码会自动带上阻尼
        dE_dtheta += (damping * k) * dtheta_dt
        
        # 4. 修改 Hessian 系数
        # 原始: k
        # 阻尼: k * damping / dt
        # 合并: k * (1 + damping/dt)
        k = k * (1.0 + damping * inv_dt)

    # ----------------------------------------------
    # 下面的代码完全不用变，因为我们已经修改了 dE_dtheta 和 k
    # ----------------------------------------------

    # 四个力：f_i = -Total_Scalar * g_i
    f0 = -dE_dtheta * g0
    f1 = -dE_dtheta * g1
    f2 = -dE_dtheta * g2
    f3 = -dE_dtheta * g3

    # 16 个 Hessian 块：H_ij = Total_K * g_i g_j^T
    h00 = k * wp.outer(g0, g0)
    h01 = k * wp.outer(g0, g1)
    h02 = k * wp.outer(g0, g2)
    h03 = k * wp.outer(g0, g3)

    h10 = k * wp.outer(g1, g0)
    h11 = k * wp.outer(g1, g1)
    h12 = k * wp.outer(g1, g2)
    h13 = k * wp.outer(g1, g3)

    h20 = k * wp.outer(g2, g0)
    h21 = k * wp.outer(g2, g1)
    h22 = k * wp.outer(g2, g2)
    h23 = k * wp.outer(g2, g3)

    h30 = k * wp.outer(g3, g0)
    h31 = k * wp.outer(g3, g1)
    h32 = k * wp.outer(g3, g2)
    h33 = k * wp.outer(g3, g3)

    return f0, f1, f2, f3, h00, h01, h02, h03, h10, h11, h12, h13, h20, h21, h22, h23, h30, h31, h32, h33

@wp.func
def zcy_evaluate_stvk_force_hessian(
    face: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3), # [新增] 用于计算速度
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_pose: wp.mat22,
    area: float,
    mu: float,
    lmbd: float,
    kd: float,      # [新增] 刚度阻尼系数 (Rayleigh Damping coeff)
    dt: float,      # [新增] 时间步长
):
    # --- 1) ~ 6) 保持原有的 StVK 弹性力与 Hessian 计算逻辑不变 ---
    
    # 1) 组装 F 的两列
    v0 = tri_indices[face, 0]
    v1 = tri_indices[face, 1]
    v2 = tri_indices[face, 2]

    x0 = pos[v0]
    x01 = pos[v1] - x0
    x02 = pos[v2] - x0

    DmInv00 = tri_pose[0, 0]
    DmInv01 = tri_pose[0, 1]
    DmInv10 = tri_pose[1, 0]
    DmInv11 = tri_pose[1, 1]

    f0 = x01 * DmInv00 + x02 * DmInv10
    f1 = x01 * DmInv01 + x02 * DmInv11

    # 2) Green 应变与阈值
    f0f0 = wp.dot(f0, f0)
    f1f1 = wp.dot(f1, f1)
    f0f1 = wp.dot(f0, f1)

    G00 = 0.5 * (f0f0 - 1.0)
    G11 = 0.5 * (f1f1 - 1.0)
    G01 = 0.5 * f0f1

    G_norm_sq = G00 * G00 + G11 * G11 + 2.0 * G01 * G01
    
    # [提前返回检查]
    if G_norm_sq < 1.0e-20:
        z = wp.vec3(0.0, 0.0, 0.0)
        Z = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return z, z, z, Z, Z, Z, Z, Z, Z, Z, Z, Z

    # 3) PK1 两列
    t = G00 + G11
    two_mu = 2.0 * mu
    lt = lmbd * t
    PK1_col0 = f0 * (two_mu * G00 + lt) + f1 * (two_mu * G01)
    PK1_col1 = f0 * (two_mu * G01) + f1 * (two_mu * G11 + lt)

    # 4) f 空间 Hessian 块
    Ic = f0f0 + f1f1
    c = -mu + (0.5 * Ic - 1.0) * lmbd
    I3 = wp.identity(n=3, dtype=float)

    f0of0 = wp.outer(f0, f0)
    f1of1 = wp.outer(f1, f1)
    f0of1 = wp.outer(f0, f1)
    f1of0 = wp.outer(f1, f0)

    A00 = lmbd * f0of0 + c * I3 + mu * (f0f0 * I3 + 2.0 * f0of0 + f1of1)
    A01 = lmbd * f0of1 + mu * (f0f1 * I3 + f1of0)
    A11 = lmbd * f1of1 + c * I3 + mu * (f1f1 * I3 + 2.0 * f1of1 + f0of0)

    # 5) 位置空间系数 alpha
    a00 = -(DmInv00 + DmInv10)
    a01 = DmInv00
    a02 = DmInv10
    b00 = -(DmInv01 + DmInv11)
    b01 = DmInv01
    b02 = DmInv11

    # 6) 三个弹性力 f_elastic
    f_a0 = -(PK1_col0 * a00 + PK1_col1 * b00) * area
    f_a1 = -(PK1_col0 * a01 + PK1_col1 * b01) * area
    f_a2 = -(PK1_col0 * a02 + PK1_col1 * b02) * area

    # 7) 九个弹性海森块 K_elastic (H00...H22)
    A01T = wp.transpose(A01)

    H00 = (a00 * a00) * A00 + (b00 * b00) * A11 + (a00 * b00) * A01 + (b00 * a00) * A01T
    H01 = (a00 * a01) * A00 + (b00 * b01) * A11 + (a00 * b01) * A01 + (b00 * a01) * A01T
    H02 = (a00 * a02) * A00 + (b00 * b02) * A11 + (a00 * b02) * A01 + (b00 * a02) * A01T

    H10 = (a01 * a00) * A00 + (b01 * b00) * A11 + (a01 * b00) * A01 + (b01 * a00) * A01T
    H11 = (a01 * a01) * A00 + (b01 * b01) * A11 + (a01 * b01) * A01 + (b01 * a01) * A01T
    H12 = (a01 * a02) * A00 + (b01 * b02) * A11 + (a01 * b02) * A01 + (b01 * a02) * A01T

    H20 = (a02 * a00) * A00 + (b02 * b00) * A11 + (a02 * b00) * A01 + (b02 * a00) * A01T
    H21 = (a02 * a01) * A00 + (b02 * b01) * A11 + (a02 * b01) * A01 + (b02 * a01) * A01T
    H22 = (a02 * a02) * A00 + (b02 * b02) * A11 + (a02 * b02) * A01 + (b02 * a02) * A01T

    # 面积缩放 Hessian (得到真实的 Stiffness Matrix K)
    H00 *= area; H01 *= area; H02 *= area
    H10 *= area; H11 *= area; H12 *= area
    H20 *= area; H21 *= area; H22 *= area

    # --- [新增逻辑] 添加 Rayleigh Damping ---
    
    # A. 计算阻尼力 (Damping Force)
    # 物理公式: f_damp = -kd * (K * v)
    if kd > 0.0:
        inv_dt = 1.0 / dt
        
        # 1. 计算三个顶点的速度
        vel0 = (pos[v0] - pos_prev[v0]) * inv_dt
        vel1 = (pos[v1] - pos_prev[v1]) * inv_dt
        vel2 = (pos[v2] - pos_prev[v2]) * inv_dt

        # 2. 计算 K * v (利用刚度矩阵与速度的乘积)
        # 注意：这里利用了你刚才问到的公式 [Haa*va + Hab*vb + Hac*vc]
        # 这样能自动处理非对角线耦合
        kv0 = H00 * vel0 + H01 * vel1 + H02 * vel2
        kv1 = H10 * vel0 + H11 * vel1 + H12 * vel2
        kv2 = H20 * vel0 + H21 * vel1 + H22 * vel2

        # 3. 将阻尼力叠加到总力上
        f_a0 += -kd * kv0
        f_a1 += -kd * kv1
        f_a2 += -kd * kv2

        # B. 计算阻尼 Hessian (Damping Hessian)
        # 物理公式: H_total = K_elastic + (kd/dt) * K_elastic
        # 系数 scale = 1.0 + kd/dt
        hessian_scale = 1.0 + kd * inv_dt
        
        H00 *= hessian_scale
        H01 *= hessian_scale
        H02 *= hessian_scale
        H10 *= hessian_scale
        H11 *= hessian_scale
        H12 *= hessian_scale
        H20 *= hessian_scale
        H21 *= hessian_scale
        H22 *= hessian_scale

    return f_a0, f_a1, f_a2, H00, H01, H02, H10, H11, H12, H20, H21, H22

@wp.func
def zcy_evaluate_edge_edge_contact_2_vertices(
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),  # [新增] 用于计算速度
    dt: float,                          # [新增] 时间步长
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    barrier_threshold: float,
):
    r"""
    Returns the edge-edge contact force and hessian with Rayleigh Damping.
    """
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]

    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    e1_vec = e1_v2_pos - e1_v1_pos
    e2_vec = e2_v2_pos - e2_v1_pos
    c1 = e1_v1_pos + e1_vec * s
    c2 = e2_v1_pos + e2_vec * t

    diff = c1 - c2
    dis = st[2]
    
    # 注意：collision_normal 指向 e1 -> e2 还是 e2 -> e1 取决于 diff 的方向
    # 这里 diff = c1 - c2，所以 normal 指向 c1 (Edge1)
    # 如果 dis 非常小，normal 可能会不稳定，wp.closest_point 内部处理了
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        # Barycentric weights specific to the vector (c1 - c2)
        # c1 = (1-s)v1 + s*v2
        # c2 = (1-t)v3 + t*v4
        # diff = (1-s)v1 + s*v2 - (1-t)v3 - t*v4
        # Coefficients: [1-s, s, -1+t, -t]
        bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness, barrier_threshold)

        # 1. 基础弹性部分 (Core Elastic Force & Hessian)
        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # --- [新增] Damping 处理 ---
        if collision_damping > 0.0:
            inv_dt = 1.0 / dt
            
            # 计算4个顶点的速度
            v_a = (e1_v1_pos - pos_prev[e1_v1]) * inv_dt
            v_b = (e1_v2_pos - pos_prev[e1_v2]) * inv_dt
            v_c = (e2_v1_pos - pos_prev[e2_v1]) * inv_dt
            v_d = (e2_v2_pos - pos_prev[e2_v2]) * inv_dt

            # 计算接触点处的相对速度 (v_c1 - v_c2)
            # 利用同样的重心坐标权重 bs 组合速度
            v_rel = v_a * bs[0] + v_b * bs[1] + v_c * bs[2] + v_d * bs[3]

            # 投影相对速度到法线方向
            # normal 从 c2 指向 c1。
            # v_rel = v_c1 - v_c2.
            # dot > 0 表示分离，dot < 0 表示靠近(挤压)
            v_proj = wp.dot(v_rel, collision_normal)

            # 只在相互靠近（挤压）时施加阻尼
            if v_proj < 0.0:
                # 1. 阻尼力: f = -kd * (K * v)
                # K * v = (k * n * n^T) * v_rel = k * n * (v_proj)
                # 这里我们直接用算好的 3x3 collision_hessian 矩阵乘向量
                damping_force_vec = -collision_damping * (collision_hessian * v_rel)
                
                # 将阻尼力叠加到总力
                collision_force += damping_force_vec

                # 2. 阻尼 Hessian: K_new = K * (1 + kd/dt)
                scale = 1.0 + collision_damping * inv_dt
                collision_hessian *= scale
        # -------------------------

        # 以下代码保持不变，负责将 3x3 的核心力和矩阵分发给 4x4 的块
        ### edge1
        collision_force_a = collision_force * bs[0]
        collision_force_b = collision_force * bs[1]

        collision_hessian_aa = collision_hessian * bs[0] * bs[0]
        collision_hessian_bb = collision_hessian * bs[1] * bs[1]
        collision_hessian_ab = collision_hessian * bs[0] * bs[1] 
        collision_hessian_ba = collision_hessian * bs[1] * bs[0] 

        ### edge2
        collision_force_c = collision_force * bs[2]
        collision_force_d = collision_force * bs[3]

        collision_hessian_cc = collision_hessian * bs[2] * bs[2]
        collision_hessian_dd = collision_hessian * bs[3] * bs[3]
        collision_hessian_cd = collision_hessian * bs[2] * bs[3] 
        collision_hessian_dc = collision_hessian * bs[3] * bs[2] 

        # edge1 to edge2
        collision_hessian_ac = bs[0] * bs[2] * collision_hessian  
        collision_hessian_ad = bs[0] * bs[3] * collision_hessian  
        collision_hessian_bc = bs[1] * bs[2] * collision_hessian  
        collision_hessian_bd = bs[1] * bs[3] * collision_hessian  

        # edge2 to edge1
        collision_hessian_ca = bs[2] * bs[0] * collision_hessian  
        collision_hessian_cb = bs[2] * bs[1] * collision_hessian  
        collision_hessian_da = bs[3] * bs[0] * collision_hessian  
        collision_hessian_db = bs[3] * bs[1] * collision_hessian  

        return True, collision_force_a, collision_force_b, collision_force_c, collision_force_d,\
                    collision_hessian_aa, collision_hessian_ab, collision_hessian_ac, collision_hessian_ad, \
                    collision_hessian_ba, collision_hessian_bb, collision_hessian_bc, collision_hessian_bd, \
                    collision_hessian_ca, collision_hessian_cb, collision_hessian_cc, collision_hessian_cd, \
                    collision_hessian_da, collision_hessian_db, collision_hessian_dc, collision_hessian_dd
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return False, collision_force, collision_force, collision_force, collision_force,\
                    collision_hessian, collision_hessian, collision_hessian, collision_hessian,\
                    collision_hessian, collision_hessian,collision_hessian, collision_hessian,\
                    collision_hessian, collision_hessian,collision_hessian, collision_hessian,\
                    collision_hessian, collision_hessian,collision_hessian, collision_hessian

@wp.func
def zcy_evaluate_vertex_triangle_collision_force_hessian_4_vertices(
    v: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),  # [新增] 用于计算速度
    dt: float,                          # [新增] 时间步长
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    barrier_threshold: float,
):
    # 获取三角形三个顶点的索引
    idx_a = tri_indices[tri, 0]
    idx_b = tri_indices[tri, 1]
    idx_c = tri_indices[tri, 2]

    a = pos[idx_a]
    b = pos[idx_b]
    c = pos[idx_c]
    p = pos[v]

    closest_p, bary, feature_type = triangle_closest_point(a, b, c, p)

    # diff 指向 P (Vertex) 的方向: P - ClosestPoint
    diff = p - closest_p
    dis = wp.length(diff)
    
    # 防止除零 (wp.closest_point 通常保证 diff 非零，但在极端重叠下需小心)
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        # bs = [-u, -v, -w, 1]
        # 对应梯度权重: Triangle Vertices (负权重), Point Vertex (正权重)
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness, barrier_threshold)

        # 1. 基础弹性力和 Hessian (3x3 Core)
        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # --- [新增] Damping 处理 ---
        if collision_damping > 0.0:
            inv_dt = 1.0 / dt
            
            # 计算4个顶点的速度
            vel_a = (a - pos_prev[idx_a]) * inv_dt
            vel_b = (b - pos_prev[idx_b]) * inv_dt
            vel_c = (c - pos_prev[idx_c]) * inv_dt
            vel_p = (p - pos_prev[v]) * inv_dt

            # 计算相对速度 (v_point - v_closest_on_triangle)
            # 因为 bs 前三项是负的，最后一项是 1.0
            # v_rel = 1.0 * vel_p + (-u)*vel_a + (-v)*vel_b + (-w)*vel_c
            # 这正好就是我们需要的物理相对速度
            v_rel = vel_a * bs[0] + vel_b * bs[1] + vel_c * bs[2] + vel_p * bs[3]

            # 投影到碰撞法线方向 (Normal 指向 P)
            # dot < 0 表示 P 和三角形正在相互靠近(挤压)
            v_proj = wp.dot(v_rel, collision_normal)

            if v_proj < 0.0:
                # 1. 阻尼力: f_damp = -kd * (K * v_rel)
                # 使用已经算好的刚度矩阵 collision_hessian
                damping_force_vec = -collision_damping * (collision_hessian * v_rel)
                
                # 叠加力
                collision_force += damping_force_vec

                # 2. 阻尼 Hessian: K_new = K * (1 + kd/dt)
                scale = 1.0 + collision_damping * inv_dt
                collision_hessian *= scale
        # -------------------------

        # 以下逻辑保持不变 (利用 bs 权重分发核心力和矩阵)
        
        # Force Distribution
        collision_force_a = collision_force * bs[0]
        collision_force_b = collision_force * bs[1]
        collision_force_c = collision_force * bs[2]
        collision_force_d = collision_force * bs[3]

        # Hessian Diagonal Blocks
        collision_hessian_aa = collision_hessian * bs[0] * bs[0]
        collision_hessian_bb = collision_hessian * bs[1] * bs[1]
        collision_hessian_cc = collision_hessian * bs[2] * bs[2]
        collision_hessian_dd = collision_hessian * bs[3] * bs[3]

        # Hessian Off-Diagonal Blocks
        # vertex (d) to triangle (a,b,c) interactions and triangle internal interactions
        collision_hessian_ab = bs[0] * bs[1] * collision_hessian 
        collision_hessian_ac = bs[0] * bs[2] * collision_hessian 
        collision_hessian_ad = bs[0] * bs[3] * collision_hessian 
        
        collision_hessian_ba = bs[1] * bs[0] * collision_hessian 
        collision_hessian_bc = bs[1] * bs[2] * collision_hessian 
        collision_hessian_bd = bs[1] * bs[3] * collision_hessian 
        
        collision_hessian_ca = bs[2] * bs[0] * collision_hessian 
        collision_hessian_cb = bs[2] * bs[1] * collision_hessian 
        collision_hessian_cd = bs[2] * bs[3] * collision_hessian 
        
        collision_hessian_da = bs[3] * bs[0] * collision_hessian 
        collision_hessian_db = bs[3] * bs[1] * collision_hessian 
        collision_hessian_dc = bs[3] * bs[2] * collision_hessian 


        return (
            True,
            collision_force_a, collision_force_b, collision_force_c, collision_force_d,\
            collision_hessian_aa, collision_hessian_ab, collision_hessian_ac, collision_hessian_ad, \
            collision_hessian_ba, collision_hessian_bb, collision_hessian_bc, collision_hessian_bd, \
            collision_hessian_ca, collision_hessian_cb, collision_hessian_cc, collision_hessian_cd, \
            collision_hessian_da, collision_hessian_db, collision_hessian_dc, collision_hessian_dd
        )
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return (
            False,
            collision_force, collision_force, collision_force, collision_force,
            collision_hessian, collision_hessian, collision_hessian, collision_hessian,
            collision_hessian, collision_hessian, collision_hessian, collision_hessian,
            collision_hessian, collision_hessian, collision_hessian, collision_hessian,
            collision_hessian, collision_hessian, collision_hessian, collision_hessian,
        )

@wp.func
def zcy_evaluate_spring_force_and_hessian(
    v0: wp.vec3, 
    v1: wp.vec3,
    l0: float,
    k: float,
):
    # 计算向量与长度
    diff = v0 - v1
    l = wp.length(diff)

    # 防止除以零
    if l < 1.0e-6:
        return wp.vec3(0.0, 0.0, 0.0), wp.mat33(0.0)

    # 方向
    n = diff / l

    # -------------------------------
    # 力：F = -k (l - l0) * n
    # -------------------------------
    f = - k * (l - l0) * n

    # -------------------------------
    # 正定 Hessian (近似几何刚度项)
    # -------------------------------
    # 这里用标准弹簧 Hessian 推导形式
    # H = k [ n n^T + ((l - l0)/l) * (I - n n^T) ]
    I = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        )
    H = k * (wp.outer(n, n) + ((l - l0) / l) * (I - wp.outer(n, n)))

    return f, H

@wp.func
def zcy_apply_conservative_bound_truncation(
    v_index: wp.int32,
    pos_new: wp.vec3,
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    truncation_threshold: float,
):
    particle_pos_prev_collision_detection = pos_prev_collision_detection[v_index]
    accumulated_displacement = pos_new - particle_pos_prev_collision_detection
    conservative_bound = particle_conservative_bounds[v_index]

    accumulated_displacement_norm = wp.length(accumulated_displacement)
    if accumulated_displacement_norm > conservative_bound and conservative_bound > truncation_threshold:
        accumulated_displacement_norm_truncated = conservative_bound
        accumulated_displacement = accumulated_displacement * (
            accumulated_displacement_norm_truncated / accumulated_displacement_norm
        )
        return particle_pos_prev_collision_detection + accumulated_displacement
    else:
        return pos_new
# zcy
# endregion: zcy


# region: zcy
# zcy
# zcy_forward
@wp.kernel
def zcy_forward_step_penetration_free(
    dt: float,
    forward_type: int,
    gravity: wp.vec3,
    prev_pos: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    all_particle_flag: wp.array(dtype=wp.int32),
    truncation_threshold: float,
):
    particle_index = wp.tid()

    if all_particle_flag[particle_index] == -1:
        return

    if forward_type == 0:
        para = 0.0
    else:
        para = 1.0

    vel_new = vel[particle_index] + para * gravity * dt
    pos_inertia = prev_pos[particle_index] + para * vel_new * dt 

    pos[particle_index] = zcy_apply_conservative_bound_truncation(
        particle_index, 
        pos_inertia, 
        pos_prev_collision_detection, 
        particle_conservative_bounds, 
        truncation_threshold
    )

@wp.kernel
def zcy_truncation_by_conservative_bounds(
    pos_new: wp.array(dtype=wp.vec3),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    pos_cur_truncation: wp.array(dtype=wp.vec3),
    truncation_threshold: float,
):
    particle_index = wp.tid()

    pos_cur_truncation[particle_index] = zcy_apply_conservative_bound_truncation(
        particle_index, 
        pos_new[particle_index], 
        pos_prev_collision_detection, 
        particle_conservative_bounds, 
        truncation_threshold
    )

# zcy_hessian
@wp.kernel
def zcy_VBD_accumulate_contact_force_and_hessian(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    # DeBUG
    DeBUG_Eigen: bool,
    DeBUG_Contact_EE: bool,
    DeBUG_Contact_VT: bool,
    barrier_threshold: float,
    pos_prev: wp.array(dtype=wp.vec3),
    dt: float,
    temp_mem1: wp.array(dtype=float),
    temp_mem2: wp.array(dtype=float),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    # self contact
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    # outputs: particle force and hessian
    # edge_contact
    edge_contact_forces: wp.array(dtype=wp.vec3),
    edge_contact_hessian_values: wp.array(dtype=wp.mat33),
    edge_contact_hessian_rows: wp.array(dtype=int),
    edge_contact_hessian_cols: wp.array(dtype=int),
    # vertex-triangle_contact
    vt_contact_forces: wp.array(dtype=wp.vec3),
    vt_contact_hessian_values: wp.array(dtype=wp.mat33),
    vt_contact_hessian_rows: wp.array(dtype=int),
    vt_contact_hessian_cols: wp.array(dtype=int),
):
    
    t_id = wp.tid()
    collision_info = collision_info_array[0]
    
    if DeBUG_Contact_EE:
        # process edge-edge collisions
        if t_id * 2 < collision_info.edge_colliding_edges.shape[0]:
            #wp.printf("t_id: %d\n", collision_info.edge_colliding_edges.shape[0])
            #wp.printf("t_id: %d, e1_idx: %d, e2_idx: %d\n", t_id, collision_info.edge_colliding_edges[2 * t_id], collision_info.edge_colliding_edges[2 * t_id + 1])
            e1_idx = collision_info.edge_colliding_edges[2 * t_id]
            e2_idx = collision_info.edge_colliding_edges[2 * t_id + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]
                e2_v1 = edge_indices[e2_idx, 2]
                e2_v2 = edge_indices[e2_idx, 3]

                (has_contact, collision_force_a, collision_force_b, collision_force_c, collision_force_d,
                        collision_hessian_aa, collision_hessian_ab, collision_hessian_ac, collision_hessian_ad, 
                        collision_hessian_ba, collision_hessian_bb, collision_hessian_bc, collision_hessian_bd, 
                        collision_hessian_ca, collision_hessian_cb, collision_hessian_cc, collision_hessian_cd, 
                        collision_hessian_da, collision_hessian_db, collision_hessian_dc, collision_hessian_dd
                ) =  zcy_evaluate_edge_edge_contact_2_vertices(
                        e1_idx,
                        e2_idx,
                        pos,
                        pos_prev,
                        dt,
                        edge_indices,
                        collision_radius,
                        soft_contact_ke,
                        soft_contact_kd,
                        friction_mu,
                        friction_epsilon,
                        edge_edge_parallel_epsilon,
                        barrier_threshold,
                    )

                if DeBUG_Eigen:
                    # 1. 左边用括号包裹，允许换行
                    (
                        collision_hessian_aa, collision_hessian_ab, collision_hessian_ac, collision_hessian_ad,
                        collision_hessian_ba, collision_hessian_bb, collision_hessian_bc, collision_hessian_bd,
                        collision_hessian_ca, collision_hessian_cb, collision_hessian_cc, collision_hessian_cd,
                        collision_hessian_da, collision_hessian_db, collision_hessian_dc, collision_hessian_dd
                    ) = filter_hessian_12x12_device(
                        # 2. 函数参数也按 4x4 排列
                        collision_hessian_aa, collision_hessian_ab, collision_hessian_ac, collision_hessian_ad, 
                        collision_hessian_ba, collision_hessian_bb, collision_hessian_bc, collision_hessian_bd, 
                        collision_hessian_ca, collision_hessian_cb, collision_hessian_cc, collision_hessian_cd, 
                        collision_hessian_da, collision_hessian_db, collision_hessian_dc, collision_hessian_dd,
                        # 3. 额外参数单独一行
                        temp_mem1, 
                        t_id
                    )


                #加两遍，除2
                if has_contact:
                    #wp.printf("has_contact: %d, e1_idx: %d, e2_idx: %d, e1_v1: %d, e1_v2: %d, e2_v1: %d, e2_v2: %d\n", 
                    #          has_contact, e1_idx, e2_idx, e1_v1, e1_v2, e2_v1, e2_v2)
                    # edge1
                    # force
                    wp.atomic_add(edge_contact_forces, e1_v1, collision_force_a*0.5 )
                    wp.atomic_add(edge_contact_forces, e1_v2, collision_force_b*0.5 )

                    # edge2
                    # force
                    wp.atomic_add(edge_contact_forces, e2_v1, collision_force_c*0.5 )
                    wp.atomic_add(edge_contact_forces, e2_v2, collision_force_d*0.5 )

                    # 假设每个 contact 预分配 16 个条目
                    # contact_base: 每个 contact 的起始索引
                    contact_index = t_id
                    contact_base = contact_index * 16  # contact_index 需根据循环传入

                    # --- edge1 ---
                    # (e1_v1, e1_v1)
                    edge_contact_hessian_rows[contact_base + 0] = e1_v1
                    edge_contact_hessian_cols[contact_base + 0] = e1_v1
                    edge_contact_hessian_values[contact_base + 0] = collision_hessian_aa*0.5
                    # (e1_v1, e1_v2)
                    edge_contact_hessian_rows[contact_base + 1] = e1_v1
                    edge_contact_hessian_cols[contact_base + 1] = e1_v2
                    edge_contact_hessian_values[contact_base + 1] = collision_hessian_ab*0.5
                    # (e1_v2, e1_v1)
                    edge_contact_hessian_rows[contact_base + 2] = e1_v2
                    edge_contact_hessian_cols[contact_base + 2] = e1_v1
                    edge_contact_hessian_values[contact_base + 2] = collision_hessian_ba*0.5
                    # (e1_v2, e1_v2)
                    edge_contact_hessian_rows[contact_base + 3] = e1_v2
                    edge_contact_hessian_cols[contact_base + 3] = e1_v2
                    edge_contact_hessian_values[contact_base + 3] = collision_hessian_bb*0.5

                    # --- edge2 ---
                    edge_contact_hessian_rows[contact_base + 4] = e2_v1
                    edge_contact_hessian_cols[contact_base + 4] = e2_v1
                    edge_contact_hessian_values[contact_base + 4] = collision_hessian_cc*0.5

                    edge_contact_hessian_rows[contact_base + 5] = e2_v1
                    edge_contact_hessian_cols[contact_base + 5] = e2_v2
                    edge_contact_hessian_values[contact_base + 5] = collision_hessian_cd*0.5

                    edge_contact_hessian_rows[contact_base + 6] = e2_v2
                    edge_contact_hessian_cols[contact_base + 6] = e2_v1
                    edge_contact_hessian_values[contact_base + 6] = collision_hessian_dc*0.5

                    edge_contact_hessian_rows[contact_base + 7] = e2_v2
                    edge_contact_hessian_cols[contact_base + 7] = e2_v2
                    edge_contact_hessian_values[contact_base + 7] = collision_hessian_dd*0.5

                    # --- edge1 <-> edge2 cross blocks ---
                    edge_contact_hessian_rows[contact_base + 8] = e1_v1
                    edge_contact_hessian_cols[contact_base + 8] = e2_v1
                    edge_contact_hessian_values[contact_base + 8] = collision_hessian_ac*0.5

                    edge_contact_hessian_rows[contact_base + 9] = e1_v1
                    edge_contact_hessian_cols[contact_base + 9] = e2_v2
                    edge_contact_hessian_values[contact_base + 9] = collision_hessian_ad*0.5

                    edge_contact_hessian_rows[contact_base + 10] = e1_v2
                    edge_contact_hessian_cols[contact_base + 10] = e2_v1
                    edge_contact_hessian_values[contact_base + 10] = collision_hessian_bc*0.5

                    edge_contact_hessian_rows[contact_base + 11] = e1_v2
                    edge_contact_hessian_cols[contact_base + 11] = e2_v2
                    edge_contact_hessian_values[contact_base + 11] = collision_hessian_bd*0.5

                    # --- edge2 <-> edge1 cross blocks ---
                    edge_contact_hessian_rows[contact_base + 12] = e2_v1
                    edge_contact_hessian_cols[contact_base + 12] = e1_v1
                    edge_contact_hessian_values[contact_base + 12] = collision_hessian_ca*0.5

                    edge_contact_hessian_rows[contact_base + 13] = e2_v1
                    edge_contact_hessian_cols[contact_base + 13] = e1_v2
                    edge_contact_hessian_values[contact_base + 13] = collision_hessian_cb*0.5

                    edge_contact_hessian_rows[contact_base + 14] = e2_v2
                    edge_contact_hessian_cols[contact_base + 14] = e1_v1
                    edge_contact_hessian_values[contact_base + 14] = collision_hessian_da*0.5

                    edge_contact_hessian_rows[contact_base + 15] = e2_v2
                    edge_contact_hessian_cols[contact_base + 15] = e1_v2
                    edge_contact_hessian_values[contact_base + 15] = collision_hessian_db*0.5

    if DeBUG_Contact_VT:
        # process vertex-triangle collisions
        if t_id * 2 < collision_info.vertex_colliding_triangles.shape[0]:
            #wp.printf("t_id: %d\n", collision_info.vertex_colliding_triangles.shape[0])
            #wp.printf("t_id: %d, v_idx: %d, t_idx: %d\n", t_id, collision_info.vertex_colliding_triangles[2 * t_id], collision_info.vertex_colliding_triangles[2 * t_id + 1])
            particle_idx = collision_info.vertex_colliding_triangles[2 * t_id]
            tri_idx = collision_info.vertex_colliding_triangles[2 * t_id + 1]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                (has_contact, collision_force_a, collision_force_b, collision_force_c, collision_force_d,
                        collision_hessian_aa, collision_hessian_ab, collision_hessian_ac, collision_hessian_ad, 
                        collision_hessian_ba, collision_hessian_bb, collision_hessian_bc, collision_hessian_bd, 
                        collision_hessian_ca, collision_hessian_cb, collision_hessian_cc, collision_hessian_cd, 
                        collision_hessian_da, collision_hessian_db, collision_hessian_dc, collision_hessian_dd
                ) = zcy_evaluate_vertex_triangle_collision_force_hessian_4_vertices(
                    particle_idx,
                    tri_idx,
                    pos,
                    pos_prev,
                    dt,
                    tri_indices,
                    collision_radius,
                    soft_contact_ke,
                    soft_contact_kd,
                    friction_mu,
                    friction_epsilon,
                    barrier_threshold,
                )

                if DeBUG_Eigen:
                    # 1. 左边用括号包裹，允许换行
                    (
                        collision_hessian_aa, collision_hessian_ab, collision_hessian_ac, collision_hessian_ad,
                        collision_hessian_ba, collision_hessian_bb, collision_hessian_bc, collision_hessian_bd,
                        collision_hessian_ca, collision_hessian_cb, collision_hessian_cc, collision_hessian_cd,
                        collision_hessian_da, collision_hessian_db, collision_hessian_dc, collision_hessian_dd
                    ) = filter_hessian_12x12_device(
                        # 2. 函数参数也按 4x4 排列
                        collision_hessian_aa, collision_hessian_ab, collision_hessian_ac, collision_hessian_ad, 
                        collision_hessian_ba, collision_hessian_bb, collision_hessian_bc, collision_hessian_bd, 
                        collision_hessian_ca, collision_hessian_cb, collision_hessian_cc, collision_hessian_cd, 
                        collision_hessian_da, collision_hessian_db, collision_hessian_dc, collision_hessian_dd,
                        # 3. 额外参数单独一行
                        temp_mem2, 
                        t_id
                    )

                if has_contact:
                    #wp.printf("has_contact: %d, p_idx: %d, t_idx: %d, t_a: %d, t_b: %d, t_c: %d\n", 
                    #          has_contact, particle_idx, tri_idx, tri_a, tri_b, tri_c)

                    contact_index = t_id
                    contact_base = contact_index * 16  # 每个 particle-tri contact 占16个条目

                    # --- 力累加 ---
                    wp.atomic_add(vt_contact_forces, particle_idx, collision_force_d)
                    wp.atomic_add(vt_contact_forces, tri_a, collision_force_a)
                    wp.atomic_add(vt_contact_forces, tri_b, collision_force_b)
                    wp.atomic_add(vt_contact_forces, tri_c, collision_force_c)

                    # --- 对角块 ---
                    # particle
                    vt_contact_hessian_rows[contact_base + 0] = particle_idx
                    vt_contact_hessian_cols[contact_base + 0] = particle_idx
                    vt_contact_hessian_values[contact_base + 0] = collision_hessian_dd

                    # tri_a
                    vt_contact_hessian_rows[contact_base + 1] = tri_a
                    vt_contact_hessian_cols[contact_base + 1] = tri_a
                    vt_contact_hessian_values[contact_base + 1] = collision_hessian_aa

                    # tri_b
                    vt_contact_hessian_rows[contact_base + 2] = tri_b
                    vt_contact_hessian_cols[contact_base + 2] = tri_b
                    vt_contact_hessian_values[contact_base + 2] = collision_hessian_bb

                    # tri_c
                    vt_contact_hessian_rows[contact_base + 3] = tri_c
                    vt_contact_hessian_cols[contact_base + 3] = tri_c
                    vt_contact_hessian_values[contact_base + 3] = collision_hessian_cc

                    # --- cross blocks ---
                    # a0
                    vt_contact_hessian_rows[contact_base + 4] = tri_a
                    vt_contact_hessian_cols[contact_base + 4] = tri_b
                    vt_contact_hessian_values[contact_base + 4] = collision_hessian_ab

                    vt_contact_hessian_rows[contact_base + 5] = tri_a
                    vt_contact_hessian_cols[contact_base + 5] = tri_c
                    vt_contact_hessian_values[contact_base + 5] = collision_hessian_ac

                    vt_contact_hessian_rows[contact_base + 6] = tri_a
                    vt_contact_hessian_cols[contact_base + 6] = particle_idx
                    vt_contact_hessian_values[contact_base + 6] = collision_hessian_ad

                    # b1
                    vt_contact_hessian_rows[contact_base + 7] = tri_b
                    vt_contact_hessian_cols[contact_base + 7] = tri_a
                    vt_contact_hessian_values[contact_base + 7] = collision_hessian_ba

                    vt_contact_hessian_rows[contact_base + 8] = tri_b
                    vt_contact_hessian_cols[contact_base + 8] = tri_c
                    vt_contact_hessian_values[contact_base + 8] = collision_hessian_bc

                    vt_contact_hessian_rows[contact_base + 9] = tri_b
                    vt_contact_hessian_cols[contact_base + 9] = particle_idx
                    vt_contact_hessian_values[contact_base + 9] = collision_hessian_bd

                    # c2
                    vt_contact_hessian_rows[contact_base + 10] = tri_c
                    vt_contact_hessian_cols[contact_base + 10] = tri_a
                    vt_contact_hessian_values[contact_base + 10] = collision_hessian_ca

                    vt_contact_hessian_rows[contact_base + 11] = tri_c
                    vt_contact_hessian_cols[contact_base + 11] = tri_b
                    vt_contact_hessian_values[contact_base + 11] = collision_hessian_cb

                    vt_contact_hessian_rows[contact_base + 12] = tri_c
                    vt_contact_hessian_cols[contact_base + 12] = particle_idx
                    vt_contact_hessian_values[contact_base + 12] = collision_hessian_cd

                    # p3
                    vt_contact_hessian_rows[contact_base + 13] = particle_idx
                    vt_contact_hessian_cols[contact_base + 13] = tri_a
                    vt_contact_hessian_values[contact_base + 13] = collision_hessian_da

                    vt_contact_hessian_rows[contact_base + 14] = particle_idx
                    vt_contact_hessian_cols[contact_base + 14] = tri_b
                    vt_contact_hessian_values[contact_base + 14] = collision_hessian_db

                    vt_contact_hessian_rows[contact_base + 15] = particle_idx
                    vt_contact_hessian_cols[contact_base + 15] = tri_c
                    vt_contact_hessian_values[contact_base + 15] = collision_hessian_dc

@wp.kernel
def zcy_accumulate_spring_force_and_hessian(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    DeBUG_Eigen: bool,
    pos_prev: wp.array(dtype=wp.vec3),
    dt: float,
    temp_mem: wp.array(dtype=float),
    # spring constraints
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    damping_ratio: float,
    # outputs: particle force and hessian
    spring_forces: wp.array(dtype=wp.vec3),
    spring_hessian_values: wp.array(dtype=wp.mat33),
    spring_hessian_rows: wp.array(dtype=int),
    spring_hessian_cols: wp.array(dtype=int)
):
    spring_index = wp.tid()
    # 获取两个端点
    i = spring_indices[spring_index * 2]
    j = spring_indices[spring_index * 2 + 1]
    v0 = pos[i]
    v1 = pos[j]

    f_ij, H = zcy_evaluate_spring_force_and_hessian(
        v0, v1,
        spring_rest_length[spring_index],
        spring_stiffness[spring_index],
    )

    # --- [Rayleigh Damping 添加部分] ---
    if damping_ratio > 0.0:
        inv_dt = 1.0 / dt
        
        # A. 计算相对速度 (v_i - v_j)
        # 既然 f_ij 是 i 点受力，我们需要基于 i 和 j 的相对运动来阻碍它
        vel_i = (v0 - pos_prev[i]) * inv_dt
        vel_j = (v1 - pos_prev[j]) * inv_dt
        rel_vel = vel_i - vel_j

        # B. 计算阻尼力
        # 公式: f_damp_on_i = -kd * (K_ii * v_i + K_ij * v_j)
        # 对于弹簧: K_ii = H, K_ij = -H
        # 所以: f_damp_on_i = -kd * H * (v_i - v_j)
        f_damp = -damping_ratio * (H * rel_vel)
        
        # 叠加到总力
        f_ij += f_damp

        # C. 缩放 Hessian
        # 系数 scale = 1.0 + kd / dt
        scale = 1.0 + damping_ratio * inv_dt
        H = H * scale
    # -----------------------------------
    H_aa = H
    H_ab = -H
    H_ba = -H
    H_bb = H

    if DeBUG_Eigen:
        H_aa, H_ab, H_ba, H_bb = filter_hessian_6x6_device(
            H, -H,
            -H, H,
            temp_mem, spring_index
        )

    # --- 累加到端点 ---
    # i: 受到 +f_ij 力
    # j: 受到 -f_ij 力
    wp.atomic_add(spring_forces, i, f_ij)
    wp.atomic_add(spring_forces, j, -f_ij)

    # 记录4个对称块
    # 每个spring_index 生成4个条目：base + [0,1,2,3]
    base = spring_index * 4

    # (i,i): +H
    spring_hessian_rows[base + 0] = i
    spring_hessian_cols[base + 0] = i
    spring_hessian_values[base + 0] = H_aa

    # (i,j): -H
    spring_hessian_rows[base + 1] = i
    spring_hessian_cols[base + 1] = j
    spring_hessian_values[base + 1] = H_ab

    # (j,i): -H
    spring_hessian_rows[base + 2] = j
    spring_hessian_cols[base + 2] = i
    spring_hessian_values[base + 2] = H_ba

    # (j,j): +H
    spring_hessian_rows[base + 3] = j
    spring_hessian_cols[base + 3] = j
    spring_hessian_values[base + 3] = H_bb

@wp.kernel
def zcy_accumulate_stvk_force_and_hessian(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    DeBUG_Eigen: bool,
    pos_prev: wp.array(dtype=wp.vec3),
    dt: float,
    temp_mem: wp.array(dtype=float),
    # stvk force and hessian
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    # outputs: particle force and hessian
    stvk_forces: wp.array(dtype=wp.vec3),
    stvk_hessian_values: wp.array(dtype=wp.mat33),
    stvk_hessian_rows: wp.array(dtype=int),
    stvk_hessian_cols: wp.array(dtype=int)
):
    tri_index = wp.tid()
    
    # 获取当前三角形的索引和顶点顺序
    a = tri_indices[tri_index, 0]
    b = tri_indices[tri_index, 1]
    c = tri_indices[tri_index, 2]

    # elastic force and hessian
    f_a, f_b, f_c, h_aa, h_ab, h_ac, h_ba, h_bb, h_bc, h_ca, h_cb, h_cc = zcy_evaluate_stvk_force_hessian(
        tri_index,
        pos,
        pos_prev,
        tri_indices,
        tri_poses[tri_index],
        tri_areas[tri_index],
        tri_materials[tri_index, 0],
        tri_materials[tri_index, 1],
        tri_materials[tri_index, 2],
        dt,
    )

    if DeBUG_Eigen:
        (
            h_aa, h_ab, h_ac, 
            h_ba, h_bb, h_bc, 
            h_ca, h_cb, h_cc
        ) = filter_hessian_9x9_device(
            h_aa, h_ab, h_ac, 
            h_ba, h_bb, h_bc, 
            h_ca, h_cb, h_cc,
            temp_mem, tri_index
        )

    # --- 累加到端点 ---
    wp.atomic_add(stvk_forces, a, f_a)
    wp.atomic_add(stvk_forces, b, f_b)
    wp.atomic_add(stvk_forces, c, f_c)

    # 记录9个对称块
    # 每个spring_index 生成9个条目：base + [0,1,2,3,4,5,6,7,8]
    base = tri_index * 9

    # (a,a):
    stvk_hessian_rows[base + 0] = a
    stvk_hessian_cols[base + 0] = a
    stvk_hessian_values[base + 0] = h_aa

    # (a,b):
    stvk_hessian_rows[base + 1] = a
    stvk_hessian_cols[base + 1] = b
    stvk_hessian_values[base + 1] = h_ab

    # (a,c):
    stvk_hessian_rows[base + 2] = a
    stvk_hessian_cols[base + 2] = c
    stvk_hessian_values[base + 2] = h_ac

    # (b,a): 
    stvk_hessian_rows[base + 3] = b
    stvk_hessian_cols[base + 3] = a
    stvk_hessian_values[base + 3] = h_ba

    # (b,b): 
    stvk_hessian_rows[base + 4] = b
    stvk_hessian_cols[base + 4] = b
    stvk_hessian_values[base + 4] = h_bb

    # (b,c): 
    stvk_hessian_rows[base + 5] = b
    stvk_hessian_cols[base + 5] = c
    stvk_hessian_values[base + 5] = h_bc

    # (c,a): 
    stvk_hessian_rows[base + 6] = c
    stvk_hessian_cols[base + 6] = a
    stvk_hessian_values[base + 6] = h_ca

    # (c,b): 
    stvk_hessian_rows[base + 7] = c
    stvk_hessian_cols[base + 7] = b
    stvk_hessian_values[base + 7] = h_cb

    # (c,c): 
    stvk_hessian_rows[base + 8] = c
    stvk_hessian_cols[base + 8] = c
    stvk_hessian_values[base + 8] = h_cc


@wp.kernel
def zcy_accumulate_bending_force_and_hessian(
    pos: wp.array(dtype=wp.vec3),
    DeBUG_Eigen: bool,
    pos_prev: wp.array(dtype=wp.vec3),
    dt: float,
    temp_mem: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angle: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    bending_forces: wp.array(dtype=wp.vec3),
    bending_hessian_values: wp.array(dtype=wp.mat33),
    bending_hessian_rows: wp.array(dtype=int),
    bending_hessian_cols: wp.array(dtype=int),
):
    edge_index = wp.tid()

    # Skip invalid edges (boundary edges with missing opposite vertices)
    if edge_indices[edge_index, 0] == -1 or edge_indices[edge_index, 1] == -1:
        return

    # 当前边的四个顶点
    i = edge_indices[edge_index, 0]
    j = edge_indices[edge_index, 1]
    k = edge_indices[edge_index, 2]
    l = edge_indices[edge_index, 3]

    # 单元评估：返回四力与 16 个 3x3 Hessian 子块
    f0, f1, f2, f3, h00, h01, h02, h03, h10, h11, h12, h13, h20, h21, h22, h23, h30, h31, h32, h33 = zcy_evaluate_dihedral_angle_based_bending_force_hessian(
        edge_index,
        pos,
        pos_prev,
        edge_indices,
        edge_rest_angle,
        edge_rest_length,
        edge_bending_properties[edge_index, 0],
        edge_bending_properties[edge_index, 1],
        dt,
    )
    # 过滤无效的Hessian块
    if DeBUG_Eigen:
        (
            h00, h01, h02, h03, h10, h11, h12, h13, h20, h21, h22, h23, h30, h31, h32, h33
        ) = filter_hessian_12x12_device(
            h00, h01, h02, h03, h10, h11, h12, h13, h20, h21, h22, h23, h30, h31, h32, h33,
            temp_mem, 
            edge_index
        )
    

    # 原子累加到四个顶点
    wp.atomic_add(bending_forces, i, f0)
    wp.atomic_add(bending_forces, j, f1)
    wp.atomic_add(bending_forces, k, f2)
    wp.atomic_add(bending_forces, l, f3)

    # COO 写入：每条边写 16 个块
    base = edge_index * 16

    bending_hessian_rows[base + 0] = i
    bending_hessian_cols[base + 0] = i
    bending_hessian_values[base + 0] = h00

    bending_hessian_rows[base + 1] = i
    bending_hessian_cols[base + 1] = j
    bending_hessian_values[base + 1] = h01

    bending_hessian_rows[base + 2] = i
    bending_hessian_cols[base + 2] = k
    bending_hessian_values[base + 2] = h02

    bending_hessian_rows[base + 3] = i
    bending_hessian_cols[base + 3] = l
    bending_hessian_values[base + 3] = h03

    bending_hessian_rows[base + 4] = j
    bending_hessian_cols[base + 4] = i
    bending_hessian_values[base + 4] = h10

    bending_hessian_rows[base + 5] = j
    bending_hessian_cols[base + 5] = j
    bending_hessian_values[base + 5] = h11

    bending_hessian_rows[base + 6] = j
    bending_hessian_cols[base + 6] = k
    bending_hessian_values[base + 6] = h12

    bending_hessian_rows[base + 7] = j
    bending_hessian_cols[base + 7] = l
    bending_hessian_values[base + 7] = h13

    bending_hessian_rows[base + 8] = k
    bending_hessian_cols[base + 8] = i
    bending_hessian_values[base + 8] = h20

    bending_hessian_rows[base + 9] = k
    bending_hessian_cols[base + 9] = j
    bending_hessian_values[base + 9] = h21

    bending_hessian_rows[base + 10] = k
    bending_hessian_cols[base + 10] = k
    bending_hessian_values[base + 10] = h22

    bending_hessian_rows[base + 11] = k
    bending_hessian_cols[base + 11] = l
    bending_hessian_values[base + 11] = h23

    bending_hessian_rows[base + 12] = l
    bending_hessian_cols[base + 12] = i
    bending_hessian_values[base + 12] = h30

    bending_hessian_rows[base + 13] = l
    bending_hessian_cols[base + 13] = j
    bending_hessian_values[base + 13] = h31

    bending_hessian_rows[base + 14] = l
    bending_hessian_cols[base + 14] = k
    bending_hessian_values[base + 14] = h32

    bending_hessian_rows[base + 15] = l
    bending_hessian_cols[base + 15] = l
    bending_hessian_values[base + 15] = h33

@wp.kernel
def zcy_assemble_inertia_and_gravity_add_force(
    pos_warp: wp.array(dtype=wp.vec3),
    pos_prev_warp: wp.array(dtype=wp.vec3),
    vel_warp: wp.array(dtype=wp.vec3),
    dt: float,
    mass: float,
    gravity: wp.vec3,
    # force
    spring_forces: wp.array(dtype=wp.vec3),
    edge_contact_forces: wp.array(dtype=wp.vec3),
    vt_contact_forces: wp.array(dtype=wp.vec3),
    bending_forces: wp.array(dtype=wp.vec3),
    # fixed particle
    free_particle_offset: wp.array(dtype=wp.int32),
    # outputs: 
    grad: wp.array(dtype=wp.vec3)
):
    tid = wp.tid() 
    free_particle = tid + free_particle_offset[tid]

    # wp.printf("Thread %d: free_particle = %d\n", tid, free_particle)

    # inertia
    inertia = pos_warp[free_particle] - pos_prev_warp[free_particle] - dt * vel_warp[free_particle]

    grad[tid] = mass / (dt * dt) * inertia - (spring_forces[free_particle] + edge_contact_forces[free_particle] + vt_contact_forces[free_particle] + bending_forces[free_particle] + mass * gravity)

# zcy_force
@wp.kernel
def zcy_accumulate_contact_force(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    # DeBUG
    DeBUG_Contact_EE: bool,
    DeBUG_Contact_VT: bool,
    barrier_threshold: float,
    pos_prev: wp.array(dtype=wp.vec3),
    dt: float,
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    # self contact
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    # outputs: particle force and hessian
    # edge_contact
    edge_contact_forces: wp.array(dtype=wp.vec3),
    # vertex-triangle_contact
    vt_contact_forces: wp.array(dtype=wp.vec3),
):
    
    t_id = wp.tid()
    collision_info = collision_info_array[0]
    
    if DeBUG_Contact_EE:
        # process edge-edge collisions
        if t_id * 2 < collision_info.edge_colliding_edges.shape[0]:
            #wp.printf("t_id: %d\n", collision_info.edge_colliding_edges.shape[0])
            #wp.printf("t_id: %d, e1_idx: %d, e2_idx: %d\n", t_id, collision_info.edge_colliding_edges[2 * t_id], collision_info.edge_colliding_edges[2 * t_id + 1])
            e1_idx = collision_info.edge_colliding_edges[2 * t_id]
            e2_idx = collision_info.edge_colliding_edges[2 * t_id + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]
                e2_v1 = edge_indices[e2_idx, 2]
                e2_v2 = edge_indices[e2_idx, 3]

                (has_contact, collision_force_a, collision_force_b, collision_force_c, collision_force_d,
                        collision_hessian_aa, collision_hessian_ab, collision_hessian_ac, collision_hessian_ad, 
                        collision_hessian_ba, collision_hessian_bb, collision_hessian_bc, collision_hessian_bd, 
                        collision_hessian_ca, collision_hessian_cb, collision_hessian_cc, collision_hessian_cd, 
                        collision_hessian_da, collision_hessian_db, collision_hessian_dc, collision_hessian_dd
                ) =  zcy_evaluate_edge_edge_contact_2_vertices(
                        e1_idx,
                        e2_idx,
                        pos,
                        pos_prev,
                        dt,
                        edge_indices,
                        collision_radius,
                        soft_contact_ke,
                        soft_contact_kd,
                        friction_mu,
                        friction_epsilon,
                        edge_edge_parallel_epsilon,
                        barrier_threshold
                    )

                #加两遍，除2
                if has_contact:
                    #wp.printf("has_contact: %d, e1_idx: %d, e2_idx: %d, e1_v1: %d, e1_v2: %d, e2_v1: %d, e2_v2: %d\n", 
                    #          has_contact, e1_idx, e2_idx, e1_v1, e1_v2, e2_v1, e2_v2)
                    # edge1
                    # force
                    wp.atomic_add(edge_contact_forces, e1_v1, collision_force_a*0.5 )
                    wp.atomic_add(edge_contact_forces, e1_v2, collision_force_b*0.5 )

                    # edge2
                    # force
                    wp.atomic_add(edge_contact_forces, e2_v1, collision_force_c*0.5 )
                    wp.atomic_add(edge_contact_forces, e2_v2, collision_force_d*0.5 )


    if DeBUG_Contact_VT:
        # process vertex-triangle collisions
        if t_id * 2 < collision_info.vertex_colliding_triangles.shape[0]:
            #wp.printf("t_id: %d\n", collision_info.vertex_colliding_triangles.shape[0])
            #wp.printf("t_id: %d, v_idx: %d, t_idx: %d\n", t_id, collision_info.vertex_colliding_triangles[2 * t_id], collision_info.vertex_colliding_triangles[2 * t_id + 1])
            particle_idx = collision_info.vertex_colliding_triangles[2 * t_id]
            tri_idx = collision_info.vertex_colliding_triangles[2 * t_id + 1]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                (has_contact, collision_force_a, collision_force_b, collision_force_c, collision_force_d,
                        collision_hessian_aa, collision_hessian_ab, collision_hessian_ac, collision_hessian_ad, 
                        collision_hessian_ba, collision_hessian_bb, collision_hessian_bc, collision_hessian_bd, 
                        collision_hessian_ca, collision_hessian_cb, collision_hessian_cc, collision_hessian_cd, 
                        collision_hessian_da, collision_hessian_db, collision_hessian_dc, collision_hessian_dd
                ) = zcy_evaluate_vertex_triangle_collision_force_hessian_4_vertices(
                    particle_idx,
                    tri_idx,
                    pos,
                    pos_prev,
                    dt,
                    tri_indices,
                    collision_radius,
                    soft_contact_ke,
                    soft_contact_kd,
                    friction_mu,
                    friction_epsilon,
                    barrier_threshold
                )

                if has_contact:
                    # --- 力累加 ---
                    wp.atomic_add(vt_contact_forces, particle_idx, collision_force_d)
                    wp.atomic_add(vt_contact_forces, tri_a, collision_force_a)
                    wp.atomic_add(vt_contact_forces, tri_b, collision_force_b)
                    wp.atomic_add(vt_contact_forces, tri_c, collision_force_c)


@wp.kernel
def zcy_accumulate_spring_force(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    dt: float,
    # spring constraints
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    damping_ratio: float,
    # outputs: particle force and hessian
    spring_forces: wp.array(dtype=wp.vec3),
):
    spring_index = wp.tid()
    # 获取两个端点
    i = spring_indices[spring_index * 2]
    j = spring_indices[spring_index * 2 + 1]
    v0 = pos[i]
    v1 = pos[j]

    f_ij, H = zcy_evaluate_spring_force_and_hessian(
        v0, v1,
        spring_rest_length[spring_index],
        spring_stiffness[spring_index],
    )

    # --- [Rayleigh Damping 添加部分] ---
    if damping_ratio > 0.0:
        inv_dt = 1.0 / dt
        
        # A. 计算相对速度 (v_i - v_j)
        # 既然 f_ij 是 i 点受力，我们需要基于 i 和 j 的相对运动来阻碍它
        vel_i = (v0 - pos_prev[i]) * inv_dt
        vel_j = (v1 - pos_prev[j]) * inv_dt
        rel_vel = vel_i - vel_j

        # B. 计算阻尼力
        # 公式: f_damp_on_i = -kd * (K_ii * v_i + K_ij * v_j)
        # 对于弹簧: K_ii = H, K_ij = -H
        # 所以: f_damp_on_i = -kd * H * (v_i - v_j)
        f_damp = -damping_ratio * (H * rel_vel)
        
        # 叠加到总力
        f_ij += f_damp

        # C. 缩放 Hessian
        # 系数 scale = 1.0 + kd / dt
        scale = 1.0 + damping_ratio * inv_dt
        H = H * scale
    # -----------------------------------

    # --- 累加到端点 ---
    # i: 受到 +f_ij 力
    # j: 受到 -f_ij 力
    wp.atomic_add(spring_forces, i, f_ij)
    wp.atomic_add(spring_forces, j, -f_ij)


@wp.kernel
def zcy_accumulate_stvk_force(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    dt: float,
    # stvk force and hessian
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    # outputs: particle force and hessian
    stvk_forces: wp.array(dtype=wp.vec3),
):
    tri_index = wp.tid()
    
    # 获取当前三角形的索引和顶点顺序
    a = tri_indices[tri_index, 0]
    b = tri_indices[tri_index, 1]
    c = tri_indices[tri_index, 2]

    # elastic force and hessian
    f_a, f_b, f_c, h_aa, h_ab, h_ac, h_ba, h_bb, h_bc, h_ca, h_cb, h_cc = zcy_evaluate_stvk_force_hessian(
        tri_index,
        pos,
        pos_prev,
        tri_indices,
        tri_poses[tri_index],
        tri_areas[tri_index],
        tri_materials[tri_index, 0],
        tri_materials[tri_index, 1],
        tri_materials[tri_index, 2],
        dt,
    )

    # --- 累加到端点 ---
    wp.atomic_add(stvk_forces, a, f_a)
    wp.atomic_add(stvk_forces, b, f_b)
    wp.atomic_add(stvk_forces, c, f_c)


@wp.kernel
def zcy_accumulate_bending_force(
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    dt: float,
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angle: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    bending_forces: wp.array(dtype=wp.vec3),
):
    edge_index = wp.tid()

    # Skip invalid edges (boundary edges with missing opposite vertices)
    if edge_indices[edge_index, 0] == -1 or edge_indices[edge_index, 1] == -1:
        return

    # 当前边的四个顶点
    i = edge_indices[edge_index, 0]
    j = edge_indices[edge_index, 1]
    k = edge_indices[edge_index, 2]
    l = edge_indices[edge_index, 3]

    # 单元评估：返回四力与 16 个 3x3 Hessian 子块
    f0, f1, f2, f3, h00, h01, h02, h03, h10, h11, h12, h13, h20, h21, h22, h23, h30, h31, h32, h33 = zcy_evaluate_dihedral_angle_based_bending_force_hessian(
        edge_index,
        pos,
        pos_prev,
        edge_indices,
        edge_rest_angle,
        edge_rest_length,
        edge_bending_properties[edge_index, 0],
        edge_bending_properties[edge_index, 1],
        dt,
    )
    
    # 原子累加到四个顶点
    wp.atomic_add(bending_forces, i, f0)
    wp.atomic_add(bending_forces, j, f1)
    wp.atomic_add(bending_forces, k, f2)
    wp.atomic_add(bending_forces, l, f3)


# zcy_line_search 
# and zcy_residual
@wp.kernel
def zcy_residual_computation(
    pos_warp: wp.array(dtype=wp.vec3),
    pos_prev_warp: wp.array(dtype=wp.vec3),
    vel_warp: wp.array(dtype=wp.vec3),
    dt: float,
    mass: float,
    gravity: wp.vec3,
    # force
    spring_forces: wp.array(dtype=wp.vec3),
    edge_contact_forces: wp.array(dtype=wp.vec3),
    vt_contact_forces: wp.array(dtype=wp.vec3),
    bending_forces: wp.array(dtype=wp.vec3),
    # fixed particle
    free_particle_offset: wp.array(dtype=wp.int32),
    # outputs: 
    residual: wp.array(dtype=wp.vec3)
):
    tid = wp.tid() 
    free_particle = tid + free_particle_offset[tid]

    # inertia
    inertia = pos_warp[free_particle] - pos_prev_warp[free_particle] - dt * vel_warp[free_particle]

    residual[tid] = mass / (dt * dt) * inertia - (spring_forces[free_particle] + edge_contact_forces[free_particle] + vt_contact_forces[free_particle] + bending_forces[free_particle] + mass * gravity)


@wp.kernel
def zcy_compute_incremental_energy(
    residual: wp.array(dtype=wp.vec3),
    dx: wp.array(dtype=wp.vec3),
    alpha: float,
    c1: float,
    # outputs: 
    incremental_energy: wp.array(dtype=float)
):
    tid = wp.tid()

    # incremental energy
    incremental_energy_local = c1 * alpha * wp.dot(residual[tid], dx[tid])

    # accumulate
    wp.atomic_add(incremental_energy, 0, incremental_energy_local)

# and zcy_energy
@wp.kernel
def zcy_accumulate_inertia_energy(
    pos_warp: wp.array(dtype=wp.vec3),
    pos_prev_warp: wp.array(dtype=wp.vec3),
    vel_warp: wp.array(dtype=wp.vec3),
    dt: float,
    mass: float,
    gravity: wp.vec3,
    # fixed particle
    free_particle_offset: wp.array(dtype=wp.int32),
    # outputs: 
    energy: wp.array(dtype=float)
):
    tid = wp.tid() 
    free_particle = tid + free_particle_offset[tid]

    # inertia
    inertia = pos_warp[free_particle] - pos_prev_warp[free_particle] - dt * vel_warp[free_particle]

    energy_inertia = 0.5 * mass * wp.dot(inertia, inertia) /dt/dt  + mass * wp.dot(pos_warp[free_particle], -gravity)

    wp.atomic_add(energy, 0, energy_inertia)

@wp.kernel
def zcy_accumulate_contact_energy(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    DeBUG_Contact_EE: bool,
    DeBUG_Contact_VT: bool,
    barrier_threshold: float,
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    # self contact
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    collision_radius: float,
    soft_contact_ke: float,
    edge_edge_parallel_epsilon: float,
    # outputs: 
    energy: wp.array(dtype=float)
):
    
    t_id = wp.tid()
    collision_info = collision_info_array[0]
    
    if DeBUG_Contact_EE:
        # process edge-edge collisions
        if t_id * 2 < collision_info.edge_colliding_edges.shape[0]:
            e1_idx = collision_info.edge_colliding_edges[2 * t_id]
            e2_idx = collision_info.edge_colliding_edges[2 * t_id + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]
                e2_v1 = edge_indices[e2_idx, 2]
                e2_v2 = edge_indices[e2_idx, 3]

                e1_v1_pos = pos[e1_v1]
                e1_v2_pos = pos[e1_v2]
                e2_v1_pos = pos[e2_v1]
                e2_v2_pos = pos[e2_v2]

                st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
                s = st[0]
                t = st[1]
                dis = st[2]

                if 0.0 < dis < collision_radius:
                    tau = collision_radius * 0.5
                    if tau > dis > barrier_threshold:
                        k2 = 0.5 * tau * tau * soft_contact_ke
                        b = 0.5 * soft_contact_ke * (collision_radius - tau) * (collision_radius - tau) + k2 * wp.log(tau)
                        energy_edge = -k2 * wp.log(dis) + b
                    else:
                        energy_edge = 0.5 * soft_contact_ke * (collision_radius - dis) * (collision_radius - dis)
                    #加两遍，除2
                    wp.atomic_add(energy, 0, energy_edge / 2.0)

    if DeBUG_Contact_VT:
        # process vertex-triangle collisions
        if t_id * 2 < collision_info.vertex_colliding_triangles.shape[0]:
            particle_idx = collision_info.vertex_colliding_triangles[2 * t_id]
            tri_idx = collision_info.vertex_colliding_triangles[2 * t_id + 1]

            if particle_idx != -1 and tri_idx != -1:           
                tri_a = pos[tri_indices[tri_idx, 0]]
                tri_b = pos[tri_indices[tri_idx, 1]]
                tri_c = pos[tri_indices[tri_idx, 2]]
                p = pos[particle_idx]

                closest_p, bary, feature_type = triangle_closest_point(tri_a, tri_b, tri_c, p)
                diff = p - closest_p
                dis = wp.length(diff)

                if 0.0 < dis < collision_radius:
                    tau = collision_radius * 0.5
                    if tau > dis > barrier_threshold:
                        k2 = 0.5 * tau * tau * soft_contact_ke
                        b = 0.5 * soft_contact_ke * (collision_radius - tau) * (collision_radius - tau) + k2 * wp.log(tau)
                        energy_vt = -k2 * wp.log(dis) + b
                    else:
                        energy_vt = 0.5 * soft_contact_ke * (collision_radius - dis) * (collision_radius - dis)

                    wp.atomic_add(energy, 0, energy_vt)

@wp.kernel
def zcy_accumulate_spring_energy(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    # spring constraints
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    # outputs: particle force and hessian
    total_energy: wp.array(dtype=float)
):
    spring_index = wp.tid()
    # 获取两个端点
    i = spring_indices[spring_index * 2]
    j = spring_indices[spring_index * 2 + 1]
    v0 = pos[i]
    v1 = pos[j]

    # 弹簧能量
    rest_len = spring_rest_length[spring_index]
    k = spring_stiffness[spring_index]
    dis = wp.length(v1 - v0)
    energy_spring = 0.5 * k * (dis - rest_len) * (dis - rest_len)
    
    # --- 累加到总能量 ---
    wp.atomic_add(total_energy, 0, energy_spring)

@wp.kernel
def zcy_accumulate_stvk_energy(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    total_energy: wp.array(dtype=float)
):
    tri_index = wp.tid()
    
    # 获取当前三角形的索引
    v0 = tri_indices[tri_index, 0]
    v1 = tri_indices[tri_index, 1]
    v2 = tri_indices[tri_index, 2]

    # 获取材料参数
    mu = tri_materials[tri_index, 0]
    lmbd = tri_materials[tri_index, 1]
    area = tri_areas[tri_index]

    # 1) 组装 F 的两列
    x0 = pos[v0]
    x01 = pos[v1] - x0
    x02 = pos[v2] - x0

    DmInv = tri_poses[tri_index]
    DmInv00 = DmInv[0, 0]
    DmInv01 = DmInv[0, 1]
    DmInv10 = DmInv[1, 0]
    DmInv11 = DmInv[1, 1]

    f0 = x01 * DmInv00 + x02 * DmInv10
    f1 = x01 * DmInv01 + x02 * DmInv11

    # 2) Green 应变 G = 0.5 * (F^T F - I)
    f0f0 = wp.dot(f0, f0)
    f1f1 = wp.dot(f1, f1)
    f0f1 = wp.dot(f0, f1)

    G00 = 0.5 * (f0f0 - 1.0)
    G11 = 0.5 * (f1f1 - 1.0)
    G01 = 0.5 * f0f1
    
    # 3) StVK 能量密度计算
    # Psi = mu * tr(G^2) + 0.5 * lambda * (tr(G))^2
    # tr(G) = G00 + G11
    # tr(G^2) = G00^2 + G11^2 + 2 * G01^2
    
    tr_G = G00 + G11
    tr_G_sq = G00 * G00 + G11 * G11 + 2.0 * G01 * G01
    
    energy_density = mu * tr_G_sq + 0.5 * lmbd * tr_G * tr_G
    
    # 累加能量 (Psi * area)
    wp.atomic_add(total_energy, 0, energy_density * area)

@wp.kernel
def zcy_accumulate_bending_energy(
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angle: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    total_energy: wp.array(dtype=float)
):
    edge_index = wp.tid()

    # Skip invalid edges
    if edge_indices[edge_index, 0] == -1 or edge_indices[edge_index, 1] == -1:
        return

    vi0 = edge_indices[edge_index, 0]
    vi1 = edge_indices[edge_index, 1]
    vi2 = edge_indices[edge_index, 2]
    vi3 = edge_indices[edge_index, 3]

    x0 = pos[vi0]
    x1 = pos[vi1]
    x2 = pos[vi2]
    x3 = pos[vi3]

    # Compute edge vectors
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    e = x3 - x2

    # Compute normals
    n1 = wp.cross(x02, x03)
    n2 = wp.cross(x13, x12)

    eps = 1.0e-6
    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)
    e_norm = wp.length(e)

    if n1_norm < eps or n2_norm < eps or e_norm < eps:
        return

    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm
    e_hat = e / e_norm

    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    cos_theta = wp.dot(n1_hat, n2_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    # Bending Energy Calculation
    # E = 0.5 * k * (theta - theta_0)^2
    # k = stiffness * rest_length
    
    stiffness = edge_bending_properties[edge_index, 0]
    k = stiffness * edge_rest_length[edge_index]
    
    diff = theta - edge_rest_angle[edge_index]
    energy = 0.5 * k * diff * diff

    wp.atomic_add(total_energy, 0, energy)

# zcy_update
@wp.kernel
def zcy_line_search_truncation(
    dx: wp.array(dtype=wp.vec3), 
    pos: wp.array(dtype=wp.vec3), 
    all_particle_flag: wp.array(dtype=wp.int32),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    truncation_threshold: float,
):
    particle_index = wp.tid()

    if all_particle_flag[particle_index] == -1:
        return

    offset = all_particle_flag[particle_index]
    pos_before_truncation = pos[particle_index] + dx[particle_index-offset]
    
    pos_truncated = zcy_apply_conservative_bound_truncation(
        particle_index, 
        pos_before_truncation, 
        pos_prev_collision_detection, 
        particle_conservative_bounds, 
        truncation_threshold
    )

    dx[particle_index-offset] = pos_truncated - pos[particle_index]


@wp.kernel
def zcy_line_search_test_position(
    pos_test: wp.array(dtype=wp.vec3), 
    dx: wp.array(dtype=wp.vec3), 
    alpha: float,
    all_particle_flag: wp.array(dtype=wp.int32),
):
    particle_index = wp.tid()

    if all_particle_flag[particle_index] == -1:
        return

    offset = all_particle_flag[particle_index]
    pos_test[particle_index] += alpha * dx[particle_index-offset]


@wp.kernel
def zcy_update_velocity(
    dt: float, 
    damping: float, 
    pos_prev: wp.array(dtype=wp.vec3), 
    pos: wp.array(dtype=wp.vec3), 
    vel: wp.array(dtype=wp.vec3), 
    all_particle_flag: wp.array(dtype=wp.int32)
):
    particle = wp.tid()

    if all_particle_flag[particle] == -1:
        return

    vel[particle] = damping * (pos[particle] - pos_prev[particle]) / dt
  
@wp.kernel
def zcy_update_position(
    pos: wp.array(dtype=wp.vec3), 
    dx: wp.array(dtype=wp.vec3), 
    all_particle_flag: wp.array(dtype=wp.int32),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    truncation_threshold: float,
):
    particle_index = wp.tid()

    if all_particle_flag[particle_index] == -1:
        return

    offset = all_particle_flag[particle_index]
    pos_before_truncation = pos[particle_index] + dx[particle_index-offset]

    pos[particle_index] = zcy_apply_conservative_bound_truncation(
        particle_index, 
        pos_before_truncation, 
        pos_prev_collision_detection, 
        particle_conservative_bounds, 
        truncation_threshold,
    )


# zcy_sparse
def warp_coo_deduplicate(rows, cols, vals):
    """
    去重 COO 格式，vals 为 3x3 矩阵块，只做 sum 聚合
    - 过滤无效 (0,0,0) 碰撞
    - 保证至少一个 (0,0) + 0 block
    - 使用 np.bincount 替代 np.add.at
    """
    rows_np = rows.numpy()
    cols_np = cols.numpy()
    vals_np = vals.numpy()  # (nnz, 3, 3)

    nnz = rows_np.shape[0]

    # -----------------------------
    # 1. 过滤无效碰撞
    #    定义：row=0, col=0, 且 3x3 block 全为 0
    # -----------------------------
    if nnz > 0:
        invalid_mask = (
            (rows_np == 0) &
            (cols_np == 0) &
            np.all(vals_np == 0, axis=(1, 2))
        )
        valid_mask = ~invalid_mask

        rows_np = rows_np[valid_mask]
        cols_np = cols_np[valid_mask]
        vals_np = vals_np[valid_mask]

    # -----------------------------
    # 2. 保证至少一个 (0,0) + 0 block
    # -----------------------------
    if rows_np.size == 0:
        rows_np = np.array([0], dtype=rows_np.dtype)
        cols_np = np.array([0], dtype=cols_np.dtype)
        vals_np = np.zeros((1, 3, 3), dtype=vals_np.dtype)

    # -----------------------------
    # 3. COO 去重
    # -----------------------------
    max_col = np.max(cols_np)
    idx = rows_np * (max_col + 1) + cols_np

    unique_idx, inv = np.unique(idx, return_inverse=True)
    n_unique = unique_idx.shape[0]

    out_rows_np = unique_idx // (max_col + 1)
    out_cols_np = unique_idx % (max_col + 1)

    # -----------------------------
    # 4. 用 bincount 累加 3x3 block
    # -----------------------------
    vals_flat = vals_np.reshape(-1, 9)  # (nnz_valid, 9)
    out_vals_flat = np.zeros((n_unique, 9), dtype=vals_np.dtype)

    for k in range(9):
        out_vals_flat[:, k] = np.bincount(
            inv,
            weights=vals_flat[:, k],
            minlength=n_unique
        )

    out_vals_np = out_vals_flat.reshape(n_unique, 3, 3)

    # 如果后面不用 Warp
    return out_rows_np, out_cols_np, out_vals_np

def remove_fixed_blocks(rows, cols, vals, flag_all_particle):
    """
    从 COO (rows, cols, vals) 中删除涉及 fixed_points 的块，
    并根据 flag_all_particle 进行行列偏移。
    
    参数:
        rows_np, cols_np : np.ndarray, shape (nnz,)
        vals_np          : np.ndarray, shape (nnz, 3, 3)
        flag_all_particle : np.ndarray, shape (n_points,)
            -1 表示固定点
             其他值表示偏移量（即删除的固定点数）
    返回:
        过滤并重编号后的 (rows_np, cols_np, vals_np)
    """
    rows_np = rows.numpy()
    cols_np = cols.numpy()
    vals_np = vals.numpy() 
    flag_all_particle = flag_all_particle.numpy()

    nnz = len(rows_np)
    keep_mask = np.ones(nnz, dtype=bool)

    # 1️⃣ 找出涉及固定点的条目
    for i in range(nnz):
        if flag_all_particle[rows_np[i]] == -1 or flag_all_particle[cols_np[i]] == -1:
            keep_mask[i] = False

    # 2️⃣ 保留非固定条目
    rows_np = rows_np[keep_mask]
    cols_np = cols_np[keep_mask]
    vals_np = vals_np[keep_mask]

    # 3️⃣ 应用偏移：新索引 = 原索引 - flag_all_particle[原索引]
    rows_np = rows_np - flag_all_particle[rows_np]
    cols_np = cols_np - flag_all_particle[cols_np]

    return (
        wp.array(rows_np, dtype=int),
        wp.array(cols_np, dtype=int),
        wp.array(vals_np, dtype=wp.mat33)
    )

def build_bsr_from_block_coo(blocks_data: np.ndarray,
                              row: np.ndarray,
                              col: np.ndarray,
                              nb: int,
                              blocksize: tuple[int, int] = (3, 3),
                              sort_blocks: bool = True) -> bsr_matrix:
    """
    用 3×3×N 的数值块数组 + 块级坐标 (row, col) 构造 BSR 矩阵。

    参数:
      - blocks_data: 形状为 (nnz_blocks, br, bc) 的数值数组，例如 (N, 3, 3)。
      - row, col: 每个块的块行与块列索引，长度为 nnz_blocks，范围在 [0, nb)。
      - nb: 块行/块列的总数（最终矩阵尺寸为 (nb*br, nb*bc)）。
      - blocksize: 每个块的尺寸 (br, bc)，默认 (3, 3)。
      - sort_blocks: 是否按 (row, col) 排序以生成规范的 indptr/indices。

    返回:
      - scipy.sparse.bsr_matrix，形状为 (nb*br, nb*bc)。
    """
    blocks_data = np.asarray(blocks_data, dtype=np.float64)
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)

    if blocks_data.ndim != 3:
        raise ValueError("blocks_data 必须是三维数组，形状为 (nnz_blocks, br, bc)")
    nnz_blocks, br, bc = blocks_data.shape
    if (br, bc) != tuple(blocksize):
        raise ValueError(f"块尺寸不匹配: blocks_data 为 {(br, bc)}, 期望 {blocksize}")
    if row.shape != (nnz_blocks,) or col.shape != (nnz_blocks,):
        raise ValueError("row/col 长度必须与块数 nnz_blocks 相同")
    if np.any(row < 0) or np.any(row >= nb) or np.any(col < 0) or np.any(col >= nb):
        raise ValueError("row/col 索引越界: 必须在 [0, nb) 范围内")

    # 规范顺序：按 (row, col) 排序，便于构建 indptr/indices
    if sort_blocks:
        order = np.lexsort((col, row))
        row = row[order]
        col = col[order]
        blocks_data = blocks_data[order]

    # 构造 BSR 压缩格式需要的 indptr/indices
    counts = np.bincount(row, minlength=nb)
    indptr = np.empty(nb + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = col

    A_bsr = bsr_matrix((blocks_data, indices, indptr), shape=(nb * br, nb * bc))
    return A_bsr

# pure numpy version
def coo_deduplicate_np(rows_np, cols_np, vals_np):
    """
    纯 NumPy COO 去重
    输入:
        rows_np : (nnz,)
        cols_np : (nnz,)
        vals_np : (nnz, 3, 3)
    输出:
        out_rows_np : (n_unique,)
        out_cols_np : (n_unique,)
        out_vals_np : (n_unique, 3, 3)
    """

    nnz = rows_np.shape[0]

    # -----------------------------
    # 1. 过滤无效 (0,0) + zero block
    # -----------------------------
    if nnz > 0:
        invalid_mask = (
            (rows_np == 0) &
            (cols_np == 0) &
            np.all(vals_np == 0, axis=(1, 2))
        )
        valid_mask = ~invalid_mask

        rows_np = rows_np[valid_mask]
        cols_np = cols_np[valid_mask]
        vals_np = vals_np[valid_mask]

    # -----------------------------
    # 2. 保证至少一个占位 (0,0)
    # -----------------------------
    if rows_np.size == 0:
        rows_np = np.array([0], dtype=rows_np.dtype)
        cols_np = np.array([0], dtype=cols_np.dtype)
        vals_np = np.zeros((1, 3, 3), dtype=vals_np.dtype)

    # -----------------------------
    # 3. COO 去重
    # -----------------------------
    max_col = np.max(cols_np)
    idx = rows_np * (max_col + 1) + cols_np

    unique_idx, inv = np.unique(idx, return_inverse=True)
    n_unique = unique_idx.shape[0]

    out_rows_np = unique_idx // (max_col + 1)
    out_cols_np = unique_idx % (max_col + 1)

    # -----------------------------
    # 4. bincount 累加 3x3 block
    # -----------------------------
    vals_flat = vals_np.reshape(-1, 9)
    out_vals_flat = np.zeros((n_unique, 9), dtype=vals_np.dtype)

    for k in range(9):
        out_vals_flat[:, k] = np.bincount(
            inv,
            weights=vals_flat[:, k],
            minlength=n_unique
        )

    out_vals_np = out_vals_flat.reshape(n_unique, 3, 3)

    return out_rows_np, out_cols_np, out_vals_np

def remove_fixed_blocks_np(rows_np, cols_np, vals_np, flag_all_particle_np):
    """
    纯 NumPy 版本：
    从 COO (rows, cols, vals) 中删除涉及 fixed_points 的块，
    并根据 flag_all_particle 进行行列偏移。

    参数:
        rows_np : (nnz,) np.ndarray
        cols_np : (nnz,) np.ndarray
        vals_np : (nnz, 3, 3) np.ndarray
        flag_all_particle_np : (n_points,) np.ndarray
            -1 表示固定点
             其他值表示该点之前被删除的固定点数（偏移量）

    返回:
        rows_np, cols_np, vals_np （均为 NumPy 数组）
    """
    # -----------------------------
    # 1️⃣ 向量化判断是否涉及 fixed 点
    # -----------------------------
    row_fixed = flag_all_particle_np[rows_np] == -1
    col_fixed = flag_all_particle_np[cols_np] == -1

    keep_mask = ~(row_fixed | col_fixed)

    # -----------------------------
    # 2️⃣ 过滤
    # -----------------------------
    rows_np = rows_np[keep_mask]
    cols_np = cols_np[keep_mask]
    vals_np = vals_np[keep_mask]

    # -----------------------------
    # 3️⃣ 应用偏移（重编号）
    #     new_idx = old_idx - offset
    # -----------------------------
    rows_np = rows_np - flag_all_particle_np[rows_np]
    cols_np = cols_np - flag_all_particle_np[cols_np]

    return rows_np, cols_np, vals_np


# check by fd
@wp.kernel
def zcy_inertia_and_gravity_grad_computation(
    pos_warp: wp.array(dtype=wp.vec3),
    pos_prev_warp: wp.array(dtype=wp.vec3),
    vel_warp: wp.array(dtype=wp.vec3),
    dt: float,
    mass: float,
    gravity: wp.vec3,
    # fixed particle
    free_particle_offset: wp.array(dtype=wp.int32),
    # outputs: 
    inertia_grad: wp.array(dtype=wp.vec3)
):
    tid = wp.tid() 
    free_particle = tid + free_particle_offset[tid]

    # inertia
    inertia = pos_warp[free_particle] - pos_prev_warp[free_particle] - dt * vel_warp[free_particle]

    inertia_grad[tid] = mass / (dt * dt) * inertia - (mass * gravity)




# zcy
# endregion: zcy


class zcy_SolverNewton(SolverBase):

    def __init__(
        self,
        # debug
        DeBUG: dict,
        # self parameters
        dt: float,
        mass: float,
        # fixed particle mask
        fixed_particle_num: int,
        free_particle_offset: wp.array(dtype=wp.int32),
        all_particle_flag: wp.array(dtype=wp.int32),
        # spring information
        spring_indices: wp.array(dtype=wp.int32), 
        spring_rest_length: wp.array(dtype=wp.float32), 
        spring_stiffness: wp.array(dtype=wp.float32),
        # defult
        model: Model,
        iterations: int = 10,
        handle_self_contact: bool = False,
        self_contact_radius: float = 0.08,
        self_contact_margin: float = 0.08,
        integrate_with_external_rigid_solver: bool = False,
        penetration_free_conservative_bound_relaxation: float = 0.42,
        friction_epsilon: float = 1e-2,
        vertex_collision_buffer_pre_alloc: int = 32,
        edge_collision_buffer_pre_alloc: int = 64,
        collision_detection_interval: int = 0,
        edge_edge_parallel_epsilon: float = 1e-5,
        use_tile_solve: bool = True
    ):
        # region: before
        super().__init__(model)
        self.iterations = iterations
        self.integrate_with_external_rigid_solver = integrate_with_external_rigid_solver
        self.collision_detection_interval = collision_detection_interval

        # add new attributes for VBD solve
        self.particle_q_prev = wp.zeros_like(model.particle_q, device=self.device)
        self.inertia = wp.zeros_like(model.particle_q, device=self.device)

        self.adjacency = self.compute_force_element_adjacency(model).to(self.device)

        self.body_particle_contact_count = wp.zeros((model.particle_count,), dtype=wp.int32, device=self.device)

        self.handle_self_contact = handle_self_contact
        self.self_contact_radius = self_contact_radius
        self.self_contact_margin = self_contact_margin


        soft_contact_max = model.shape_count * model.particle_count
        if handle_self_contact:
            if self_contact_margin < self_contact_radius:
                raise ValueError(
                    "self_contact_margin is smaller than self_contact_radius, this will result in missing contacts and cause instability.\n"
                    "It is advisable to make self_contact_margin 1.5-2 times larger than self_contact_radius."
                )

            self.conservative_bound_relaxation = penetration_free_conservative_bound_relaxation
            self.pos_prev_collision_detection = wp.zeros_like(model.particle_q, device=self.device)
            self.particle_conservative_bounds = wp.full((model.particle_count,), dtype=float, device=self.device)

            self.trimesh_collision_detector = TriMeshCollisionDetector(
                self.model,
                vertex_collision_buffer_pre_alloc=vertex_collision_buffer_pre_alloc,
                edge_collision_buffer_pre_alloc=edge_collision_buffer_pre_alloc,
                edge_edge_parallel_epsilon=edge_edge_parallel_epsilon,
            )

            self.trimesh_collision_info = wp.array(
                [self.trimesh_collision_detector.collision_info], dtype=TriMeshCollisionInfo, device=self.device
            )

            self.collision_evaluation_kernel_launch_size = max(
                self.model.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                self.model.edge_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                soft_contact_max,
            )
        else:
            self.collision_evaluation_kernel_launch_size = soft_contact_max
    # endregion: before

        # region: my
        # debug
        self.DeBUG = DeBUG
        self.dt = dt
        self.mass = mass

        # particle information
        self.num_particle = self.model.particle_count

        # spring information
        self.spring_indices = spring_indices
        self.spring_rest_length = spring_rest_length
        self.spring_stiffness = spring_stiffness
        print('\n', self.spring_indices.shape, self.spring_rest_length.shape, self.spring_stiffness.shape)
        print(self.model.tri_indices.shape, self.model.tri_poses.shape, self.model.tri_materials.shape, self.model.tri_areas.shape)

        # fixed particle
        self.fixed_particle_num = fixed_particle_num
        self.free_particle_num = self.num_particle - fixed_particle_num
        self.free_particle_offset = free_particle_offset
        self.all_particle_flag = all_particle_flag

        # sparse hessian
        self.spring = self.DeBUG['spring_type']
        self.num_spring = self.spring_rest_length.shape[0]
        self.num_triangles = self.model.tri_indices.shape[0]

        # spaces for particle force and hessian
        self.particle_forces = wp.zeros(self.free_particle_num, dtype=wp.vec3, device=self.device)
        self.particle_hessians = wp.zeros(self.free_particle_num, dtype=wp.mat33, device=self.device)

        self.friction_epsilon = friction_epsilon
        
        # spring
        if self.spring:
            self.spring_forces = wp.zeros(self.num_particle, dtype=wp.vec3, device=self.device)
            self.spring_hessian_values = wp.zeros(self.num_spring*4, dtype=wp.mat33, device=self.device)
            self.spring_hessian_rows = wp.zeros(self.num_spring*4, dtype=int, device=self.device)
            self.spring_hessian_cols = wp.zeros(self.num_spring*4, dtype=int, device=self.device)
        else:
            self.spring_forces = wp.zeros(self.num_particle, dtype=wp.vec3, device=self.device)
            self.spring_hessian_values = wp.zeros(self.num_triangles*9, dtype=wp.mat33, device=self.device)
            self.spring_hessian_rows = wp.zeros(self.num_triangles*9, dtype=int, device=self.device)
            self.spring_hessian_cols = wp.zeros(self.num_triangles*9, dtype=int, device=self.device)

        # bending
        self.bending_forces = wp.zeros(self.num_particle, dtype=wp.vec3, device=self.device)
        self.bending_hessian_values = wp.zeros(self.num_spring*16, dtype=wp.mat33, device=self.device)
        self.bending_hessian_rows = wp.zeros(self.num_spring*16, dtype=int, device=self.device)
        self.bending_hessian_cols = wp.zeros(self.num_spring*16, dtype=int, device=self.device)

        # static matrix
        self.zcy_compute_static_matrix(dt, mass)

        # contact
        self.zcy_pre_and_refit_contact_size()

        # line search
        self.energy = 0.0
        self.residual = wp.zeros(shape=(self.free_particle_num,), dtype=wp.vec3)

        # endregion: my

# zcy
    def zcy_simulate_one_step(
        self,  pos_warp, pos_prev_warp, vel_warp, dt: float, mass: float, damping: float, num_iter: int, tolerance: float, time_step: int
    ):
        # collision detection before initialization to compute conservative bounds for initialization
        self.zcy_collision_detection_penetration_free(pos_prev_warp)
        
        # forward
        self.zcy_forward_step_penetration_free(pos_warp, pos_prev_warp, vel_warp, dt, forward_type=self.DeBUG['forward_type'])

        # after initialization, we need new collision detection to update the bounds
        # collision detection
        self.zcy_collision_detection_penetration_free(pos_warp)

        # residual_start
        residual_forward, residual_norm_forward = self.zcy_compute_residual(pos_warp, pos_prev_warp, vel_warp, dt, mass)
        energy_forward = self.zcy_compute_energy(pos_warp, pos_prev_warp, vel_warp, dt, mass)

        print('residual_norm_forward:', residual_norm_forward, 'energy_forward:', energy_forward)
        
        # debug_information_log
        if self.DeBUG['DeBUG0']:
            log_residual_path = f"running_log/run_{self.DeBUG['record_name']}_residual_log.txt"
            with open(log_residual_path, "a", encoding="utf-8") as f:
                    f.write(f'\n--- time_step: {time_step} ---\n')
                    f.write(f'forward: residual_norm_forward: {residual_norm_forward}, energy_forward: {energy_forward}\n')
                    
        if self.DeBUG['DeBUG']:
            log_residual_path = f"running_log/run_{self.DeBUG['record_name']}_residual_log.txt"
            with open(log_residual_path, "a", encoding="utf-8") as f:
                    f.write(f'\n--- time_step: {time_step} ---\n')
                    f.write(f'forward: residual_norm_forward: {residual_norm_forward}, energy_forward: {energy_forward}\n')

            if self.DeBUG['max_information']:
                log_warning_path = f"running_log/run_{self.DeBUG['record_name']}_warning_log.txt"

            if self.DeBUG['record_hessian']:
                path_hessian = f"running_log/run_{self.DeBUG['record_name']}_hessian_isbeing_log.txt"
                with open(path_hessian, "a") as f:
                        f.write(f'\n--- time_step: {time_step} ---\n')

        # line search start
        residual0 = residual_forward
        residual_norm0 = residual_norm_forward
        energy0 = energy_forward

        for _iter in range(num_iter):

            if self.DeBUG['DeBUG'] & self.DeBUG['record_hessian']:
                with open(path_hessian, "a") as f:
                    f.write(f'--- iter: {_iter} ---\n')

            # break
            # if residual_norm_forward < 1e-5:
            #    break

            # collision detection
            self.zcy_collision_detection_penetration_free(pos_warp)

            # assemble matrix and vector
            A =   self.zcy_assemble_matrix(pos_warp, pos_prev_warp, dt)
            b = - self.zcy_assemble_vector(pos_warp, pos_prev_warp, vel_warp, dt, mass)
  
            dx = spsolve(A.tocsr(), b.numpy().reshape(self.free_particle_num*3).astype(np.float64))
            dx = wp.array(dx.reshape(self.free_particle_num,3), dtype=wp.vec3)

            if self.DeBUG['DeBUG'] & self.DeBUG['record_hessian']:
                A_dense = A.tocsr().toarray()
                A_sym = (A_dense + A_dense.T) * 0.5

                # ---------- 先判断对称性 ----------
                is_symmetric = bool(np.allclose(A_dense, A_dense.T, atol=1e-8))

                # ---------- cond 数 ----------
                try:
                    cond = float(np.linalg.cond(A_dense))
                except Exception:
                    cond = float("inf")

                # ---------- SPD 判断 ----------
                eig_min = float("nan")
                eig_max = float("nan")
                is_spd = False

                try:
                    eigvals = np.linalg.eigvalsh(A_sym)
                    eig_min = float(eigvals.min())
                    eig_max = float(eigvals.max())
                    is_spd = bool(eig_min > 0.0 and is_symmetric)
                except Exception:
                    # eigvalsh失败 → 尝试 Cholesky
                    try:
                        np.linalg.cholesky(A_sym)
                        is_spd = bool(is_symmetric)
                    except Exception:
                        is_spd = False

                with open(path_hessian, "a") as f:
                    f.write(f'\nA.condition_number_and_spd:\n')
                    f.write(f'[cond, is_symmetric, is_spd, eig_min, eig_max]\n')
                    f.write(str([cond, is_symmetric, is_spd, eig_min, eig_max]) + "\n\n")


            # dx
            dx_truncated = wp.zeros_like(dx)
            dx_truncated.assign(dx)

            ### line search
            # region: line search
            # 0.1.initialization
            # truncation dx
            wp.launch(
                kernel=zcy_line_search_truncation,
                inputs=[dx_truncated, 
                        # input
                        pos_warp, 
                        self.all_particle_flag, 
                        self.pos_prev_collision_detection, 
                        self.particle_conservative_bounds, 
                        self.DeBUG['truncation_threshold'],
                        ], 
                dim=self.num_particle,
                device=self.device,
            )

            if self.DeBUG['DeBUG']:
                dx_debug = dx.numpy()
                pos_debug = pos_warp.numpy()

            # 0.parameters
            c1 = 1e-3
            alpha = 1.0
            gamma = 0.5
            # 0.line search 
            for _line_search_times in range(self.DeBUG['line_search_max_step']):  
                # test alpha position
                pos_warp_test_alpha = wp.zeros_like(pos_warp)
                pos_warp_test_alpha.assign(pos_warp)
                wp.launch(
                    kernel=zcy_line_search_test_position,
                    inputs=[pos_warp_test_alpha, 
                            # input
                            dx_truncated,
                            alpha,
                            self.all_particle_flag, 
                            ], 
                    dim=self.num_particle,
                    device=self.device,
                )

                if self.DeBUG['DeBUG']:
                    dx_debug = dx.numpy()
                    pos_debug0 = pos_warp_test_alpha.numpy()

                # collision detection
                self.trimesh_collision_detector.refit(pos_warp_test_alpha)
                self.trimesh_collision_detector.vertex_triangle_collision_detection(self.self_contact_margin)
                self.trimesh_collision_detector.edge_edge_collision_detection(self.self_contact_margin)
                
                # 1.2.compute energy1
                energy1 = self.zcy_compute_energy(pos_warp_test_alpha, pos_prev_warp, vel_warp, dt, mass)
                # 1.3.compute incremental energy
                incremental_energy = self.zcy_compute_incremental_energy(residual0, dx, alpha, c1)
                # 1.4.check armijo condition
                residual1, residual_norm1 = self.zcy_compute_residual(pos_warp_test_alpha, pos_prev_warp, vel_warp, dt, mass)

                # break condition
                energy_condition = (
                    energy1.numpy().item() < energy0.numpy().item() + incremental_energy.numpy().item() 
                    and not self.DeBUG['line_search_control_residual']
                )
                numerical_precision_condition0 = (
                    (np.abs(incremental_energy.numpy().item()/(abs(energy0.numpy().item())+1e-12)) 
                    < self.DeBUG['numerical_precision_rel_tolerance']
                    or np.abs(incremental_energy.numpy().item()) < 
                    self.DeBUG['numerical_precision_abs_tolerance'])
                    and self.DeBUG['numerical_precision_condition']
                )
                energy_residual_condition = (
                    energy1.numpy().item() < energy0.numpy().item() + incremental_energy.numpy().item()
                    and residual_norm1 < residual_norm0 + 1e-6
                    and self.DeBUG['line_search_control_residual']
                )

                if energy_condition or energy_residual_condition or numerical_precision_condition0:
                    break
                else:
                    alpha *= gamma

                # line search warning
                if self.DeBUG['DeBUG'] & self.DeBUG['max_information']:
                    if _line_search_times == self.DeBUG['line_search_max_step'] or incremental_energy.numpy().item() > 0.0 : 
                        # 初级信息
                        with open(log_warning_path, "a", encoding="utf-8") as f:
                            f.write(f'time_step: {time_step} \n')
                            f.write(f'iter: {_iter} \n')
                            f.write(f'line_search_times: {_line_search_times} \n')
                            f.write(f'incremental_energy: {incremental_energy} \n')

                        if self.DeBUG['max_warning']:
                            # 优化信息
                            A_dense = A.tocsr().toarray()
                            A_sym = (A_dense + A_dense.T) * 0.5
                            try:
                                cond = float(np.linalg.cond(A_dense))
                            except Exception:
                                cond = float("inf")
                            is_symmetric = bool(np.allclose(A_dense, A_dense.T, atol=1e-8))
                            is_spd = False
                            eig_min = float("nan")
                            eig_max = float("nan")
                            try:
                                eigvals = np.linalg.eigvalsh(A_sym)
                                eig_min = float(eigvals.min())
                                eig_max = float(eigvals.max())
                                is_spd = bool(eig_min > 0.0 and is_symmetric)
                            except Exception:
                                try:
                                    np.linalg.cholesky(A_sym)
                                    is_spd = bool(is_symmetric)
                                except Exception:
                                    is_spd = False

                            # 写入报错信息：
                            with open(log_warning_path, "a", encoding="utf-8") as f:
                                f.write(f'cond: {cond} \n')
                                f.write(f'is_symmetric: {is_symmetric} \n')
                                f.write(f'is_spd: {is_spd} \n')
                                f.write(f'eig_min: {eig_min} \n')
                                f.write(f'eig_max: {eig_max} \n\n\n')

                            raise RuntimeError(f"\n--- warning: {time_step} line search reach max iter {_line_search_times} or incremental_energy > 0.0 {incremental_energy.numpy()[0]} ---\n")

            # endregion

            energy0, energy1 = energy1, energy0
            residual_norm0, residual_norm1 = residual_norm1, residual_norm0
            residual0, residual1 = residual1, residual0

            # 1.5.update pos_warp
            pos_warp.assign(pos_warp_test_alpha)

            if self.DeBUG['DeBUG']:
                    pos_debug1 = pos_warp.numpy()
            
            print('residual_norm:', residual_norm0, '|energy:', energy0, '|incremental_energy:', incremental_energy, '|alpha:', alpha)
            
            if self.DeBUG['DeBUG'] or self.DeBUG['DeBUG0']:
                with open(log_residual_path, "a", encoding="utf-8") as f:
                        # 写入当前迭代信息
                        f.write(f'residual_norm: {residual_norm0} |energy: {energy0} |incremental_energy: {incremental_energy} |alpha: {alpha}\n')
                
            absolute_residual_condition = (
                residual_norm0 < self.DeBUG['convergence_abs_tolerance']
            )
            relative_residual_condition = (
                residual_norm0/(residual_norm_forward + 1e-12) < self.DeBUG['convergence_rel_tolerance']
            )
            numerical_precision_condition1 = (
                (abs(energy0.numpy().item()- energy1.numpy().item())/(abs(energy0.numpy().item()) + 1e-12) < self.DeBUG['numerical_precision_rel_tolerance']
                or abs(energy0.numpy().item()- energy1.numpy().item()) < self.DeBUG['numerical_precision_abs_tolerance'])
                and self.DeBUG['numerical_precision_condition']
            )
            
            if absolute_residual_condition or relative_residual_condition:
                break

            if self.DeBUG['DeBUG']:
                if numerical_precision_condition0 or numerical_precision_condition1:
                    with open(log_warning_path, "a", encoding="utf-8") as f:
                        f.write(f'time_step: {time_step}; iter: {_iter}; line_search_times: {_line_search_times} \n')
                        f.write(f'condition0:{numerical_precision_condition0}; condition1:{numerical_precision_condition1}\n')
                        f.write(f'residual_norm0: {residual_norm0} |energy0: {energy0} |incremental_energy: {incremental_energy} |alpha: {alpha}\n')
                        f.write(f'"Warning: Newton iteration stalled. Energy implies convergence but Residual is high."\n\n')
                    break

            # region: iteration information 
            if self.DeBUG['DeBUG'] & self.DeBUG['max_information']:
                if _iter == num_iter - 1:
                    # 初级信息
                    with open(log_warning_path, "a", encoding="utf-8") as f:
                        f.write(f'time_step: {time_step} \n')
                        f.write(f'iter: {_iter} \n')
                        f.write(f'line_search_times: {_line_search_times} \n')
                        f.write(f'incremental_energy: {incremental_energy} \n')

                    if self.DeBUG['max_warning']:
                        # 优化信息
                        A_dense = A.tocsr().toarray()
                        A_sym = (A_dense + A_dense.T) * 0.5
                        try:
                            cond = float(np.linalg.cond(A_dense))
                        except Exception:
                            cond = float("inf")
                        is_symmetric = bool(np.allclose(A_dense, A_dense.T, atol=1e-8))
                        is_spd = False
                        eig_min = float("nan")
                        eig_max = float("nan")
                        try:
                            eigvals = np.linalg.eigvalsh(A_sym)
                            eig_min = float(eigvals.min())
                            eig_max = float(eigvals.max())
                            is_spd = bool(eig_min > 0.0 and is_symmetric)
                        except Exception:
                            try:
                                np.linalg.cholesky(A_sym)
                                is_spd = bool(is_symmetric)
                            except Exception:
                                is_spd = False

                        with open(log_warning_path, "a", encoding="utf-8") as f:
                            f.write(f'cond: {cond} \n')
                            f.write(f'is_symmetric: {is_symmetric} \n')
                            f.write(f'is_spd: {is_spd} \n')
                            f.write(f'eig_min: {eig_min} \n')
                            f.write(f'eig_max: {eig_max} \n\n\n')
                    
                        raise RuntimeError(f"\n--- warning: {time_step} iteration reach max iter {_iter} ---\n")
            # endregion

        wp.launch(
            kernel=zcy_update_velocity,
            inputs=[dt, damping, pos_prev_warp, pos_warp, vel_warp, self.all_particle_flag],
            dim=self.num_particle,
            device=self.device,
        )

    def zcy_assemble_vector(self, pos_warp, pos_prev_warp, vel_warp, dt, mass):
        
        # inertia and gravity
        grad = wp.zeros(shape=(self.free_particle_num,), dtype=wp.vec3)

        wp.launch(
            kernel=zcy_assemble_inertia_and_gravity_add_force,
            inputs=[
                pos_warp,
                pos_prev_warp,
                vel_warp,
                dt,
                mass,
                self.model.gravity,
                # force
                self.spring_forces,
                self.edge_contact_forces,
                self.vt_contact_forces,
                self.bending_forces,
                # fixed particle
                self.free_particle_offset,
                # outputs: 
                grad
            ],
            dim=self.free_particle_num,
            device=self.device,
        )

        return grad

    def zcy_assemble_matrix(self, pos_warp, pos_prev_warp, dt):
        # contact hessian
        (edge_contact_hessian_rows, edge_contact_hessian_cols, edge_contact_hessian_values,
         vt_contact_hessian_rows, vt_contact_hessian_cols, vt_contact_hessian_values 
        )= self.zcy_compute_contact_hessian_force(pos_warp, pos_prev_warp, dt)
        # spring hessian
        spring_hessian_rows, spring_hessian_cols, spring_hessian_values = self.zcy_compute_spring_hessian_force(pos_warp, pos_prev_warp, dt)
        # bending hessian
        bending_hessian_rows, bending_hessian_cols, bending_hessian_values = self.zcy_compute_bending_hessian_force(pos_warp, pos_prev_warp, dt)
        
        A_rows = np.concatenate(
            (self.A_rows, 
            spring_hessian_rows, 
            edge_contact_hessian_rows, 
            vt_contact_hessian_rows, 
            bending_hessian_rows), axis=0
        )
        A_cols = np.concatenate(
            (self.A_cols, 
            spring_hessian_cols,
            edge_contact_hessian_cols,
            vt_contact_hessian_cols,
            bending_hessian_cols), axis=0
        )
        A_values = np.concatenate(
            (self.A_values, 
            spring_hessian_values, 
            edge_contact_hessian_values, 
            vt_contact_hessian_values, 
            bending_hessian_values), axis=0
        )

        A_rows, A_cols, A_values = coo_deduplicate_np(A_rows.astype(int), A_cols.astype(int), A_values)

        A_rows, A_cols, A_values = remove_fixed_blocks_np(A_rows, A_cols, A_values, self.all_particle_flag.numpy())

        A = build_bsr_from_block_coo(
            A_values, A_rows, A_cols, 
            nb=self.free_particle_num, blocksize=(3, 3)
        )

        return A

    def zcy_compute_residual(self, pos_warp, pos_prev_warp, vel_warp, dt, mass):
        # compute force for residual
        self.zcy_assemble_force_for_residual(pos_warp, pos_prev_warp, dt)
        
        # inertia and gravity
        residual = wp.zeros(shape=(self.free_particle_num,), dtype=wp.vec3)

        wp.launch(
            kernel=zcy_residual_computation,
            inputs=[
                pos_warp,
                pos_prev_warp,
                vel_warp,
                dt,
                mass,
                self.model.gravity,
                # force
                self.spring_forces,
                self.edge_contact_forces,
                self.vt_contact_forces,
                self.bending_forces,
                # fixed particle
                self.free_particle_offset,
                # outputs: 
                residual
            ],
            dim=self.free_particle_num,
            device=self.device,
        )

        residual_norm = np.linalg.norm(np.linalg.norm(residual.numpy(), axis=1))
        return residual, residual_norm

    def zcy_assemble_force_for_residual(self, pos_warp, pos_prev_warp, dt):
 
        # region: 1.contact force
        # edge_contact
        self.edge_contact_forces.zero_()
        # vertex-triangle_contact
        self.vt_contact_forces.zero_()
        # DeBUG_array
        if self.DeBUG['Contact']:
            # dim
            wp.launch(
                kernel=zcy_accumulate_contact_force,
                dim=self.num_contact,
                inputs=[
                    pos_warp,
                    # DeBUG
                    self.DeBUG['Contact_EE'],
                    self.DeBUG['Contact_VT'],
                    self.DeBUG['barrier_threshold'],
                    pos_prev_warp,
                    dt,
                    self.model.tri_indices,
                    self.model.edge_indices,
                    # self-contact
                    self.trimesh_collision_info,
                    self.self_contact_radius,
                    self.model.soft_contact_ke,
                    self.model.soft_contact_kd,
                    self.model.soft_contact_mu,
                    self.friction_epsilon,
                    self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                ],
                outputs=[
                    # edge_contact
                    self.edge_contact_forces,
                    # vertex-triangle_contact
                    self.vt_contact_forces,
                ],
                device=self.device,
            )
    # endregion

        # region: 2.spring force
        self.spring_forces.zero_()

        if self.DeBUG['Spring']:
            if self.spring :
                wp.launch(
                    kernel=zcy_accumulate_spring_force,
                    inputs=[
                        pos_warp,
                        pos_prev_warp,
                        dt,
                        # spring constraints
                        self.spring_indices,
                        self.spring_rest_length,
                        self.spring_stiffness,
                        self.model.spring_damping,
                        # outputs: particle force and hessian
                        self.spring_forces,
                    ],
                    dim=self.num_spring,
                    device=self.device,
                )
            else :
                wp.launch(
                    kernel=zcy_accumulate_stvk_force,
                    inputs=[
                        pos_warp,
                        pos_prev_warp,
                        dt,
                        # stvk force and hessian
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        # outputs: particle force and hessian
                        self.spring_forces,
                    ],
                    dim=self.num_triangles,
                    device=self.device,
                )
    # endregion

        # region: 3.bending force
        self.bending_forces.zero_()
        if self.DeBUG['Bending']:     
            wp.launch(
                kernel=zcy_accumulate_bending_force,
                inputs=[
                    pos_warp,
                    pos_prev_warp,
                    dt,
                    # bending force and hessian
                    self.model.edge_indices,
                    self.model.edge_rest_angle,
                    self.model.edge_rest_length,
                    self.model.edge_bending_properties,
                    # outputs: particle force and hessian
                    self.bending_forces,
                ],
                dim=self.num_spring,
                device=self.device,
            )
    # endregion


    def zcy_compute_energy(self, pos_warp, pos_prev_warp, vel_warp, dt, mass):

        energy = wp.zeros(shape=(1,), dtype=float)

        # inertia energy
        wp.launch(
            kernel=zcy_accumulate_inertia_energy,
            inputs=[
                pos_warp,
                pos_prev_warp,
                vel_warp,
                dt,
                mass,
                self.model.gravity,
                # fixed particle
                self.free_particle_offset,
                # outputs: 
                energy
            ],
            dim=self.free_particle_num,
            device=self.device,
        )

        # contact energy
        if self.DeBUG['Contact']:
            wp.launch(
                kernel=zcy_accumulate_contact_energy,
                inputs=[
                    pos_warp,
                    # DeBUG
                    self.DeBUG['Contact_EE'],
                    self.DeBUG['Contact_VT'],
                    self.DeBUG['barrier_threshold'],
                    self.model.tri_indices,
                    self.model.edge_indices,
                    # self-contact
                    self.trimesh_collision_info,
                    self.self_contact_radius,
                    self.model.soft_contact_ke,
                    self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                    # outputs: 
                    energy
                ],
                dim=self.num_contact,
                device=self.device,
            )

        # elastic energy
        if self.DeBUG['Spring']:
            if self.spring :
                wp.launch(
                    kernel=zcy_accumulate_spring_energy,
                    inputs=[
                        pos_warp,
                        # spring constraints
                        self.spring_indices,
                        self.spring_rest_length,
                        self.spring_stiffness,
                        # outputs: 
                        energy
                    ],
                    dim=self.num_spring,
                    device=self.device,
                )
            else :
                wp.launch(
                    kernel=zcy_accumulate_stvk_energy,
                    inputs=[
                        pos_warp,
                        # stvk force and hessian
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        # outputs: 
                        energy
                    ],
                    dim=self.num_triangles,
                    device=self.device,
                )

        # bending energy
        if self.DeBUG['Bending']:
            wp.launch(
                kernel=zcy_accumulate_bending_energy,
                inputs=[
                    pos_warp,
                    # bending force and hessian
                    self.model.edge_indices,
                    self.model.edge_rest_angle,
                    self.model.edge_rest_length,
                    self.model.edge_bending_properties,
                    # output
                    energy
                ],
            dim=self.num_spring,
            device=self.device,
        )

        return energy


    def zcy_compute_incremental_energy(self, residual, dx, alpha, c1):
        # init
        incremental_energy = wp.zeros(shape=(1,), dtype=float)

        wp.launch(
            kernel=zcy_compute_incremental_energy,
            inputs=[
                residual,
                dx,
                alpha,
                c1,
                # outputs: 
                incremental_energy
            ],
            dim=self.free_particle_num,
            device=self.device,
        )

        return incremental_energy


    def zcy_compute_static_matrix(self, dt, mass):
        # inertia and gravity
        self.A_rows = np.array([i for i in range(self.num_particle)])
        self.A_cols = np.array([i for i in range(self.num_particle)])
        self.A_values = np.array([np.eye(3) * mass / (dt * dt) for _ in range(self.num_particle)])

        if not self.DeBUG['Inertia_Hessian']:
            self.A_values.fill(0.0)
        

    def zcy_compute_contact_hessian_force(
        self, pos_warp, pos_prev_warp, dt,
    ):
        # edge_contact
        self.edge_contact_forces.zero_()
        self.edge_contact_hessian_values.zero_()
        self.edge_contact_hessian_rows.zero_()
        self.edge_contact_hessian_cols.zero_()
        # vertex-triangle_contact
        self.vt_contact_forces.zero_()
        self.vt_contact_hessian_values.zero_()
        self.vt_contact_hessian_rows.zero_()
        self.vt_contact_hessian_cols.zero_()

        # eigen filtering
        # 288 = 144 (矩阵) + 144 (特征向量)
        temp_mem1_size = self.num_ee_contact * 288
        temp_mem2_size = self.num_vt_contact * 288
        temp_buffer1 = wp.zeros(temp_mem1_size, dtype=float, device=self.device)
        temp_buffer2 = wp.zeros(temp_mem2_size, dtype=float, device=self.device)

        # DeBUG_array
        if self.DeBUG['Contact']:
            # dim
            wp.launch(
                kernel=zcy_VBD_accumulate_contact_force_and_hessian,
                dim=self.num_contact,
                inputs=[
                    pos_warp,
                    # DeBUG
                    self.DeBUG['Eigen'],
                    self.DeBUG['Contact_EE'],
                    self.DeBUG['Contact_VT'],
                    self.DeBUG['barrier_threshold'],
                    pos_prev_warp,
                    dt,
                    temp_buffer1,
                    temp_buffer2,
                    self.model.tri_indices,
                    self.model.edge_indices,
                    # self-contact
                    self.trimesh_collision_info,
                    self.self_contact_radius,
                    self.model.soft_contact_ke,
                    self.model.soft_contact_kd,
                    self.model.soft_contact_mu,
                    self.friction_epsilon,
                    self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                ],
                outputs=[
                    # edge_contact
                    self.edge_contact_forces,
                    self.edge_contact_hessian_values,
                    self.edge_contact_hessian_rows,
                    self.edge_contact_hessian_cols,
                    # vertex-triangle_contact
                    self.vt_contact_forces,
                    self.vt_contact_hessian_values,
                    self.vt_contact_hessian_rows,
                    self.vt_contact_hessian_cols,
                ],
                device=self.device,
            )

        # edge
        edge_contact_hessian_rows, edge_contact_hessian_cols, edge_contact_hessian_values = warp_coo_deduplicate(
            self.edge_contact_hessian_rows, self.edge_contact_hessian_cols, self.edge_contact_hessian_values)
        #print('\n---edge---')
        #print(f"\nedge_contact_hessian_rows={edge_contact_hessian_rows}, edge_contact_hessian_rows.shape={edge_contact_hessian_rows.shape}")
        #print(f"\nedge_contact_hessian_cols={edge_contact_hessian_cols}, edge_contact_hessian_cols.shape={edge_contact_hessian_cols.shape}")
        #print(f"\nedge_contact_hessian_values={edge_contact_hessian_values}, edge_contact_hessian_values.shape={edge_contact_hessian_values.shape}")
        
        # vt
        #np.savetxt("debug_rows.txt", self.vt_contact_hessian_rows, fmt="%d")
        #np.savetxt("debug_cols.txt", self.vt_contact_hessian_cols, fmt="%d")
        vt_contact_hessian_rows, vt_contact_hessian_cols, vt_contact_hessian_values = warp_coo_deduplicate(
            self.vt_contact_hessian_rows, self.vt_contact_hessian_cols, self.vt_contact_hessian_values)
        #print('\n---vt---')
        #print(f"\nvt_contact_hessian_rows={vt_contact_hessian_rows}, vt_contact_hessian_rows.shape={vt_contact_hessian_rows.shape}")
        #print(f"\nvt_contact_hessian_cols={vt_contact_hessian_cols}, vt_contact_hessian_cols.shape={vt_contact_hessian_cols.shape}")
        #print(f"\nvt_contact_hessian_values={vt_contact_hessian_values}, vt_contact_hessian_values.shape={vt_contact_hessian_values.shape}")

        if self.DeBUG['DeBUG'] & self.DeBUG['record_hessian']:
            path_hessian = f"running_log/run_{self.DeBUG['record_name']}_hessian_isbeing_log.txt"
            contact_ee_hessian = np.abs(edge_contact_hessian_values).max()
            contact_ee_residual = np.abs(edge_contact_forces.numpy()).max()
            contact_vt_hessian = np.abs(vt_contact_hessian_values).max()
            contact_vt_residual = np.abs(vt_contact_forces.numpy()).max()

            with open(path_hessian, "a") as f:
                f.write(f"contact_ee_hessian={contact_ee_hessian}, contact_ee_residual={contact_ee_residual}\n")
                f.write(f"contact_vt_hessian={contact_vt_hessian}, contact_vt_residual={contact_vt_residual}\n")
                f.write(f"edge_contact_hessian_rows={edge_contact_hessian_rows}\n")
                f.write(f"edge_contact_hessian_cols={edge_contact_hessian_cols}\n")
                f.write(f"edge_contact_hessian_values={edge_contact_hessian_values}\n")
                f.write(f"vt_contact_hessian_rows={vt_contact_hessian_rows}\n")
                f.write(f"vt_contact_hessian_cols={vt_contact_hessian_cols}\n")
                f.write(f"vt_contact_hessian_values={vt_contact_hessian_values}\n")


        return edge_contact_hessian_rows, edge_contact_hessian_cols, edge_contact_hessian_values, vt_contact_hessian_rows, vt_contact_hessian_cols, vt_contact_hessian_values


    def zcy_compute_spring_hessian_force(
        self, pos_warp, pos_prev_warp, dt
    ):
        # choose energy
        # spring
        self.spring_forces.zero_()
        self.spring_hessian_values.zero_()
        self.spring_hessian_rows.zero_()
        self.spring_hessian_cols.zero_()

        # dim
        if self.DeBUG['Spring']:
            if self.spring :
                # eigen filtering
                # 72 = 36 (矩阵) + 36 (特征向量)
                temp_mem_size = self.num_spring * 72
                temp_buffer = wp.zeros(temp_mem_size, dtype=float, device=self.device)

                wp.launch(
                    kernel=zcy_accumulate_spring_force_and_hessian,
                    inputs=[
                        pos_warp,
                        # DeBUG
                        self.DeBUG['Eigen'],
                        pos_prev_warp,
                        dt,
                        temp_buffer,
                        # spring constraints
                        self.spring_indices,
                        self.spring_rest_length,
                        self.spring_stiffness,
                        self.model.spring_damping,
                        # outputs: particle force and hessian
                        self.spring_forces,
                        self.spring_hessian_values,
                        self.spring_hessian_rows,
                        self.spring_hessian_cols
                    ],
                    dim=self.num_spring,
                    device=self.device,
                )
            else :
                # eigen filtering
                # 162 = 81 (矩阵) + 81 (特征向量)
                temp_mem_size = self.num_triangles * 162
                temp_buffer = wp.zeros(temp_mem_size, dtype=float, device=self.device)

                wp.launch(
                    kernel=zcy_accumulate_stvk_force_and_hessian,
                    inputs=[
                        pos_warp,
                        # DeBUG
                        self.DeBUG['Eigen'],
                        pos_prev_warp,
                        dt,
                        temp_buffer,
                        # stvk force and hessian
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        # outputs: particle force and hessian
                        self.spring_forces,
                        self.spring_hessian_values,
                        self.spring_hessian_rows,
                        self.spring_hessian_cols
                    ],
                    dim=self.num_triangles,
                    device=self.device,
                )
        
        # spring
        #print('\n---spring---')
        spring_hessian_rows, spring_hessian_cols, spring_hessian_values = warp_coo_deduplicate(
            self.spring_hessian_rows, self.spring_hessian_cols, self.spring_hessian_values)

        if self.DeBUG['DeBUG'] & self.DeBUG['record_hessian']:
            path_hessian = f"running_log/run_{self.DeBUG['record_name']}_hessian_isbeing_log.txt"
            spring_hessian = np.abs(spring_hessian_values).max()
            spring_residual = np.abs(spring_forces.numpy()).max()

            with open(path_hessian, "a") as f:
                f.write(f"spring_hessian={spring_hessian}, spring_residual={spring_residual}\n")
                f.write(f"spring_hessian_rows={spring_hessian_rows}\n")
                f.write(f"spring_hessian_cols={spring_hessian_cols}\n")
                f.write(f"spring_hessian_values={spring_hessian_values}\n")

        return spring_hessian_rows, spring_hessian_cols, spring_hessian_values


    def zcy_compute_bending_hessian_force(
        self, pos_warp, pos_prev_warp, dt
    ):
        # bending
        self.bending_forces.zero_()
        self.bending_hessian_values.zero_()
        self.bending_hessian_rows.zero_()
        self.bending_hessian_cols.zero_()

        # eigen filtering
        # 288 = 144 (矩阵) + 144 (特征向量)
        temp_mem_size = self.num_spring * 288
        temp_buffer = wp.zeros(temp_mem_size, dtype=float, device=self.device)

        if self.DeBUG['Bending']:
            # dim             
            wp.launch(
                kernel=zcy_accumulate_bending_force_and_hessian,
                inputs=[
                    pos_warp,
                    # DeBUG
                    self.DeBUG['Eigen'],
                    pos_prev_warp,
                    dt,
                    temp_buffer,
                    # bending force and hessian
                    self.model.edge_indices,
                    self.model.edge_rest_angle,
                    self.model.edge_rest_length,
                    self.model.edge_bending_properties,
                    # outputs: particle force and hessian
                    self.bending_forces,
                    self.bending_hessian_values,
                    self.bending_hessian_rows,
                    self.bending_hessian_cols
                ],
                dim=self.num_spring,
                device=self.device,
            )
        
        # bending
        #print('\n---bending---')
        bending_hessian_rows, bending_hessian_cols, bending_hessian_values = warp_coo_deduplicate(
            self.bending_hessian_rows, self.bending_hessian_cols, self.bending_hessian_values)
        #print(f"\nbending_hessian_rows={bending_hessian_rows}, bending_hessian_rows.shape={bending_hessian_rows.shape}")
        #print(f"\nbending_hessian_cols={bending_hessian_cols}, bending_hessian_cols.shape={bending_hessian_cols.shape}")
        #print(f"\nbending_hessian_values={bending_hessian_values}, bending_hessian_values.shape={bending_hessian_values.shape}")
        if self.DeBUG['DeBUG'] & self.DeBUG['record_hessian']:
            path_hessian = f"running_log/run_{self.DeBUG['record_name']}_hessian_isbeing_log.txt"
            bending_hessian = np.abs(bending_hessian_values).max()
            bending_residual = np.abs(bending_forces.numpy()).max()

            with open(path_hessian, "a") as f:
                f.write(f"bending_hessian={bending_hessian}, bending_residual={bending_residual}\n")
                f.write(f"bending_hessian_rows={bending_hessian_rows}\n")
                f.write(f"bending_hessian_cols={bending_hessian_cols}\n")
                f.write(f"bending_hessian_values={bending_hessian_values}\n")
        
        return bending_hessian_rows, bending_hessian_cols, bending_hessian_values


    def zcy_forward_step_penetration_free(
        self, pos_warp, pos_prev_warp, vel_warp, dt: float, forward_type: int = 0
    ):
        model=self.model

        # give the gravity to the model
        print(model.gravity)    

        # pos_prev_warp give information to update pos_warp
        wp.launch(
            kernel=zcy_forward_step_penetration_free,
            inputs=[
                dt,
                forward_type,
                model.gravity,
                pos_prev_warp,
                pos_warp,
                vel_warp,
                self.pos_prev_collision_detection,
                self.particle_conservative_bounds,
                self.all_particle_flag,
                self.DeBUG['truncation_threshold'],
            ],
            dim=model.particle_count,
            device=self.device,
        )

    def zcy_collision_detection_penetration_free(self, pos_warp):
        self.trimesh_collision_detector.refit(pos_warp)
        self.trimesh_collision_detector.vertex_triangle_collision_detection(self.self_contact_margin)
        self.trimesh_collision_detector.edge_edge_collision_detection(self.self_contact_margin)

        self.pos_prev_collision_detection.assign(pos_warp)
        wp.launch(
            kernel=compute_particle_conservative_bound,
            inputs=[
                self.conservative_bound_relaxation,
                self.self_contact_margin,
                self.adjacency,
                self.trimesh_collision_detector.collision_info,
            ],
            outputs=[
                self.particle_conservative_bounds,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        # self.zcy_pre_and_refit_contact_size()

    def zcy_truncation_by_conservative_bound(self, pos_new):

        pos_old = wp.clone(pos_new)

        wp.launch(
            kernel=zcy_truncation_by_conservative_bounds,
            inputs=[
                pos_old,
                self.pos_prev_collision_detection,
                self.particle_conservative_bounds,
            ],
            outputs=[
                pos_new,
                self.DeBUG['truncation_threshold'],
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

    def zcy_pre_and_refit_contact_size(self):
        # contact num
        self.num_ee_contact = int(self.trimesh_collision_detector.collision_info.edge_colliding_edges.shape[0]/2)
        self.num_vt_contact = int(self.trimesh_collision_detector.collision_info.vertex_colliding_triangles.shape[0]/2)
        print(f"ee_max_num={self.trimesh_collision_detector.collision_info.edge_colliding_edges_buffer_sizes.numpy().max()}")
        print(f"vt_max_num={self.trimesh_collision_detector.collision_info.vertex_colliding_triangles_buffer_sizes.numpy().max()}")
        print(f"num_ee_contact={self.num_ee_contact}, num_vt_contact={self.num_vt_contact}")
        self.num_contact = int(max(self.num_ee_contact, self.num_vt_contact))
        # edge_contact
        self.edge_contact_forces = wp.zeros(self.num_particle, dtype=wp.vec3, device=self.device)
        self.edge_contact_hessian_values = wp.zeros(self.num_ee_contact*16, dtype=wp.mat33, device=self.device)
        self.edge_contact_hessian_rows = wp.zeros(self.num_ee_contact*16, dtype=int, device=self.device)
        self.edge_contact_hessian_cols = wp.zeros(self.num_ee_contact*16, dtype=int, device=self.device)
        # vertex-triangle_contact
        self.vt_contact_forces = wp.zeros(self.num_particle, dtype=wp.vec3, device=self.device)
        self.vt_contact_hessian_values = wp.zeros(self.num_vt_contact*16, dtype=wp.mat33, device=self.device)
        self.vt_contact_hessian_rows = wp.zeros(self.num_vt_contact*16, dtype=int, device=self.device)
        self.vt_contact_hessian_cols = wp.zeros(self.num_vt_contact*16, dtype=int, device=self.device)

# zcy_check_grad_and_hessian_via_fd
    def zcy_check_grad_and_hessian_via_fd(self, pos_warp, pos_prev_warp, vel_warp, dt: float, Check_Switch:dict):
        '''
        input: 
            pos_warp: current position
            pos_prev_warp: previous position
            vel_warp: current velocity
            dt: time step
            Check_Switch: switch of every energy(inertia, elastic, bending and collision)
        
        output:
            # 1.energy, grad and hessian of my solver
                energy: energy of my solver
                grad: gradient of my solver
                hessian: hessian of my solver

            # 2.energy, grad and hessian of finite difference
                grad_fd_of_enegy: gradient of finite difference
                hessian_fd_of_energy: hessian of finite difference
                hessian_fd_of_grad: hessian of finite difference

            # 3.verification metric
                grad_error_norm_of_energy_fd: norm of gradient error of finite difference and mysolver
                hessian_error_norm_of_grad_fd: norm of hessian error of finite difference and mysolver
        '''
        # preprocess
        self.zcy_collision_detection_penetration_free(pos_warp)

        # 0. Switch
        self.DeBUG['Spring'] = Check_Switch['Elastic']
        self.DeBUG['Bending'] = Check_Switch['Bending']
        self.DeBUG['Contact'] = Check_Switch['Contact']
        self.DeBUG['Contact_EE'] = Check_Switch['Contact_EE']
        self.DeBUG['Contact_VT'] = Check_Switch['Contact_VT']
        self.DeBUG['Inertia_Hessian'] = Check_Switch['Inertia']
        self.DeBUG['Eigen'] = False
        self.perturbation_epsilon = Check_Switch['perturbation_epsilon']
        self.spring = 1 if Check_Switch['Spring_Elastic'] else 0
        mass = self.mass

        print('Finish preprocess, compute energy in progress...')
    
        # compute energy
        energy = self.zcy_compute_energy_for_fd_check(pos_warp, pos_prev_warp, vel_warp, dt, mass, Check_Switch)
        print(f'Finish compute energy, energy={energy}')

        # compute grad
        grad = self.zcy_compute_grad_for_fd_check(pos_warp, pos_prev_warp, vel_warp, dt, mass, Check_Switch)
        print(f'Finish compute grad, grad={grad.shape}')

        # compute hessian
        self.zcy_compute_static_matrix(dt, mass)
        hessian = self.zcy_compute_hessian_for_fd_check(pos_warp, pos_prev_warp, dt, Check_Switch)
        print(f'Finish compute hessian, hessian={hessian.shape}')
        
        # compute grad_fd_by_fd_enegy
        grad_fd_by_fd_energy = self.zcy_compute_grad_fd_by_fd_energy(pos_warp, pos_prev_warp, vel_warp, dt, mass, Check_Switch)
        print(f'Finish compute grad_fd_by_fd_energy, grad_fd_by_fd_energy={grad_fd_by_fd_energy.shape}')

        # compute hessian_fd_by_fd_grad
        hessian_fd_by_fd_grad = self.zcy_compute_hessian_fd_by_fd_grad(pos_warp, pos_prev_warp, vel_warp, dt, mass, Check_Switch)
        
        print(f'Finish compute hessian_fd_by_fd_grad, hessian_fd_by_fd_grad={hessian_fd_by_fd_grad.shape}')

        # compute error norm of grad
        grad_np = grad.numpy() if hasattr(grad, "numpy") else np.asarray(grad)
        grad_fd_np = grad_fd_by_fd_energy.numpy() if hasattr(grad_fd_by_fd_energy, "numpy") else np.asarray(grad_fd_by_fd_energy)
        _rel_norm = float(np.max(np.abs(grad_np))) if float(np.max(np.abs(grad_np))) > 1.0 else 1.0
        grad_error_norm_of_energy_fd = float(np.max(np.abs(grad_np - grad_fd_np))) / _rel_norm

        # compute error norm of hessian
        if hasattr(hessian, "toarray"):
            h_dense = hessian.toarray()
        else:
            h_dense = np.asarray(hessian)

        n = self.free_particle_num
        hessian_np = h_dense.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
        hessian_fd_np = (
            hessian_fd_by_fd_grad.numpy()
            if hasattr(hessian_fd_by_fd_grad, "numpy")
            else np.asarray(hessian_fd_by_fd_grad)
        )
        _rel_norm = float(np.max(np.abs(hessian_np))) if float(np.max(np.abs(hessian_np))) > 1.0 else 1.0
        hessian_error_norm_of_grad_fd = float(np.max(np.abs(hessian_np - hessian_fd_np))) / _rel_norm

        return (
            energy.numpy()[0],
            grad.numpy(),
            hessian,
            grad_fd_by_fd_energy,
            hessian_fd_by_fd_grad,
            grad_error_norm_of_energy_fd,
            hessian_error_norm_of_grad_fd,
        )

    def zcy_collision_detection_for_check(self, pos_warp):
        self.trimesh_collision_detector.refit(pos_warp)
        self.trimesh_collision_detector.vertex_triangle_collision_detection(self.self_contact_margin)
        self.trimesh_collision_detector.edge_edge_collision_detection(self.self_contact_margin)


    def zcy_compute_energy_for_fd_check(self, pos_warp, pos_prev_warp, vel_warp, dt, mass, Check_Switch):

        energy = wp.zeros(shape=(1,), dtype=float)

        # inertia energy
        if Check_Switch['Inertia']:
            wp.launch(
                kernel=zcy_accumulate_inertia_energy,
                inputs=[
                    pos_warp,
                    pos_prev_warp,
                    vel_warp,
                    dt,
                    mass,
                    self.model.gravity,
                    # fixed particle
                    self.free_particle_offset,
                    # outputs: 
                    energy
                ],
                dim=self.free_particle_num,
                device=self.device,
            )

        # contact energy
        if Check_Switch['Contact']:
            # collision detection
            self.zcy_collision_detection_for_check(pos_warp)

            wp.launch(
                kernel=zcy_accumulate_contact_energy,
                inputs=[
                    pos_warp,
                    # DeBUG
                    Check_Switch['Contact_EE'],
                    Check_Switch['Contact_VT'],
                    self.DeBUG['barrier_threshold'],
                    self.model.tri_indices,
                    self.model.edge_indices,
                    # self-contact
                    self.trimesh_collision_info,
                    self.self_contact_radius,
                    self.model.soft_contact_ke,
                    self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                    # outputs: 
                    energy
                ],
                dim=self.num_contact,
                device=self.device,
            )

        # elastic energy
        if Check_Switch['Elastic']:
            if Check_Switch['Spring_Elastic']:
                wp.launch(
                    kernel=zcy_accumulate_spring_energy,
                    inputs=[
                        pos_warp,
                        # spring constraints
                        self.spring_indices,
                        self.spring_rest_length,
                        self.spring_stiffness,
                        # outputs: 
                        energy
                    ],
                    dim=self.num_spring,
                    device=self.device,
                )
            elif Check_Switch['Stvk_Elastic']:
                wp.launch(
                    kernel=zcy_accumulate_stvk_energy,
                    inputs=[
                        pos_warp,
                        # stvk force and hessian
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        # outputs: 
                        energy
                    ],
                    dim=self.num_triangles,
                    device=self.device,
                )

        # bending energy
        if Check_Switch['Bending']:
            wp.launch(
                kernel=zcy_accumulate_bending_energy,
                inputs=[
                    pos_warp,
                    # bending force and hessian
                    self.model.edge_indices,
                    self.model.edge_rest_angle,
                    self.model.edge_rest_length,
                    self.model.edge_bending_properties,
                    # output
                    energy
                ],
            dim=self.num_spring,
            device=self.device,
        )

        return energy

    def zcy_compute_grad_for_fd_check(self, pos_warp, pos_prev_warp, vel_warp, dt, mass, Check_Switch):
        self.spring_forces.zero_()
        self.edge_contact_forces.zero_()
        self.vt_contact_forces.zero_()
        self.bending_forces.zero_()

        if Check_Switch["Contact"]:
            # collision detection
            self.zcy_collision_detection_for_check(pos_warp)
            
            wp.launch(
                kernel=zcy_accumulate_contact_force,
                dim=self.num_contact,
                inputs=[
                    pos_warp,
                    Check_Switch["Contact_EE"],
                    Check_Switch["Contact_VT"],
                    self.DeBUG["barrier_threshold"],
                    pos_prev_warp,
                    dt,
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.trimesh_collision_info,
                    self.self_contact_radius,
                    self.model.soft_contact_ke,
                    self.model.soft_contact_kd,
                    self.model.soft_contact_mu,
                    self.friction_epsilon,
                    self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                ],
                outputs=[
                    self.edge_contact_forces,
                    self.vt_contact_forces,
                ],
                device=self.device,
            )

        if Check_Switch["Elastic"]:
            if Check_Switch["Spring_Elastic"]:
                wp.launch(
                    kernel=zcy_accumulate_spring_force,
                    inputs=[
                        pos_warp,
                        pos_prev_warp,
                        dt,
                        self.spring_indices,
                        self.spring_rest_length,
                        self.spring_stiffness,
                        self.model.spring_damping,
                        self.spring_forces,
                    ],
                    dim=self.num_spring,
                    device=self.device,
                )
            elif Check_Switch["Stvk_Elastic"]:
                wp.launch(
                    kernel=zcy_accumulate_stvk_force,
                    inputs=[
                        pos_warp,
                        pos_prev_warp,
                        dt,
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        self.spring_forces,
                    ],
                    dim=self.num_triangles,
                    device=self.device,
                )

        if Check_Switch["Bending"]:
            wp.launch(
                kernel=zcy_accumulate_bending_force,
                inputs=[
                    pos_warp,
                    pos_prev_warp,
                    dt,
                    self.model.edge_indices,
                    self.model.edge_rest_angle,
                    self.model.edge_rest_length,
                    self.model.edge_bending_properties,
                    self.bending_forces,
                ],
                dim=self.num_spring,
                device=self.device,
            )

        grad = wp.zeros(shape=(self.free_particle_num,), dtype=wp.vec3)
        inertia_on = bool(Check_Switch.get("Inertia", True))
        mass_eff = float(mass) if inertia_on else 0.0
        gravity_eff = self.model.gravity if inertia_on else wp.vec3(0.0, 0.0, 0.0)

        wp.launch(
            kernel=zcy_residual_computation,
            inputs=[
                pos_warp,
                pos_prev_warp,
                vel_warp,
                dt,
                mass_eff,
                gravity_eff,
                self.spring_forces,
                self.edge_contact_forces,
                self.vt_contact_forces,
                self.bending_forces,
                self.free_particle_offset,
                grad,
            ],
            dim=self.free_particle_num,
            device=self.device,
        )

        return grad

    def zcy_compute_hessian_for_fd_check(self, pos_warp, pos_prev_warp, dt, Check_Switch):
        # collision detection
        self.zcy_collision_detection_for_check(pos_warp)

        Hessian =   self.zcy_assemble_matrix(pos_warp, pos_prev_warp, dt)

        return Hessian

    def zcy_compute_grad_fd_by_fd_energy(self, pos_warp, pos_prev_warp, vel_warp, dt, mass, Check_Switch):
        h = self.perturbation_epsilon
        pos0 = pos_warp.numpy() if hasattr(pos_warp, "numpy") else np.asarray(pos_warp)
        offsets = self.free_particle_offset.numpy() if hasattr(self.free_particle_offset, "numpy") else np.asarray(self.free_particle_offset)

        grad = np.zeros((self.free_particle_num, 3), dtype=pos0.dtype)
        for tid in range(self.free_particle_num):
            p_idx = tid + int(offsets[tid])
            for axis in range(3):
                pos_plus = pos0.copy()
                pos_minus = pos0.copy()
                pos_plus[p_idx, axis] += h
                pos_minus[p_idx, axis] -= h

                pos_plus_warp = wp.array(pos_plus, dtype=wp.vec3, device=self.device)
                pos_minus_warp = wp.array(pos_minus, dtype=wp.vec3, device=self.device)

                e_plus = self.zcy_compute_energy_for_fd_check(
                    pos_plus_warp, pos_prev_warp, vel_warp, dt, mass, Check_Switch
                )
                e_minus = self.zcy_compute_energy_for_fd_check(
                    pos_minus_warp, pos_prev_warp, vel_warp, dt, mass, Check_Switch
                )

                e_plus_s = float(e_plus.numpy()[0]) if hasattr(e_plus, "numpy") else float(np.asarray(e_plus)[0])
                e_minus_s = float(e_minus.numpy()[0]) if hasattr(e_minus, "numpy") else float(np.asarray(e_minus)[0])
                grad[tid, axis] = (e_plus_s - e_minus_s) / (2.0 * h)

        return grad

    def zcy_compute_hessian_fd_by_fd_grad(self, pos_warp, pos_prev_warp, vel_warp, dt, mass, Check_Switch):
        h = self.perturbation_epsilon
        pos0 = pos_warp.numpy() if hasattr(pos_warp, "numpy") else np.asarray(pos_warp)
        offsets = self.free_particle_offset.numpy() if hasattr(self.free_particle_offset, "numpy") else np.asarray(self.free_particle_offset)

        hessian = np.zeros((self.free_particle_num, self.free_particle_num, 3, 3), dtype=pos0.dtype)
        for jdx in range(self.free_particle_num):
            p_j = jdx + int(offsets[jdx])
            for axis_j in range(3):
                pos_plus = pos0.copy()
                pos_minus = pos0.copy()
                pos_plus[p_j, axis_j] += h
                pos_minus[p_j, axis_j] -= h

                pos_plus_warp = wp.array(pos_plus, dtype=wp.vec3, device=self.device)
                pos_minus_warp = wp.array(pos_minus, dtype=wp.vec3, device=self.device)

                g_plus = self.zcy_compute_grad_for_fd_check(
                    pos_plus_warp, pos_prev_warp, vel_warp, dt, mass, Check_Switch
                )
                g_minus = self.zcy_compute_grad_for_fd_check(
                    pos_minus_warp, pos_prev_warp, vel_warp, dt, mass, Check_Switch
                )

                g_plus_np = g_plus.numpy() if hasattr(g_plus, "numpy") else np.asarray(g_plus)
                g_minus_np = g_minus.numpy() if hasattr(g_minus, "numpy") else np.asarray(g_minus)
                hessian[:, jdx, :, axis_j] = (g_plus_np - g_minus_np) / (2.0 * h)

        return hessian
      
# zcy

    def compute_force_element_adjacency(self, model):
        adjacency = ForceElementAdjacencyInfo()
        edges_array = model.edge_indices.to("cpu")
        spring_array = model.spring_indices.to("cpu")
        face_indices = model.tri_indices.to("cpu")

        with wp.ScopedDevice("cpu"):
            if edges_array.size:
                # build vertex-edge adjacency data
                num_vertex_adjacent_edges = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=self.count_num_adjacent_edges,
                    inputs=[edges_array, num_vertex_adjacent_edges],
                    dim=1,
                )

                num_vertex_adjacent_edges = num_vertex_adjacent_edges.numpy()
                vertex_adjacent_edges_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_edges_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_edges)[:]
                vertex_adjacent_edges_offsets[0] = 0
                adjacency.v_adj_edges_offsets = wp.array(vertex_adjacent_edges_offsets, dtype=wp.int32)

                # temporal variables to record how much adjacent edges has been filled to each vertex
                vertex_adjacent_edges_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                edge_adjacency_array_size = 2 * num_vertex_adjacent_edges.sum()
                # vertex order: o0: 0, o1: 1, v0: 2, v1: 3,
                adjacency.v_adj_edges = wp.empty(shape=(edge_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_edges,
                    inputs=[
                        edges_array,
                        adjacency.v_adj_edges_offsets,
                        vertex_adjacent_edges_fill_count,
                        adjacency.v_adj_edges,
                    ],
                    dim=1,
                )
            else:
                adjacency.v_adj_edges_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_edges = wp.empty(shape=(0,), dtype=wp.int32)

            if face_indices.size:
                # compute adjacent triangles
                # count number of adjacent faces for each vertex
                num_vertex_adjacent_faces = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                wp.launch(kernel=self.count_num_adjacent_faces, inputs=[face_indices, num_vertex_adjacent_faces], dim=1)

                # preallocate memory based on counting results
                num_vertex_adjacent_faces = num_vertex_adjacent_faces.numpy()
                vertex_adjacent_faces_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_faces_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_faces)[:]
                vertex_adjacent_faces_offsets[0] = 0
                adjacency.v_adj_faces_offsets = wp.array(vertex_adjacent_faces_offsets, dtype=wp.int32)

                vertex_adjacent_faces_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                face_adjacency_array_size = 2 * num_vertex_adjacent_faces.sum()
                # (face, vertex_order) * num_adj_faces * num_particles
                # vertex order: v0: 0, v1: 1, o0: 2, v2: 3
                adjacency.v_adj_faces = wp.empty(shape=(face_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_faces,
                    inputs=[
                        face_indices,
                        adjacency.v_adj_faces_offsets,
                        vertex_adjacent_faces_fill_count,
                        adjacency.v_adj_faces,
                    ],
                    dim=1,
                )
            else:
                adjacency.v_adj_faces_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_faces = wp.empty(shape=(0,), dtype=wp.int32)

            if spring_array.size:
                # build vertex-springs adjacency data
                num_vertex_adjacent_spring = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=self.count_num_adjacent_springs,
                    inputs=[spring_array, num_vertex_adjacent_spring],
                    dim=1,
                )

                num_vertex_adjacent_spring = num_vertex_adjacent_spring.numpy()
                vertex_adjacent_springs_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_springs_offsets[1:] = np.cumsum(num_vertex_adjacent_spring)[:]
                vertex_adjacent_springs_offsets[0] = 0
                adjacency.v_adj_springs_offsets = wp.array(vertex_adjacent_springs_offsets, dtype=wp.int32)

                # temporal variables to record how much adjacent springs has been filled to each vertex
                vertex_adjacent_springs_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                adjacency.v_adj_springs = wp.empty(shape=(num_vertex_adjacent_spring.sum(),), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_springs,
                    inputs=[
                        spring_array,
                        adjacency.v_adj_springs_offsets,
                        vertex_adjacent_springs_fill_count,
                        adjacency.v_adj_springs,
                    ],
                    dim=1,
                )

            else:
                adjacency.v_adj_springs_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_springs = wp.empty(shape=(0,), dtype=wp.int32)

        return adjacency

    def rebuild_bvh(self, state: State):
        """This function will rebuild the BVHs used for detecting self-contacts using the input `state`.

        When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        quality. In these cases, rebuilding the entire tree is necessary to achieve better querying efficiency.

        Args:
            state (newton.State):  The state whose particle positions (:attr:`State.particle_q`) will be used for rebuilding the BVHs.
        """
        if self.handle_self_contact:
            self.trimesh_collision_detector.rebuild(state.particle_q)

    @wp.kernel
    def count_num_adjacent_edges(
        edges_array: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_edges: wp.array(dtype=wp.int32)
    ):
        for edge_id in range(edges_array.shape[0]):
            o0 = edges_array[edge_id, 0]
            o1 = edges_array[edge_id, 1]

            v0 = edges_array[edge_id, 2]
            v1 = edges_array[edge_id, 3]

            num_vertex_adjacent_edges[v0] = num_vertex_adjacent_edges[v0] + 1
            num_vertex_adjacent_edges[v1] = num_vertex_adjacent_edges[v1] + 1

            if o0 != -1:
                num_vertex_adjacent_edges[o0] = num_vertex_adjacent_edges[o0] + 1
            if o1 != -1:
                num_vertex_adjacent_edges[o1] = num_vertex_adjacent_edges[o1] + 1

    @wp.kernel
    def fill_adjacent_edges(
        edges_array: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_edges_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_edges_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_edges: wp.array(dtype=wp.int32),
    ):
        for edge_id in range(edges_array.shape[0]):
            v0 = edges_array[edge_id, 2]
            v1 = edges_array[edge_id, 3]

            fill_count_v0 = vertex_adjacent_edges_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_edges_offsets[v0]
            vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 2
            vertex_adjacent_edges_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_edges_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_edges_offsets[v1]
            vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 3
            vertex_adjacent_edges_fill_count[v1] = fill_count_v1 + 1

            o0 = edges_array[edge_id, 0]
            if o0 != -1:
                fill_count_o0 = vertex_adjacent_edges_fill_count[o0]
                buffer_offset_o0 = vertex_adjacent_edges_offsets[o0]
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2 + 1] = 0
                vertex_adjacent_edges_fill_count[o0] = fill_count_o0 + 1

            o1 = edges_array[edge_id, 1]
            if o1 != -1:
                fill_count_o1 = vertex_adjacent_edges_fill_count[o1]
                buffer_offset_o1 = vertex_adjacent_edges_offsets[o1]
                vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2 + 1] = 1
                vertex_adjacent_edges_fill_count[o1] = fill_count_o1 + 1

    @wp.kernel
    def count_num_adjacent_faces(
        face_indices: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_faces: wp.array(dtype=wp.int32)
    ):
        for face in range(face_indices.shape[0]):
            v0 = face_indices[face, 0]
            v1 = face_indices[face, 1]
            v2 = face_indices[face, 2]

            num_vertex_adjacent_faces[v0] = num_vertex_adjacent_faces[v0] + 1
            num_vertex_adjacent_faces[v1] = num_vertex_adjacent_faces[v1] + 1
            num_vertex_adjacent_faces[v2] = num_vertex_adjacent_faces[v2] + 1

    @wp.kernel
    def fill_adjacent_faces(
        face_indices: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_faces_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_faces_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_faces: wp.array(dtype=wp.int32),
    ):
        for face in range(face_indices.shape[0]):
            v0 = face_indices[face, 0]
            v1 = face_indices[face, 1]
            v2 = face_indices[face, 2]

            fill_count_v0 = vertex_adjacent_faces_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_faces_offsets[v0]
            vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2] = face
            vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 0
            vertex_adjacent_faces_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_faces_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_faces_offsets[v1]
            vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2] = face
            vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 1
            vertex_adjacent_faces_fill_count[v1] = fill_count_v1 + 1

            fill_count_v2 = vertex_adjacent_faces_fill_count[v2]
            buffer_offset_v2 = vertex_adjacent_faces_offsets[v2]
            vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2] = face
            vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2 + 1] = 2
            vertex_adjacent_faces_fill_count[v2] = fill_count_v2 + 1

    @wp.kernel
    def count_num_adjacent_springs(
        springs_array: wp.array(dtype=wp.int32), num_vertex_adjacent_springs: wp.array(dtype=wp.int32)
    ):
        num_springs = springs_array.shape[0] / 2
        for spring_id in range(num_springs):
            v0 = springs_array[spring_id * 2]
            v1 = springs_array[spring_id * 2 + 1]

            num_vertex_adjacent_springs[v0] = num_vertex_adjacent_springs[v0] + 1
            num_vertex_adjacent_springs[v1] = num_vertex_adjacent_springs[v1] + 1

    @wp.kernel
    def fill_adjacent_springs(
        springs_array: wp.array(dtype=wp.int32),
        vertex_adjacent_springs_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_springs_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_springs: wp.array(dtype=wp.int32),
    ):
        num_springs = springs_array.shape[0] / 2
        for spring_id in range(num_springs):
            v0 = springs_array[spring_id * 2]
            v1 = springs_array[spring_id * 2 + 1]

            fill_count_v0 = vertex_adjacent_springs_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_springs_offsets[v0]
            vertex_adjacent_springs[buffer_offset_v0 + fill_count_v0] = spring_id
            vertex_adjacent_springs_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_springs_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_springs_offsets[v1]
            vertex_adjacent_springs[buffer_offset_v1 + fill_count_v1] = spring_id
            vertex_adjacent_springs_fill_count[v1] = fill_count_v1 + 1
