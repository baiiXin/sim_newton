import torch


class ImplicitEulerLoss:
    def __init__(
        self,
        rest_pos,
        edge_index,
        face_index,
        density,
        mu,
        lambda_,
        k_bending,
        gravity=(0.0, 0.0, -9.81),
        dt=None,
        pinned_idx=None,
        eps=1e-12,
    ):
        """
        Minimal implicit Euler loss class.

        Parameters
        ----------
        rest_pos : Tensor-like, [N, 3]
            Rest pose / material-space vertex positions.
        edge_index : Tensor-like, [E, 2]
            Undirected mesh edges. Currently stored but not used by the minimal losses.
        face_index : Tensor-like, [F, 3]
            Triangle indices. Assumed consistently oriented.
        density : float
            Areal density. Vertex mass = density * lumped vertex area.
        mu : float
            StVK Lamé shear parameter.
        lambda_ : float
            StVK Lamé first parameter.
        k_bending : float
            Simple dihedral-angle bending stiffness.
        gravity : Tensor-like, [3]
            Gravity acceleration vector.
        dt : float or None
            Optional fixed timestep. If None, forward will later require dt.
        pinned_idx : Tensor-like or None
            Fixed vertex indices.
        eps : float
            Numerical epsilon.
        """

        # --------
        # 1. Basic tensor setup
        # --------
        if torch.is_tensor(rest_pos):
            rest_pos = rest_pos.clone()
            if not rest_pos.dtype.is_floating_point:
                rest_pos = rest_pos.to(dtype=torch.get_default_dtype())
        else:
            rest_pos = torch.tensor(rest_pos, dtype=torch.get_default_dtype())

        self.device = rest_pos.device
        self.dtype = rest_pos.dtype
        self.eps = eps

        self.rest_pos = rest_pos

        self.edge_index = torch.as_tensor(
            edge_index,
            dtype=torch.long,
            device=self.device,
        )

        self.face_index = torch.as_tensor(
            face_index,
            dtype=torch.long,
            device=self.device,
        )

        self._validate_inputs()

        self.num_vertices = self.rest_pos.shape[0]
        self.num_faces = self.face_index.shape[0]

        # --------
        # 2. Fixed physical parameters
        # --------
        self.density = torch.as_tensor(
            density,
            dtype=self.dtype,
            device=self.device,
        )

        self.mu = torch.as_tensor(
            mu,
            dtype=self.dtype,
            device=self.device,
        )

        self.lambda_ = torch.as_tensor(
            lambda_,
            dtype=self.dtype,
            device=self.device,
        )

        self.k_bending = torch.as_tensor(
            k_bending,
            dtype=self.dtype,
            device=self.device,
        )

        self.gravity = torch.as_tensor(
            gravity,
            dtype=self.dtype,
            device=self.device,
        )

        if self.gravity.shape != (3,):
            raise ValueError(f"gravity must have shape [3], got {self.gravity.shape}")

        self.dt = None
        if dt is not None:
            self.dt = torch.as_tensor(
                dt,
                dtype=self.dtype,
                device=self.device,
            )

        # --------
        # 3. Pinned / free vertices
        # --------
        self.free_mask = torch.ones(
            self.num_vertices,
            dtype=torch.bool,
            device=self.device,
        )

        if pinned_idx is None:
            self.pinned_idx = None
        else:
            self.pinned_idx = torch.as_tensor(
                pinned_idx,
                dtype=torch.long,
                device=self.device,
            )
            self.free_mask[self.pinned_idx] = False

        # --------
        # 4. Geometry precomputation
        # --------
        self.face_area, self.Dm_inv = self._compute_face_area_and_Dm_inv()

        self.vertex_area = self._compute_lumped_vertex_area()

        # mass is fixed in the minimal version
        self.mass = self.density * self.vertex_area

        self.hinges = self._build_hinges_from_faces()

        self.theta0 = self._dihedral_angle(self.rest_pos, self.hinges)

    def _validate_inputs(self):
        if self.rest_pos.ndim != 2 or self.rest_pos.shape[1] != 3:
            raise ValueError(f"rest_pos must have shape [N, 3], got {self.rest_pos.shape}")

        if self.edge_index.ndim != 2 or self.edge_index.shape[1] != 2:
            raise ValueError(f"edge_index must have shape [E, 2], got {self.edge_index.shape}")

        if self.face_index.ndim != 2 or self.face_index.shape[1] != 3:
            raise ValueError(f"face_index must have shape [F, 3], got {self.face_index.shape}")

        n = self.rest_pos.shape[0]

        if self.face_index.numel() > 0:
            if self.face_index.min() < 0 or self.face_index.max() >= n:
                raise ValueError("face_index contains vertex indices out of range.")

        if self.edge_index.numel() > 0:
            if self.edge_index.min() < 0 or self.edge_index.max() >= n:
                raise ValueError("edge_index contains vertex indices out of range.")

    def _compute_face_area_and_Dm_inv(self):
        """
        Precompute per-face rest area and inverse material shape matrix.

        face_area : [F]
        Dm_inv    : [F, 2, 2]
        """
        faces = self.face_index

        Xi = self.rest_pos[faces[:, 0]]
        Xj = self.rest_pos[faces[:, 1]]
        Xk = self.rest_pos[faces[:, 2]]

        e1 = Xj - Xi
        e2 = Xk - Xi

        cross = torch.cross(e1, e2, dim=-1)
        face_area = 0.5 * torch.linalg.norm(cross, dim=-1)

        if torch.any(face_area <= self.eps):
            min_area = face_area.min().item()
            raise ValueError(f"Degenerate triangle detected. min face area = {min_area}")

        # Build local 2D rest triangle:
        #
        # ui = (0, 0)
        # uj = (||e1||, 0)
        # uk = (a, b)
        #
        # Dm = [[||e1||, a],
        #       [0,      b]]
        e1_len = torch.linalg.norm(e1, dim=-1)

        if torch.any(e1_len <= self.eps):
            min_len = e1_len.min().item()
            raise ValueError(f"Degenerate triangle edge detected. min edge length = {min_len}")

        a = (e1 * e2).sum(dim=-1) / e1_len

        e2_len_sq = (e2 * e2).sum(dim=-1)
        b_sq = torch.clamp(e2_len_sq - a * a, min=0.0)
        b = torch.sqrt(b_sq)

        if torch.any(b <= self.eps):
            min_b = b.min().item()
            raise ValueError(f"Degenerate local triangle basis detected. min b = {min_b}")

        F = faces.shape[0]
        Dm = torch.zeros(
            F,
            2,
            2,
            dtype=self.dtype,
            device=self.device,
        )

        Dm[:, 0, 0] = e1_len
        Dm[:, 0, 1] = a
        Dm[:, 1, 0] = 0.0
        Dm[:, 1, 1] = b

        Dm_inv = torch.linalg.inv(Dm)

        return face_area, Dm_inv

    def _compute_lumped_vertex_area(self):
        """
        vertex_area[i] = sum_{f incident to i} face_area[f] / 3
        """
        vertex_area = torch.zeros(
            self.num_vertices,
            dtype=self.dtype,
            device=self.device,
        )

        contrib = self.face_area / 3.0

        faces = self.face_index
        vertex_area.scatter_add_(0, faces[:, 0], contrib)
        vertex_area.scatter_add_(0, faces[:, 1], contrib)
        vertex_area.scatter_add_(0, faces[:, 2], contrib)

        return vertex_area

    def _build_hinges_from_faces(self):
        """
        Build bending hinges from consistently oriented faces.

        Each hinge is [i, j, k, l], where:
            i, j : shared edge, oriented according to the first face
            k    : opposite vertex in the first face
            l    : opposite vertex in the second face

        Boundary edges are skipped.
        Non-manifold edges raise an error.
        """
        faces_cpu = self.face_index.detach().cpu().tolist()

        edge_to_entries = {}

        for f_id, (a, b, c) in enumerate(faces_cpu):
            # Oriented face edges with opposite vertex.
            oriented_edges = [
                (a, b, c),
                (b, c, a),
                (c, a, b),
            ]

            for u, v, opp in oriented_edges:
                key = (min(u, v), max(u, v))
                edge_to_entries.setdefault(key, []).append((u, v, opp, f_id))

        hinges = []

        for key, entries in edge_to_entries.items():
            if len(entries) == 1:
                # Boundary edge: no bending hinge.
                continue

            if len(entries) > 2:
                raise ValueError(f"Non-manifold edge detected: edge {key} has {len(entries)} incident faces.")

            e0, e1 = entries

            i, j, k, _ = e0
            u, v, l, _ = e1

            # With consistent winding, the second face should contain the reversed edge.
            # We do not hard fail here, because for the minimal pipeline we assume
            # clean constructed examples. The same convention is used for theta0 and theta.
            hinges.append([i, j, k, l])

        if len(hinges) == 0:
            return torch.empty(
                0,
                4,
                dtype=torch.long,
                device=self.device,
            )

        return torch.tensor(
            hinges,
            dtype=torch.long,
            device=self.device,
        )

    def _dihedral_angle(self, x, hinges):
        """
        Compute signed dihedral angles for hinges.

        Parameters
        ----------
        x : [N, 3]
        hinges : [H, 4]

        Returns
        -------
        theta : [H]
        """
        if hinges.numel() == 0:
            return torch.empty(
                0,
                dtype=self.dtype,
                device=self.device,
            )

        i = hinges[:, 0]
        j = hinges[:, 1]
        k = hinges[:, 2]
        l = hinges[:, 3]

        xi = x[i]
        xj = x[j]
        xk = x[k]
        xl = x[l]

        edge = xj - xi
        edge_norm = torch.linalg.norm(edge, dim=-1, keepdim=True).clamp_min(self.eps)
        edge_hat = edge / edge_norm

        n0_raw = torch.cross(xj - xi, xk - xi, dim=-1)
        n1_raw = torch.cross(xl - xi, xj - xi, dim=-1)

        n0 = n0_raw / torch.linalg.norm(n0_raw, dim=-1, keepdim=True).clamp_min(self.eps)
        n1 = n1_raw / torch.linalg.norm(n1_raw, dim=-1, keepdim=True).clamp_min(self.eps)

        sin_theta = (torch.cross(n0, n1, dim=-1) * edge_hat).sum(dim=-1)
        cos_theta = (n0 * n1).sum(dim=-1).clamp(-1.0 + self.eps, 1.0 - self.eps)

        theta = torch.atan2(sin_theta, cos_theta)
        return theta

    def _check_state_tensor(self, x, name="x"):
        if not torch.is_tensor(x):
            raise TypeError(f"{name} must be a torch.Tensor.")

        if x.shape != self.rest_pos.shape:
            raise ValueError(
                f"{name} must have shape {self.rest_pos.shape}, got {x.shape}"
            )

        if x.device != self.device:
            raise ValueError(
                f"{name} must be on device {self.device}, got {x.device}"
            )

        if x.dtype != self.dtype:
            raise ValueError(
                f"{name} must have dtype {self.dtype}, got {x.dtype}"
            )

    def _get_dt(self, dt=None):
        if dt is None:
            if self.dt is None:
                raise ValueError("dt was not provided and self.dt is None.")
            return self.dt

        return torch.as_tensor(
            dt,
            dtype=self.dtype,
            device=self.device,
        )

    def inertia_loss(self, x, x_prev, v_prev, dt=None):
        """
        L_inertia = sum_i 0.5 * m_i / dt^2 * ||x_i - (x_prev_i + dt * v_prev_i)||^2

        Only free vertices participate.
        """
        self._check_state_tensor(x, "x")
        self._check_state_tensor(x_prev, "x_prev")
        self._check_state_tensor(v_prev, "v_prev")

        dt = self._get_dt(dt)

        x_hat = x_prev + dt * v_prev
        diff = x - x_hat

        loss_per_vertex = (
            0.5
            * self.mass
            / (dt * dt)
            * (diff * diff).sum(dim=-1)
        )

        return loss_per_vertex[self.free_mask].sum()

    def gravity_loss(self, x):
        """
        L_gravity = -sum_i m_i * g^T x_i

        Only free vertices participate.
        """
        self._check_state_tensor(x, "x")

        dot = (x * self.gravity).sum(dim=-1)
        loss_per_vertex = -self.mass * dot

        return loss_per_vertex[self.free_mask].sum()

    def elastic_loss(self, x):
        """
        StVK triangle FEM stretching energy.

        L_elastic = sum_f A_f * [
            mu * ||G_f||_F^2
            + 0.5 * lambda * tr(G_f)^2
        ]

        G_f = 0.5 * (F_f^T F_f - I)
        F_f = D_s_f D_m_f^{-1}

        All triangles participate in this minimal version.
        """
        self._check_state_tensor(x, "x")

        faces = self.face_index

        xi = x[faces[:, 0]]
        xj = x[faces[:, 1]]
        xk = x[faces[:, 2]]

        # D_s: [F, 3, 2]
        Ds = torch.stack(
            [
                xj - xi,
                xk - xi,
            ],
            dim=-1,
        )

        # Fmat: [F, 3, 2]
        Fmat = Ds @ self.Dm_inv

        # C = F^T F: [F, 2, 2]
        C = Fmat.transpose(-1, -2) @ Fmat

        I = torch.eye(
            2,
            dtype=self.dtype,
            device=self.device,
        ).expand_as(C)

        G = 0.5 * (C - I)

        trG = G[:, 0, 0] + G[:, 1, 1]
        normG2 = (G * G).sum(dim=(-1, -2))

        psi = self.mu * normG2 + 0.5 * self.lambda_ * trG * trG

        return (self.face_area * psi).sum()

    def bending_loss(self, x):
        """
        Simple dihedral-angle bending energy.

        L_bending = sum_h 0.5 * k_bending * wrap(theta_h - theta0_h)^2

        No geometric scaling weight in this minimal version.
        All hinges participate.
        """
        self._check_state_tensor(x, "x")

        if self.hinges.numel() == 0:
            return torch.zeros(
                (),
                dtype=self.dtype,
                device=self.device,
            )

        theta = self._dihedral_angle(x, self.hinges)

        dtheta_raw = theta - self.theta0

        # Wrapped angle difference in (-pi, pi]
        dtheta = torch.atan2(
            torch.sin(dtheta_raw),
            torch.cos(dtheta_raw),
        )

        return (0.5 * self.k_bending * dtheta * dtheta).sum()

    def forward(self, x, x_prev, v_prev, dt=None):
        """
        Compute all currently enabled losses.

        Collision is intentionally skipped in the minimal pipeline.
        """
        loss_inertia = self.inertia_loss(x, x_prev, v_prev, dt)
        loss_gravity = self.gravity_loss(x)
        loss_elastic = self.elastic_loss(x)
        loss_bending = self.bending_loss(x)

        loss_total = (
            loss_inertia
            + loss_gravity
            + loss_elastic
            + loss_bending
        )

        return {
            "total": loss_total,
            "inertia": loss_inertia,
            "gravity": loss_gravity,
            "elastic": loss_elastic,
            "bending": loss_bending,
        }

    def residual(
        self,
        x,
        x_prev,
        v_prev,
        dt=None,
        normalize_by_mass=False,
        create_graph=False,
    ):
        """
        Compute implicit Euler residual.

        raw residual:
            r = grad_x L_total(x)

        If normalize_by_mass=True:
            r_i = r_i / m_i

        Pinned vertices are set to zero.
        """

        self._check_state_tensor(x, "x")
        self._check_state_tensor(x_prev, "x_prev")
        self._check_state_tensor(v_prev, "v_prev")

        # residual 需要梯度，即使外部处于 torch.no_grad() 环境，
        # 这里也必须重新启用 grad。
        with torch.enable_grad():
            if x.requires_grad:
                x_req = x
            else:
                x_req = x.detach().clone().requires_grad_(True)

            losses = self.forward(
                x=x_req,
                x_prev=x_prev,
                v_prev=v_prev,
                dt=dt,
            )

            total_loss = losses["total"]

            grad = torch.autograd.grad(
                total_loss,
                x_req,
                create_graph=create_graph,
                retain_graph=create_graph,
                only_inputs=True,
            )[0]

        residual_vec = grad

        if normalize_by_mass:
            residual_vec = residual_vec / self.mass[:, None].clamp_min(self.eps)

        # Pinned vertices do not participate in residual.
        residual_vec = residual_vec.clone()
        residual_vec[~self.free_mask] = 0.0

        norm_per_vertex = torch.linalg.norm(residual_vec, dim=-1)
        free_norm = norm_per_vertex[self.free_mask]

        if free_norm.numel() == 0:
            mean_norm = torch.zeros((), dtype=self.dtype, device=self.device)
            max_norm = torch.zeros((), dtype=self.dtype, device=self.device)
            l2_norm = torch.zeros((), dtype=self.dtype, device=self.device)
        else:
            mean_norm = free_norm.mean()
            max_norm = free_norm.max()
            l2_norm = torch.linalg.norm(residual_vec[self.free_mask].reshape(-1))

        return {
            "vector": residual_vec,
            "norm_per_vertex": norm_per_vertex,
            "mean": mean_norm,
            "max": max_norm,
            "l2": l2_norm,
        }  

