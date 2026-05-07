### truncate the displacement
import warp as wp
import numpy as np
#import trimesh
import os

# cpmpute bounds
import newton
from newton._src.solvers.zcy_newton.zcy_solver_newton import zcy_SolverNewton
from assets.generate_cloth import generate_cloth_mesh, generate_unique_springs


class Spring:
    def __init__(self, num=None, ele=None, stiff_k=None, rest_len=None):
        self.num = num # 弹簧数量；1
        self.ele = ele # 弹簧连接的质点编号；[[0, 1], [1, 2], [2, 3], [3, 4]]
        self.rest_len = rest_len # 弹簧的初始长度；[1.0, 1.0, 1.0, 1.0]
        self.stiff_k = stiff_k  # 弹簧的刚度；1

class Mass:
    def __init__(self, num=int, 
                 pos_cur=None, vel_cur=None, pos_prev=None, vel_prev=None,
                 ele=None, mass=None, 
                 force=None, Hessian=None, Mass_k=None,
                 damp=None, gravity=None, Spring=Spring, dt=None, 
                 tolerance_newton=None, cloth_size=0, DeBUG=None):

        # warp_vbd_self_collison_init
        wp.init()
        if wp.is_cuda_available():
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = wp.get_device(device)
        wp.set_device(self.device) 

        self.num = num # 质点数量；1
        self.ele = ele # 三角元；[[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        self.pos_cur = pos_cur # 质点位置；[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        self.vel_cur = vel_cur # 质点速度；[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self.pos_prev = pos_prev # 预测质点位置
        self.vel_prev = vel_prev # 预测质点速度
        self.force = force # 力向量--牛顿迭代--线性方程组--b
        self.Hessian = Hessian # 矩阵--牛顿迭代--线性方程组--A
        self.mass = mass # 质点质量；1
        self.damp = damp # 阻尼系数；1
        self.gravity = gravity # 重力加速度；9.8
        self.Spring = Spring # 弹簧
        self.dt = dt # 时间步长
        self.tolerance_newton = tolerance_newton # 牛顿迭代的容差
        self.iterations = 10
        # self.fixed_num = 9
        self.space_dim = 3
        self.load = False
        self.cloth_size = cloth_size
        self.DeBUG = DeBUG

        # fixed points
        # rotation fixed points
        #cloth_size = 9

        # 初始化
        mid_index = cloth_size // 2
        mid_cloth = mid_index * cloth_size
        self.fixed_idx = [mid_cloth, mid_cloth + cloth_size -1] #[0, 1, 2] #[360, 440] #[0, 4] #[10, 14] #[72, 80] #[0, 8] #[36, 44]
        self._compute_fixed_information()

        # 缩放
        self.scale=1.0

        # contact parameters
        self.contact_radius=0.08
        self.contact_margin=0.08

        # 初始值
        #self.pos_cur[:, [1, 2]] = self.pos_cur[:, [2, 1]]
        self.pos_prev = self.pos_cur.copy()*self.scale
        self.vel_prev = self.vel_cur.copy()*self.scale

        self.pos_warp = [wp.vec3(self.pos_cur[i,:]) for i in range(self.num)]

        self.builder = newton.ModelBuilder()
        self.builder.add_cloth_mesh(
                    pos=wp.vec3(0.0, 0.0, 0.0),
                    rot=wp.quat_identity(),
                    scale=self.scale,
                    vertices=self.pos_warp,
                    indices=self.ele.reshape(-1),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    density=0.2,
                    tri_ke=1.0e3,
                    tri_ka=1.0e3,
                    tri_kd=2.0e-7 * self.DeBUG['Damping'],
                    edge_ke=1e-3,
                    edge_kd=1e-4 * self.DeBUG['Damping'],
        )
        self.builder.add_ground_plane()
        self.builder.color(include_bending=True)
        self.model = self.builder.finalize(self.device)

        # contact parameters
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e-2 * self.DeBUG['Damping']
        self.model.soft_contact_mu = 0.02

        # model.gravity
        self.model.gravity = wp.vec3(0.0, 0.0, -self.gravity)
        self.model.spring_damping = 1.0e-2 * self.DeBUG['Damping']
        print('self.model.g', self.model.gravity)

        # spring information
        self.spring_indices = [x for row in self.Spring.ele for x in row]
        self.spring_indices = wp.array(self.spring_indices, dtype=wp.int32)
        self.spring_rest_length = wp.array(self.Spring.rest_len, dtype=wp.float32)
        self.spring_stiffness = [self.Spring.stiff_k for i in range(len(self.spring_rest_length))]
        self.spring_stiffness = wp.array(self.spring_stiffness, dtype=wp.float32)

        print('spring_indices', type(self.spring_indices))

        self.integrator = zcy_SolverNewton(
                    model=self.model,
                    # DeBUG
                    DeBUG = self.DeBUG,
                    # self parameters
                    dt = self.dt,
                    mass = self.mass,
                    # fixed particle information
                    fixed_particle_num = self.fixed_particle_num,
                    free_particle_offset = self.free_particle_offset,
                    all_particle_flag = self.all_particle_flag,
                    # other
                    iterations=self.iterations, 
                    # before
                    handle_self_contact=True,
                    self_contact_radius=self.contact_radius,
                    self_contact_margin=self.contact_margin,
                    spring_indices = self.spring_indices, 
                    spring_rest_length = self.spring_rest_length, 
                    spring_stiffness = self.spring_stiffness
        )

        # state
        #self.state_0 = self.model.state()
        #self.state_1 = self.model.state()
        # transform
        self.pos_cur *= self.scale
        self.vel_cur *= self.scale
        self.pos_warp = wp.array(self.pos_cur, dtype=wp.vec3)
        self.pos_prev_warp = wp.array(self.pos_prev, dtype=wp.vec3)
        self.vel_warp = wp.array(self.vel_cur, dtype=wp.vec3)
        self.vel_prev_warp = wp.array(self.vel_prev, dtype=wp.vec3)

        # 检查
        print('model.tri_indices', self.model.tri_indices.shape)
        print('model.edge_indices', self.model.edge_indices.shape)
        
    def Single_Newton_Method(self, Spring: Spring, fixed_num, ite_num, space_dim=3, time_step=0):
        
        # Newton Method (Implicit Euler)
        # 计时
        times_ms = []
        # 残差曲线
        Newton_step = []
        Error_dx_norm = []
        Residual_norm = []
        Energy_norm = []

        # iteration
        print('\n---cloth---', self.iterations)

        # step
        self.integrator.zcy_simulate_one_step(
            pos_warp = self.pos_warp,
            pos_prev_warp = self.pos_prev_warp,
            vel_warp = self.vel_warp,
            dt = self.dt, 
            mass = self.mass, 
            damping = self.damp, 
            num_iter = ite_num,
            tolerance = self.tolerance_newton,
            time_step = time_step,
        )

        self.pos_cur = self.pos_warp.numpy()
        self.vel_cur = self.vel_warp.numpy()

        self.pos_warp, self.pos_prev_warp = self.pos_prev_warp, self.pos_warp

        return Newton_step, times_ms, Error_dx_norm, Residual_norm, Energy_norm

    def _compute_fixed_information(self):
        self.fixed_particle_num = len(self.fixed_idx)

        self.all_particle_flag = []
        self.free_particle_offset = []
        flag = 0
        for i in range(self.num):
            if i in self.fixed_idx:
                self.all_particle_flag.append(-1)
                flag += 1
            else:
                self.all_particle_flag.append(flag)
                self.free_particle_offset.append(flag)

        self.all_particle_flag = wp.array(self.all_particle_flag, dtype=wp.int32)
        self.free_particle_offset = wp.array(self.free_particle_offset, dtype=wp.int32)


def main():
    # =================== 数据构造部分 ===================
    # 你的数据（假设已加载）
    # cloth_data: List of (Nm, 3) arrays
    # triangles: (Ne, 3) array of vertex indices
    # ===== 仿真参数 =====
    # 网格生成
    a = -2.0
    b = 2.0
    c = -2.0
    d = 2.0
    h1 = 0.5
    h2 = 0.5

    cloth_size = int((b-a)/h1+1)
    fixed_num = 2 # int((b-a)/h1+1)

    # 材料参数
    z=0.0
    mass_m = 0.0083
    stiff_k = 1000

    # 阻尼参数
    damp = 1.0
    gravity = 9.8

    # simulation
    # 初始参数
    dt = 0.003
    N = 1000
    ite_num = 5
    tolerance_newton = 1e-4

    # DeBUG 
    DeBUG = {
        'DeBUG': False,
        'DeBUG0': False,
        'record_hessian': False,
        'max_information': True,
        'max_warning': False,
        'Spring': True,
        'Bending': True,
        'Contact': True,
        'Contact_EE': True,
        'Contact_VT': True,
        'Inertia_Hessian': True,
        'Eigen': True,
        'line_search_max_step': 1,
        'line_search_control_residual': False,
        'convergence_abs_tolerance': 1e-4,
        'convergence_rel_tolerance': 1e-4,
        'numerical_precision_condition': True,
        'numerical_precision_abs_tolerance': 1e-12,
        'numerical_precision_rel_tolerance': 1e-12,
        'barrier_threshold': 0.0,
        'truncation_threshold': 0.0,
        'Damping': 0.0,
        'spring_type': 0,
        'forward_type': 1,
        'record_name': 'cloth2'
    }

    # 仿真计算
    # Mass
    P, T = generate_cloth_mesh(a, b, c, d, h1, h2, z)
    Mass_num = P.shape[0]
    Mass_X = P.astype(np.float64, copy=False)
    Mass_E = T.astype(np.int32, copy=False)
    Mass_V = np.zeros_like(Mass_X)
    Mass_m = mass_m
    print("Mass_num", Mass_num)
    # Spring
    Ns, Springs, Lens, Ks = generate_unique_springs(P, T, stiff_k)
    Spring_num = Ns
    Spring_ele = Springs.astype(np.int32, copy=False)
    Spring_len = Lens.astype(np.float64, copy=False)
    Spring_stiff_k = Ks.astype(np.float64, copy=False)
    print("Spring_num", Spring_num)

    # 创建弹簧
    mySpring = Spring(
        num=Spring_num,
        ele=Spring_ele,
        rest_len=Spring_len,
        stiff_k=Spring_stiff_k
    )
    #print(Mass_X)
    # 创建质点
    myMass = Mass(
        num=Mass_num,
        ele=Mass_E,
        pos_cur=Mass_X.copy(),
        vel_cur=Mass_V.copy(),
        pos_prev=Mass_X.copy(),
        vel_prev=Mass_V.copy(),
        mass=Mass_m,
        damp=damp,
        gravity=gravity,
        Spring=mySpring,
        dt=dt,
        tolerance_newton=tolerance_newton,
        cloth_size=cloth_size,
        DeBUG=DeBUG,
    )

    # 储存结果
    cloth_data = [myMass.pos_cur.astype(np.float64).copy()]
    cloth_vel = [myMass.vel_cur.astype(np.float64).copy()]

    # 计算
    for i in range(N):
        print("\n\n=====Time step: ", i, "=====")
        [Newton_steps, times_ms, Error_dx_norm, Residual_norm, Energy_norm] = myMass.Single_Newton_Method(mySpring, fixed_num, ite_num, time_step=i)
        cloth_data.append(myMass.pos_cur.astype(np.float64).copy())
        cloth_vel.append(myMass.vel_cur.astype(np.float64).copy())

        if i % 50 == 0:
            # =================== 保存 verts ===================
            save_dir = os.path.join(os.path.dirname(__file__), "output/data")
            os.makedirs(save_dir, exist_ok=True)
            run_id = DeBUG['record_name']
            path_data = os.path.join(save_dir, f"cloth_data_{run_id}.npy")
            np.save(path_data, np.array(cloth_data))

    # =================== 保存 verts ===================
    save_dir = os.path.join(os.path.dirname(__file__), "output/data")
    os.makedirs(save_dir, exist_ok=True)
    run_id = DeBUG['record_name']
    path_data = os.path.join(save_dir, f"cloth_data_{run_id}.npy")
    path_topy = os.path.join(save_dir, f"cloth_topy_{run_id}.npy")
    np.save(path_data, np.array(cloth_data))
    np.save(path_topy, Mass_E.astype(np.int32))



if __name__ == "__main__":
    main()
