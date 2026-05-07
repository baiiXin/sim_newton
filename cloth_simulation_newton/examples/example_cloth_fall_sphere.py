### truncate the displacement
import warp as wp
import numpy as np

#import trimesh
import os

def select_file(folder, suffix=None):
    files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and (suffix is None or f.endswith(suffix))
    ]
    files.sort()

    for i, f in enumerate(files):
        print(f"[{i}] {f}")

    idx = int(input("请选择文件编号: "))
    return os.path.join(folder, files[idx])

# 取自身目录并拼接
EXAMPLES = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(EXAMPLES, "assets")
OUTPUT = os.path.join(EXAMPLES, "output")

# cpmpute bounds
import newton
from newton._src.solvers.zcy_newton.zcy_solver_newton import zcy_SolverNewton

class Cloth:
    def __init__(self):
        # 读取文件+取文件名
        load_file = select_file(ASSETS)
        SPACIAL_NAME = os.path.splitext(os.path.basename(load_file))[0]
        
        data = np.load(load_file, allow_pickle=True)
        mesh = data["mesh"].item()

        # 初始化 位置和速度
        self.pos_cur = mesh["vertices_all"].astype(np.float64, copy=False)
        self.vel_cur = np.zeros_like(self.pos_cur)
        self.num = self.pos_cur.shape[0] 
        self.ele = mesh["triangles_all"].astype(np.int32, copy=False)

        # 初始化 质量和阻尼
        self.mass = 0.0083
        self.damp = 1.0
        self.gravity = 9.8
        self.dt = 0.01
        self.All_Time_Step = 3000

        self.tolerance_newton = 1e-4
        self.iterations = 50
        self.DeBUG =  {
            'DeBUG': True,
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
            'line_search_max_step': 15,
            'line_search_control_residual': False,
            'convergence_abs_tolerance': 1e-2,
            'convergence_rel_tolerance': 1e-4,
            'numerical_precision_condition': True,
            'numerical_precision_abs_tolerance': 1e-12,
            'numerical_precision_rel_tolerance': 1e-16,
            'barrier_threshold': 0.0,
            'truncation_threshold': 0.0,
            'Damping': 0.0,
            'spring_type': 0,
            'forward_type': 1,
            'record_name': SPACIAL_NAME+'soft10000'
        }
        # fixed points
        self.fixed_idx = mesh["fixed_index"]
        self._compute_fixed_information()

        # 缩放
        self.scale=1.0

        # contact parameters
        self.contact_radius=0.05
        self.contact_margin=0.05

        # 初始值
        self.pos_prev = self.pos_cur.copy()*self.scale
        self.vel_prev = self.vel_cur.copy()*self.scale

        # warp_vbd_self_collison_init
        wp.init()
        if wp.is_cuda_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = wp.get_device(device)

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
                    tri_kd=2.0e-2 * self.DeBUG['Damping'],
                    edge_ke=1e-3,
                    edge_kd=1e-2 * self.DeBUG['Damping'],
        )
        self.builder.add_ground_plane()
        self.builder.color(include_bending=True)
        self.model = self.builder.finalize()

        # contact parameters
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e-2 * self.DeBUG['Damping']
        self.model.soft_contact_mu = 0.2

        # model.gravity
        self.model.gravity = wp.vec3(0.0, 0.0, -self.gravity)
        self.model.spring_damping = 1.0e-2 * self.DeBUG['Damping']
        print('self.model.g', self.model.gravity)

        # spring information
        self.spring_indices =  wp.zeros(10, dtype=wp.int32, device=self.device)
        self.spring_rest_length = wp.zeros(10, dtype=wp.float32, device=self.device)
        self.spring_stiffness = wp.zeros(10, dtype=wp.float32, device=self.device)

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
        
    def Single_Newton_Method(self, time_step=0):
        
        # Newton Method (Implicit Euler)
        # iteration
        self.integrator.zcy_simulate_one_step(
            pos_warp = self.pos_warp,
            pos_prev_warp = self.pos_prev_warp,
            vel_warp = self.vel_warp,
            dt = self.dt, 
            mass = self.mass, 
            damping = self.damp, 
            num_iter = self.iterations,
            tolerance = self.tolerance_newton,
            time_step = time_step,
        )

        self.pos_cur = self.pos_warp.numpy()
        self.vel_cur = self.vel_warp.numpy()

        self.pos_warp, self.pos_prev_warp = self.pos_prev_warp, self.pos_warp

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
    # 创建仿真对象
    cloth = Cloth()
    
    # 储存结果
    cloth_data = [cloth.pos_cur.astype(np.float64).copy()]
    cloth_vel = [cloth.vel_cur.astype(np.float64).copy()]

    # 计算
    for i in range(cloth.All_Time_Step):
        print("\n\n=====Time step: ", i, "=====")
        cloth.Single_Newton_Method(time_step=i)
        cloth_data.append(cloth.pos_cur.astype(np.float64).copy())
        cloth_vel.append(cloth.vel_cur.astype(np.float64).copy())

        if i % 50 == 0:
            # =================== 保存 verts ===================
            save_dir = os.path.join(os.path.dirname(__file__), "output/data")
            os.makedirs(save_dir, exist_ok=True)
            run_id = cloth.DeBUG['record_name']
            path_data = os.path.join(save_dir, f"cloth_data_{run_id}.npy")
            np.save(path_data, np.array(cloth_data))

    # =================== 保存 verts ===================
    save_dir = os.path.join(os.path.dirname(__file__), "output/data")
    os.makedirs(save_dir, exist_ok=True)
    run_id = cloth.DeBUG['record_name']
    path_data = os.path.join(save_dir, f"cloth_data_{run_id}.npy")
    np.save(path_data, np.array(cloth_data))



if __name__ == "__main__":
    main()