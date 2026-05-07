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



import sys
from pathlib import Path
# 1️⃣ 找到你想加入的目录（绝对路径）
ROOT = Path(__file__).resolve().parents[2]  # 父目录，或者父的父目录
# 2️⃣ 转成字符串并插入 sys.path
sys.path.insert(0, str(ROOT))

# cpmpute bounds
import newton
from newton._src.solvers.zcy_newton.zcy_solver_newton import zcy_SolverNewton
from cloth_simulation_newton.examples.assets.generate_cloth import generate_cloth_mesh, generate_unique_springs


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
        # 初始化
        self.fixed_idx = [] #[0, 1, 2] #[360, 440] #[0, 4] #[10, 14] #[72, 80] #[0, 8] #[36, 44]
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
                    tri_kd=2.0e-2 * self.DeBUG['Damping'],
                    edge_ke=1e-3,
                    edge_kd=1e-2 * self.DeBUG['Damping'],
        )
        self.builder.add_ground_plane()
        self.builder.color(include_bending=True)
        self.model = self.builder.finalize(self.device)

        # contact parameters
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e-2 * self.DeBUG['Damping']
        self.model.soft_contact_mu = 0.2

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
    fixed_num = 0 # int((b-a)/h1+1)

    # 材料参数
    mass_m = 1
    stiff_k = 8000

    # 阻尼参数
    dump = 1.00
    gravity = 9.8

    # simulation
    # 初始参数
    dt = 0.01
    N = 300
    ite_num = 100
    tolerance_newton =  1e-4

    # DeBUG 
    DeBUG = {
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
        'Eigen': False,
        'line_search_max_step': 15,
        'line_search_control_residual': False,
        'convergence_abs_tolerance': 1e-2,
        'convergence_rel_tolerance': 1e-4,
        'numerical_precision_condition': True,
        'numerical_precision_abs_tolerance': 1e-14,
        'numerical_precision_rel_tolerance': 1e-18,
        'barrier_threshold': 0.0,
        'truncation_threshold': 0.0,
        'Damping': 0.0,
        'spring_type': 0,
        'forward_type': 1,
        'record_name': 'two_triangles'
    }

    Mass_X = np.array([[0.0,0.0,0.0], [5.0,0.0,0.0], [0.0,5.0,0.0], [1.0,1.0,0.5], [2.0,1.0,1.5], [1.0,2.0,1.5]])

    Mass_V = np.array([[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]])

    Mass_E = np.array([[0,1,2], [3,4,5]])

    Spring_ele = np.array([[0,1], [0,2], [1,2], [3,4], [3,5], [4,5]])

    Spring_len = np.array([np.linalg.norm(Mass_X[Spring_ele[i,0]] - Mass_X[Spring_ele[i,1]]) for i in range(Spring_ele.shape[0])])

    # 创建弹簧
    mySpring = Spring(
        num=Spring_ele.shape[0],
        ele=Spring_ele,
        rest_len=Spring_len,
        stiff_k=stiff_k
    )
    #print(Mass_X)
    # 创建质点
    myMass = Mass(
        num=Mass_X.shape[0],
        ele=Mass_E,
        pos_cur=Mass_X.copy(),
        vel_cur=Mass_V.copy(),
        pos_prev=Mass_X.copy(),
        vel_prev=Mass_V.copy(),
        mass=mass_m,
        damp=dump,
        gravity=gravity,
        Spring=mySpring,
        dt=dt,
        tolerance_newton=tolerance_newton,
        cloth_size=3,
        DeBUG=DeBUG,
    )

    # load files
    # 取自身目录并拼接
    Project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    Examples = os.path.join(Project, "examples")
    OUTPUT = os.path.join(Examples, "output")
    DATA = os.path.join(OUTPUT, "data")
    load_file = select_file(DATA)
    
    # read
    pos_frames = np.load(load_file)
    print(f"总共{pos_frames.shape[0]}帧")
    # 读取文件+取文件名 
    frame = int(input("请输入要检查的帧号: "))
    pos_cur = pos_frames[frame]
    pos_prev = pos_frames[frame-1] if frame > 0 else pos_frames[frame]
    vel = (pos_cur - pos_prev) / dt

    # 转换为wp.array
    pos_warp = wp.array(pos_cur, dtype=wp.vec3)
    pos_prev_warp = wp.array(pos_prev, dtype=wp.vec3)
    vel_warp = wp.array(vel, dtype=wp.vec3)
    print(pos_warp.shape)

    input("请按任意键继续...")
   
   
    # 初始化
    Check_Switch0 = {
        # inertia
        "Inertia": False,
        # elastic
        "Elastic": False,
        "Spring_Elastic": False,
        "Stvk_Elastic": True,
        # bending
        "Bending": False,
        # contact
        "Contact": False,
        "Contact_EE": True,
        "Contact_VT": True,
        # numerical precision
        "perturbation_epsilon": 1e-3,
    }

    Check_select = {
        "Inertia": True,
        "Elastic": True,
        "Bending": True,
        "Contact": True,
        "All": True,
    }

    for key, value in Check_select.items():
        Check_Switch = Check_Switch0.copy()

        if key == "All":
            for k, v in Check_select.items():
                Check_Switch[k] = v
        else:
            # 更新当前键对应的字典
            Check_Switch[key] = value
        
        # 这里可以使用 Check_Switch 做后续操作
        # print(f"{key}: {Check_Switch}")

        # 计算
        (   energy,
            grad,
            hessian,
            grad_fd_by_fd_energy,
            hessian_fd_by_fd_grad,
            grad_error_norm_of_energy_fd,
            hessian_error_norm_of_grad_fd,
        ) = myMass.integrator.zcy_check_grad_and_hessian_via_fd(pos_warp, pos_prev_warp, vel_warp, dt, Check_Switch)

        print(f'{key} energy', energy)
        print(f'{key} grad', grad.shape)
        print(f'{key} hessian', hessian.shape)
        print(f'{key} grad_fd_by_fd_energy', grad_fd_by_fd_energy.shape)
        print(f'{key} hessian_fd_by_fd_grad', hessian_fd_by_fd_grad.shape)
        print(f'{key} grad_error_norm_of_energy_fd', grad_error_norm_of_energy_fd)
        print(f'{key} hessian_error_norm_of_grad_fd', hessian_error_norm_of_grad_fd)
        print()



if __name__ == "__main__":
    main()
