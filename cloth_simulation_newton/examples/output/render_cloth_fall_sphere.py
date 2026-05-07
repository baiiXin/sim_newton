import polyscope as ps
import numpy as np
import imageio
import os # 用于处理路径

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
EXAMPLES =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(EXAMPLES, "assets")
OUTPUT = os.path.join(EXAMPLES, "output")
DATA = os.path.join(OUTPUT, "data")

# 读取文件+取文件名
load_assets_file = select_file(ASSETS)
load_data_file = select_file(DATA)
SPACIAL_NAME = os.path.splitext(os.path.basename(load_assets_file))[0]

# ==========================================
# 1. 路径设置 (根据你的环境修改)
# ==========================================

assets_path = load_assets_file
data_path = load_data_file
output_video_path = os.path.join(OUTPUT, 'video', f'{SPACIAL_NAME}.mp4')

# 确保输出目录存在
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# ==========================================
# 2. 初始化 Polyscope
# ==========================================
ps.init()
ps.set_window_size(1200, 800)
ps.set_up_dir("z_up") # 确认你的数据是 Z 轴向上的
ps.set_ground_plane_mode("none")

# ==========================================
# 3. 读取并准备数据
# ==========================================
print(f"Loading assets from: {assets_path}")
try:
    data_mesh = np.load(assets_path, allow_pickle=True)
    mesh = data_mesh["mesh"].item()
except FileNotFoundError:
    print("错误: 找不到 assets 文件，请检查路径！")
    exit()

# 提取静态拓扑数据 (Vertices & Faces)
# 注意：Vertices 仅用于初始帧注册，Faces 用于构建网格
sphere_v = mesh["sphere"]["vertices"]
sphere_f = mesh["sphere"]["triangles"]
print(sphere_v.shape, sphere_f.shape)
print(sphere_v.dtype, sphere_f.dtype)

cloth1_v = mesh["cloth1"]["vertices"]
cloth1_f = mesh["cloth1"]["triangles"]
cloth2_v = mesh["cloth2"]["vertices"]
cloth2_f = mesh["cloth2"]["triangles"]
cloth3_v = mesh["cloth3"]["vertices"]
cloth3_f = mesh["cloth3"]["triangles"]
ground_v = mesh["ground"]["vertices"]
ground_f = mesh["ground"]["triangles"]

# 注册初始网格
ps.register_surface_mesh("Sphere", sphere_v, sphere_f, color=(0.9, 0.3, 0.3), smooth_shade=False)
ps.register_surface_mesh("Cloth1", cloth1_v, cloth1_f, color=(0.3, 0.3, 0.3), smooth_shade=False)
ps.register_surface_mesh("Cloth2", cloth2_v, cloth2_f, color=(0.2, 0.8, 0.2), smooth_shade=False)
ps.register_surface_mesh("Cloth3", cloth3_v, cloth3_f, color=(0.2, 0.2, 0.8), smooth_shade=False)
ps.register_surface_mesh("Ground", ground_v, ground_f, color=(0.5, 0.5, 0.5), smooth_shade=False)

# ==========================================
# 4. 读取动画数据
# ==========================================
print(f"Loading simulation data from: {data_path}")
try:
    sim_data = np.load(data_path, allow_pickle=True) # 改个名避免混淆
except FileNotFoundError:
    print("错误: 找不到 data 文件，请检查路径！")
    exit()

print(f"Data shape: {sim_data.shape}")

# 预先计算好索引切片 (Slices)，避免在循环里重复计算
def get_slice(name):
    start = mesh[name]["start_pos_index"]
    num = mesh[name]["pos_num"]
    return slice(start, start + num)

slice_c1 = get_slice("cloth1")
slice_c2 = get_slice("cloth2")
slice_c3 = get_slice("cloth3")
slice_sph = get_slice("sphere")
slice_gnd = get_slice("ground")

# ==========================================
# 5. 动画回调逻辑
# ==========================================
t = 0
frames = [] 
max_frames = sim_data.shape[0]
is_recording = True

def callback():
    global t, frames, is_recording
    
    # -------------------------------------------------
    # [关键修改] 先检查是否录制完成，防止数组越界
    # -------------------------------------------------
    if t >= max_frames:
        if is_recording:
            is_recording = False
            print("\nSimulation finished. Saving MP4... please wait.")
            try:
                # 使用 imageio[ffmpeg] 保存
                imageio.mimsave(output_video_path, frames, fps=100)
                print(f"Done! Video saved as:\n{output_video_path}")
            except Exception as e:
                print(f"Save failed: {e}")
            
            # 停止回调，防止死循环
            ps.set_user_callback(None)
        return

    # --- 1. 获取当前帧数据 ---
    # 使用切片比 numpy indexing (array) 稍微快一点点，也更清晰
    current_frame_data = sim_data[t]
    
    pos_cloth_move1 = current_frame_data[slice_c1]
    pos_cloth_move2 = current_frame_data[slice_c2]
    pos_cloth_move3 = current_frame_data[slice_c3]
    pos_sphere_move = current_frame_data[slice_sph]
    pos_ground_move = current_frame_data[slice_gnd]

    # --- 2. 更新 Polyscope ---
    ps.get_surface_mesh("Cloth1").update_vertex_positions(pos_cloth_move1)
    ps.get_surface_mesh("Cloth2").update_vertex_positions(pos_cloth_move2)
    ps.get_surface_mesh("Cloth3").update_vertex_positions(pos_cloth_move3)
    ps.get_surface_mesh("Sphere").update_vertex_positions(pos_sphere_move)
    ps.get_surface_mesh("Ground").update_vertex_positions(pos_ground_move)

    # --- 3. 录制帧 ---
    screenshot = ps.screenshot_to_buffer(transparent_bg=False)
    frames.append(screenshot)
    
    # 打印进度 (用 \r 实现单行刷新，避免刷屏)
    print(f"Processing frame: {t+1} / {max_frames}", end='\r')
    
    # --- 4. 推进时间 ---
    t += 1

# --- 设置视角 ---
# 假设你的物体在 (0,0,0) 附近，且是 Z轴向上
# camera_pos: x=3, y=-3, z=3 (从斜上方看)
# target: 看向原点 (0,0,0)
ps.look_at((12.0, 0.0, 12.0), (0.0, 0.0, 3.0))

# 开始录制
ps.set_user_callback(callback)
ps.show()