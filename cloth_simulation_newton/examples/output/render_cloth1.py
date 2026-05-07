import polyscope as ps
import numpy as np
import imageio
import os # 用于处理路径

# 取 examples 目录并拼接
EXAMPLES = "/data/zhoucy/sim_newton/cloth_simulation_newton/examples"
OUTPUT = os.path.join(EXAMPLES, "output")
DATA = os.path.join(OUTPUT, "data")

# ==========================================
# 1. 路径设置
# ==========================================

data_path = os.path.join(DATA, "cloth_data_cloth1.npy")
topy_path = os.path.join(DATA, "cloth_topy_cloth1.npy")
output_video_path = os.path.join(OUTPUT, 'video', 'cloth1.mp4')
FPS = 20
Smooth_shade = False

# 确保输出目录存在
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# ==========================================
# 2. 初始化 Polyscope
# ==========================================
# 服务器没有 DISPLAY 时，使用 EGL headless 后端
ps.init("openGL3_egl")
ps.set_window_size(1200, 800)
ps.set_up_dir("z_up") # 确认你的数据是 Z 轴向上的
ps.set_ground_plane_mode("none")

# ==========================================
# 3. 读取并准备数据
# ==========================================
print(f"Loading topology from: {topy_path}")
try:
    cloth1_f = np.load(topy_path, allow_pickle=True)
except FileNotFoundError:
    print("错误: 找不到 topy 文件，请检查路径！")
    exit()

print(f"Loading simulation data from: {data_path}")
try:
    sim_data = np.load(data_path, allow_pickle=True)
except FileNotFoundError:
    print("错误: 找不到 data 文件，请检查路径！")
    exit()

print(f"Topology shape: {cloth1_f.shape}")
print(f"Data shape: {sim_data.shape}")
print(cloth1_f.dtype, sim_data.dtype)

# 提取初始帧顶点，用于注册网格
cloth1_v = sim_data[0]

# 注册初始网格
ps.register_surface_mesh("Cloth1", cloth1_v, cloth1_f, color=(0.5, 0.7, 1.0), smooth_shade=Smooth_shade)

# ==========================================
# 4. 读取动画数据
# ==========================================
# 这里 cloth_data_cloth1.npy 已经只包含 cloth1 的每帧顶点位置
# 因此不再需要从总数据里根据 start_pos_index / pos_num 做切片

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
    # 先检查是否录制完成，防止数组越界
    # -------------------------------------------------
    if t >= max_frames:
        if is_recording:
            is_recording = False
            print("\nSimulation finished. Saving MP4... please wait.")
            try:
                # 使用 imageio[ffmpeg] 保存
                imageio.mimsave(output_video_path, frames, fps=FPS)
                print(f"Done! Video saved as:\n{output_video_path}")
            except Exception as e:
                print(f"Save failed: {e}")

            # 停止回调，防止死循环
            ps.set_user_callback(None)
        return

    # --- 1. 获取当前帧数据 ---
    current_frame_data = sim_data[t]

    # --- 2. 更新 Polyscope ---
    ps.get_surface_mesh("Cloth1").update_vertex_positions(current_frame_data)

    # --- 3. 录制帧 ---
    screenshot = ps.screenshot_to_buffer(transparent_bg=False)
    frames.append(screenshot)

    # 打印进度
    print(f"Processing frame: {t+1} / {max_frames}", end='\r')

    # --- 4. 推进时间 ---
    t += 1

# --- 设置视角 ---
# 假设你的物体在 (0,0,0) 附近，且是 Z轴向上
# camera_pos: x=3, y=-3, z=3 (从斜上方看)
# target: 看向原点 (0,0,0)
ps.look_at((8.0, 8.0, -8.0), (0.0, -2.0, 0.0))

# 开始录制
ps.set_user_callback(callback)
ps.show()
