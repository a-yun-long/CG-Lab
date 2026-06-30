"""
Work6: 质点-弹簧布料模拟与数值积分对比
使用 Taichi 框架实现 GPU 加速的物理模拟
"""
import taichi as ti
import numpy as np

# ===================== 配置参数 =====================
NUM_ROWS = 20
NUM_COLS = 20
NUM_PARTICLES = NUM_ROWS * NUM_COLS
SPACING = 0.05  # 质点间距

# 弹簧参数
KS = 1e3        # 劲度系数
KD = 0.5        # 阻尼系数
GRAVITY = ti.Vector([0.0, -9.8, 0.0])
MAX_VELOCITY = 5.0  # 速度上限，防止数值爆炸

# 时间步长
DT = 2e-3

# 隐式欧拉定点迭代次数
IMPLICIT_ITERATIONS = 5

# 积分方法枚举
METHOD_EXPLICIT = 0
METHOD_SEMI_IMPLICIT = 1
METHOD_IMPLICIT = 2
METHOD_NAMES = ["Explicit Euler", "Semi-Implicit Euler", "Implicit Euler (Fixed-Point)"]

# ===================== Taichi 初始化 =====================
# 使用 ti.cpu 保证在无头/远程环境中也能运行
# 本地有显卡时可改为 ti.init(arch=ti.gpu)
ti.init(arch=ti.cpu, kernel_profiler=False, offline_cache=False)

# ===================== 场定义 =====================
# 质点状态
pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
vel = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
force = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
mass = ti.field(dtype=ti.f32, shape=NUM_PARTICLES)
fixed = ti.field(dtype=ti.i32, shape=NUM_PARTICLES)

# 隐式欧拉用的临时存储
old_pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
old_vel = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)

# 弹簧拓扑（结构弹簧：上下左右）
NUM_SPRINGS_H = NUM_ROWS * (NUM_COLS - 1)
NUM_SPRINGS_V = (NUM_ROWS - 1) * NUM_COLS
NUM_SPRINGS = NUM_SPRINGS_H + NUM_SPRINGS_V

spring_p1 = ti.field(dtype=ti.i32, shape=NUM_SPRINGS)
spring_p2 = ti.field(dtype=ti.i32, shape=NUM_SPRINGS)
spring_rest_len = ti.field(dtype=ti.f32, shape=NUM_SPRINGS)

# 渲染用的线段索引
line_indices = ti.field(dtype=ti.i32, shape=NUM_SPRINGS * 2)

# 原子计数器
spring_counter = ti.field(dtype=ti.i32, shape=())


# ===================== 初始化 Kernels =====================
@ti.kernel
def init_particles_kernel(offset_x: ti.f32, offset_y: ti.f32, offset_z: ti.f32):
    """初始化质点位置、速度、质量和固定约束"""
    for i, j in ti.ndrange(NUM_ROWS, NUM_COLS):
        idx = i * NUM_COLS + j
        pos[idx] = ti.Vector([
            j * SPACING - offset_x,
            offset_y,
            i * SPACING - offset_z
        ])
        vel[idx] = ti.Vector([0.0, 0.0, 0.0])
        force[idx] = ti.Vector([0.0, 0.0, 0.0])
        mass[idx] = 1.0
        # 固定第一行的左右两个角点
        if i == 0 and (j == 0 or j == NUM_COLS - 1):
            fixed[idx] = 1
        else:
            fixed[idx] = 0


@ti.kernel
def init_springs_kernel():
    """使用原子操作安全地初始化弹簧拓扑"""
    for i, j in ti.ndrange(NUM_ROWS, NUM_COLS):
        idx = i * NUM_COLS + j

        # 水平弹簧（右邻居）
        if j + 1 < NUM_COLS:
            s = ti.atomic_add(spring_counter[None], 1)
            neighbor = i * NUM_COLS + (j + 1)
            spring_p1[s] = idx
            spring_p2[s] = neighbor
            spring_rest_len[s] = SPACING
            line_indices[s * 2] = idx
            line_indices[s * 2 + 1] = neighbor

        # 垂直弹簧（下邻居）
        if i + 1 < NUM_ROWS:
            s = ti.atomic_add(spring_counter[None], 1)
            neighbor = (i + 1) * NUM_COLS + j
            spring_p1[s] = idx
            spring_p2[s] = neighbor
            spring_rest_len[s] = SPACING
            line_indices[s * 2] = idx
            line_indices[s * 2 + 1] = neighbor


@ti.kernel
def reset_spring_counter():
    """重置弹簧计数器"""
    spring_counter[None] = 0


def init_cloth():
    """Python 侧按顺序调用初始化 Kernels，保证 GPU 状态同步"""
    reset_spring_counter()
    offset_x = (NUM_COLS - 1) * SPACING / 2.0
    offset_z = (NUM_ROWS - 1) * SPACING / 2.0
    init_particles_kernel(offset_x, 1.0, offset_z)
    init_springs_kernel()


# ===================== 物理计算 ti.func =====================
@ti.func
def compute_forces():
    """
    计算每个质点受到的合力（重力 + 阻尼 + 弹簧力）。
    作为 ti.func 会在编译时内联到调用它的 Kernel 中，减少 GPU 函数调用开销。
    """
    # 1. 重置力并施加重力和阻尼
    for i in range(NUM_PARTICLES):
        if fixed[i] == 0:
            force[i] = GRAVITY * mass[i] - KD * vel[i]
        else:
            force[i] = ti.Vector([0.0, 0.0, 0.0])

    # 2. 累加弹簧力（使用 ti.atomic_add 避免多线程写入冲突）
    for s in range(NUM_SPRINGS):
        p1 = spring_p1[s]
        p2 = spring_p2[s]

        dx = pos[p1] - pos[p2]
        dist = dx.norm()

        if dist > 1e-6:
            # 胡克定律: f = -ks * (|x1-x2| - l) * (x1-x2) / |x1-x2|
            f_magnitude = -KS * (dist - spring_rest_len[s])
            f = f_magnitude * (dx / dist)

            ti.atomic_add(force[p1], f)
            ti.atomic_add(force[p2], -f)


@ti.func
def clamp_velocity():
    """
    限制质点最大速度，防止显式欧拉等不稳定方法出现数值爆炸。
    作为 ti.func 内联到 Kernel 中。
    """
    for i in range(NUM_PARTICLES):
        if fixed[i] == 0:
            v = vel[i]
            speed = v.norm()
            if speed > MAX_VELOCITY:
                vel[i] = v * (MAX_VELOCITY / speed)


# ===================== 数值积分 Kernels =====================
@ti.kernel
def step_explicit(dt: ti.f32):
    """
    显式欧拉 (Explicit Euler):
    x_{t+1} = x_t + v_t * dt
    v_{t+1} = v_t + a_t * dt
    """
    compute_forces()

    for i in range(NUM_PARTICLES):
        if fixed[i] == 0:
            acc = force[i] / mass[i]
            vel[i] = vel[i] + acc * dt
            pos[i] = pos[i] + vel[i] * dt

    clamp_velocity()


@ti.kernel
def step_semi_implicit(dt: ti.f32):
    """
    半隐式欧拉 (Semi-Implicit / Symplectic Euler):
    v_{t+1} = v_t + a_t * dt
    x_{t+1} = x_t + v_{t+1} * dt
    先更新速度，再用新速度更新位置。比显式欧拉更稳定，且保持能量近似守恒。
    """
    compute_forces()

    for i in range(NUM_PARTICLES):
        if fixed[i] == 0:
            acc = force[i] / mass[i]
            vel[i] = vel[i] + acc * dt
            pos[i] = pos[i] + vel[i] * dt

    clamp_velocity()


@ti.kernel
def step_implicit_iter(dt: ti.f32, num_iterations: ti.i32):
    """
    隐式欧拉 (Implicit Euler / Backward Euler) - 定点迭代近似:
    v_{t+1} = v_t + a_{t+1} * dt
    x_{t+1} = x_t + v_{t+1} * dt

    使用定点迭代法近似求解隐式方程:
    1. 保存 t 时刻状态
    2. 初始猜测 v^0 = v_t, x^0 = x_t + v_t * dt
    3. 迭代: 基于当前猜测计算力 -> 更新 v 和 x
    """
    # 保存 t 时刻状态
    for i in range(NUM_PARTICLES):
        old_pos[i] = pos[i]
        old_vel[i] = vel[i]

    # 初始猜测：显式一步（作为迭代的起点）
    compute_forces()
    for i in range(NUM_PARTICLES):
        if fixed[i] == 0:
            acc = force[i] / mass[i]
            vel[i] = old_vel[i] + acc * dt
            pos[i] = old_pos[i] + vel[i] * dt

    # 定点迭代
    for _ in range(num_iterations):
        compute_forces()
        for i in range(NUM_PARTICLES):
            if fixed[i] == 0:
                acc = force[i] / mass[i]
                vel[i] = old_vel[i] + acc * dt
                pos[i] = old_pos[i] + vel[i] * dt

    clamp_velocity()


@ti.kernel
def apply_wind_kernel(w: ti.types.vector(3, ti.f32)):
    """施加风力"""
    for i in range(NUM_PARTICLES):
        if fixed[i] == 0:
            force[i] += w


# ===================== 主程序 =====================
def main():
    # 允许 GUI 中修改这些全局参数
    global KS, KD, DT, MAX_VELOCITY

    print("=" * 50)
    print("Work6: Mass-Spring Cloth Simulation")
    print("=" * 50)
    print(f"Grid: {NUM_ROWS}x{NUM_COLS}, Particles: {NUM_PARTICLES}, Springs: {NUM_SPRINGS}")
    print(f"ks={KS}, kd={KD}, dt={DT}, max_vel={MAX_VELOCITY}")
    print("Controls: Button1=Explicit, Button2=Semi-Implicit, Button3=Implicit")
    print("          R=Reset, Space=Pause, Arrow keys=wind")
    print("=" * 50)

    # 初始化布料
    init_cloth()

    # GGUI 窗口
    window = ti.ui.Window("Cloth Simulation - Mass Spring", (1280, 720), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    gui = window.get_gui()
    camera = ti.ui.Camera()

    # 相机初始位置
    camera.position(0.0, 0.8, 1.5)
    camera.lookat(0.0, 0.5, 0.0)
    camera.up(0.0, 1.0, 0.0)

    # 状态变量
    current_method = METHOD_SEMI_IMPLICIT
    paused = False
    frame_count = 0

    # 风力（可通过键盘控制）
    wind_strength = ti.Vector.field(3, dtype=ti.f32, shape=())
    wind_strength[None] = ti.Vector([0.0, 0.0, 0.0])

    while window.running:
        # ---------- GUI 控制面板 ----------
        with gui.sub_window("Control Panel", 0.02, 0.02, 0.28, 0.45):
            gui.text("=" * 20)
            gui.text("Integration Method")
            gui.text("=" * 20)

            if gui.button("1. Explicit Euler"):
                current_method = METHOD_EXPLICIT
                print(f"[Switch] -> Explicit Euler")

            if gui.button("2. Semi-Implicit Euler"):
                current_method = METHOD_SEMI_IMPLICIT
                print(f"[Switch] -> Semi-Implicit Euler")

            if gui.button("3. Implicit Euler (Fixed-Point)"):
                current_method = METHOD_IMPLICIT
                print(f"[Switch] -> Implicit Euler")

            gui.text("")
            gui.text(f"Current: {METHOD_NAMES[current_method]}")
            gui.text(f"Paused: {'Yes' if paused else 'No'}")
            gui.text(f"Frame: {frame_count}")
            gui.text("")

            if gui.button("Pause / Resume"):
                paused = not paused
                print(f"[Pause] -> {paused}")

            if gui.button("Reset Cloth"):
                init_cloth()
                frame_count = 0
                wind_strength[None] = ti.Vector([0.0, 0.0, 0.0])
                print("[Reset] Cloth reset")

            gui.text("")
            gui.text("=" * 20)
            gui.text("Parameters")
            gui.text("=" * 20)

            KS = gui.slider_float("Stiffness (ks)", KS, 1e2, 5e3)
            KD = gui.slider_float("Damping (kd)", KD, 0.0, 5.0)
            DT = gui.slider_float("Time Step (dt)", DT, 1e-4, 1e-2)
            MAX_VELOCITY = gui.slider_float("Max Velocity", MAX_VELOCITY, 1.0, 20.0)

            gui.text("")
            gui.text("Wind: Arrow keys")
            w = wind_strength[None]
            gui.text(f"  Wind: ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})")

        # ---------- 键盘输入 ----------
        if window.is_pressed(ti.ui.SPACE):
            paused = not paused
            print(f"[Pause] -> {paused}")

        if window.is_pressed('r') or window.is_pressed('R'):
            init_cloth()
            frame_count = 0
            wind_strength[None] = ti.Vector([0.0, 0.0, 0.0])
            print("[Reset] Cloth reset")

        # 风力控制
        wind = wind_strength[None]
        if window.is_pressed(ti.ui.UP):
            wind[2] -= 0.5
        if window.is_pressed(ti.ui.DOWN):
            wind[2] += 0.5
        if window.is_pressed(ti.ui.LEFT):
            wind[0] -= 0.5
        if window.is_pressed(ti.ui.RIGHT):
            wind[0] += 0.5
        wind_strength[None] = wind

        # ---------- 物理更新 ----------
        if not paused:
            # 每帧执行多步子步进，提高稳定性
            steps_per_frame = 5
            dt_sub = DT / steps_per_frame

            for _ in range(steps_per_frame):
                if wind.norm() > 0.1:
                    apply_wind_kernel(wind_strength[None])

                if current_method == METHOD_EXPLICIT:
                    step_explicit(dt_sub)
                elif current_method == METHOD_SEMI_IMPLICIT:
                    step_semi_implicit(dt_sub)
                else:
                    step_implicit_iter(dt_sub, IMPLICIT_ITERATIONS)

            frame_count += 1

        # ---------- 渲染 ----------
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # 设置光照
        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.8, 0.8, 0.8))
        scene.point_light(pos=(-0.5, 1.5, -0.5), color=(0.4, 0.4, 0.4))

        # 渲染质点为粉色小球
        scene.particles(pos, radius=0.008, color=(1.0, 0.5, 0.6))

        # 渲染弹簧为淡蓝色线段
        scene.lines(
            vertices=pos,
            width=1.0,
            indices=line_indices,
            color=(0.5, 0.7, 1.0)
        )

        # 绘制地面参考网格
        # 使用 scene.lines 画一个简单地面

        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    main()
