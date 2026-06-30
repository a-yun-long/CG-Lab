"""
Work6 测试脚本：在无头环境中运行布料模拟并保存截图
使用 matplotlib 绘制 3D 结果，不依赖 GGUI 窗口
"""
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入主程序的物理模块（会自动执行 ti.init）
from main import (
    NUM_ROWS, NUM_COLS, NUM_PARTICLES, NUM_SPRINGS, SPACING,
    KS, KD, GRAVITY, MAX_VELOCITY, DT,
    METHOD_EXPLICIT, METHOD_SEMI_IMPLICIT, METHOD_IMPLICIT,
    pos, vel, force, mass, fixed,
    spring_p1, spring_p2, spring_rest_len,
    init_cloth, step_explicit, step_semi_implicit, step_implicit_iter,
    apply_wind_kernel,
)

def get_pos_numpy():
    """将 Taichi field 复制为 numpy 数组"""
    return pos.to_numpy()


def plot_cloth(ax, positions, title, elev=20, azim=45):
    """用 matplotlib 绘制布料 3D 状态"""
    ax.clear()

    # 绘制弹簧线段
    for s in range(NUM_SPRINGS):
        p1 = spring_p1[s]
        p2 = spring_p2[s]
        x = [positions[p1, 0], positions[p2, 0]]
        y = [positions[p1, 1], positions[p2, 1]]
        z = [positions[p1, 2], positions[p2, 2]]
        ax.plot(x, y, z, color='cornflowerblue', linewidth=0.8, alpha=0.8)

    # 绘制质点
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c='hotpink', s=15, depthshade=True)

    # 标记固定点
    fixed_np = fixed.to_numpy()
    fixed_pos = positions[fixed_np == 1]
    if len(fixed_pos) > 0:
        ax.scatter(fixed_pos[:, 0], fixed_pos[:, 1], fixed_pos[:, 2],
                   c='red', s=50, marker='s', label='Fixed')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.8, 1.2])
    ax.set_zlim([-0.6, 0.6])
    ax.view_init(elev=elev, azim=azim)
    if len(fixed_pos) > 0:
        ax.legend()


def run_simulation(method, num_frames, dt, title_prefix, output_prefix, wind=None):
    """运行指定积分方法的模拟并保存截图"""
    init_cloth()

    # 施加风力（可选）
    if wind is not None:
        apply_wind_kernel(wind)

    steps_per_frame = 5
    dt_sub = dt / steps_per_frame

    snapshots = []
    snapshot_frames = [0, num_frames // 4, num_frames // 2, num_frames * 3 // 4, num_frames - 1]

    for frame in range(num_frames):
        for _ in range(steps_per_frame):
            if method == METHOD_EXPLICIT:
                step_explicit(dt_sub)
            elif method == METHOD_SEMI_IMPLICIT:
                step_semi_implicit(dt_sub)
            else:
                step_implicit_iter(dt_sub, 5)

        if frame in snapshot_frames:
            snapshots.append((frame, get_pos_numpy()))

    # 绘制并保存
    fig = plt.figure(figsize=(18, 4))
    for idx, (frame, positions) in enumerate(snapshots):
        ax = fig.add_subplot(1, 5, idx + 1, projection='3d')
        plot_cloth(ax, positions, f'{title_prefix}\nFrame {frame}')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_sequence.png', dpi=150)
    plt.close()
    print(f"Saved {output_prefix}_sequence.png")

    # 保存最终状态单独大图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    final_pos = snapshots[-1][1]
    plot_cloth(ax, final_pos, f'{title_prefix} - Final State', elev=25, azim=60)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_final.png', dpi=150)
    plt.close()
    print(f"Saved {output_prefix}_final.png")


def main():
    import os
    os.makedirs('Work6/images', exist_ok=True)

    print("=" * 50)
    print("Work6 Simulation Test (Headless Mode)")
    print("=" * 50)

    # 1. 初始状态
    init_cloth()
    init_pos = get_pos_numpy()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_cloth(ax, init_pos, 'Initial Cloth State', elev=25, azim=45)
    plt.tight_layout()
    plt.savefig('Work6/images/initial_state.png', dpi=150)
    plt.close()
    print("Saved images/initial_state.png")

    # 2. 三种积分方法对比（无风，短时间）
    print("\nRunning Explicit Euler...")
    run_simulation(METHOD_EXPLICIT, 120, DT,
                   'Explicit Euler', 'Work6/images/explicit')

    print("\nRunning Semi-Implicit Euler...")
    run_simulation(METHOD_SEMI_IMPLICIT, 120, DT,
                   'Semi-Implicit Euler', 'Work6/images/semi_implicit')

    print("\nRunning Implicit Euler...")
    run_simulation(METHOD_IMPLICIT, 120, DT,
                   'Implicit Euler', 'Work6/images/implicit')

    # 3. 大风力下的对比（展示稳定性差异）
    print("\nRunning with strong wind ( Explicit vs Semi-Implicit )...")
    wind = ti.Vector([3.0, -1.0, 2.0])

    run_simulation(METHOD_EXPLICIT, 100, DT,
                   'Explicit + Wind', 'Work6/images/explicit_wind', wind=wind)
    run_simulation(METHOD_SEMI_IMPLICIT, 100, DT,
                   'Semi-Implicit + Wind', 'Work6/images/semi_implicit_wind', wind=wind)

    # 4. 高阻尼 vs 低阻尼对比（半隐式）
    print("\nRunning damping comparison...")
    global KD

    init_cloth()
    KD = 0.1
    for frame in range(80):
        for _ in range(5):
            step_semi_implicit(DT / 5)
    low_damp = get_pos_numpy()

    init_cloth()
    KD = 5.0
    for frame in range(80):
        for _ in range(5):
            step_semi_implicit(DT / 5)
    high_damp = get_pos_numpy()

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    plot_cloth(ax1, low_damp, 'Low Damping (kd=0.1)', elev=25, azim=60)
    ax2 = fig.add_subplot(122, projection='3d')
    plot_cloth(ax2, high_damp, 'High Damping (kd=5.0)', elev=25, azim=60)
    plt.tight_layout()
    plt.savefig('Work6/images/damping_comparison.png', dpi=150)
    plt.close()
    print("Saved Work6/images/damping_comparison.png")

    # 恢复默认阻尼
    KD = 0.5

    print("\nAll test images saved to Work6/images/")
    print("Run main.py for interactive GGUI version.")


if __name__ == '__main__':
    main()
