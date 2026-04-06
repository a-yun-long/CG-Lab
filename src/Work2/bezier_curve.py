import taichi as ti
import numpy as np

# --- 任务 1: 初始化与显存预分配 ---
ti.init(arch=ti.gpu)

RES = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

# 像素缓冲区 (Frame Buffer)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))
# 曲线坐标缓冲区 (CPU -> GPU 的桥梁)
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
# 控制点显示缓冲区 (对象池)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
# 索引缓冲区 (用于连接折线)
line_indices_field = ti.field(dtype=ti.i32, shape=2 * (MAX_CONTROL_POINTS - 1))

# --- 任务 2: 实现 De Casteljau 算法 ---
def de_casteljau(points, t):
    new_points = points
    while len(new_points) > 1:
        next_iter = []
        for i in range(len(new_points) - 1):
            # 线性插值公式: (1-t)P_i + tP_{i+1}
            p = (1.0 - t) * new_points[i] + t * new_points[i+1]
            next_iter.append(p)
        new_points = next_iter
    return new_points[0]

# --- 任务 3: 编写 GPU 绘制内核 ---
@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        pos = curve_points_field[i]
        x = ti.cast(pos[0] * RES, ti.i32)
        y = ti.cast(pos[1] * RES, ti.i32)
        if 0 <= x < RES and 0 <= y < RES:
            pixels[x, y] = ti.Vector([0.0, 1.0, 0.0]) # 绿色

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

# --- 任务 4 & 5: 主循环与交互 ---
def main():
    window = ti.ui.Window("Bézier Curve - De Casteljau", (RES, RES))
    canvas = window.get_canvas()
    control_points = []

    while window.running:
        # 1. 处理输入事件
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.LMB: # 鼠标左键添加点
                if len(control_points) < MAX_CONTROL_POINTS:
                    curr_mouse_pos = window.get_cursor_pos()
                    control_points.append(np.array([curr_mouse_pos[0], curr_mouse_pos[1]], dtype=np.float32))
            elif window.event.key == 'c': # 键盘 C 清空
                control_points = []
        
        # 2. 逻辑计算与 GPU 数据准备
        clear_pixels()

        if len(control_points) >= 2:
            # A. 计算贝塞尔曲线采样点
            samples = []
            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                p = de_casteljau(control_points, t)
                samples.append(p)
            curve_points_field.from_numpy(np.array(samples, dtype=np.float32))
            
            # B. 准备折线索引 (用于绘制控制多边形)
            indices_list = []
            for i in range(len(control_points) - 1):
                indices_list.append(i)
                indices_list.append(i + 1)
            indices_np = np.zeros(2 * (MAX_CONTROL_POINTS - 1), dtype=np.int32)
            indices_np[:len(indices_list)] = indices_list
            line_indices_field.from_numpy(indices_np)
            
            # C. 调用内核点亮像素
            draw_curve_kernel(NUM_SEGMENTS + 1)

        # 3. 更新 UI 对象池 (控制点红点)
        display_pts = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
        for i in range(len(control_points)):
            display_pts[i] = control_points[i]
        gui_points.from_numpy(display_pts)

        # 4. 最终画面渲染 (注意层级顺序)
        canvas.set_image(pixels) # 最底层：绿色曲线
        
        if len(control_points) > 0:
            # 绘制原始折线（控制多边形）
            if len(control_points) >= 2:
                canvas.lines(gui_points, width=0.0015, indices=line_indices_field, color=(0.4, 0.4, 0.4))
            
            # 绘制控制点（红色圆圈）
            canvas.circles(gui_points, radius=0.01, color=(1.0, 0.0, 0.0))

        window.show()

if __name__ == "__main__":
    main()