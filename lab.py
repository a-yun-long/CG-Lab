import taichi as ti
import math

# 初始化 Taichi，显式指定 CPU 后端
# 如果你的电脑有显卡且驱动正常，可以改为 ti.gpu 来提速
ti.init(arch=ti.cpu)

# 存储原始顶点（3D）和变换后的屏幕坐标（2D）
vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=3)

@ti.func
def get_model_matrix(angle: ti.f32):
    """
    模型变换：绕 Z 轴旋转
    """
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    """
    视图变换：将相机平移至原点
    """
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """
    透视投影矩阵：从视锥体挤压到标准立方体
    """
    # 按照右手坐标系习惯，看向 -Z 轴
    n = -zNear
    f = -zFar
    
    # 计算视锥体边界
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    r = aspect_ratio * t
    
    # 1. 挤压矩阵 (Perspective -> Orthographic)
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # 2. 正交投影矩阵 (缩放至 [-1, 1])
    # 由于对称性 (l=-r, b=-t)，平移项为0，简化为缩放
    M_ortho = ti.Matrix([
        [1.0/r, 0.0, 0.0, 0.0],
        [0.0, 1.0/t, 0.0, 0.0],
        [0.0, 0.0, 2.0/(n-f), -(n+f)/(n-f)],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle: ti.f32):
    """
    在并行后端计算 MVP 变换
    """
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    
    # 组合 MVP 矩阵
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    mvp = proj @ view @ model
    
    for i in range(3):
        v4 = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
        v_clip = mvp @ v4
        
        # 透视除法：归一化到 NDC 坐标 [-1, 1]
        v_ndc = v_clip / v_clip[3]
        
        # 视口变换：从 NDC [-1, 1] 映射到 GUI 坐标 [0, 1]
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    # 填入实验要求的三角形顶点坐标
    vertices[0] = [2.0, 0.0, -2.0]
    vertices[1] = [0.0, 2.0, -2.0]
    vertices[2] = [-2.0, 0.0, -2.0]
    
    gui = ti.GUI("Zan Lu's CG Lab - MVP Transform", res=(700, 700))
    angle = 0.0
    
    print(">>> 程序已启动！按 A / D 键旋转三角形，按 Esc 退出。")
    
    while gui.running:
        # 处理按键事件
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10.0
            elif gui.event.key == 'd':
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # 调用内核进行计算
        compute_transform(angle)
        
        # 绘制线框三角形
        a, b, c = screen_coords[0], screen_coords[1], screen_coords[2]
        gui.line(a, b, radius=2, color=0xFF0000) # 红
        gui.line(b, c, radius=2, color=0x00FF00) # 绿
        gui.line(c, a, radius=2, color=0x0000FF) # 蓝
        
        gui.show()

if __name__ == '__main__':
    main()