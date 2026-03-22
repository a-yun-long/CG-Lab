import taichi as ti
import math

# 初始化 Taichi
ti.init(arch=ti.cpu)

# ---------------------------------------------------------
# 1. 数据定义
# ---------------------------------------------------------
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)      # 8个顶点
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8) # 8个屏幕坐标

# 将 12 条边分组成三个 field，每组 4 条边，方便上色
edges_back = ti.Vector.field(2, dtype=ti.i32, shape=4)  # 后框 (红色)
edges_front = ti.Vector.field(2, dtype=ti.i32, shape=4) # 前框 (绿色)
edges_side = ti.Vector.field(2, dtype=ti.i32, shape=4)  # 侧边 (蓝色)

# ---------------------------------------------------------
# 2. 矩阵变换逻辑
# ---------------------------------------------------------
@ti.func
def get_model_matrix(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c, s = ti.cos(rad), ti.sin(rad)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    n, f = -zNear, -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    r = aspect_ratio * t
    
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    M_ortho = ti.Matrix([
        [1.0/r, 0.0, 0.0, 0.0],
        [0.0, 1.0/t, 0.0, 0.0],
        [0.0, 0.0, 2.0/(n-f), -(n+f)/(n-f)],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    mvp = get_projection_matrix(45.0, 1.0, 0.1, 50.0) @ get_view_matrix(eye_pos) @ get_model_matrix(angle)
    
    for i in range(8):
        v4 = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

# ---------------------------------------------------------
# 3. 主程序
# ---------------------------------------------------------
def main():
    # 初始化顶点坐标
    v_list = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1], [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
    for i in range(8): vertices[i] = v_list[i]

    # 初始化边的索引
    b_l, f_l, s_l = [[0,1],[1,2],[2,3],[3,0]], [[4,5],[5,6],[6,7],[7,4]], [[0,4],[1,5],[2,6],[3,7]]
    for i in range(4):
        edges_back[i], edges_front[i], edges_side[i] = b_l[i], f_l[i], s_l[i]

    gui = ti.GUI("Zan Lu's Colored Cube", res=(700, 700))
    angle = 0.0

    while gui.running:
        # 按键处理
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a': angle += 5.0
            elif gui.event.key == 'd': angle -= 5.0
            elif gui.event.key == 't': angle = 0.0
            elif gui.event.key == ti.GUI.ESCAPE: gui.running = False
        
        compute_transform(angle)

        # 分组绘制颜色
        for i in range(4): # 红色后框
            gui.line(screen_coords[edges_back[i][0]], screen_coords[edges_back[i][1]], radius=2, color=0xFF0000)
        for i in range(4): # 绿色前框
            gui.line(screen_coords[edges_front[i][0]], screen_coords[edges_front[i][1]], radius=2, color=0x00FF00)
        for i in range(4): # 蓝色侧边
            gui.line(screen_coords[edges_side[i][0]], screen_coords[edges_side[i][1]], radius=2, color=0x0000FF)
            
        gui.show()

if __name__ == '__main__':
    main()