import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=float, shape=(res_x, res_y))

light_pos = ti.Vector.field(3, dtype=float, shape=())
max_bounces = ti.field(dtype=int, shape=())

light_pos[None] = [0.0, 5.0, 0.0]
max_bounces[None] = 4 # 玻璃折射通常需要更多的弹射次数才能看清内部

@ti.func
def intersect_sphere(pos, d, center, radius):
    oc = pos - center
    a = tm.dot(d, d)
    b = 2.0 * tm.dot(oc, d)
    c = tm.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    
    hit_dist = 1e10
    if discriminant > 0:
        sqrtd = tm.sqrt(discriminant)
        t1 = (-b - sqrtd) / (2.0 * a)
        t2 = (-b + sqrtd) / (2.0 * a)
        if t1 > 1e-4:
            hit_dist = t1
        elif t2 > 1e-4:
            hit_dist = t2
    return hit_dist

@ti.func
def intersect_plane(pos, d, plane_y):
    hit_dist = 1e10
    if abs(d.y) > 1e-4:
        t = (plane_y - pos.y) / d.y
        if t > 1e-4:
            hit_dist = t
    return hit_dist

@ti.func
def scene_intersect(pos, d):
    """
    材质 ID 规则: 0=无, 1=平面(漫反射), 2=左球(玻璃折射), 3=右球(镜面反射)
    """
    closest_dist = 1e10
    hit_mat_id = 0 
    hit_pos = tm.vec3(0.0)
    hit_normal = tm.vec3(0.0)
    
    # 1. 击中无限大平面
    t_plane = intersect_plane(pos, d, -1.0)
    if t_plane < closest_dist:
        closest_dist = t_plane
        hit_mat_id = 1
        hit_pos = pos + d * t_plane
        hit_normal = tm.vec3(0.0, 1.0, 0.0)
        
    # 2. 击中左侧玻璃球 (原为红球)
    center_glass = tm.vec3(-1.5, 0.0, 0.0)
    t_glass = intersect_sphere(pos, d, center_glass, 1.0)
    if t_glass < closest_dist:
        closest_dist = t_glass
        hit_mat_id = 2
        hit_pos = pos + d * t_glass
        hit_normal = tm.normalize(hit_pos - center_glass)
        
    # 3. 击中右侧银色镜面球
    center_silver = tm.vec3(1.5, 0.0, 0.0)
    t_silver = intersect_sphere(pos, d, center_silver, 1.0)
    if t_silver < closest_dist:
        closest_dist = t_silver
        hit_mat_id = 3
        hit_pos = pos + d * t_silver
        hit_normal = tm.normalize(hit_pos - center_silver)
        
    return closest_dist, hit_mat_id, hit_pos, hit_normal

@ti.kernel
def render():
    for i, j in pixels:
        final_pixel_color = tm.vec3(0.0)
        
        # 选做内容 2：MSAA 抗锯齿，像素内随机多次采样
        samples = 4
        for s in range(samples):
            u = (i + ti.random()) / res_x * 2.0 - 1.0
            v = (j + ti.random()) / res_y * 2.0 - 1.0
            u *= float(res_x) / float(res_y)
            
            ro = tm.vec3(0.0, 1.0, 5.0) 
            rd = tm.normalize(tm.vec3(u, v, -1.0)) 
            
            throughput = tm.vec3(1.0) 
            final_color = tm.vec3(0.0)
            
            for bounce in range(max_bounces[None]):
                dist, mat_id, hit_pos, hit_normal = scene_intersect(ro, rd)
                
                if mat_id == 0: # 背景
                    final_color += throughput * tm.vec3(0.05, 0.05, 0.05) 
                    break
                    
                if mat_id == 1: # 漫反射平面
                    ix = tm.floor(hit_pos.x)
                    iz = tm.floor(hit_pos.z)
                    albedo = tm.vec3(0.8) if int(ix + iz) % 2 == 0 else tm.vec3(0.2)
                        
                    light_dir = tm.normalize(light_pos[None] - hit_pos)
                    light_dist = tm.length(light_pos[None] - hit_pos)
                    
                    shadow_ro = hit_pos + hit_normal * 1e-4
                    s_dist, s_mat, _, _ = scene_intersect(shadow_ro, light_dir)
                    
                    in_shadow = 0.0
                    if s_mat != 0 and s_dist < light_dist:
                        in_shadow = 1.0
                        
                    n_dot_l = tm.max(tm.dot(hit_normal, light_dir), 0.0)
                    ambient = 0.1 * albedo
                    diffuse = albedo * n_dot_l * (1.0 - in_shadow)
                    
                    final_color += throughput * (ambient + diffuse)
                    break 
                    
                elif mat_id == 2: # 选做内容 1：玻璃材质与折射
                    ior = 1.5 # 玻璃的折射率
                    cos_theta_i = tm.dot(rd, hit_normal)
                    
                    outward_normal = hit_normal
                    eta = 1.0 / ior # 默认从空气进入玻璃
                    
                    # 判断是从外面射入，还是从里面射出
                    if cos_theta_i > 0: 
                        outward_normal = -hit_normal # 光在玻璃内部，法线翻转
                        eta = ior / 1.0 # 从玻璃回到空气
                    else:
                        cos_theta_i = -cos_theta_i
                    
                    # 斯涅尔定律计算判别式
                    sin2_theta_t = eta * eta * (1.0 - cos_theta_i * cos_theta_i)
                    
                    if sin2_theta_t > 1.0:
                        # 判别式大于 1.0，发生【全反射】 (TIR)
                        rd = rd - 2.0 * tm.dot(rd, outward_normal) * outward_normal
                        ro = hit_pos + outward_normal * 1e-4 # 沿着反射方向偏移避免自相交
                    else:
                        # 正常【折射】
                        cos_theta_t = tm.sqrt(1.0 - sin2_theta_t)
                        rd = eta * rd + (eta * cos_theta_i - cos_theta_t) * outward_normal
                        # ⚠️ 核心避坑：光线穿透了物体，偏移量必须是反向法线方向，否则会卡在表面
                        ro = hit_pos - outward_normal * 1e-4 
                        throughput *= 0.9 # 玻璃吸收一点光线
                        
                elif mat_id == 3: # 镜面材质
                    ro = hit_pos + hit_normal * 1e-4 
                    rd = rd - 2.0 * tm.dot(rd, hit_normal) * hit_normal 
                    throughput *= 0.8 
                    
            final_pixel_color += final_color
            
        pixels[i, j] = final_pixel_color / float(samples)

def main():
    window = ti.ui.Window("Ray Tracer - Glass & MSAA", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    lx, ly, lz = 0.0, 5.0, 0.0
    mb = 4
    
    while window.running:
        gui.begin("Settings", 0.02, 0.02, 0.35, 0.25)
        gui.text("Light Position")
        lx = gui.slider_float("Light X", lx, -5.0, 5.0)
        ly = gui.slider_float("Light Y", ly,  1.0, 10.0)
        lz = gui.slider_float("Light Z", lz, -5.0, 5.0)
        light_pos[None] = [lx, ly, lz]
        
        gui.text("Ray Tracing Options")
        mb = gui.slider_int("Max Bounces", mb, 1, 6)
        max_bounces[None] = mb
        gui.end()
        
        render()
        canvas.set_image(pixels)
        window.show()

if __name__ == '__main__':
    main()