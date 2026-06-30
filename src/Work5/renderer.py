"""
软光栅化渲染器：纯 PyTorch 实现
"""
import torch
import torch.nn.functional as F
import math


def look_at(eye, at, up):
    """
    构造 look-at 视图矩阵 (world -> camera)
    eye: 相机位置 (3,)
    at: 目标位置 (3,)
    up: 上方向 (3,)
    return: (4, 4) 变换矩阵
    """
    z = F.normalize(at - eye, dim=0)
    x = F.normalize(torch.cross(z, up, dim=0), dim=0)
    y = torch.cross(x, z, dim=0)

    # 注意 z 轴指向相机后方，所以这里取 -z
    R = torch.stack([x, y, -z], dim=0)
    t = -R @ eye

    W2C = torch.eye(4, device=eye.device, dtype=eye.dtype)
    W2C[:3, :3] = R
    W2C[:3, 3] = t
    return W2C


def perspective_projection(fov=60.0, aspect=1.0, near=0.1, far=100.0):
    """
    构造 OpenGL 风格的透视投影矩阵
    """
    fov_rad = fov * math.pi / 180.0
    f = 1.0 / math.tan(fov_rad / 2.0)

    P = torch.zeros(4, 4, dtype=torch.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = (2 * far * near) / (near - far)
    P[3, 2] = -1.0
    return P


def get_camera_from_angles(distance, elevation, azimuth, device='cpu'):
    """
    根据球坐标角度生成相机外参
    elevation: 仰角（度）
    azimuth: 方位角（度）
    return: (R, T) 相机旋转和平移
    """
    elev_rad = elevation * math.pi / 180.0
    azim_rad = azimuth * math.pi / 180.0

    # 相机位置（球坐标转直角坐标）
    x = distance * math.cos(elev_rad) * math.sin(azim_rad)
    y = distance * math.sin(elev_rad)
    z = distance * math.cos(elev_rad) * math.cos(azim_rad)

    eye = torch.tensor([x, y, z], dtype=torch.float32, device=device)
    at = torch.zeros(3, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)

    W2C = look_at(eye, at, up)
    R = W2C[:3, :3].unsqueeze(0)
    T = W2C[:3, 3].unsqueeze(0)
    return R, T


def transform_points(points, R, T):
    """
    points: (N, 3)
    R: (1, 3, 3) 或 (3, 3)
    T: (1, 3) 或 (3,)
    return: (N, 3) 变换后的点
    """
    if R.dim() == 3:
        R = R.squeeze(0)
    if T.dim() == 2:
        T = T.squeeze(0)
    return (R @ points.T).T + T


def edge_function(v0, v1, p):
    """
    计算边函数（叉积的 z 分量），用于判断点 p 在边 v0->v1 的哪一侧
    v0, v1, p: (..., 2)
    return: (...)
    """
    return (p[..., 0] - v0[..., 0]) * (v1[..., 1] - v0[..., 1]) - \
           (p[..., 1] - v0[..., 1]) * (v1[..., 0] - v0[..., 0])


class SoftSilhouetteRenderer:
    """
    软剪影渲染器
    """
    def __init__(self, image_size=128, fov=60.0, sigma=1e-4, tau=0.5, device='cpu'):
        self.image_size = image_size
        self.fov = fov
        self.sigma = sigma  # 边缘模糊程度
        self.tau = tau      # z-buffer 锐度
        self.device = device

        # 创建像素坐标网格
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, image_size, device=device),
            torch.linspace(-1, 1, image_size, device=device),
            indexing='ij'
        )
        self.pixel_coords = torch.stack([x, y], dim=-1)  # (H, W, 2)

    def render(self, mesh, R, T):
        """
        渲染网格的剪影图
        mesh: Mesh 对象
        R: (1, 3, 3)
        T: (1, 3)
        return: (H, W) 剪影图，值域 [0, 1]
        """
        verts = mesh.verts
        faces = mesh.faces
        num_faces = faces.shape[0]

        # 1. 相机变换
        verts_cam = transform_points(verts, R, T)

        # 2. 透视投影到 NDC
        P = perspective_projection(self.fov, 1.0, 0.1, 100.0).to(self.device)
        # 齐次坐标
        verts_hom = torch.cat([verts_cam, torch.ones(verts_cam.shape[0], 1, device=self.device)], dim=-1)
        verts_ndc_hom = (P @ verts_hom.T).T
        # 透视除法
        verts_ndc = verts_ndc_hom[:, :3] / (verts_ndc_hom[:, 3:4] + 1e-8)

        # 3. 屏幕空间坐标（NDC [-1,1] 映射到像素坐标 [-1,1] 保持一致）
        verts_screen = verts_ndc[:, :2]  # (V, 2)
        verts_depth = verts_ndc[:, 2]    # (V,)

        # 初始化累加器
        color_sum = torch.zeros(self.image_size, self.image_size, device=self.device)
        weight_sum = torch.zeros(self.image_size, self.image_size, device=self.device)

        # 4. 逐个三角形处理（为了内存效率，每次处理一个三角形）
        for f_idx in range(num_faces):
            face = faces[f_idx]
            v0 = verts_screen[face[0]]
            v1 = verts_screen[face[1]]
            v2 = verts_screen[face[2]]

            # 背面剔除：如果三角形在屏幕空间面积接近0或背面，跳过
            area = edge_function(v0.unsqueeze(0), v1.unsqueeze(0), v2.unsqueeze(0)).item()
            if area <= 0:
                continue

            # 计算包围盒（NDC空间 [-1,1]）
            min_xy = torch.min(torch.min(v0, v1), v2)
            max_xy = torch.max(torch.max(v0, v1), v2)

            # 映射到像素索引范围
            min_px = ((min_xy + 1.0) / 2.0 * self.image_size).long()
            max_px = ((max_xy + 1.0) / 2.0 * self.image_size).long()

            # 限制范围并增加一点padding
            min_px = torch.clamp(min_px - 2, 0, self.image_size - 1)
            max_px = torch.clamp(max_px + 2, 0, self.image_size - 1)

            if min_px[0] >= max_px[0] or min_px[1] >= max_px[1]:
                continue

            # 提取该区域的像素坐标
            local_pixels = self.pixel_coords[min_px[1]:max_px[1]+1, min_px[0]:max_px[0]+1]

            # 计算边函数（有符号面积的两倍）
            e0 = edge_function(v1, v2, local_pixels)
            e1 = edge_function(v2, v0, local_pixels)
            e2 = edge_function(v0, v1, local_pixels)

            # 软覆盖率：使用 Sigmoid
            # 在三角形内部时边函数>0，外部<0
            # 我们除以三角形面积进行归一化，使得边缘过渡与三角形大小无关
            area = area + 1e-8
            w0 = torch.sigmoid(e0 / (self.sigma * area))
            w1 = torch.sigmoid(e1 / (self.sigma * area))
            w2 = torch.sigmoid(e2 / (self.sigma * area))
            coverage = w0 * w1 * w2

            # 深度：使用重心坐标插值
            # 归一化边函数作为重心坐标
            b0 = e0 / area
            b1 = e1 / area
            b2 = e2 / area
            # clamp 到合理范围用于深度计算
            b0 = torch.clamp(b0, 0, 1)
            b1 = torch.clamp(b1, 0, 1)
            b2 = torch.clamp(b2, 0, 1)
            # 重新归一化
            sum_b = b0 + b1 + b2 + 1e-8
            b0, b1, b2 = b0/sum_b, b1/sum_b, b2/sum_b

            d0 = verts_depth[face[0]]
            d1 = verts_depth[face[1]]
            d2 = verts_depth[face[2]]
            depth = b0 * d0 + b1 * d1 + b2 * d2

            # Soft z-buffer 权重：越近权重越大
            # 使用 exp(-depth / tau)，但 depth 在 NDC 中是 [-1,1]
            # 我们需要将其映射到正数范围
            z_weight = torch.exp(-depth / self.tau)

            # 累加：weight_sum 也乘以 coverage，避免被不相关的三角形稀释
            contrib = coverage * z_weight
            color_sum[min_px[1]:max_px[1]+1, min_px[0]:max_px[0]+1] += contrib
            weight_sum[min_px[1]:max_px[1]+1, min_px[0]:max_px[0]+1] += coverage * z_weight

        # 归一化
        silhouette = color_sum / (weight_sum + 1e-8)
        # clamp 到 [0,1]
        silhouette = torch.clamp(silhouette, 0, 1)
        return silhouette

    def render_batch(self, mesh, cameras):
        """
        批量渲染多个视角
        cameras: list of (R, T) tuples
        return: (N, H, W) 剪影图批次
        """
        images = []
        for R, T in cameras:
            img = self.render(mesh, R, T)
            images.append(img)
        return torch.stack(images, dim=0)
