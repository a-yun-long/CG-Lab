"""快速生成 README 展示图片（只渲染，不优化）"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from mesh_utils import load_obj_simple, normalize_mesh, create_icosphere, Mesh
from renderer import SoftSilhouetteRenderer, get_camera_from_angles

os.makedirs('images', exist_ok=True)
device = 'cpu'

# 加载目标网格
target_verts, target_faces = load_obj_simple("data/cow_mesh/cow.obj")
target_verts, _, _ = normalize_mesh(target_verts)
target_mesh = Mesh(target_verts, target_faces, device=device)

# 初始化源球体
source_verts, source_faces = create_icosphere(level=2)
source_mesh = Mesh(source_verts, source_faces, device=device)

# 渲染器
renderer = SoftSilhouetteRenderer(image_size=128, device=device)

# 1. 保存目标剪影多视角
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
views = [(0, 180), (10, 140), (20, 100), (0, 220), (-10, 260), (15, 160), (-15, 200), (5, 120)]
for idx, (elev, azim) in enumerate(views):
    R, T = get_camera_from_angles(2.7, elev, azim, device=device)
    sil = renderer.render(target_mesh, R, T)
    ax = axes[idx // 4, idx % 4]
    ax.imshow(sil.cpu().numpy(), cmap='gray')
    ax.set_title(f'elev={elev}, azim={azim}')
    ax.axis('off')
plt.suptitle('Target Cow Silhouettes (Multiple Views)', fontsize=14)
plt.tight_layout()
plt.savefig('images/target_silhouettes.png', dpi=150)
plt.close()
print("Saved images/target_silhouettes.png")

# 2. 保存初始球体剪影
R, T = get_camera_from_angles(2.7, 0, 180, device=device)
sphere_sil = renderer.render(source_mesh, R, T)
plt.figure(figsize=(5, 5))
plt.imshow(sphere_sil.cpu().numpy(), cmap='gray')
plt.title('Initial Sphere Silhouette')
plt.axis('off')
plt.tight_layout()
plt.savefig('images/sphere_silhouette.png', dpi=150)
plt.close()
print("Saved images/sphere_silhouette.png")

# 3. 保存球体 vs 奶牛对比
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
cow_sil = renderer.render(target_mesh, R, T)
axes[0].imshow(sphere_sil.cpu().numpy(), cmap='gray')
axes[0].set_title('Initial Sphere')
axes[1].imshow(cow_sil.cpu().numpy(), cmap='gray')
axes[1].set_title('Target Cow')
diff = np.abs(sphere_sil.cpu().numpy() - cow_sil.cpu().numpy())
axes[2].imshow(diff, cmap='hot')
axes[2].set_title('Difference')
for ax in axes:
    ax.axis('off')
plt.suptitle('Shape Optimization Goal', fontsize=14)
plt.tight_layout()
plt.savefig('images/sphere_vs_cow.png', dpi=150)
plt.close()
print("Saved images/sphere_vs_cow.png")

# 4. 软光栅化边界效果展示（放大局部）
renderer_detail = SoftSilhouetteRenderer(image_size=256, sigma=5e-5, device=device)
cow_sil_detail = renderer_detail.render(target_mesh, R, T)
# 裁剪局部
local = cow_sil_detail[100:180, 80:160].cpu().numpy()
plt.figure(figsize=(6, 6))
plt.imshow(local, cmap='gray')
plt.title('Soft Rasterization Boundary (Zoomed)')
plt.axis('off')
plt.tight_layout()
plt.savefig('images/soft_boundary.png', dpi=150)
plt.close()
print("Saved images/soft_boundary.png")

# 5. 损失曲线示意图（模拟数据展示概念）
fig, ax = plt.subplots(figsize=(8, 5))
iters = np.arange(300)
loss_sil = 0.4 * np.exp(-iters / 80) + 0.02
loss_lap = 0.05 + 0.02 * np.sin(iters / 20) * np.exp(-iters / 150)
loss_edge = 0.08 * np.exp(-iters / 100) + 0.01
ax.plot(iters, loss_sil, label='Silhouette Loss', linewidth=2)
ax.plot(iters, loss_lap, label='Laplacian Loss', linewidth=2)
ax.plot(iters, loss_edge, label='Edge Loss', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Typical Loss Curves During Optimization')
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('images/loss_curve_demo.png', dpi=150)
plt.close()
print("Saved images/loss_curve_demo.png")

print("\nAll README images generated in images/")
