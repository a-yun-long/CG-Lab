"""快速测试基础功能"""
import torch
from mesh_utils import load_obj_simple, normalize_mesh, create_icosphere, Mesh
from renderer import SoftSilhouetteRenderer, get_camera_from_angles

print("Testing mesh loading...")
target_verts, target_faces = load_obj_simple("data/cow_mesh/cow.obj")
print(f"Loaded cow: {target_verts.shape}, {target_faces.shape}")

target_verts, _, _ = normalize_mesh(target_verts)
print(f"Normalized verts range: {target_verts.min():.3f} to {target_verts.max():.3f}")

print("\nTesting icosphere generation...")
source_verts, source_faces = create_icosphere(level=2)
print(f"Icosphere: {source_verts.shape}, {source_faces.shape}")

print("\nTesting mesh class...")
mesh = Mesh(target_verts, target_faces)
print(f"Edges: {mesh.edges().shape}")
print(f"Face normals: {mesh.face_normals().shape}")

print("\nTesting renderer...")
renderer = SoftSilhouetteRenderer(image_size=64, device='cpu')
R, T = get_camera_from_angles(2.7, 0, 180, device='cpu')
sil = renderer.render(mesh, R, T)
print(f"Silhouette shape: {sil.shape}, min={sil.min():.3f}, max={sil.max():.3f}")

# 保存渲染结果查看
import matplotlib.pyplot as plt
plt.imsave('test_silhouette.png', sil.detach().cpu().numpy(), cmap='gray')
print("Saved test_silhouette.png")

# 测试球体渲染
sphere_mesh = Mesh(source_verts, source_faces)
R2, T2 = get_camera_from_angles(2.7, 0, 180, device='cpu')
sil2 = renderer.render(sphere_mesh, R2, T2)
plt.imsave('test_sphere_silhouette.png', sil2.detach().cpu().numpy(), cmap='gray')
print("Saved test_sphere_silhouette.png")

print("\nBasic tests passed!")
