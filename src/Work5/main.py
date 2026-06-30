"""
可微网格优化主程序：将球体通过软光栅化优化成奶牛形状
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio

from mesh_utils import load_obj_simple, normalize_mesh, create_icosphere, Mesh
from renderer import SoftSilhouetteRenderer, get_camera_from_angles
from losses import (
    silhouette_loss,
    mesh_laplacian_smoothing,
    mesh_edge_loss,
    mesh_normal_consistency,
)


def generate_cameras(num_views=20, distance=2.7, elev_range=30, azim_range=180, device='cpu'):
    """
    生成多个视角的相机参数
    """
    cameras = []
    elevs = torch.linspace(-elev_range, elev_range, num_views)
    azims = torch.linspace(-azim_range, azim_range, num_views) + 180.0
    for i in range(num_views):
        R, T = get_camera_from_angles(distance, elevs[i].item(), azims[i].item(), device=device)
        cameras.append((R, T))
    return cameras


def save_silhouette_image(silhouette, filepath):
    """保存剪影图为灰度图像"""
    img = (silhouette.detach().cpu().numpy() * 255).astype(np.uint8)
    plt.imsave(filepath, img, cmap='gray')


def save_mesh_obj(mesh, filepath):
    """保存网格为 OBJ 文件"""
    verts = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    with open(filepath, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def optimize_mesh(
    target_mesh,
    source_mesh,
    cameras,
    renderer,
    num_iters=2000,
    lr=0.01,
    w_lap=0.1,
    w_edge=0.1,
    w_normal=0.01,
    save_interval=100,
    output_dir='./output',
    device='cpu'
):
    """
    执行网格优化
    """
    os.makedirs(output_dir, exist_ok=True)

    # 预渲染目标剪影
    print("预渲染目标剪影...")
    target_silhouettes = []
    for R, T in tqdm(cameras):
        sil = renderer.render(target_mesh, R, T)
        target_silhouettes.append(sil)
    target_silhouettes = torch.stack(target_silhouettes, dim=0).to(device)

    # 保存目标剪影参考图
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for idx, ax in enumerate(axes.flat):
        if idx < len(target_silhouettes):
            ax.imshow(target_silhouettes[idx].cpu().numpy(), cmap='gray')
            ax.set_title(f"View {idx}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_silhouettes.png'))
    plt.close()

    # 将源网格的顶点偏移量设为可优化参数
    deform_verts = torch.zeros_like(source_mesh.verts, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([deform_verts], lr=lr)

    # 记录初始边长目标（用于边长正则化）
    with torch.no_grad():
        temp_mesh = Mesh(source_mesh.verts + deform_verts, source_mesh.faces, device=device)
        target_edge_length = None  # 使用动态平均边长

    # 优化循环
    print(f"开始优化，共 {num_iters} 轮...")
    loss_history = {
        'total': [],
        'silhouette': [],
        'laplacian': [],
        'edge': [],
        'normal': []
    }

    frames = []

    for iter_idx in tqdm(range(num_iters)):
        optimizer.zero_grad()

        # 当前形变后的网格
        current_verts = source_mesh.verts + deform_verts
        current_mesh = Mesh(current_verts, source_mesh.faces, device=device)

        # 渲染当前剪影
        pred_silhouettes = []
        for R, T in cameras:
            sil = renderer.render(current_mesh, R, T)
            pred_silhouettes.append(sil)
        pred_silhouettes = torch.stack(pred_silhouettes, dim=0)

        # 计算损失
        loss_sil = silhouette_loss(pred_silhouettes, target_silhouettes)
        loss_lap = mesh_laplacian_smoothing(current_mesh)
        loss_edge = mesh_edge_loss(current_mesh, target_length=target_edge_length)
        loss_normal = mesh_normal_consistency(current_mesh)

        loss_total = loss_sil + w_lap * loss_lap + w_edge * loss_edge + w_normal * loss_normal

        # 反向传播
        loss_total.backward()
        optimizer.step()

        # 记录
        loss_history['total'].append(loss_total.item())
        loss_history['silhouette'].append(loss_sil.item())
        loss_history['laplacian'].append(loss_lap.item())
        loss_history['edge'].append(loss_edge.item())
        loss_history['normal'].append(loss_normal.item())

        # 定期保存
        if iter_idx % save_interval == 0 or iter_idx == num_iters - 1:
            print(f"\nIter {iter_idx}: total={loss_total.item():.6f}, "
                  f"sil={loss_sil.item():.6f}, lap={loss_lap.item():.6f}, "
                  f"edge={loss_edge.item():.6f}, normal={loss_normal.item():.6f}")

            # 保存当前网格
            save_mesh_obj(current_mesh, os.path.join(output_dir, f'mesh_iter_{iter_idx:04d}.obj'))

            # 保存当前某个视角的剪影对比图
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            with torch.no_grad():
                view_idx = len(cameras) // 2
                pred_view = pred_silhouettes[view_idx].detach().cpu().numpy()
                target_view = target_silhouettes[view_idx].detach().cpu().numpy()
                axes[0].imshow(pred_view, cmap='gray')
                axes[0].set_title(f'Predicted (iter {iter_idx})')
                axes[1].imshow(target_view, cmap='gray')
                axes[1].set_title('Target')
                diff = np.abs(pred_view - target_view)
                axes[2].imshow(diff, cmap='hot')
                axes[2].set_title(f'Difference (max={diff.max():.3f})')
                for ax in axes:
                    ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'compare_iter_{iter_idx:04d}.png'))
            plt.close()

            # 收集帧用于GIF
            frames.append((pred_view * 255).astype(np.uint8))

    # 保存优化过程GIF
    if len(frames) > 1:
        imageio.mimsave(os.path.join(output_dir, 'optimization_process.gif'), frames, duration=0.3)

    # 保存损失曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, values in loss_history.items():
        ax.plot(values, label=key)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Optimization Loss Curve')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # 保存最终网格
    save_mesh_obj(current_mesh, os.path.join(output_dir, 'final_mesh.obj'))

    print(f"优化完成！结果保存在 {output_dir}")
    return current_mesh


def main():
    parser = argparse.ArgumentParser(description='Differentiable Mesh Optimization')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--image_size', type=int, default=64, help='Rendered image size')
    parser.add_argument('--num_views', type=int, default=12, help='Number of camera views')
    parser.add_argument('--num_iters', type=int, default=300, help='Optimization iterations')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--w_lap', type=float, default=0.2, help='Laplacian smoothing weight')
    parser.add_argument('--w_edge', type=float, default=0.1, help='Edge length weight')
    parser.add_argument('--w_normal', type=float, default=0.01, help='Normal consistency weight')
    parser.add_argument('--sigma', type=float, default=1e-4, help='Soft rasterization edge sigma')
    parser.add_argument('--sphere_level', type=int, default=2, help='Icosphere subdivision level')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--data_dir', type=str, default='./data/cow_mesh', help='Data directory')
    parser.add_argument('--save_interval', type=int, default=30, help='Save interval')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载目标网格
    cow_obj_path = os.path.join(args.data_dir, 'cow.obj')
    if not os.path.exists(cow_obj_path):
        print(f"错误：找不到目标模型 {cow_obj_path}")
        print("请确保已下载 cow.obj 到 data/cow_mesh/ 目录")
        return

    print("加载目标网格...")
    target_verts, target_faces = load_obj_simple(cow_obj_path)
    target_verts, _, _ = normalize_mesh(target_verts)
    target_mesh = Mesh(target_verts, target_faces, device=device)
    print(f"目标网格: {target_mesh.num_verts} 顶点, {target_mesh.num_faces} 面")

    # 2. 初始化源网格（球体）
    print("初始化源球体...")
    source_verts, source_faces = create_icosphere(level=args.sphere_level)
    source_mesh = Mesh(source_verts, source_faces, device=device)
    print(f"源网格: {source_mesh.num_verts} 顶点, {source_mesh.num_faces} 面")

    # 3. 生成多视角相机
    print(f"生成 {args.num_views} 个相机视角...")
    cameras = generate_cameras(num_views=args.num_views, device=device)

    # 4. 初始化渲染器
    print("初始化软光栅化渲染器...")
    renderer = SoftSilhouetteRenderer(
        image_size=args.image_size,
        sigma=args.sigma,
        device=device
    )

    # 5. 执行优化
    optimize_mesh(
        target_mesh=target_mesh,
        source_mesh=source_mesh,
        cameras=cameras,
        renderer=renderer,
        num_iters=args.num_iters,
        lr=args.lr,
        w_lap=args.w_lap,
        w_edge=args.w_edge,
        w_normal=args.w_normal,
        save_interval=max(1, args.num_iters // 10),
        output_dir=args.output_dir,
        device=device
    )


if __name__ == '__main__':
    main()
