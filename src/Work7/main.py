"""
Work7: SMPL 模型加载、可视化与手写 LBS 实现
================================================
本脚本完成以下任务：
1. 加载 SMPL 模型并输出基础信息
2. 可视化模板网格与蒙皮权重（单关节热力图 + 全关节主导权重分布）
3. 可视化形状校正与关节回归（v_shaped + J_regressor）
4. 可视化姿态校正 B_P(theta)（pose_offsets）
5. 手写完整 LBS 并可视化最终结果
6. 生成 2x2 总对比图
7. 手写 LBS 与官方前向结果一致性验证

依赖：smplx, torch, numpy, matplotlib, trimesh
"""
import os
import pickle
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import smplx

warnings.filterwarnings('ignore')

# ===================== 配置 =====================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'SMPL_NEUTRAL.pkl')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 相机视角参数（matplotlib 3D）
ELEV = 10
AZIM = -70

# ============================================================
# 辅助函数
# ============================================================

def ensure_model_exists():
    """检查模型文件是否存在"""
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] SMPL model not found at: {MODEL_PATH}")
        print("Please download SMPL_NEUTRAL.pkl from:")
        print("  - 师大云盘 (recommended for this course)")
        print("  - https://smpl.is.tue.mpg.de/ (official, requires registration)")
        print("Then place it at: Work7/models/SMPL_NEUTRAL.pkl")
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")


def plot_mesh(ax, vertices, faces, colors=None, vertex_colors=None,
              cmap='viridis', title='', elev=ELEV, azim=AZIM,
              show_axis=True, alpha=0.9):
    """
    使用 matplotlib 绘制三角网格。
    colors: 面片颜色数组 (N_faces,)
    vertex_colors: 顶点颜色数组 (N_verts,)
    """
    ax.clear()
    if vertex_colors is not None:
        # 使用顶点颜色（通过 facecolors 插值）
        norm = Normalize(vmin=vertex_colors.min(), vmax=vertex_colors.max())
        face_colors = vertex_colors[faces].mean(axis=1)
        cmap_obj = plt.get_cmap(cmap)
        rgba = cmap_obj(norm(face_colors))
        mesh = Poly3DCollection(vertices[faces], alpha=alpha, facecolors=rgba,
                                edgecolors='none', linewidth=0)
    elif colors is not None:
        norm = Normalize(vmin=colors.min(), vmax=colors.max())
        cmap_obj = plt.get_cmap(cmap)
        rgba = cmap_obj(norm(colors))
        mesh = Poly3DCollection(vertices[faces], alpha=alpha, facecolors=rgba,
                                edgecolors='none', linewidth=0)
    else:
        mesh = Poly3DCollection(vertices[faces], alpha=alpha,
                                facecolor='lightgray', edgecolor='darkgray',
                                linewidth=0.1)
    ax.add_collection3d(mesh)

    # 自动设置坐标范围
    margin = 0.1
    ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
    ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)
    ax.set_zlim(vertices[:, 2].min() - margin, vertices[:, 2].max() + margin)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    if not show_axis:
        ax.set_axis_off()
    return mesh


def plot_mesh_with_joints(ax, vertices, faces, joints, joint_colors='red',
                          joint_size=20, title='', elev=ELEV, azim=AZIM):
    """绘制网格 + 关节点"""
    plot_mesh(ax, vertices, faces, title=title, elev=elev, azim=azim)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c=joint_colors, s=joint_size, depthshade=False, edgecolors='black', linewidths=0.5)


def rodrigues_to_rotation_matrix(axis_angle):
    """
    将轴角 (N, 3) 转换为旋转矩阵 (N, 3, 3)
    使用罗德里格斯公式
    """
    batch_size = axis_angle.shape[0]
    angle = torch.norm(axis_angle + 1e-8, dim=1, keepdim=True)
    rot_dir = axis_angle / angle

    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)

    # 外积矩阵
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=axis_angle.dtype, device=axis_angle.device)

    K = torch.cat([
        zeros, -rz, ry,
        rz, zeros, -rx,
        -ry, rx, zeros
    ], dim=1).view(batch_size, 3, 3)

    ident = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).unsqueeze(0)
    rot_mat = ident + sin * K + (1 - cos) * (K @ K)
    return rot_mat


def batch_rodrigues(rot_vecs):
    """smplx 风格的 batch 轴角转旋转矩阵 (N, 3) -> (N, 3, 3)"""
    return rodrigues_to_rotation_matrix(rot_vecs)


def transform_mat(R, t):
    """从旋转矩阵 R (3,3) 和平移 t (3,) 构造 4x4 齐次变换矩阵"""
    T = torch.eye(4, dtype=R.dtype, device=R.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def get_global_transform(kintree, local_rotmats, local_trans):
    """
    根据运动学树计算每个关节的全局变换。
    kintree: (2, J) 其中 kintree[0, j] 是父关节索引
    local_rotmats: (J, 3, 3)
    local_trans: (J, 3)
    返回: global_transforms (J, 4, 4)
    """
    J = local_rotmats.shape[0]
    global_T = torch.zeros(J, 4, 4, dtype=local_rotmats.dtype, device=local_rotmats.device)

    for j in range(J):
        local_T = transform_mat(local_rotmats[j], local_trans[j])
        parent = kintree[0, j]
        if parent == -1 or parent == 4294967295:  # 根节点（uint32 的最大值或 -1）
            global_T[j] = local_T
        else:
            global_T[j] = global_T[parent] @ local_T
    return global_T


# ============================================================
# 主流程
# ============================================================
def main():
    ensure_model_exists()

    print("=" * 60)
    print("Work7: SMPL Model Loading, Visualization & Hand-written LBS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 任务 1：加载 SMPL 模型并输出基础信息
    # ------------------------------------------------------------------
    print("\n[Task 1] Loading SMPL model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = smplx.create(
        MODEL_PATH,
        model_type='smpl',
        gender='neutral',
        batch_size=1
    ).to(device)

    num_verts = model.v_template.shape[0]
    num_faces = model.faces.shape[0]
    num_joints = model.J_regressor.shape[0]
    num_betas = model.shapedirs.shape[-1]

    print(f"  Vertices : {num_verts}")
    print(f"  Faces    : {num_faces}")
    print(f"  Joints   : {num_joints}")
    print(f"  Betas dim: {num_betas}")

    # 同时读取原始 pickle 以获取 kintree_table（smplx 0.1.28 的接口差异）
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    if isinstance(data, dict):
        raw_model = data
    else:
        raw_model = data['model']

    kintree_table = raw_model['kintree_table']  # (2, 24)
    faces_np = model.faces.cpu().numpy().astype(np.int32)
    v_template_np = model.v_template.cpu().numpy()

    # ------------------------------------------------------------------
    # 任务 2：可视化模板网格与蒙皮权重
    # ------------------------------------------------------------------
    print("\n[Task 2] Visualizing template mesh & skinning weights...")

    # --- (a) 单关节权重热力图 ---
    weights_np = model.lbs_weights.cpu().numpy()  # (6890, 24)
    selected_joint = 16  # 例如：左肩或右手（可根据需要调整）
    joint_weights = weights_np[:, selected_joint]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_mesh(ax, v_template_np, faces_np, vertex_colors=joint_weights,
              cmap='hot', title=f'Template Mesh + Joint {selected_joint} Weights', alpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stage_a_template_weights.png'), dpi=150)
    plt.close()
    print("  Saved: stage_a_template_weights.png")

    # --- (b) 全关节主导权重分布图（可选） ---
    dominant_joint = weights_np.argmax(axis=1)  # 每个顶点的主导关节
    dominant_weight = weights_np.max(axis=1)    # 主导权重值

    # 使用 tab20 色图给不同关节分配不同颜色，再用主导权重调整亮度
    cmap_tab20 = plt.get_cmap('tab20')
    joint_colors = cmap_tab20(dominant_joint / 24.0)[:, :3]
    # 用权重亮度调整
    brightness = dominant_weight[:, None]
    face_colors = (joint_colors[faces_np].mean(axis=1) * (0.3 + 0.7 * brightness[faces_np].mean(axis=1)[:, None]))
    face_colors = np.clip(face_colors, 0, 1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(v_template_np[faces_np], alpha=0.9,
                            facecolors=face_colors, edgecolors='none', linewidth=0)
    ax.add_collection3d(mesh)
    ax.set_xlim(v_template_np[:, 0].min() - 0.1, v_template_np[:, 0].max() + 0.1)
    ax.set_ylim(v_template_np[:, 1].min() - 0.1, v_template_np[:, 1].max() + 0.1)
    ax.set_zlim(v_template_np[:, 2].min() - 0.1, v_template_np[:, 2].max() + 0.1)
    ax.set_title('Dominant Joint Weight Distribution')
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_joint_weights.png'), dpi=150)
    plt.close()
    print("  Saved: all_joint_weights.png")

    # ------------------------------------------------------------------
    # 任务 3：可视化形状校正与关节回归
    # ------------------------------------------------------------------
    print("\n[Task 3] Shape correction & joint regression...")

    # 设置非零 shape 参数
    betas = torch.zeros(1, num_betas, device=device)
    betas[0, 0] = 2.0   # 变高/变矮
    betas[0, 1] = -1.5  # 变胖/变瘦
    betas[0, 2] = 0.8

    # 官方前向：仅 shape，无 pose
    with torch.no_grad():
        output_shape = model(betas=betas, body_pose=torch.zeros(1, 69, device=device),
                             global_orient=torch.zeros(1, 3, device=device), return_verts=True)
    v_shaped = output_shape.vertices[0].cpu().numpy()
    J_regressed = output_shape.joints[0, :24].cpu().numpy()  # SMPL 有 24 个关节

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_mesh_with_joints(ax, v_shaped, faces_np, J_regressed,
                          title='(b) v_shaped + Regressed Joints', elev=ELEV, azim=AZIM)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stage_b_shaped_joints.png'), dpi=150)
    plt.close()
    print("  Saved: stage_b_shaped_joints.png")

    # ------------------------------------------------------------------
    # 任务 4：可视化姿态校正 B_P(theta)
    # ------------------------------------------------------------------
    print("\n[Task 4] Pose corrective offsets B_P(theta)...")

    # 设置非零姿态：抬手、弯肘、略微扭转躯干
    body_pose = torch.zeros(1, 69, device=device)
    global_orient = torch.zeros(1, 3, device=device)

    # 抬右手（关节 17 或 19，取决于索引定义，这里用轴角表示）
    # SMPL 23 个 body pose joints (excluding root) = 69 dim
    # 关节索引映射（近似）：13=左肩, 16=右肩, 18=左肘, 19=右肘
    body_pose[0, 16 * 3 + 0] = 0.3   # 右肩抬起
    body_pose[0, 16 * 3 + 2] = 1.2   # 右肩外展
    body_pose[0, 19 * 3 + 0] = 1.0   # 右肘弯曲
    body_pose[0, 3 * 3 + 2] = 0.4    # 躯干扭转

    # 官方前向：shape + pose
    with torch.no_grad():
        output_pose = model(betas=betas, body_pose=body_pose,
                            global_orient=global_orient, return_verts=True)
    v_posed_official = output_pose.vertices[0].cpu().numpy()
    J_posed_official = output_pose.joints[0, :24].cpu().numpy()

    # 手动计算 pose_offsets
    # 1. 轴角 -> 旋转矩阵
    full_pose = torch.cat([global_orient, body_pose], dim=1)  # (1, 72)
    rot_mats = batch_rodrigues(full_pose[0].view(-1, 3))  # (24, 3, 3)

    # 2. pose_feature = R - I
    pose_feature = (rot_mats - torch.eye(3, device=device).unsqueeze(0)).view(24, -1)  # (24, 9)
    # posedirs 的维度是 (6890, 3, 207)，需要 reshape
    posedirs = model.posedirs  # (6890*3, 207) 或 (6890, 3, 207) 取决于 smplx 版本
    if posedirs.dim() == 2:
        posedirs = posedirs.view(num_verts, 3, 207)

    # 3. pose_offsets = posedirs @ pose_feature[1:].flatten()  (去掉根关节)
    pose_offsets = torch.einsum('vij,j->vi', posedirs[:, :, :207], pose_feature[1:].flatten())
    pose_offsets_np = pose_offsets.cpu().numpy()
    offset_magnitude = np.linalg.norm(pose_offsets_np, axis=1)

    # v_shaped（从之前计算）
    v_shaped_torch = output_shape.vertices[0]
    v_posed_manual = (v_shaped_torch + pose_offsets).cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_mesh(ax, v_posed_manual, faces_np, vertex_colors=offset_magnitude,
              cmap='plasma', title='(c) Pose Offsets Magnitude |v_posed - v_shaped|',
              elev=ELEV, azim=AZIM)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stage_c_pose_offsets.png'), dpi=150)
    plt.close()
    print("  Saved: stage_c_pose_offsets.png")

    # ------------------------------------------------------------------
    # 任务 5 & 7：手写完整 LBS + 与官方结果一致性验证
    # ------------------------------------------------------------------
    print("\n[Task 5 & 7] Hand-written LBS & consistency check...")

    # 提取模型参数（全部转为 torch，device）
    v_template_t = model.v_template.to(device)          # (6890, 3)
    shapedirs_t = model.shapedirs.to(device)            # (6890, 3, 10)
    posedirs_t = model.posedirs.to(device)              # (6890*3, 207) -> reshape
    if posedirs_t.dim() == 2:
        posedirs_t = posedirs_t.view(num_verts, 3, 207)
    J_regressor_t = model.J_regressor.to(device)        # (24, 6890)
    weights_t = model.lbs_weights.to(device)            # (6890, 24)

    # 1. 计算 v_shaped = v_template + shapedirs @ betas
    v_shaped_t = v_template_t + torch.einsum('vij,bj->bvi', shapedirs_t, betas)[0]

    # 2. 回归关节 J = J_regressor @ v_shaped
    J_t = torch.einsum('jv,bvj->bj', J_regressor_t, v_shaped_t.unsqueeze(0))[0]  # (24, 3)

    # 3. 计算 pose_offsets
    rot_mats = batch_rodrigues(full_pose[0].view(-1, 3))  # (24, 3, 3)
    pose_feature = (rot_mats - torch.eye(3, device=device).unsqueeze(0)).view(24, -1)  # (24, 9)
    pose_offsets_t = torch.einsum('vij,j->vi', posedirs_t, pose_feature[1:].flatten())
    v_posed_t = v_shaped_t + pose_offsets_t

    # 4. 运动学：计算每个关节的全局变换
    # 局部平移：根节点 = J[0]，其余 = J[j] - J[parent]
    local_trans = torch.zeros_like(J_t)
    local_trans[0] = J_t[0]
    for j in range(1, 24):
        parent = kintree_table[0, j]
        local_trans[j] = J_t[j] - J_t[parent]

    # 局部旋转 = rot_mats
    global_T = get_global_transform(kintree_table, rot_mats, local_trans)  # (24, 4, 4)

    # 5. LBS：将 v_posed 变换到每个关节的局部坐标系，加权求和
    # v_posed_homo = (V, 4)
    v_posed_homo = torch.cat([v_posed_t, torch.ones(num_verts, 1, device=device)], dim=1)  # (6890, 4)

    # 对于每个关节 j，计算 T_j @ (v_posed - J_j) + J_j
    # 等价于 T_j @ v_posed + (J_j - T_j @ J_j)，但更简单的是直接在世界坐标系中计算
    # 实际上标准公式是：v = sum_j w_j * (T_j @ (v_posed - J_j) + J_j)
    # 其中 T_j 是世界变换
    verts_lbs = torch.zeros_like(v_posed_t)
    for j in range(24):
        T_j = global_T[j]
        # 将 v_posed 转换到关节 j 的绑定姿势
        v_j = (T_j[:3, :3] @ (v_posed_t - J_t[j]).T).T + T_j[:3, 3] + J_t[j]
        verts_lbs += weights_t[:, j:j+1] * v_j

    verts_lbs_np = verts_lbs.cpu().numpy()

    # 6. 与官方结果比较
    official_verts = output_pose.vertices[0].cpu().numpy()
    diff = np.abs(verts_lbs_np - official_verts)
    mae = diff.mean()
    max_ae = diff.max()
    rmse = np.sqrt((diff ** 2).mean())

    print(f"  Consistency Check:")
    print(f"    MAE  = {mae:.6e}")
    print(f"    RMSE = {rmse:.6e}")
    print(f"    MaxAE= {max_ae:.6e}")

    # 保存 summary.txt
    summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SMPL Model Summary & LBS Consistency Check\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Path: {MODEL_PATH}\n")
        f.write(f"Vertices : {num_verts}\n")
        f.write(f"Faces    : {num_faces}\n")
        f.write(f"Joints   : {num_joints}\n")
        f.write(f"Betas dim: {num_betas}\n\n")
        f.write("LBS Consistency (Hand-written vs Official Forward):\n")
        f.write(f"  Mean Absolute Error : {mae:.6e}\n")
        f.write(f"  Root Mean Sq Error  : {rmse:.6e}\n")
        f.write(f"  Max Absolute Error  : {max_ae:.6e}\n")
        if max_ae < 1e-4:
            f.write("\n[PASS] Hand-written LBS matches official output within tolerance.\n")
        else:
            f.write("\n[NOTE] Discrepancy detected (possible numerical differences).\n")
    print(f"  Saved: summary.txt")

    # 可视化最终 LBS 结果
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_mesh_with_joints(ax, verts_lbs_np, faces_np, J_posed_official,
                          title='(d) Final LBS Skinned Mesh', elev=ELEV, azim=AZIM)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stage_d_lbs_result.png'), dpi=150)
    plt.close()
    print("  Saved: stage_d_lbs_result.png")

    # ------------------------------------------------------------------
    # 任务 6：生成总对比图
    # ------------------------------------------------------------------
    print("\n[Task 6] Generating comparison grid...")

    fig = plt.figure(figsize=(16, 14))

    # (a) template + weights
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plot_mesh(ax1, v_template_np, faces_np, vertex_colors=joint_weights,
              cmap='hot', title='(a) Template + Weights', alpha=0.95)

    # (b) shape + joints
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_mesh_with_joints(ax2, v_shaped, faces_np, J_regressed,
                          title='(b) Shaped + Joints', elev=ELEV, azim=AZIM)

    # (c) pose offsets
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    plot_mesh(ax3, v_posed_manual, faces_np, vertex_colors=offset_magnitude,
              cmap='plasma', title='(c) Pose Offsets', alpha=0.95)

    # (d) final skinned mesh
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    plot_mesh_with_joints(ax4, verts_lbs_np, faces_np, J_posed_official,
                          title='(d) Final Skinned Mesh', elev=ELEV, azim=AZIM)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_grid.png'), dpi=150)
    plt.close()
    print("  Saved: comparison_grid.png")

    print("\n" + "=" * 60)
    print("All tasks completed. Outputs saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
