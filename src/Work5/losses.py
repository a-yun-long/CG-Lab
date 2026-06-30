"""
网格正则化损失：拉普拉斯平滑、边长一致性、法线一致性
"""
import torch
import torch.nn.functional as F


def mesh_laplacian_smoothing(mesh):
    """
    拉普拉斯平滑损失：约束每个顶点靠近其邻居的平均位置
    利用 Mesh 中预计算的邻接信息进行向量化计算
    """
    verts = mesh.verts
    neighbor_mean = mesh.laplacian_target()
    laplacian = verts - neighbor_mean
    loss = (laplacian ** 2).sum() / mesh.num_verts
    return loss


def mesh_edge_loss(mesh, target_length=None):
    """
    边长一致性损失：惩罚过长或过短的边
    """
    edges = mesh.edges()
    verts = mesh.verts

    v0 = verts[edges[:, 0]]
    v1 = verts[edges[:, 1]]
    edge_lengths = torch.norm(v1 - v0, dim=1)

    if target_length is None:
        target_length = edge_lengths.mean()

    loss = ((edge_lengths - target_length) ** 2).mean()
    return loss


def mesh_normal_consistency(mesh):
    """
    法线一致性损失：约束相邻三角形面的法线方向接近
    利用 Mesh 中预计算的相邻面对进行向量化计算
    """
    face_normals = mesh.face_normals()
    adj_pairs = mesh.adj_face_pairs()

    if adj_pairs.shape[0] == 0:
        return torch.tensor(0.0, device=mesh.device)

    n1 = face_normals[adj_pairs[:, 0]]
    n2 = face_normals[adj_pairs[:, 1]]

    # 1 - dot(n1, n2) 当法线归一化时等价于 0.5 * ||n1 - n2||^2
    diff = 1.0 - (n1 * n2).sum(dim=1)
    loss = diff.mean()
    return loss


def silhouette_loss(pred_silhouettes, target_silhouettes):
    """
    剪影 MSE 损失
    pred_silhouettes: (N, H, W)
    target_silhouettes: (N, H, W)
    """
    return F.mse_loss(pred_silhouettes, target_silhouettes)
