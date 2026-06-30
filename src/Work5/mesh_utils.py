"""
网格工具模块：OBJ加载、球体生成、邻接关系、法线计算
"""
import torch
import torch.nn.functional as F
import numpy as np


def load_obj_simple(filepath):
    """
    简化版OBJ加载器，只加载顶点和面
    """
    verts = []
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v':
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                face = []
                for p in parts[1:]:
                    face.append(int(p.split('/')[0]) - 1)
                faces.append(face)
    verts = torch.tensor(verts, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.long)
    return verts, faces


def normalize_mesh(verts):
    """
    将网格归一化到以原点为中心、半径为1的球内
    """
    center = verts.mean(dim=0)
    verts = verts - center
    scale = verts.abs().max()
    verts = verts / scale
    return verts, center, scale


def create_icosphere(level=3):
    """
    通过细分二十面体生成球体网格
    level: 细分次数，level=2 约有 320 个面，level=3 约有 1280 个面
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float32)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int64)

    for _ in range(level):
        verts, faces = subdivide_mesh(verts, faces)

    return torch.from_numpy(verts).float(), torch.from_numpy(faces).long()


def subdivide_mesh(verts, faces):
    """对网格进行一次细分（Loop简化版）"""
    verts = list(verts)
    faces = list(faces)
    edge_map = {}
    new_faces = []

    def get_midpoint(v1, v2):
        key = tuple(sorted((v1, v2)))
        if key not in edge_map:
            mid = (verts[v1] + verts[v2]) / 2.0
            mid = mid / np.linalg.norm(mid)
            edge_map[key] = len(verts)
            verts.append(mid)
        return edge_map[key]

    for face in faces:
        v0, v1, v2 = face
        a = get_midpoint(v0, v1)
        b = get_midpoint(v1, v2)
        c = get_midpoint(v2, v0)
        new_faces.append([v0, a, c])
        new_faces.append([v1, b, a])
        new_faces.append([v2, c, b])
        new_faces.append([a, b, c])

    return np.array(verts, dtype=np.float32), np.array(new_faces, dtype=np.int64)


def compute_face_normals(verts, faces):
    """计算每个面的法线"""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals = F.normalize(normals, dim=-1)
    return normals


def compute_vertex_normals(verts, faces):
    """计算每个顶点的法线（通过相邻面法线平均）"""
    face_normals = compute_face_normals(verts, faces)
    vertex_normals = torch.zeros_like(verts)
    for i in range(3):
        vertex_normals.index_add_(0, faces[:, i], face_normals)
    return F.normalize(vertex_normals, dim=-1)


class Mesh:
    """网格数据结构，预计算邻接信息以提高效率"""

    def __init__(self, verts, faces, device='cpu'):
        self.verts = verts.to(device)
        self.faces = faces.to(device)
        self.device = device
        self._edges = None
        self._edge_lengths = None
        self._neighbor_sum_idx = None
        self._neighbor_counts = None
        self._face_normals = None
        self._adj_face_pairs = None
        self._build_adjacency()

    def _build_adjacency(self):
        """预计算邻接信息"""
        num_verts = self.num_verts
        faces = self.faces

        # 边列表
        edges = torch.cat([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]]
        ], dim=0)
        edges = torch.sort(edges, dim=1)[0]
        edges = torch.unique(edges, dim=0)
        self._edges = edges

        # 双向边用于邻域计算
        edges_bi = torch.cat([edges, edges.flip(1)], dim=0)

        # 计算每个顶点的邻居数量
        self._neighbor_counts = torch.zeros(num_verts, dtype=torch.int64, device=self.device)
        self._neighbor_counts.index_add_(0, edges_bi[:, 0], torch.ones(edges_bi.shape[0], dtype=torch.int64, device=self.device))

        # 构建邻居索引数组（用于向量化拉普拉斯计算）
        max_degree = int(self._neighbor_counts.max().item())
        self._neighbor_idx = torch.full((num_verts, max_degree), -1, dtype=torch.long, device=self.device)
        temp_counts = torch.zeros(num_verts, dtype=torch.long, device=self.device)
        for e in edges:
            v0, v1 = e[0].item(), e[1].item()
            self._neighbor_idx[v0, temp_counts[v0]] = v1
            self._neighbor_idx[v1, temp_counts[v1]] = v0
            temp_counts[v0] += 1
            temp_counts[v1] += 1

        # 预计算相邻面对（用于法线一致性）
        # 构建边到面的映射
        edge_to_faces = {}
        for i in range(self.num_faces):
            for j in range(3):
                v0, v1 = faces[i, j].item(), faces[i, (j + 1) % 3].item()
                e = tuple(sorted((v0, v1)))
                if e not in edge_to_faces:
                    edge_to_faces[e] = []
                edge_to_faces[e].append(i)

        # 收集共享边的面对
        adj_pairs = []
        for e, fids in edge_to_faces.items():
            if len(fids) == 2:
                adj_pairs.append(fids)
        if len(adj_pairs) > 0:
            self._adj_face_pairs = torch.tensor(adj_pairs, dtype=torch.long, device=self.device)
        else:
            self._adj_face_pairs = torch.empty((0, 2), dtype=torch.long, device=self.device)

    def to(self, device):
        self.verts = self.verts.to(device)
        self.faces = self.faces.to(device)
        self.device = device
        if self._edges is not None:
            self._edges = self._edges.to(device)
        if self._neighbor_idx is not None:
            self._neighbor_idx = self._neighbor_idx.to(device)
        if self._neighbor_counts is not None:
            self._neighbor_counts = self._neighbor_counts.to(device)
        if self._adj_face_pairs is not None:
            self._adj_face_pairs = self._adj_face_pairs.to(device)
        return self

    def clone(self):
        return Mesh(self.verts.clone(), self.faces.clone(), self.device)

    @property
    def num_verts(self):
        return self.verts.shape[0]

    @property
    def num_faces(self):
        return self.faces.shape[0]

    def offset_verts_(self, offset):
        self.verts = self.verts + offset
        return self

    def scale_verts_(self, scale):
        self.verts = self.verts * scale
        return self

    def edges(self):
        return self._edges

    def laplacian_target(self):
        """向量化计算邻居平均位置"""
        valid_mask = self._neighbor_idx >= 0
        neighbor_verts = self.verts[self._neighbor_idx.clamp(min=0)]
        neighbor_verts = neighbor_verts * valid_mask.unsqueeze(-1).float()
        neighbor_sum = neighbor_verts.sum(dim=1)
        neighbor_mean = neighbor_sum / self._neighbor_counts.unsqueeze(-1).clamp(min=1).float()
        return neighbor_mean

    def face_normals(self):
        if self._face_normals is None or self._face_normals.requires_grad != self.verts.requires_grad:
            self._face_normals = compute_face_normals(self.verts, self.faces)
        return self._face_normals

    def vertex_normals(self):
        return compute_vertex_normals(self.verts, self.faces)

    def adj_face_pairs(self):
        return self._adj_face_pairs
