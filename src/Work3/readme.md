

# Graphics Lab: Ray Casting & Local Illumination

基于 Taichi 框架实现的交互式光线投射（Ray Casting）与局部光照（Local Illumination）渲染器。本项目无需任何外部 3D 模型文件，完全通过数学隐式方程定义几何体，并从零实现了光线求交、深度竞争、光照计算以及硬阴影等核心图形学算法。

## 📂 项目结构 (Directory Structure)
'''''' text
CG-Lab (仓库根目录)
├── src
│   ├── Work0 (粒子仿真)
│   │   ├── main.py          # 实验零核心代码
│   │   └── demo.gif         # 粒子仿真演示
│   ├── Work1 (MVP变换)
│   │   ├── main.py          # 实验一核心代码
│   │   └── demo.gif         # 实验一演示
│   └── Work2 (贝塞尔曲线)
│   |    ├── bezier_curve.py  # 实验二核心代码
│   |    ├── screenshot.gif   # 曲线绘制演示
│   |    └── README.md        # 实验二详细说明
|   └── Work3 (Graphics Lab: Ray Casting & Local Illumination)
|        ├──demo.gif
|        ├──pVSpb.py
|        ├──readme.md
|        └──imgui.ini
|
├── .gitignore               # Git 忽略配置文件
└── README.md                # 仓库总索引
''''''

##  🚀 运行效果
![演示](./demo.gif)

## ✨ 特性与实验目标

本项目达成了以下核心计算机图形学目标：
- **隐式曲面求交 (Implicit Surface Intersection):** 利用数学解析式计算光线与三维几何体（红色球体、紫色圆锥）的精确交点。
- **深度竞争 (Z-buffering Equivalent):** 实现多物体场景下的射线距离 $t$ 竞争，保证正确的空间遮挡关系。
- **Phong 局部光照模型:** 完整实现 Ambient（环境光）、Diffuse（漫反射）和 Specular（镜面高光）分量的计算与叠加。
- **(选做) Blinn-Phong 光照升级:** 引入半程向量 $\mathbf{H}$，解决大入射角下的高光截断问题，提升物理真实感。
- **(选做) 硬阴影 (Hard Shadows):** 通过向光源发射暗影射线（Shadow Ray），计算自遮挡与物体间的遮挡阴影。
- **实时交互 UI:** 结合 Taichi UI 模块，实现光照参数 ($K_a, K_d, K_s$, Shininess) 与渲染模式的实时动态调节。

---

## 📐 数学原理与算法基础

### 1. 局部光照方程
本项目的基础着色遵循经典 Phong 经验模型，像素最终颜色由三个独立的反射分量叠加而成：
$$I = I_{ambient} + I_{diffuse} + I_{specular}$$

* **环境光 (Ambient):** 模拟场景底光。
    $$I_{ambient} = K_a \times C_{light} \times C_{object}$$
* **漫反射 (Diffuse):** 遵循 Lambert 余弦定律，与光线入射角相关。
    $$I_{diffuse} = K_d \times \max(0, \mathbf{N} \cdot \mathbf{L}) \times C_{light} \times C_{object}$$
* **镜面高光 (Specular):** 模拟光滑表面的反光。
    $$I_{specular} = K_s \times \max(0, \mathbf{R} \cdot \mathbf{V})^n \times C_{light}$$

*(注：$\mathbf{N}$ 为表面法向量，$\mathbf{L}$ 为光源方向，$\mathbf{V}$ 为视线方向，$\mathbf{R}$ 为反射向量，$n$ 为高光指数。所有参与计算的向量均已归一化)*

### 2. Blinn-Phong 模型 (选做优化)
在 UI 勾选 `Use Blinn-Phong` 时，高光计算将切换为使用**半程向量 (Halfway Vector)** $\mathbf{H}$：
$$\mathbf{H} = \frac{\mathbf{L} + \mathbf{V}}{||\mathbf{L} + \mathbf{V}||}$$
$$I_{specular\_blinn} = K_s \times \max(0, \mathbf{N} \cdot \mathbf{H})^n \times C_{light}$$

---

## 🛠️ 环境配置与运行指南

### 依赖要求
- Python 3.8+
- Taichi (提供高性能 GPU 运算加速)

### 安装与运行
你可以使用 pip 或新兴的包管理器（如 UV）来快速安装依赖并运行代码。

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. 安装 Taichi
pip install taichi  # 或者使用 uv pip install taichi

# 3. 运行渲染器
python main.py
```

### UI 交互控制
运行程序后，屏幕左上角将出现一个参数控制面板。你可以使用鼠标拖动滑块实时观察渲染变化：
* **Ka (Ambient):** 调节环境光底色亮度。
* **Kd (Diffuse):** 调节漫反射强度。
* **Ks (Specular):** 调节镜面高光强度。
* **Shininess:** 调节高光聚焦程度（值越大，光斑越小且越锐利）。
* **Use Blinn-Phong [Checkbox]:** 切换基础 Phong 模型与 Blinn-Phong 模型。
* **Enable Hard Shadow [Checkbox]:** 开启/关闭阴影射线测试。

---

## 📊 实验结果分析 (Phong vs Blinn-Phong)

在本次实验中，我们重点对比了标准 Phong 模型与 Blinn-Phong 模型在高光区域边缘的视觉表现差异。

**实验现象与分析：**
1. **Phong 模型的局限性：** 当光源从极大掠射角（入射角接近 $90^\circ$）照射球体或圆锥边缘时，标准 Phong 模型计算得出的反射向量 $\mathbf{R}$ 可能会指向物体内部，导致 $\mathbf{R} \cdot \mathbf{V}$ 为负数。经过 `max(0, ...)` 截断后，**物体边缘的高光会突然生硬消失，形成不自然的黑色断层**。
2. **Blinn-Phong 的物理改进：** 切换至 Blinn-Phong 模型后，系统改用半程向量 $\mathbf{H}$ 进行 $\mathbf{N} \cdot \mathbf{H}$ 点乘。由于 $\mathbf{H}$ 始终位于 $\mathbf{L}$ 和 $\mathbf{V}$ 之间，无论视点和光源角度如何，都不会出现点积为负并被强行截断的情况。
3. **视觉结论：** 实验结果清晰表明，**Blinn-Phong 的高光在几何体边缘的过渡更加平滑、柔和，光斑范围略大**，成功消除了边缘的硬性黑斑，在视觉上更符合真实物理世界中微表面（Microfacet）的反射规律，且由于省略了全反射向量 $\mathbf{R}$ 的计算，具有更高的运算效率。

---

## 🗂️ 核心代码结构

* `get_sphere_intersect(ro, rd)`: 求解射线与球体的二次方程，返回最小的正根 $t$ 及交点法线。
* `get_cone_intersect(ro, rd)`: 求解射线与圆锥侧面及底面的解析交点，并处理高度截断 ($y \in [-1.4, 1.2]$)。
* `intersect_scene(ro, rd)`: 遍历所有图元进行求交测试，执行类似 Z-buffer 的深度竞争，返回最近交点信息。
* `render()`: Taichi Kernel 函数，并行执行光线投射，计算环境光、漫反射、镜面高光（含模型分支），并执行硬阴影的二次光线追踪测试。

