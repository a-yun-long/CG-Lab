# 实验三：贝塞尔曲线与 De Casteljau 算法

## 📂 项目结构 (Directory Structure)

```text
CG-Lab (仓库根目录)
├── src
│   ├── Work0 (粒子仿真)
│   │   ├── main.py          # 实验一核心代码
│   │   └── demo.gif         # 粒子仿真演示
|   |   |——Work1 (MVP变换)
│   │   ├── main.py          # 实验一核心代码
│   │   └── demo.gif         # 粒子仿真演示
│   └── Work2 (贝塞尔曲线)
│       ├── bezier_curve.py  # 实验二核心代码
│       ├── screenshot.gif         # 曲线绘制演示
│       └── README.md        # 实验二详细说明
├── .gitignore               # Git 忽略配置文件
└── README.md                # 仓库总索引

本实验基于 **Python + Taichi** 实现了交互式的贝塞尔曲线绘制，深入理解了 De Casteljau 算法的几何意义及光栅化基础。

## 🚀 运行效果
![运行演示](./src/Work2/screenshot.gif)
## 🛠️ 环境配置
本项目使用现代化的 **uv** 进行包管理。

1. 安装 uv: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. 运行项目: `uv run bezier_curve.py`

## 🧠 核心算法：De Casteljau
贝塞尔曲线的生成基于递归线性插值。对于给定的参数 $t \in [0, 1]$，其核心迭代公式为：
$$P'_i = (1 - t)P_i + tP_{i+1}$$
通过不断递归缩小点的规模，最终求得曲线上确切的坐标点。

## ✨ 实现功能
- [x] **交互式绘制**：鼠标左键实时添加控制点。
- [x] **实时渲染**：基于 Taichi Kernel 的 GPU 并行光栅化。
- [x] **对象池技巧**：预分配固定大小的 Field 优化内存。
- [x] **快捷操作**：按下 `C` 键一键清空画布。

## 📂 项目结构
- `main.py`: 核心实现代码
- `pyproject.toml`: 依赖管理配置文件
- `.gitignore`: 忽略无关文件
