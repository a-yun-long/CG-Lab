# CG-Lab: 万有引力粒子群仿真!
## 📺 效果演示
![粒子仿真演示](./demo.gif)


## 🛠️ 项目环境
- **IDE**: Trae
- **工具链**: uv
- **物理引擎**: Taichi (ti.init(arch=ti.gpu))

## 📂 项目架构
项目采用 `src` 布局，结构如下：
- `config.py`: 仿真参数（引力常数、粒子数等）
- `physics.py`: 基于 Taichi kernel 的并行计算逻辑
- `main.py`: GUI 渲染与交互入口

## 🚀 如何运行
```bash
uv run -m src.Work0.main

