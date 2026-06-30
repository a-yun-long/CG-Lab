# Work7: SMPL 模型加载、可视化与手写 LBS 实现

## 实验目标

- 成功加载 SMPL 模型，理解其数据结构（顶点、面片、关节、形状/姿态参数）。
- 可视化模板网格与蒙皮权重，理解 LBS（Linear Blend Skinning）权重的空间分布。
- 掌握形状校正（Shape Blend Shapes）与关节回归（Joint Regressor）的原理。
- 理解姿态校正（Pose Blend Shapes）的作用与计算方法。
- 手写完整的 LBS 前向过程，并与 `smplx` 官方前向结果进行一致性验证。

## 项目结构

```
Work7/
├── main.py              # 主程序：完成全部 7 个任务
├── download_model.py    # 辅助脚本：尝试自动下载 SMPL 模型
├── models/              # 模型文件存放目录（需手动放置 SMPL_NEUTRAL.pkl）
├── outputs/             # 输出结果目录
│   ├── stage_a_template_weights.png
│   ├── all_joint_weights.png
│   ├── stage_b_shaped_joints.png
│   ├── stage_c_pose_offsets.png
│   ├── stage_d_lbs_result.png
│   ├── comparison_grid.png
│   └── summary.txt
└── README.md            # 本文件
```

## 环境配置

```bash
pip install smplx torch numpy matplotlib
```

或使用 conda：

```bash
conda activate graphics
pip install smplx matplotlib
```

## 模型准备

SMPL_NEUTRAL.pkl 需要手动获取（受版权保护，无法自动分发）：

1. **师大云盘**（推荐）：下载课程资料中的 `SMPL_NEUTRAL.pkl`
2. **SMPL 官网**（https://smpl.is.tue.mpg.de/）：注册后下载 `SMPL_python_v.1.1.0.zip`，解压后将 `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` **重命名**为 `SMPL_NEUTRAL.pkl`

将文件放置到：
```
Work7/models/SMPL_NEUTRAL.pkl
```

也可尝试运行辅助脚本（部分镜像可能已失效）：
```bash
python download_model.py
```

## 运行方法

```bash
cd Work7
python main.py
```

运行后会在 `outputs/` 目录下生成所有要求的图片和验证报告。

## 核心实现说明

### 任务 1：模型基础信息

使用 `smplx.create(..., model_type='smpl', gender='neutral')` 加载模型，并打印：
- 顶点数：6890
- 面片数：13776
- 关节数：24
- Betas 维度：10

### 任务 2：权重可视化

- **单关节热力图**：从 `lbs_weights` (6890, 24) 中选取指定关节，用 `hot` 色图映射到顶点颜色
- **全关节主导权重分布**：每个顶点取最大权重的关节索引，用 `tab20` 色图区分不同关节控制区域

### 任务 3：形状校正与关节回归

设置非零 `betas`（如 `[2.0, -1.5, 0.8, ...]`），计算：
```
v_shaped = v_template + shapedirs @ betas
J = J_regressor @ v_shaped
```
可视化变形后的网格与回归出的关节点。

### 任务 4：姿态校正 B_P(θ)

1. 将轴角 `body_pose` / `global_orient` 转为旋转矩阵（罗德里格斯公式）
2. 构造 `pose_feature = R - I`
3. 计算 `pose_offsets = posedirs @ pose_feature[1:].flatten()`
4. `v_posed = v_shaped + pose_offsets`
5. 用 `plasma` 色图可视化 `||pose_offsets||` 的大小

### 任务 5：手写完整 LBS

核心步骤：
1. 计算 `v_shaped` 和 `J`
2. 计算 `v_posed = v_shaped + pose_offsets`
3. 根据 `kintree_table` 构建运动学树，计算每个关节的局部变换
4. 前向运动学遍历树，得到全局变换 `T_j` (4x4)
5. LBS 加权：
```
verts = sum_j w_j * (T_j[:3,:3] @ (v_posed - J_j) + T_j[:3,3] + J_j)
```

### 任务 6：总对比图

将 4 个阶段的结果排成 2x2 网格，一目了然展示从模板 -> 形状校正 -> 姿态校正 -> 最终蒙皮的完整流程。

### 任务 7：一致性验证

使用与手写实现完全相同的 `betas`、`global_orient`、`body_pose` 调用 `model(...)`，得到官方 `output.vertices`，然后逐顶点比较：
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MaxAE** (Max Absolute Error)

结果保存到 `outputs/summary.txt`。

## 思考要点

1. **为什么一个顶点不只受一个关节影响？**
   - 真实人体的皮肤/肌肉在关节处是平滑过渡的，如果每个顶点只绑定到一个关节，会出现明显的"裂缝"和刚性折叠。多关节加权能实现自然的变形过渡。

2. **为什么关节位置要从形状后的网格回归，而不是固定不变？**
   - 不同体型的人（高/矮、胖/瘦）关节位置会随之变化。例如变胖后，髋关节和肩关节的位置会外扩。`J_regressor` 保证了关节始终位于身体内部合理位置。

3. **为什么 LBS 之前还要加 pose_offsets？**
   - 标准 LBS 只能产生刚性旋转效果，无法模拟肌肉隆起、皮肤褶皱等非刚性细节。`pose_offsets` 是数据驱动的姿态相关修正，让肘部弯曲时看起来更真实。

4. **为什么最终顶点要写成加权和，而不是只选择最大权重的关节？**
   - 取最大权重会导致硬边界（hard segmentation），在关节交界处产生不连续。加权求和保证了平滑过渡，符合人体解剖学特征。

## 参考效果

运行后 `outputs/` 目录应包含：
- `stage_a_template_weights.png`：模板网格 + 单关节权重热力图
- `all_joint_weights.png`：全关节主导权重分布（可选辅助图）
- `stage_b_shaped_joints.png`：形状校正后网格 + 回归关节
- `stage_c_pose_offsets.png`：姿态修正量大小可视化
- `stage_d_lbs_result.png`：最终蒙皮结果
- `comparison_grid.png`：四阶段总对比图
- `summary.txt`：模型信息 + 手写 LBS 误差报告
