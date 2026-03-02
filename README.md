# 3D CLBM-IBM 运动翼型仿真

> D3Q19 Cascaded LBM + Immersed Boundary Method，NACA 0012 翼型预设俯仰-沉浮运动

## 演示

<video src="vel2d.mp4" controls width="700"></video>

## 项目简介

在三维结构化格子上对 **NACA 0012 翼型**的预设运动（俯仰 + 沉浮）进行流固耦合仿真。翼型沿 z 方向均匀延伸，x-y 平面上按解析运动学公式做周期性振荡。

| 参数 | 值 | 说明 |
|------|----|------|
| 格子尺寸 | 600 × 200 × 50 | x × y × z |
| Reynolds 数 | 200 | 基于弦长 |
| 来流速度 | 0.05 lu/step | 格子单位 |
| 弦长 | 80 lu | |
| 初始迎角 | 8° | |
| 沉浮振幅 | A_heave | 配置于 config_moving.py |
| 俯仰振幅 | A_pitch | 配置于 config_moving.py |
| 缩减频率 | k = 0.5 | |
| 仿真步数 | 30,000 | |

## 方法

- **流体求解器**：CLBM D3Q19，Guo 体力修正
- **运动模型**：预设刚体运动，解析计算每步位移和速度（无需数值微分）
- **IBM 耦合**：每步更新标记点位置（旋转 + 平移），计算刚体速度约束
- **边界条件**：Zou-He 速度入口、零梯度出口、y/z 周期

## 文件结构

```
3d_moving_airfoil/
├── config_moving.py   — 运动参数（振幅、频率、相位差）+ D3Q19 参数 + 几何
├── clbm_moving.py     — CLBM D3Q19 求解器（导入 config_moving）
├── airfoil_moving.py  — 运动翼型 IBM（每步更新标记位置与速度）
├── main_moving.py     — 主循环 + 翼型轮廓叠加可视化 + MP4 输出
└── clbm_2d.py         — D2Q9 CLBM 二维求解器参考版本（依赖 config_moving 的 2D 参数子集）
```

> `clbm_2d.py` 是开发过程中的二维参考实现，从 `config_moving` 中提取 Nx、Ny 等二维参数。完整的二维项目请参见 `fsi_lbm_fvm_2d/`。

## 运行方式

```bash
pip install taichi numpy matplotlib imageio imageio-ffmpeg
python main_moving.py
```

输出文件：`clbm_airfoil_moving_3d.mp4`——中间截面速度场 + 翼型轮廓动画。
