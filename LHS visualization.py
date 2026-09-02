"""
LHS visualization.py — Visualize Latin Hypercube Sampling principle and distributions
Author: Hantao He | Project: photovoltaic_prediction | 2026-09
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats.qmc import LatinHypercube

# -------------------------- 全局字体&绘图参数 --------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20    # 全局文字
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelpad'] = 8

n_sample = 16
np.random.seed(42)

# 生成LHS采样点
lhs2d = LatinHypercube(d=2, scramble=True, rng=42)
pts2d = lhs2d.random(n=n_sample)

lhs3d = LatinHypercube(d=3, scramble=True, rng=42)
pts3d = lhs3d.random(n=n_sample)

# -------------------------- 创建画布 --------------------------
fig = plt.figure(figsize=(18, 12), dpi=300)

# ========== 左子图 (a) 2D LHS ==========
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlabel("Variable $X_1$")
ax1.set_ylabel("Variable $X_2$")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# 2D等分网格
divs = np.linspace(0, 1, n_sample + 1)
for x in divs:
    ax1.axvline(x, c='#dddddd', lw=0.4, ls='--')
for y in divs:
    ax1.axhline(y, c='#dddddd', lw=0.4, ls='--')

# 浅蓝色垂直虚线、浅橙色水平虚线
for x, y in pts2d:
    ax1.plot([x, x], [0, y], c='#b8d8f0', lw=0.6, ls='--')
    ax1.plot([0, x], [y, y], c='#ffddbb', lw=0.6, ls='--')

# 坐标轴三角标记
x_mid = (divs[:-1] + divs[1:]) / 2
ax1.scatter(x_mid, np.zeros_like(x_mid), marker='^', c='#4477bb', s=22, zorder=4)
y_mid = (divs[:-1] + divs[1:]) / 2
ax1.scatter(np.zeros_like(y_mid), y_mid, marker='>', c='#ee8833', s=22, zorder=4)

# 红色采样点
ax1.scatter(pts2d[:, 0], pts2d[:, 1], c='#bb3333', s=32, zorder=5, edgecolors='k', linewidth=0.2)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.tick_params(axis='both', length=0)

# ========== 右子图 (b) 3D LHS ==========
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_xlabel("$X_1$")
ax2.set_ylabel("$X_2$")
ax2.set_zlabel("$\epsilon_X$")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_zlim(0, 1)

# 稀疏三维网格
grid_coarse = np.linspace(0, 1, 6)
for y in grid_coarse:
    for z in grid_coarse:
        ax2.plot([0,1], [y,y], [z,z], c='#dddddd', lw=0.3)
for x in grid_coarse:
    for z in grid_coarse:
        ax2.plot([x,x], [0,1], [z,z], c='#dddddd', lw=0.3)
for x in grid_coarse:
    for y in grid_coarse:
        ax2.plot([x,x], [y,y], [0,1], c='#dddddd', lw=0.3)

# 3D红点
ax2.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2], c='#bb3333', s=32, zorder=5, edgecolors='k', linewidth=0.2)

# 调整3D视角
ax2.view_init(elev=25, azim=-62)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.tick_params(axis='both', length=0)

# -------------------------- 统一布局与绝对标题对齐 --------------------------
# 1. 先固定子图的管理间距
plt.tight_layout(pad=3.0)
plt.subplots_adjust(bottom=0.22, wspace=0.25) # 稍微收紧了 bottom 留白

# 2. 使用 fig.text 在画布绝对位置绘制标题（Y坐标固定为 0.08，保证绝对水平对齐）
fig.text(0.28, 0.08, "(a) 2D visualization of LHS", ha='center', va='top')
fig.text(0.74, 0.08, "(b) 3D visualization of LHS", ha='center', va='top')

plt.show()