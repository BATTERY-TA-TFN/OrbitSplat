from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plyfile import PlyData


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "realcap" / "pikaqiu_video54_colmap_undistorted"
RAW = DATA / "sparse" / "0" / "points3D.ply"
FILTERED = DATA / "colmap_object.ply"
RESULT = ROOT / "output" / "gs_init" / "pikaqiu_video54_colmap_hq20k" / "render" / "ours_20000" / "renders" / "00030.png"
OUTPUT = ROOT / "output" / "colmap_mask_filtering_figure.png"


def load_ply(path):
    vertex = PlyData.read(path)["vertex"]
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])
    rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]) / 255.0
    return xyz, rgb


def equal_axes(ax, xyz, padding=0.08):
    low = np.percentile(xyz, 1, axis=0)
    high = np.percentile(xyz, 99, axis=0)
    center = (low + high) / 2
    radius = (high - low).max() * (0.5 + padding)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def point_panel(ax, xyz, rgb, title, subtitle, limits_xyz):
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=7, depthshade=False)
    equal_axes(ax, limits_xyz)
    ax.view_init(elev=18, azim=-74)
    ax.set_facecolor("#F6F8FB")
    ax.grid(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=20, fontweight="bold", color="#172033", pad=18)
    ax.text2D(0.5, 0.02, subtitle, transform=ax.transAxes, ha="center", fontsize=12, color="#667085")


def main():
    raw_xyz, raw_rgb = load_ply(RAW)
    filtered_xyz, filtered_rgb = load_ply(FILTERED)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    fig = plt.figure(figsize=(18, 6.8), facecolor="#F7F9FC")
    fig.suptitle("多视图掩膜约束的 COLMAP 点云过滤", x=0.045, y=0.965, ha="left", fontsize=28, fontweight="bold", color="#172033")
    fig.text(0.046, 0.89, "将稀疏点投影至多个前景掩膜，仅保留满足多视图前景一致性的点", fontsize=14, color="#667085")

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    point_panel(ax1, raw_xyz, raw_rgb, "① 原始 COLMAP 稀疏点云", f"{len(raw_xyz):,} points · 含桌面与背景点", raw_xyz)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    point_panel(ax2, filtered_xyz, filtered_rgb, "② 掩膜一致性过滤后", f"{len(filtered_xyz):,} points · 保留目标物体支持点", raw_xyz)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(Image.open(RESULT))
    ax3.axis("off")
    ax3.set_title("③ 使用过滤点云训练的结果", fontsize=20, fontweight="bold", color="#172033", pad=18)
    ax3.text(0.5, -0.075, "过滤点云作为初始化几何，继续优化 3D Gaussians", transform=ax3.transAxes, ha="center", fontsize=12, color="#667085")

    for x in [0.35, 0.675]:
        fig.text(x, 0.48, "→", fontsize=40, color="#D97706", fontweight="bold", ha="center")

    fig.text(
        0.5,
        0.035,
        f"过滤保留率：{len(filtered_xyz)}/{len(raw_xyz)} = {len(filtered_xyz) / len(raw_xyz) * 100:.1f}%　　目的：减少背景点对物体几何初始化的污染",
        ha="center",
        fontsize=14,
        color="#166534",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#EAF6EF", edgecolor="none"),
    )
    plt.subplots_adjust(left=0.03, right=0.98, top=0.82, bottom=0.14, wspace=0.12)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor())
    print(OUTPUT)


if __name__ == "__main__":
    main()
