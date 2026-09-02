import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


SH_C0 = 0.28209479177387814


def xyz(vertices) -> np.ndarray:
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]])


def dc_rgb(vertices) -> np.ndarray:
    return 0.5 + SH_C0 * np.column_stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stabilize view-dependent color in a Gaussian region.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--distance", type=float, default=0.035)
    parser.add_argument("--guide-axis", choices=("x", "y", "z"), default=None)
    parser.add_argument("--guide-min-quantile", type=float, default=None)
    parser.add_argument("--guide-max-quantile", type=float, default=None)
    parser.add_argument("--yellow-only", action="store_true")
    parser.add_argument("--target-blend", type=float, default=0.25)
    parser.add_argument("--rest-scale", type=float, default=0.25)
    args = parser.parse_args()

    ply = PlyData.read(args.input)
    vertices = ply["vertex"].data.copy()
    guide = PlyData.read(args.guide)["vertex"].data
    guide_xyz = xyz(guide)
    if args.guide_axis is not None:
        axis = {"x": 0, "y": 1, "z": 2}[args.guide_axis]
        keep = np.ones(len(guide_xyz), dtype=bool)
        if args.guide_min_quantile is not None:
            keep &= guide_xyz[:, axis] >= np.quantile(guide_xyz[:, axis], args.guide_min_quantile)
        if args.guide_max_quantile is not None:
            keep &= guide_xyz[:, axis] <= np.quantile(guide_xyz[:, axis], args.guide_max_quantile)
        guide_xyz = guide_xyz[keep]

    distance, _ = cKDTree(guide_xyz).query(xyz(vertices), workers=-1)
    selected = distance <= args.distance
    rgb = dc_rgb(vertices)
    yellow = (rgb[:, 0] > rgb[:, 2] + 0.12) & (rgb[:, 1] > rgb[:, 2] + 0.08)
    if args.yellow_only:
        selected &= yellow

    luminance = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    target_pool = yellow & (luminance >= np.quantile(luminance[yellow], 0.65))
    target = np.median(rgb[target_pool], axis=0)
    stabilized = rgb[selected] * (1.0 - args.target_blend) + target * args.target_blend
    dc = (stabilized - 0.5) / SH_C0
    for channel in range(3):
        vertices[f"f_dc_{channel}"][selected] = dc[:, channel]
    for name in vertices.dtype.names:
        if name.startswith("f_rest_"):
            vertices[name][selected] *= args.rest_scale

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=ply.text,
        byte_order=ply.byte_order,
    ).write(output)
    print(f"Guide points: {len(guide_xyz)}")
    print(f"Selected yellow Gaussians: {int(selected.sum())}/{len(vertices)}")
    print(f"Target RGB: {target}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
