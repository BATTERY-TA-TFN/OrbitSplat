import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def xyz(vertices) -> np.ndarray:
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace a spatial part of a Gaussian PLY using distance to a guide point cloud."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--part-model", required=True)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--distance", type=float, default=0.035)
    parser.add_argument("--guide-axis", choices=("x", "y", "z"), default=None)
    parser.add_argument("--guide-min-quantile", type=float, default=None)
    parser.add_argument("--guide-max-quantile", type=float, default=None)
    parser.add_argument("--part-min-opacity", type=float, default=0.0)
    parser.add_argument("--part-max-scale", type=float, default=None)
    parser.add_argument(
        "--part-isotropic-scale",
        type=float,
        default=None,
        help="Replace selected part Gaussian scales with this isotropic world-space scale.",
    )
    parser.add_argument(
        "--part-zero-sh-rest",
        action="store_true",
        help="Clear direction-dependent SH coefficients on selected part Gaussians.",
    )
    parser.add_argument("--part-min-luminance", type=float, default=None)
    parser.add_argument(
        "--part-yellow-only",
        action="store_true",
        help="Only select Gaussians whose DC color is yellow-dominant.",
    )
    parser.add_argument(
        "--add-only",
        action="store_true",
        help="Keep every base Gaussian and only add selected part Gaussians.",
    )
    parser.add_argument(
        "--part-min-base-distance",
        type=float,
        default=None,
        help="Only add part Gaussians at least this far from the existing base model.",
    )
    args = parser.parse_args()

    base_ply = PlyData.read(args.base)
    part_ply = PlyData.read(args.part_model)
    guide_ply = PlyData.read(args.guide)
    base = base_ply["vertex"].data
    part = part_ply["vertex"].data
    guide = guide_ply["vertex"].data
    if base.dtype != part.dtype:
        raise RuntimeError("Base and part Gaussian PLY schemas differ.")

    guide_xyz = xyz(guide)
    if args.guide_axis is not None:
        axis = {"x": 0, "y": 1, "z": 2}[args.guide_axis]
        keep_guide = np.ones(len(guide_xyz), dtype=bool)
        if args.guide_min_quantile is not None:
            keep_guide &= guide_xyz[:, axis] >= np.quantile(guide_xyz[:, axis], args.guide_min_quantile)
        if args.guide_max_quantile is not None:
            keep_guide &= guide_xyz[:, axis] <= np.quantile(guide_xyz[:, axis], args.guide_max_quantile)
        guide_xyz = guide_xyz[keep_guide]
        print(f"Guide selected: {len(guide_xyz)}/{len(guide)}")
    tree = cKDTree(guide_xyz)
    base_distance, _ = tree.query(xyz(base), workers=-1)
    part_distance, _ = tree.query(xyz(part), workers=-1)
    keep_base = np.ones(len(base), dtype=bool) if args.add_only else base_distance > args.distance
    keep_part = part_distance <= args.distance
    if args.part_min_base_distance is not None:
        base_tree = cKDTree(xyz(base))
        part_base_distance, _ = base_tree.query(xyz(part), workers=-1)
        keep_part &= part_base_distance >= args.part_min_base_distance
    opacity = 1.0 / (1.0 + np.exp(-part["opacity"]))
    keep_part &= opacity >= args.part_min_opacity
    if args.part_max_scale is not None:
        scales = np.exp(np.column_stack([part["scale_0"], part["scale_1"], part["scale_2"]]))
        keep_part &= scales.max(axis=1) <= args.part_max_scale
    if args.part_min_luminance is not None or args.part_yellow_only:
        rgb = 0.5 + 0.28209479177387814 * np.column_stack(
            [part["f_dc_0"], part["f_dc_1"], part["f_dc_2"]]
        )
        if args.part_min_luminance is not None:
            luminance = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
            keep_part &= luminance >= args.part_min_luminance
        if args.part_yellow_only:
            keep_part &= (rgb[:, 0] > rgb[:, 2] + 0.12) & (rgb[:, 1] > rgb[:, 2] + 0.08)
    selected_part = part[keep_part].copy()
    if args.part_isotropic_scale is not None:
        log_scale = np.log(args.part_isotropic_scale)
        for index in range(3):
            selected_part[f"scale_{index}"] = log_scale
    if args.part_zero_sh_rest:
        for name in selected_part.dtype.names:
            if name.startswith("f_rest_"):
                selected_part[name] = 0
    merged = np.concatenate([base[keep_base], selected_part])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(merged, "vertex")],
        text=base_ply.text,
        byte_order=base_ply.byte_order,
    ).write(output)
    print(f"Base kept: {int(keep_base.sum())}/{len(base)}")
    print(f"Part selected: {int(keep_part.sum())}/{len(part)}")
    print(f"Merged: {len(merged)}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
