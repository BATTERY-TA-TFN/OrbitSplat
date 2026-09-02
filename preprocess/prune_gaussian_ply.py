import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-opacity", type=float, default=0.02)
    parser.add_argument("--max-scale", type=float, default=0.05)
    args = parser.parse_args()

    ply = PlyData.read(args.input)
    vertices = ply["vertex"].data
    opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"]))
    scales = np.exp(
        np.column_stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]])
    )
    max_scale = scales.max(axis=1)
    keep = (opacity >= args.min_opacity) & (max_scale <= args.max_scale)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices[keep], "vertex")],
        text=ply.text,
        byte_order=ply.byte_order,
    ).write(output)

    print(f"Kept {int(keep.sum())}/{len(vertices)} gaussians")
    print(f"Removed low opacity: {int((opacity < args.min_opacity).sum())}")
    print(f"Removed oversized: {int((max_scale > args.max_scale).sum())}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
