import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


SH_C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    ply = PlyData.read(args.input)
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
    rgb = np.clip((dc * SH_C0 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    out = np.empty(xyz.shape[0], dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["red"], out["green"], out["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(output)
    print(f"Wrote {output} with {xyz.shape[0]} points")


if __name__ == "__main__":
    main()
