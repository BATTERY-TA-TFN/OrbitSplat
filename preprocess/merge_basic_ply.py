import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


FIELDS = ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue")


def read_vertices(path: Path) -> np.ndarray:
    vertices = PlyData.read(path)["vertex"]
    return np.column_stack([vertices[field] for field in FIELDS])


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge basic colored point-cloud PLY files.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    arrays = [read_vertices(Path(path)) for path in args.inputs]
    merged = np.concatenate(arrays, axis=0)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    output = np.empty(len(merged), dtype=dtype)
    output[:] = list(map(tuple, merged))
    PlyData([PlyElement.describe(output, "vertex")]).write(args.output)
    print(f"Merged {[len(array) for array in arrays]} -> {len(output)} points")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
