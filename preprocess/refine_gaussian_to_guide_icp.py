import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def xyz(vertices) -> np.ndarray:
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]])


def rigid_transform(source: np.ndarray, target: np.ndarray):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    u, _, vt = np.linalg.svd((target - target_mean).T @ (source - source_mean))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--keep-quantile", type=float, default=0.65)
    args = parser.parse_args()

    ply = PlyData.read(args.input)
    vertices = ply["vertex"].data.copy()
    guide = PlyData.read(args.guide)["vertex"].data
    points = xyz(vertices)
    tree = cKDTree(xyz(guide))
    total_rotation = np.eye(3)
    total_translation = np.zeros(3)

    for _ in range(args.iterations):
        distances, indices = tree.query(points, workers=-1)
        keep = distances <= np.quantile(distances, args.keep_quantile)
        rotation, translation = rigid_transform(points[keep], xyz(guide)[indices[keep]])
        points = (rotation @ points.T).T + translation
        total_rotation = rotation @ total_rotation
        total_translation = rotation @ total_translation + translation

    vertices["x"], vertices["y"], vertices["z"] = points.T
    if all(f"rot_{index}" in vertices.dtype.names for index in range(4)):
        wxyz = np.column_stack([vertices[f"rot_{index}"] for index in range(4)])
        transformed = Rotation.from_matrix(total_rotation) * Rotation.from_quat(wxyz[:, [1, 2, 3, 0]])
        transformed_wxyz = transformed.as_quat()[:, [3, 0, 1, 2]]
        for index in range(4):
            vertices[f"rot_{index}"] = transformed_wxyz[:, index]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=ply.text,
        byte_order=ply.byte_order,
    ).write(output)
    final_distances = tree.query(points, workers=-1)[0]
    print(f"Final distance quantiles: {np.quantile(final_distances, [0, .25, .5, .75, 1])}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
