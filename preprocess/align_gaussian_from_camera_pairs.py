import argparse
from pathlib import Path

import numpy as np
import pycolmap
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def camera_center(image) -> np.ndarray:
    return np.asarray(image.cam_from_world().inverse().translation)


def similarity(source: np.ndarray, target: np.ndarray):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    u, singular, vt = np.linalg.svd(
        target_centered.T @ source_centered / len(source)
    )
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = singular.sum() / np.mean(np.sum(source_centered**2, axis=1))
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-reconstruction", required=True)
    parser.add_argument("--target-reconstruction", required=True)
    parser.add_argument("--pairs", required=True, help="CSV lines: source_image,target_image")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_reconstruction = pycolmap.Reconstruction(args.source_reconstruction)
    target_reconstruction = pycolmap.Reconstruction(args.target_reconstruction)
    source_images = {image.name: image for image in source_reconstruction.images.values()}
    target_images = {image.name: image for image in target_reconstruction.images.values()}

    pairs = []
    for line in Path(args.pairs).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        source_name, target_name = [value.strip() for value in line.split(",", 1)]
        pairs.append((source_name, target_name))
    source_centers = np.asarray([camera_center(source_images[a]) for a, _ in pairs])
    target_centers = np.asarray([camera_center(target_images[b]) for _, b in pairs])
    scale, rotation, translation = similarity(source_centers, target_centers)
    residuals = np.linalg.norm(
        (scale * (rotation @ source_centers.T)).T + translation - target_centers,
        axis=1,
    )

    ply = PlyData.read(args.input)
    vertices = ply["vertex"].data.copy()
    positions = np.column_stack([vertices["x"], vertices["y"], vertices["z"]])
    positions = (scale * (rotation @ positions.T)).T + translation
    vertices["x"], vertices["y"], vertices["z"] = positions.T

    if all(f"rot_{index}" in vertices.dtype.names for index in range(4)):
        quaternions_wxyz = np.column_stack(
            [vertices[f"rot_{index}"] for index in range(4)]
        )
        quaternions_xyzw = quaternions_wxyz[:, [1, 2, 3, 0]]
        transformed = Rotation.from_matrix(rotation) * Rotation.from_quat(quaternions_xyzw)
        transformed_wxyz = transformed.as_quat()[:, [3, 0, 1, 2]]
        for index in range(4):
            vertices[f"rot_{index}"] = transformed_wxyz[:, index]

    if all(f"scale_{index}" in vertices.dtype.names for index in range(3)):
        log_scale = np.log(scale)
        for index in range(3):
            vertices[f"scale_{index}"] += log_scale

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=ply.text,
        byte_order=ply.byte_order,
    ).write(output)
    print(f"Pairs: {len(pairs)}")
    print(f"Scale: {scale:.8f}")
    print(f"Median camera residual: {np.median(residuals):.6f}")
    print(f"Max camera residual: {residuals.max():.6f}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
