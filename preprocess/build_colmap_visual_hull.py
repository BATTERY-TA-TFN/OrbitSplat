import argparse
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_erosion


def read_ply_xyz(path: Path) -> np.ndarray:
    vertices = PlyData.read(path)["vertex"]
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    data = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    attributes = np.concatenate([xyz, np.zeros_like(xyz), rgb], axis=1)
    data[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(data, "vertex")]).write(path)


def project(points: np.ndarray, image, camera):
    transform = image.cam_from_world()
    cam_points = points @ transform.rotation.matrix().T + transform.translation
    z = cam_points[:, 2]
    uv = np.empty((len(points), 2), dtype=np.float64)
    uv[:, 0] = camera.focal_length_x * cam_points[:, 0] / z + camera.principal_point_x
    uv[:, 1] = camera.focal_length_y * cam_points[:, 1] / z + camera.principal_point_y
    return uv, z


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a silhouette-carved object surface using real COLMAP cameras."
    )
    parser.add_argument("-s", "--source-path", required=True)
    parser.add_argument("--bounds-ply", default="colmap_object.ply")
    parser.add_argument("--mask-dir", default="masks")
    parser.add_argument("--grid-size", type=int, default=144)
    parser.add_argument("--padding", type=float, default=0.12)
    parser.add_argument("--min-ratio", type=float, default=0.90)
    parser.add_argument(
        "--surface-band",
        type=int,
        default=1,
        help="Surface thickness in voxels. Larger values better preserve thin structures.",
    )
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--output", default="colmap_visual_hull.ply")
    parser.add_argument(
        "--allow-missing-masks",
        action="store_true",
        help="Skip COLMAP images without a mask, useful for part-specific reconstruction.",
    )
    args = parser.parse_args()

    source = Path(args.source_path)
    reconstruction = pycolmap.Reconstruction(source / "sparse" / "0")
    bounds_xyz = read_ply_xyz(source / args.bounds_ply)
    lower = bounds_xyz.min(axis=0)
    upper = bounds_xyz.max(axis=0)
    padding = (upper - lower) * args.padding
    lower -= padding
    upper += padding

    axes = [np.linspace(lower[i], upper[i], args.grid_size, dtype=np.float32) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    points = grid.reshape(-1, 3)
    inside_count = np.zeros(len(points), dtype=np.uint16)
    valid_count = np.zeros(len(points), dtype=np.uint16)

    entries = []
    for image in sorted(reconstruction.images.values(), key=lambda item: item.name):
        camera = reconstruction.cameras[image.camera_id]
        stem = Path(image.name).stem
        mask = cv2.imread(str(source / args.mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(str(source / "images" / image.name), cv2.IMREAD_COLOR)
        if mask is None and args.allow_missing_masks:
            continue
        if mask is None or color is None:
            raise RuntimeError(f"Missing image or mask for {image.name}")
        uv, depth = project(points, image, camera)
        x = np.rint(uv[:, 0]).astype(np.int32)
        y = np.rint(uv[:, 1]).astype(np.int32)
        valid = (
            (depth > 0)
            & (x >= 0) & (x < mask.shape[1])
            & (y >= 0) & (y < mask.shape[0])
        )
        valid_count[valid] += 1
        valid_ids = np.flatnonzero(valid)
        inside_count[valid_ids] += mask[y[valid], x[valid]] > 127
        entries.append((image, camera, color))

    required = np.ceil(valid_count * args.min_ratio).astype(np.uint16)
    occupied = (valid_count == len(entries)) & (inside_count >= required)
    occupied_grid = occupied.reshape((args.grid_size,) * 3)
    eroded = occupied_grid
    for _ in range(args.surface_band):
        eroded = binary_erosion(eroded)
    surface_grid = occupied_grid & ~eroded
    surface = grid[surface_grid]
    if len(surface) == 0:
        raise RuntimeError("Visual hull is empty. Lower --min-ratio or increase --padding.")

    if len(surface) > args.max_points:
        rng = np.random.default_rng(0)
        surface = surface[rng.choice(len(surface), args.max_points, replace=False)]

    color_samples = []
    for image, camera, color in entries:
        uv, depth = project(surface, image, camera)
        x = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, color.shape[1] - 1)
        y = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, color.shape[0] - 1)
        sample = color[y, x][:, ::-1]
        color_samples.append(sample)
    rgb = np.median(np.stack(color_samples, axis=0), axis=0).astype(np.uint8)

    output = source / args.output
    write_ply(output, surface.astype(np.float32), rgb)
    print(f"Bounds: {lower} -> {upper}")
    print(f"Occupied voxels: {occupied.sum()}")
    print(f"Surface points: {len(surface)}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
