import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def look_at(position, target):
    forward = target - position
    forward /= np.linalg.norm(forward)
    world_down = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_down, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


def write_ply(path, points, colors):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    rows = np.empty(len(points), dtype=dtype)
    values = np.concatenate([points, np.zeros_like(points), colors], axis=1)
    rows[:] = list(map(tuple, values))
    PlyData([PlyElement.describe(rows, "vertex")]).write(path)


def project(points, c2w, focal, size):
    w2c = np.linalg.inv(c2w)
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera[:, 2]
    uv = camera[:, :2] / np.maximum(z[:, None], 1e-6) * focal + size / 2
    return uv, z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", required=True)
    parser.add_argument("-o", "--output-path", required=True)
    parser.add_argument("--size", type=int, default=1536)
    parser.add_argument("--padding", type=float, default=1.35)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--focal-ratio", type=float, default=1.5)
    parser.add_argument("--grid", type=int, default=100)
    parser.add_argument("--min-views", type=int, default=8)
    parser.add_argument("--clockwise", action="store_true")
    parser.add_argument(
        "--angles-file",
        help="Optional text file containing one rotation angle in degrees per image",
    )
    args = parser.parse_args()

    source = Path(args.source_path)
    output = Path(args.output_path)
    image_paths = sorted((source / "images").glob("*"))
    mask_paths = sorted((source / "masks").glob("*"))
    if len(image_paths) != len(mask_paths) or not image_paths:
        raise RuntimeError("images and masks must contain the same non-zero number of files")

    output_images = output / "images"
    output_masks = output / "masks"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    boxes = []
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path).convert("L"))
        ys, xs = np.where(mask > 127)
        boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    centers = np.array([[(x0 + x1) / 2, (y0 + y1) / 2] for x0, y0, x1, y1 in boxes])
    center = np.median(centers, axis=0)
    crop_size = int(np.ceil(max(max(x1 - x0, y1 - y0) for x0, y0, x1, y1 in boxes) * args.padding))

    cropped_images = []
    cropped_masks = []
    for index, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths), 1):
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        left = int(round(center[0] - crop_size / 2))
        top = int(round(center[1] - crop_size / 2))
        box = (left, top, left + crop_size, top + crop_size)
        image = image.crop(box).resize((args.size, args.size), Image.Resampling.LANCZOS)
        mask = mask.crop(box).resize((args.size, args.size), Image.Resampling.NEAREST)
        image_name = f"{index:03d}.png"
        image.save(output_images / image_name)
        mask.save(output_masks / image_name)
        cropped_images.append(np.array(image))
        cropped_masks.append(np.array(mask) > 127)

    count = len(cropped_images)
    if args.angles_file:
        angles = np.atleast_1d(np.loadtxt(args.angles_file, dtype=np.float64))
        if len(angles) != count:
            raise RuntimeError(
                f"angles file contains {len(angles)} values, expected {count}"
            )
        angles = np.deg2rad(angles)
    else:
        angles = 2 * np.pi * np.arange(count) / count
    np.savetxt(output / f"sparse_{count}.txt", np.arange(count), fmt="%d")
    np.savetxt(output / "sparse_test.txt", np.array([], dtype=np.int32), fmt="%d")

    focal = args.size * args.focal_ratio
    target = np.zeros(3)
    cameras = []
    c2ws = []
    direction = -1.0 if args.clockwise else 1.0
    for index in range(count):
        angle = direction * angles[index]
        position = np.array([args.radius * np.sin(angle), 0.0, -args.radius * np.cos(angle)])
        c2w = np.eye(4)
        c2w[:3, :3] = look_at(position, target)
        c2w[:3, 3] = position
        c2ws.append(c2w)
        cameras.append({
            "id": index,
            "img_name": f"{index + 1:03d}.png",
            "width": args.size,
            "height": args.size,
            "position": position.tolist(),
            "rotation": c2w[:3, :3].tolist(),
            "fy": focal,
            "fx": focal,
        })

    pose_name = f"turntable_{count}.json"
    with open(output / pose_name, "w", encoding="utf-8") as file:
        json.dump(cameras, file, indent=4)

    extent_x = args.radius * (args.size * 0.42) / focal
    extent_y = args.radius * (args.size * 0.48) / focal
    axis = np.linspace(-1.0, 1.0, args.grid)
    yy, xx, zz = np.meshgrid(axis * extent_y, axis * extent_x, axis * extent_x, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    votes = np.zeros(len(points), dtype=np.int16)
    color_sum = np.zeros((len(points), 3), dtype=np.float64)
    color_votes = np.zeros(len(points), dtype=np.int16)

    for image, mask, c2w in zip(cropped_images, cropped_masks, c2ws):
        uv, depth = project(points, c2w, focal, args.size)
        pixels = np.rint(uv).astype(np.int32)
        valid = (
            (depth > 0) &
            (pixels[:, 0] >= 0) & (pixels[:, 0] < args.size) &
            (pixels[:, 1] >= 0) & (pixels[:, 1] < args.size)
        )
        ids = np.where(valid)[0]
        inside = mask[pixels[ids, 1], pixels[ids, 0]]
        inside_ids = ids[inside]
        votes[inside_ids] += 1
        color_sum[inside_ids] += image[pixels[inside_ids, 1], pixels[inside_ids, 0]]
        color_votes[inside_ids] += 1

    keep = votes >= min(args.min_views, count)
    points = points[keep]
    colors = color_sum[keep] / np.maximum(color_votes[keep, None], 1)
    write_ply(output / f"turntable_{count}.ply", points, colors.astype(np.uint8))
    print(f"Wrote {count} centered views, {pose_name}, and {len(points)} visual-hull points to {output}")


if __name__ == "__main__":
    main()
