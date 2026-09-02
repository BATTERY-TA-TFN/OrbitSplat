import argparse
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from plyfile import PlyData, PlyElement


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    data = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    normals = np.zeros_like(xyz)
    data[:] = list(map(tuple, np.concatenate([xyz, normals, rgb], axis=1)))
    PlyData([PlyElement.describe(data, "vertex")]).write(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", required=True)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-ratio", type=float, default=0.5)
    parser.add_argument("--output", default="colmap_object.ply")
    parser.add_argument("--mask-dir", default="masks")
    args = parser.parse_args()

    source = Path(args.source_path)
    reconstruction = pycolmap.Reconstruction(source / "sparse" / "0")
    masks = {}
    for image in reconstruction.images.values():
        mask = cv2.imread(
            str(source / args.mask_dir / f"{Path(image.name).stem}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise RuntimeError(f"Missing mask for {image.name}")
        masks[image.image_id] = mask

    kept_xyz = []
    kept_rgb = []
    support_histogram = {}
    for point in reconstruction.points3D.values():
        support = 0
        valid = 0
        for element in point.track.elements:
            image = reconstruction.images[element.image_id]
            xy = image.points2D[element.point2D_idx].xy
            x, y = int(round(xy[0])), int(round(xy[1]))
            mask = masks[element.image_id]
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                valid += 1
                support += int(mask[y, x] > 127)
        ratio = support / max(valid, 1)
        support_histogram[support] = support_histogram.get(support, 0) + 1
        if support >= args.min_support and ratio >= args.min_ratio:
            kept_xyz.append(point.xyz)
            kept_rgb.append(point.color)

    xyz = np.asarray(kept_xyz, dtype=np.float32)
    rgb = np.asarray(kept_rgb, dtype=np.uint8)
    if len(xyz) == 0:
        raise RuntimeError("No points survived mask filtering.")
    output = source / args.output
    write_ply(output, xyz, rgb)
    print(f"Kept {len(xyz)}/{reconstruction.num_points3D()} points")
    print(f"Support histogram: {dict(sorted(support_histogram.items()))}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
