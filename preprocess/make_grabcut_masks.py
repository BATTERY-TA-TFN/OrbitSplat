import argparse
from pathlib import Path

import cv2
import numpy as np


def largest_seed_component(seed: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    if count <= 1:
        return seed

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def expand_bbox_from_mask(mask: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        rect_w = int(width * 0.58)
        rect_h = int(height * 0.58)
        return (width - rect_w) // 2, (height - rect_h) // 2, rect_w, rect_h

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad_x = int(width * 0.10)
    pad_top = int(height * 0.12)
    pad_bottom = int(height * 0.16)
    x = max(0, x0 - pad_x)
    y = max(0, y0 - pad_top)
    right = min(width - 1, x1 + pad_x)
    bottom = min(height - 1, y1 + pad_bottom)
    return x, y, right - x + 1, bottom - y + 1


def make_mask(image_path: Path, output_path: Path, preview_path: Path, rect_scale: float,
              iterations: int, hue_min: int, hue_max: int) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_seed = cv2.inRange(hsv, np.array([hue_min, 60, 70]), np.array([hue_max, 255, 255]))
    color_seed = cv2.morphologyEx(color_seed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    color_seed = largest_seed_component(color_seed)

    x, y, rect_w, rect_h = expand_bbox_from_mask(color_seed, width, height)
    grabcut_mask = np.full((height, width), cv2.GC_BGD, np.uint8)
    grabcut_mask[y:y + rect_h, x:x + rect_w] = cv2.GC_PR_BGD

    probable_fg = cv2.dilate(color_seed, np.ones((35, 35), np.uint8), iterations=2)
    definite_fg = cv2.erode(color_seed, np.ones((9, 9), np.uint8), iterations=1)
    grabcut_mask[probable_fg > 0] = cv2.GC_PR_FGD
    grabcut_mask[definite_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)

    mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, _, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1 and color_seed.any():
        overlapping = set(np.unique(labels[color_seed > 0]).tolist())
        overlapping.discard(0)
        if overlapping:
            keep = np.isin(labels, list(overlapping))
            mask = np.where(keep, 255, 0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)

    preview = image.copy()
    overlay = np.zeros_like(preview)
    overlay[:, :, 1] = mask
    preview = cv2.addWeighted(preview, 0.7, overlay, 0.3, 0)
    cv2.imwrite(str(preview_path), preview)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", default="data/realcap/rabbit")
    parser.add_argument("--rect-scale", type=float, default=0.58)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--hue-min", type=int, default=80)
    parser.add_argument("--hue-max", type=int, default=105)
    args = parser.parse_args()

    scene_dir = Path(args.source_path)
    image_dir = scene_dir / "images"
    mask_dir = scene_dir / "masks"
    preview_dir = scene_dir / "mask_previews"

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    for image_path in image_paths:
        output_path = mask_dir / f"{image_path.stem}.png"
        preview_path = preview_dir / f"{image_path.stem}.png"
        make_mask(
            image_path, output_path, preview_path, args.rect_scale,
            args.iterations, args.hue_min, args.hue_max
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
