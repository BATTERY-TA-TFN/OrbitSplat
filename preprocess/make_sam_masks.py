import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


def resolve_checkpoint(filename: str) -> str:
    local_path = Path("models") / filename
    if local_path.exists():
        return str(local_path)

    external_model_dir = os.environ.get("GAUSSIANOBJECT_MODEL_DIR")
    if external_model_dir:
        external_path = Path(external_model_dir) / filename
        if external_path.exists():
            return str(external_path)

    return str(local_path)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(bool)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = (flood[1:-1, 1:-1] == 0) & (mask_u8 == 0)
    return mask_u8.astype(bool) | holes


def auto_box(image_bgr: np.ndarray, padding: float) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([80, 55, 60]), np.array([108, 255, 255]))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    blue = largest_component(blue)

    ys, xs = np.where(blue)
    if len(xs) == 0:
        box_w = int(width * 0.34)
        box_h = int(height * 0.48)
        x0 = (width - box_w) // 2
        y0 = (height - box_h) // 2
        return np.array([x0, y0, x0 + box_w, y0 + box_h])

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    pad_x = int(max(box_w, width * 0.08) * padding)
    pad_top = int(max(box_h, height * 0.10) * padding)
    pad_bottom = int(max(box_h, height * 0.14) * padding)

    return np.array([
        max(0, x0 - pad_x),
        max(0, y0 - pad_top),
        min(width - 1, x1 + pad_x),
        min(height - 1, y1 + pad_bottom),
    ])


def box_from_seed_mask(mask_path: Path, image_shape: tuple[int, int], padding: float) -> np.ndarray | None:
    seed = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if seed is None:
        return None
    height, width = image_shape
    if seed.shape != (height, width):
        seed = cv2.resize(seed, (width, height), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(seed > 127)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad_x = int((x1 - x0 + 1) * padding)
    pad_y = int((y1 - y0 + 1) * padding)
    return np.array([
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width - 1, x1 + pad_x),
        min(height - 1, y1 + pad_y),
    ], dtype=np.float32)


def load_boxes(path: str) -> dict[str, np.ndarray]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {name: np.array(box, dtype=np.float32) for name, box in raw.items()}


def choose_mask(masks: np.ndarray, scores: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray:
    if seed is not None and seed.any():
        seed = seed.astype(bool)
        overlaps = []
        for mask, score in zip(masks, scores):
            intersection = np.logical_and(mask, seed).sum()
            union = np.logical_or(mask, seed).sum()
            coverage = intersection / max(seed.sum(), 1)
            iou = intersection / max(union, 1)
            overlaps.append(coverage + iou + float(score) * 0.05)
        return masks[int(np.argmax(overlaps))]
    order = np.argsort(scores)[::-1]
    for idx in order:
        mask = masks[idx]
        area = mask.mean()
        if 0.005 < area < 0.35:
            return mask
    return masks[order[0]]


def load_seed_mask(mask_path: Path, image_shape: tuple[int, int]) -> np.ndarray | None:
    seed = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if seed is None:
        return None
    height, width = image_shape
    if seed.shape != (height, width):
        seed = cv2.resize(seed, (width, height), interpolation=cv2.INTER_NEAREST)
    return seed > 127


def seed_points(seed: np.ndarray | None) -> np.ndarray | None:
    if seed is None or not seed.any():
        return None
    distance = cv2.distanceTransform(seed.astype(np.uint8), cv2.DIST_L2, 5)
    points = []
    work = distance.copy()
    suppress_radius = max(12, int(min(seed.shape) * 0.04))
    for _ in range(3):
        _, max_value, _, max_location = cv2.minMaxLoc(work)
        if max_value <= 0:
            break
        points.append(max_location)
        cv2.circle(work, max_location, suppress_radius, 0, -1)
    return np.array(points, dtype=np.float32) if points else None


def save_preview(image_bgr: np.ndarray, mask: np.ndarray, box: np.ndarray, path: Path) -> None:
    preview = image_bgr.copy()
    overlay = np.zeros_like(preview)
    overlay[:, :, 1] = (mask.astype(np.uint8) * 255)
    preview = cv2.addWeighted(preview, 0.72, overlay, 0.28, 0)
    x0, y0, x1, y1 = box.astype(int)
    cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", default="data/realcap/rabbit")
    parser.add_argument("--checkpoint", default="sam_vit_h_4b8939.pth")
    parser.add_argument("--model-type", default="vit_h")
    parser.add_argument("--boxes-json", default="")
    parser.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                        help="Use one fixed box for every image.")
    parser.add_argument("--points", nargs="+", type=float,
                        help="Fixed positive point coordinates as X1 Y1 X2 Y2 ...")
    parser.add_argument("--seed-masks-dir", default="",
                        help="Use rough masks only to create SAM boxes; SAM replaces their boundaries.")
    parser.add_argument("--padding", type=float, default=0.28)
    args = parser.parse_args()

    scene_dir = Path(args.source_path)
    image_dir = scene_dir / "images"
    mask_dir = scene_dir / "masks"
    preview_dir = scene_dir / "sam_mask_previews"
    checkpoint = resolve_checkpoint(args.checkpoint)
    if not Path(checkpoint).exists():
        raise RuntimeError(
            f"Missing SAM checkpoint: {checkpoint}\n"
            "Run: .\\models\\download_preprocess_models_windows.ps1 -DownloadSam"
        )

    boxes = load_boxes(args.boxes_json)
    sam = sam_model_registry[args.model_type](checkpoint=checkpoint).cuda()
    predictor = SamPredictor(sam)

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        box = np.array(args.box, dtype=np.float32) if args.box else boxes.get(image_path.name)
        if box is None:
            box = boxes.get(image_path.stem)
        seed = None
        if args.seed_masks_dir:
            seed_path = Path(args.seed_masks_dir) / f"{image_path.stem}.png"
            seed = load_seed_mask(seed_path, image_bgr.shape[:2])
        if box is None and args.seed_masks_dir:
            box = box_from_seed_mask(seed_path, image_bgr.shape[:2], args.padding)
        if box is None:
            box = auto_box(image_bgr, args.padding).astype(np.float32)

        points = seed_points(seed)
        if args.points:
            if len(args.points) % 2:
                raise ValueError("--points requires X Y pairs")
            points = np.array(args.points, dtype=np.float32).reshape(-1, 2)
        labels = np.ones(len(points), dtype=np.int32) if points is not None else None
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            multimask_output=True,
        )
        mask = choose_mask(masks, scores, seed)
        mask = fill_holes(largest_component(mask)).astype(np.uint8) * 255

        mask_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_dir / f"{image_path.stem}.png"), mask)
        save_preview(image_bgr, mask > 0, box, preview_dir / f"{image_path.stem}.png")
        print(f"Wrote {mask_dir / f'{image_path.stem}.png'}")


if __name__ == "__main__":
    main()
