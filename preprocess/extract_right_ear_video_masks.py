import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def largest_object_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([15, 70, 65]), np.array([48, 255, 255]))
    red = cv2.inRange(hsv, np.array([0, 80, 45]), np.array([14, 255, 255]))
    red |= cv2.inRange(hsv, np.array([165, 80, 45]), np.array([179, 255, 255]))
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 105]))
    mask = yellow | red | dark
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros(mask.shape, np.uint8)
    valid = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, count)]
    return ((labels == max(valid)[1]).astype(np.uint8) * 255)


def right_ear_mask(image: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(object_mask > 0)
    if len(xs) == 0:
        return np.zeros(object_mask.shape, np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    width, height = x1 - x0 + 1, y1 - y0 + 1
    center = np.array([(x0 + x1) * 0.5, y0 + height * 0.38])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark = ((hsv[:, :, 2] < 100) & (object_mask > 0)).astype(np.uint8)
    # Eyes and cheeks sit lower than either ear cap.
    dark[y0 + int(height * 0.44):, :] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    candidates = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < max(20, width * height * 0.0005):
            continue
        point = centroids[label]
        horizontal = abs(point[0] - center[0]) / max(width, 1)
        vertical = abs(point[1] - center[1]) / max(height, 1)
        if point[1] > y0 + height * 0.43:
            continue
        # The toy's right ear is the strongly sideways-projecting ear.
        candidates.append((horizontal - 0.18 * vertical + area / (width * height), label))
    if not candidates:
        return np.zeros(object_mask.shape, np.uint8)
    tip_label = max(candidates)[1]
    tip = (labels == tip_label).astype(np.uint8) * 255
    tx, ty = centroids[tip_label]

    direction = center - np.array([tx, ty])
    length = float(np.linalg.norm(direction))
    if length < 1:
        return tip
    unit = direction / length
    root = np.array([tx, ty]) + unit * min(length * 0.58, width * 0.34)
    radius = max(5, int(width * 0.052))
    corridor = np.zeros(object_mask.shape, np.uint8)
    cv2.line(
        corridor,
        (int(round(tx)), int(round(ty))),
        (int(round(root[0])), int(round(root[1]))),
        255,
        radius * 2,
    )
    cv2.circle(corridor, (int(round(tx)), int(round(ty))), radius * 2, 255, -1)
    ear = cv2.bitwise_and(object_mask, corridor)
    ear |= tip
    ear = cv2.morphologyEx(ear, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return ear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--resize-width", type=int, default=1080)
    parser.add_argument("--search-frames", type=int, default=18)
    args = parser.parse_args()

    output = Path(args.output)
    images_dir = output / "images"
    masks_dir = output / "right_ear_masks"
    previews_dir = output / "previews"
    for directory in (images_dir, masks_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = np.linspace(0, frame_count - 1, args.count, dtype=int)
    previews = []
    manifest = []
    for index, target in enumerate(targets, 1):
        best = None
        best_id = target
        best_score = -1.0
        for frame_id in range(max(0, target - args.search_frames), min(frame_count, target + args.search_frames + 1), 3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            score = sharpness(frame)
            if score > best_score:
                best, best_id, best_score = frame, frame_id, score
        if best is None:
            continue
        if args.resize_width and best.shape[1] != args.resize_width:
            scale = args.resize_width / best.shape[1]
            best = cv2.resize(best, (args.resize_width, int(round(best.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        object_mask = largest_object_mask(best)
        ear_mask = right_ear_mask(best, object_mask)
        stem = f"{index:03d}"
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(str(masks_dir / f"{stem}.png"), ear_mask)

        ys, xs = np.where(object_mask > 0)
        if len(xs):
            pad = 30
            crop = best[max(0, ys.min() - pad):min(best.shape[0], ys.max() + pad),
                        max(0, xs.min() - pad):min(best.shape[1], xs.max() + pad)].copy()
            mask_crop = ear_mask[max(0, ys.min() - pad):min(best.shape[0], ys.max() + pad),
                                 max(0, xs.min() - pad):min(best.shape[1], xs.max() + pad)]
            overlay = np.zeros_like(crop)
            overlay[:, :, 1] = mask_crop
            crop = cv2.addWeighted(crop, 0.70, overlay, 0.30, 0)
            crop = cv2.resize(crop, (240, 320), interpolation=cv2.INTER_AREA)
            cv2.putText(crop, f"{stem} f{best_id}", (7, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2)
            previews.append(crop)
            cv2.imwrite(str(previews_dir / f"{stem}.jpg"), crop)
        manifest.append(f"{stem},{best_id},{best_score:.2f},{int((ear_mask > 0).sum())}")
    cap.release()

    columns = 6
    rows = math.ceil(len(previews) / columns)
    sheet = np.full((rows * 320, columns * 240, 3), 255, np.uint8)
    for index, preview in enumerate(previews):
        row, col = divmod(index, columns)
        sheet[row * 320:(row + 1) * 320, col * 240:(col + 1) * 240] = preview
    cv2.imwrite(str(output / "right_ear_masks_contact.jpg"), sheet)
    (output / "manifest.csv").write_text("image,video_frame,sharpness,mask_pixels\n" + "\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} frames and right-ear masks to {output}")


if __name__ == "__main__":
    main()
