import argparse
from pathlib import Path

import cv2
import numpy as np


def find_foreground_bbox(image_paths, threshold=245):
    xs = []
    ys = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        foreground = np.any(image < threshold, axis=2)
        y, x = np.where(foreground)
        if len(x) == 0:
            continue
        xs.extend([int(x.min()), int(x.max())])
        ys.extend([int(y.min()), int(y.max())])

    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def expand_square_bbox(bbox, width, height, padding):
    x0, y0, x1, y1 = bbox
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    size = int(max(box_w, box_h) * (1.0 + padding * 2))
    size = max(size, 32)

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    x0 = cx - size // 2
    y0 = cy - size // 2
    x1 = x0 + size
    y1 = y0 + size

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height

    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def write_video(image_paths, output_path, fps, zoom, padding):
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read image: {image_paths[0]}")
    height, width = first.shape[:2]

    crop = None
    if zoom:
        bbox = find_foreground_bbox(image_paths)
        if bbox is not None:
            crop = expand_square_bbox(bbox, width, height, padding)

    out_width, out_height = width, height
    if crop is not None:
        out_width, out_height = width, height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, out_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    for path in image_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {path}")
        if crop is not None:
            x0, y0, x1, y1 = crop
            frame = frame[y0:y1, x0:x1]
            frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
        writer.write(frame)
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir")
    parser.add_argument("--output", default="")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--zoom", action="store_true")
    parser.add_argument("--padding", type=float, default=0.35)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    image_paths = sorted(frames_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")

    output_path = Path(args.output) if args.output else frames_dir.parent / ("renders_zoom.mp4" if args.zoom else "renders_fixed.mp4")
    write_video(image_paths, output_path, args.fps, args.zoom, args.padding)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
