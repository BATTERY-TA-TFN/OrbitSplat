import argparse
from pathlib import Path

import cv2
import numpy as np


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--trim-start", type=float, default=0.05)
    parser.add_argument("--trim-end", type=float, default=0.95)
    parser.add_argument("--search-seconds", type=float, default=0.25)
    parser.add_argument("--resize-width", type=int, default=1080)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        videos = sorted(input_path.glob("*.mp4"), key=lambda path: path.stat().st_size, reverse=True)
        if not videos:
            raise RuntimeError(f"No mp4 videos found in {input_path}")
        input_path = videos[0]

    output_dir = Path(args.output) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError(f"Failed to read video metadata: {input_path}")

    search = max(1, int(args.search_seconds * fps))
    targets = np.linspace(
        int(frame_count * args.trim_start),
        int(frame_count * args.trim_end),
        args.count,
        endpoint=False,
        dtype=int,
    )
    for index, target in enumerate(targets, 1):
        best_frame = None
        best_score = -1.0
        for frame_id in range(max(0, target - search), min(frame_count, target + search + 1), 3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            score = sharpness(frame)
            if score > best_score:
                best_score = score
                best_frame = frame
        if best_frame is None:
            raise RuntimeError(f"Failed to extract frame near {target}")
        if args.resize_width and best_frame.shape[1] != args.resize_width:
            scale = args.resize_width / best_frame.shape[1]
            best_frame = cv2.resize(
                best_frame,
                (args.resize_width, int(round(best_frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        output_path = output_dir / f"{index:03d}.jpg"
        cv2.imwrite(str(output_path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
        print(f"Wrote {output_path} sharpness={best_score:.1f}")
    cap.release()

    np.savetxt(Path(args.output) / f"sparse_{args.count}.txt", np.arange(args.count), fmt="%d")
    np.savetxt(Path(args.output) / "sparse_test.txt", np.array([], dtype=np.int32), fmt="%d")
    print(f"Extracted {args.count} frames from {input_path}")


if __name__ == "__main__":
    main()
