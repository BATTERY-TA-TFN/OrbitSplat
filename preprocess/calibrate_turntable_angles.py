import argparse
from pathlib import Path

import cv2
import numpy as np


def load_gray(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read {path}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def load_mask(path: Path, size: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read {path}")
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def robust_motion(flow: np.ndarray, mask: np.ndarray) -> float:
    magnitude = np.linalg.norm(flow, axis=2)
    values = magnitude[mask]
    if len(values) == 0:
        return 0.0
    cutoff = np.percentile(values, 95)
    values = values[values <= cutoff]
    return float(np.percentile(values, 70))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--flow-size", type=int, default=384)
    parser.add_argument("--smooth-passes", type=int, default=3)
    parser.add_argument("--min-step-ratio", type=float, default=0.35)
    parser.add_argument("--max-step-ratio", type=float, default=2.2)
    args = parser.parse_args()

    source = Path(args.source_path)
    image_paths = sorted((source / "images").glob("*"))
    mask_paths = sorted((source / "masks").glob("*"))
    if len(image_paths) != len(mask_paths) or len(image_paths) < 3:
        raise RuntimeError("images and masks must contain the same number of files")

    images = [load_gray(path, args.flow_size) for path in image_paths]
    masks = [load_mask(path, args.flow_size) for path in mask_paths]
    kernel = np.ones((5, 5), np.uint8)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    motions = []
    for index in range(len(images)):
        next_index = (index + 1) % len(images)
        flow = dis.calc(images[index], images[next_index], None)
        common_mask = masks[index] & masks[next_index]
        common_mask = cv2.erode(common_mask.astype(np.uint8), kernel) > 0
        motions.append(robust_motion(flow, common_mask))

    steps = np.asarray(motions, dtype=np.float64)
    median = np.median(steps)
    steps = np.clip(
        steps,
        median * args.min_step_ratio,
        median * args.max_step_ratio,
    )
    for _ in range(args.smooth_passes):
        steps = 0.25 * np.roll(steps, 1) + 0.5 * steps + 0.25 * np.roll(steps, -1)

    steps *= 360.0 / steps.sum()
    angles = np.concatenate([[0.0], np.cumsum(steps[:-1])])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, angles, fmt="%.8f")

    uniform_step = 360.0 / len(images)
    print(f"Wrote {len(angles)} calibrated angles to {output}")
    print(
        f"Step range: {steps.min():.3f} to {steps.max():.3f} degrees "
        f"(uniform would be {uniform_step:.3f})"
    )
    print(f"Step standard deviation: {steps.std():.3f} degrees")


if __name__ == "__main__":
    main()
