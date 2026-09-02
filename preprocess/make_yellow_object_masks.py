import argparse
from pathlib import Path

import cv2
import numpy as np


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(bool)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest


def fill_holes(mask):
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", required=True)
    args = parser.parse_args()

    scene = Path(args.source_path)
    mask_dir = scene / "masks"
    preview_dir = scene / "mask_previews_color"
    mask_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted((scene / "images").glob("*")):
        image = cv2.imread(str(image_path))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array([18, 105, 90]), np.array([42, 255, 255]))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        yellow = largest_component(yellow).astype(np.uint8) * 255

        near = cv2.dilate(yellow, np.ones((45, 45), np.uint8), iterations=1)
        red = cv2.inRange(hsv, np.array([0, 90, 60]), np.array([12, 255, 255]))
        red |= cv2.inRange(hsv, np.array([165, 90, 60]), np.array([179, 255, 255]))
        dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 90]))
        details = cv2.bitwise_and(red | dark, near)

        mask = yellow | details
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8), iterations=2)
        mask = largest_component(mask).astype(np.uint8) * 255
        mask = fill_holes(mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        cv2.imwrite(str(mask_dir / f"{image_path.stem}.png"), mask)
        overlay = np.zeros_like(image)
        overlay[:, :, 1] = mask
        preview = cv2.addWeighted(image, 0.72, overlay, 0.28, 0)
        cv2.imwrite(str(preview_dir / f"{image_path.stem}.png"), preview)
        print(f"Wrote {mask_dir / f'{image_path.stem}.png'}")


if __name__ == "__main__":
    main()
