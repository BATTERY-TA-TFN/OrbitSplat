import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", required=True)
    parser.add_argument("--output-dir", default="upright_ear_tip_masks")
    parser.add_argument("--preview", default="output/ear_tip_annotation/pikaqiu_video54/contact.jpg")
    args = parser.parse_args()

    source = Path(args.source_path)
    output = source / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    previews = []
    written = 0

    for image_path in sorted((source / "images").glob("*.jpg")):
        stem = image_path.stem
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        object_mask = cv2.imread(str(source / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.where(object_mask > 127)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        object_h = y1 - y0 + 1

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dark = (
            (hsv[:, :, 2] < 105)
            & (hsv[:, :, 1] < 180)
            & (object_mask > 127)
        ).astype(np.uint8)
        # Eye/nose/stripes are lower; ear caps live in the upper half.
        dark[y0 + int(object_h * 0.48):, :] = 0
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
        candidates = []
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < 40:
                continue
            cx, cy = centroids[label]
            # The upright ear cap is generally the larger, more central top component.
            centrality = abs(cx - (x0 + x1) / 2) / max(x1 - x0, 1)
            score = area * (1.3 - min(centrality, 1.0))
            candidates.append((score, label))
        if not candidates:
            continue
        label = max(candidates)[1]
        mask = (labels == label).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        cv2.imwrite(str(output / f"{stem}.png"), mask)
        written += 1

        preview = image.copy()
        overlay = np.zeros_like(preview)
        overlay[:, :, 1] = mask
        preview = cv2.addWeighted(preview, 0.72, overlay, 0.28, 0)
        preview = preview[max(0, y0 - 30):min(image.shape[0], y1 + 30),
                          max(0, x0 - 30):min(image.shape[1], x1 + 30)]
        preview = cv2.resize(preview, (240, 320), interpolation=cv2.INTER_AREA)
        cv2.putText(preview, stem, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        previews.append(preview)

    columns = 6
    rows = (len(previews) + columns - 1) // columns
    sheet = np.full((rows * 320, columns * 240, 3), 255, dtype=np.uint8)
    for index, preview in enumerate(previews):
        row, col = divmod(index, columns)
        sheet[row * 320:(row + 1) * 320, col * 240:(col + 1) * 240] = preview
    preview_path = Path(args.preview)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), sheet)
    print(f"Wrote {written} upright ear-tip masks to {output}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
