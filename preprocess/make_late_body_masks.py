import argparse
from pathlib import Path

import cv2
import numpy as np


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create late-view masks that retain the main head/body while removing thin appendages."
    )
    parser.add_argument("-s", "--source-path", required=True)
    parser.add_argument("--first-frame", type=int, default=31)
    parser.add_argument("--last-frame", type=int, default=54)
    parser.add_argument("--output-dir", default="late_body_masks")
    parser.add_argument("--preview", default="output/body_annotation/pikaqiu_video54/late_body_contact.jpg")
    args = parser.parse_args()

    source = Path(args.source_path)
    output = source / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    previews = []

    for mask_path in sorted((source / "masks").glob("*.png")):
        frame = int(mask_path.stem)
        if frame < args.first_frame or frame > args.last_frame:
            continue
        image = cv2.imread(str(source / "images" / f"{mask_path.stem}.jpg"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # A large elliptical opening disconnects ears and tail while retaining the
        # spherical head/body. Restore a little boundary thickness afterwards.
        opened = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
        )
        body = largest_component(opened)
        body = cv2.dilate(
            body,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        body = cv2.bitwise_and(body, mask)
        cv2.imwrite(str(output / mask_path.name), body)

        ys, xs = np.where(mask > 127)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        preview = image.copy()
        overlay = np.zeros_like(preview)
        overlay[:, :, 1] = body
        preview = cv2.addWeighted(preview, 0.70, overlay, 0.30, 0)
        preview = preview[
            max(0, y0 - 30):min(image.shape[0], y1 + 30),
            max(0, x0 - 30):min(image.shape[1], x1 + 30),
        ]
        preview = cv2.resize(preview, (280, 320), interpolation=cv2.INTER_AREA)
        cv2.putText(preview, mask_path.stem, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        previews.append(preview)

    columns = 6
    rows = (len(previews) + columns - 1) // columns
    sheet = np.full((rows * 320, columns * 280, 3), 255, dtype=np.uint8)
    for index, preview in enumerate(previews):
        row, col = divmod(index, columns)
        sheet[row * 320:(row + 1) * 320, col * 280:(col + 1) * 280] = preview
    preview_path = Path(args.preview)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), sheet)
    print(f"Wrote {len(previews)} late body masks to {output}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
