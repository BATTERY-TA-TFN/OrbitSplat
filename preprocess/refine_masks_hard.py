import argparse
from pathlib import Path

import cv2
import numpy as np


def make_contact_sheet(items: list[np.ndarray], output: Path, columns: int = 4) -> None:
    if not items:
        return
    cell_h, cell_w = items[0].shape[:2]
    rows = (len(items) + columns - 1) // columns
    sheet = np.full((rows * cell_h, columns * cell_w, 3), 255, dtype=np.uint8)
    for index, item in enumerate(items):
        row, col = divmod(index, columns)
        sheet[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = item
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", required=True)
    parser.add_argument("--input-dir", default="masks")
    parser.add_argument("--output-dir", default="masks_hard_erode3")
    parser.add_argument("--erode-pixels", type=int, default=3)
    parser.add_argument("--preview", default="output/video_inspection/masks_hard_erode3_contact.jpg")
    args = parser.parse_args()

    source = Path(args.source_path)
    input_dir = source / args.input_dir
    output_dir = source / args.output_dir
    image_dir = source / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_size = args.erode_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    preview_ids = {0, 7, 14, 21, 28, 35, 44, 51}
    previews = []
    area_ratios = []

    mask_paths = sorted(input_dir.glob("*.png"))
    for index, mask_path in enumerate(mask_paths):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read {mask_path}")
        hard = np.where(mask >= 127, 255, 0).astype(np.uint8)
        refined = cv2.erode(hard, kernel, iterations=1)
        cv2.imwrite(str(output_dir / mask_path.name), refined)

        original_area = max(int((hard > 0).sum()), 1)
        area_ratios.append(float((refined > 0).sum()) / original_area)

        if index in preview_ids:
            image_path = image_dir / f"{mask_path.stem}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image = cv2.resize(image, (240, 426), interpolation=cv2.INTER_AREA)
            old_small = cv2.resize(hard, (240, 426), interpolation=cv2.INTER_NEAREST)
            new_small = cv2.resize(refined, (240, 426), interpolation=cv2.INTER_NEAREST)
            old_edge = cv2.morphologyEx(old_small, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            new_edge = cv2.morphologyEx(new_small, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            image[old_edge > 0] = (0, 0, 255)
            image[new_edge > 0] = (0, 255, 0)
            previews.append(image)

    make_contact_sheet(previews, Path(args.preview))
    print(f"Wrote {len(mask_paths)} masks to {output_dir}")
    print(f"Mean retained area: {np.mean(area_ratios):.4f}")
    print(f"Preview: {args.preview} (red=old, green=new)")


if __name__ == "__main__":
    main()
