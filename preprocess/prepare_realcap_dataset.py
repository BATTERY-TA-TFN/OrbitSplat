import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def normalize_image(src: Path, dst: Path) -> None:
    image = Image.open(src)
    image = ImageOps.exif_transpose(image).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst, quality=95)


def clean_mask(src: Path, dst: Path) -> None:
    mask = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {src}")
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", required=True)
    parser.add_argument("--image-input", default="image")
    parser.add_argument("--mask-input", default="mask")
    parser.add_argument("--images-output", default="images")
    parser.add_argument("--masks-output", default="masks")
    args = parser.parse_args()

    root = Path(args.source_path)
    image_input = root / args.image_input
    mask_input = root / args.mask_input
    images_output = root / args.images_output
    masks_output = root / args.masks_output

    image_paths = sorted(path for path in image_input.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise RuntimeError(f"No images found in {image_input}")

    for image_path in image_paths:
        mask_path = mask_input / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise RuntimeError(f"Missing mask for {image_path.name}: {mask_path}")

        image_dst = images_output / image_path.name
        mask_dst = masks_output / mask_path.name
        normalize_image(image_path, image_dst)
        clean_mask(mask_path, mask_dst)
        print(f"Prepared {image_dst} and {mask_dst}")


if __name__ == "__main__":
    main()
