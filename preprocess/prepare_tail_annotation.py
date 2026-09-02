import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_FRAMES = "017,019,021,023,025,027,029,031,033,045,047,049,051"


def parse_frames(value: str) -> list[str]:
    return [item.strip().zfill(3) for item in value.split(",") if item.strip()]


def prepare(source: Path, frames: list[str], output: Path, padding: float) -> None:
    crop_dir = output / "crops"
    mask_dir = output / "masks_crop"
    preview_dir = output / "preview"
    crop_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    previews = []

    for stem in frames:
        image = cv2.imread(str(source / "images" / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        object_mask = cv2.imread(str(source / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or object_mask is None:
            raise RuntimeError(f"Missing image or mask for frame {stem}")
        ys, xs = np.where(object_mask > 127)
        if len(xs) == 0:
            raise RuntimeError(f"Empty object mask for frame {stem}")

        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        pad_x = int((x1 - x0) * padding)
        pad_y = int((y1 - y0) * padding)
        x0, x1 = max(0, x0 - pad_x), min(image.shape[1], x1 + pad_x)
        y0, y1 = max(0, y0 - pad_y), min(image.shape[0], y1 + pad_y)

        crop = image[y0:y1, x0:x1]
        cv2.imwrite(str(crop_dir / f"{stem}.png"), crop)
        blank_path = mask_dir / f"{stem}.png"
        if not blank_path.exists():
            cv2.imwrite(str(blank_path), np.zeros(crop.shape[:2], dtype=np.uint8))

        display = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_AREA)
        cv2.putText(display, stem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        previews.append(display)
        manifest[stem] = {
            "image_shape": list(image.shape[:2]),
            "crop": [int(x0), int(y0), int(x1), int(y1)],
        }

    columns = 4
    rows = (len(previews) + columns - 1) // columns
    sheet = np.full((rows * 320, columns * 320, 3), 255, dtype=np.uint8)
    for index, preview in enumerate(previews):
        row, col = divmod(index, columns)
        sheet[row * 320:(row + 1) * 320, col * 320:(col + 1) * 320] = preview
    cv2.imwrite(str(output / "contact_sheet.jpg"), sheet)
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Prepared {len(frames)} frames in {output}")
    print(f"Paint tail-only masks in {mask_dir}")


def finalize(source: Path, output: Path, tail_mask_dir: str) -> None:
    with open(output / "manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    destination = source / tail_mask_dir
    destination.mkdir(parents=True, exist_ok=True)
    previews = []

    for stem, info in manifest.items():
        height, width = info["image_shape"]
        x0, y0, x1, y1 = info["crop"]
        crop_mask = cv2.imread(str(output / "masks_crop" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if crop_mask is None:
            raise RuntimeError(f"Missing crop mask for frame {stem}")
        crop_mask = cv2.resize(crop_mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = np.where(crop_mask > 127, 255, 0).astype(np.uint8)
        object_mask = cv2.imread(str(source / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if object_mask is None:
            raise RuntimeError(f"Missing object mask for frame {stem}")
        full_mask = cv2.bitwise_and(full_mask, np.where(object_mask > 127, 255, 0).astype(np.uint8))
        cv2.imwrite(str(destination / f"{stem}.png"), full_mask)
        image = cv2.imread(str(source / "images" / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        overlay = np.zeros_like(image)
        overlay[:, :, 1] = full_mask
        preview = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        preview = cv2.resize(preview[y0:y1, x0:x1], (320, 320), interpolation=cv2.INTER_AREA)
        cv2.putText(preview, stem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        previews.append(preview)
    columns = 4
    rows = (len(previews) + columns - 1) // columns
    sheet = np.full((rows * 320, columns * 320, 3), 255, dtype=np.uint8)
    for index, preview in enumerate(previews):
        row, col = divmod(index, columns)
        sheet[row * 320:(row + 1) * 320, col * 320:(col + 1) * 320] = preview
    cv2.imwrite(str(output / "tail_masks_contact_sheet.jpg"), sheet)
    print(f"Wrote {len(manifest)} full-resolution tail masks to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", required=True)
    parser.add_argument("--output", default="output/tail_annotation/pikaqiu_video54")
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--padding", type=float, default=0.18)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--tail-mask-dir", default="tail_masks")
    args = parser.parse_args()

    source = Path(args.source_path)
    output = Path(args.output)
    if args.finalize:
        finalize(source, output, args.tail_mask_dir)
    else:
        prepare(source, parse_frames(args.frames), output, args.padding)


if __name__ == "__main__":
    main()
