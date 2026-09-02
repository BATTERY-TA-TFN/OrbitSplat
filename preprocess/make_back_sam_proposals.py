import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


FRAMES = ("017", "019", "021", "023", "025", "027", "029", "031", "033", "047", "048", "049")


def resolve_checkpoint(filename: str) -> Path:
    local = Path("models") / filename
    if local.exists():
        return local
    model_dir = os.environ.get("GAUSSIANOBJECT_MODEL_DIR")
    if model_dir and (Path(model_dir) / filename).exists():
        return Path(model_dir) / filename
    return local


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", required=True)
    parser.add_argument("--checkpoint", default="sam_vit_b_01ec64.pth")
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--output-dir", default="back_masks")
    parser.add_argument("--preview", default="output/back_annotation/pikaqiu_video54/back_masks_contact.jpg")
    args = parser.parse_args()

    source = Path(args.source_path)
    checkpoint = resolve_checkpoint(args.checkpoint)
    if not checkpoint.exists():
        raise RuntimeError(f"Missing SAM checkpoint: {checkpoint}")
    model = sam_model_registry[args.model_type](checkpoint=str(checkpoint)).cuda()
    predictor = SamPredictor(model)
    output_dir = source / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    previews = []

    for stem in FRAMES:
        image = cv2.imread(str(source / "images" / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        object_mask = cv2.imread(str(source / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.where(object_mask > 127)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        width, height = x1 - x0 + 1, y1 - y0 + 1

        # The spherical head occupies the middle-upper part of the object box.
        box = np.array([
            x0 + 0.08 * width,
            y0 + 0.20 * height,
            x0 + 0.92 * width,
            y0 + 0.73 * height,
        ], dtype=np.float32)
        point = np.array([[
            x0 + 0.52 * width,
            y0 + 0.48 * height,
        ]], dtype=np.float32)

        predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(
            point_coords=point,
            point_labels=np.ones(1, dtype=np.int32),
            box=box,
            multimask_output=True,
        )
        candidates = []
        for mask, score in zip(masks, scores):
            clipped = mask & (object_mask > 127)
            area_ratio = clipped.sum() / max((object_mask > 127).sum(), 1)
            # Prefer a large head component while rejecting whole-object proposals.
            quality = float(score) - abs(area_ratio - 0.50)
            candidates.append((quality, clipped))
        mask = largest_component(max(candidates, key=lambda item: item[0])[1])
        cv2.imwrite(str(output_dir / f"{stem}.png"), mask.astype(np.uint8) * 255)

        preview = image.copy()
        overlay = np.zeros_like(preview)
        overlay[:, :, 1] = mask.astype(np.uint8) * 255
        preview = cv2.addWeighted(preview, 0.70, overlay, 0.30, 0)
        preview = preview[max(0, y0 - 30):min(image.shape[0], y1 + 30),
                          max(0, x0 - 30):min(image.shape[1], x1 + 30)]
        preview = cv2.resize(preview, (320, 320), interpolation=cv2.INTER_AREA)
        cv2.putText(preview, stem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        previews.append(preview)
        print(f"Wrote {output_dir / f'{stem}.png'}")

    columns = 4
    rows = (len(previews) + columns - 1) // columns
    sheet = np.full((rows * 320, columns * 320, 3), 255, dtype=np.uint8)
    for index, preview in enumerate(previews):
        row, col = divmod(index, columns)
        sheet[row * 320:(row + 1) * 320, col * 320:(col + 1) * 320] = preview
    preview_path = Path(args.preview)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), sheet)
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
