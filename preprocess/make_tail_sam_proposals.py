import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


# Normalized boxes tightly surround the tail in the prepared annotation crops.
TAIL_BOXES = {
    "017": (0.47, 0.58, 0.77, 0.96),
    "019": (0.45, 0.57, 0.76, 0.96),
    "021": (0.43, 0.58, 0.75, 0.96),
    "023": (0.38, 0.58, 0.73, 0.96),
    "025": (0.27, 0.62, 0.67, 0.96),
    "027": (0.22, 0.62, 0.65, 0.96),
    "029": (0.12, 0.62, 0.55, 0.96),
    "031": (0.02, 0.60, 0.45, 0.97),
    "033": (0.00, 0.57, 0.34, 0.97),
    "045": (0.79, 0.57, 1.00, 0.98),
    "047": (0.66, 0.63, 1.00, 0.98),
    "049": (0.17, 0.64, 0.57, 0.99),
    "051": (0.00, 0.51, 0.34, 0.99),
}


def resolve_checkpoint(filename: str) -> Path:
    local = Path("models") / filename
    if local.exists():
        return local
    model_dir = os.environ.get("GAUSSIANOBJECT_MODEL_DIR")
    if model_dir and (Path(model_dir) / filename).exists():
        return Path(model_dir) / filename
    return local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", default="output/tail_annotation/pikaqiu_video54")
    parser.add_argument("--checkpoint", default="sam_vit_b_01ec64.pth")
    parser.add_argument("--model-type", default="vit_b")
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir)
    checkpoint = resolve_checkpoint(args.checkpoint)
    if not checkpoint.exists():
        raise RuntimeError(f"Missing SAM checkpoint: {checkpoint}")
    model = sam_model_registry[args.model_type](checkpoint=str(checkpoint)).cuda()
    predictor = SamPredictor(model)

    preview_dir = annotation_dir / "sam_tail_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    previews = []
    for stem, normalized_box in TAIL_BOXES.items():
        image = cv2.imread(str(annotation_dir / "crops" / f"{stem}.png"), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Missing crop {stem}")
        height, width = image.shape[:2]
        x0, y0, x1, y1 = normalized_box
        box = np.array([x0 * width, y0 * height, x1 * width, y1 * height], dtype=np.float32)
        center = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]], dtype=np.float32)

        predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(
            point_coords=center,
            point_labels=np.ones(1, dtype=np.int32),
            box=box,
            multimask_output=True,
        )
        mask = masks[int(np.argmax(scores))]
        cv2.imwrite(
            str(annotation_dir / "masks_crop" / f"{stem}.png"),
            mask.astype(np.uint8) * 255,
        )

        preview = image.copy()
        overlay = np.zeros_like(preview)
        overlay[:, :, 1] = mask.astype(np.uint8) * 255
        preview = cv2.addWeighted(preview, 0.7, overlay, 0.3, 0)
        cv2.rectangle(
            preview,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 255),
            2,
        )
        cv2.imwrite(str(preview_dir / f"{stem}.png"), preview)
        small = cv2.resize(preview, (320, 320), interpolation=cv2.INTER_AREA)
        cv2.putText(small, stem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        previews.append(small)
        print(f"Wrote tail proposal {stem}, score={scores.max():.3f}")

    columns = 4
    rows = (len(previews) + columns - 1) // columns
    sheet = np.full((rows * 320, columns * 320, 3), 255, dtype=np.uint8)
    for index, preview in enumerate(previews):
        row, col = divmod(index, columns)
        sheet[row * 320:(row + 1) * 320, col * 320:(col + 1) * 320] = preview
    cv2.imwrite(str(annotation_dir / "sam_tail_contact_sheet.jpg"), sheet)


if __name__ == "__main__":
    main()
