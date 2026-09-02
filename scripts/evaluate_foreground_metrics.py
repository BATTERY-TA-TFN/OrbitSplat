import json
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "realcap" / "pikaqiu_video54_colmap_undistorted"
MODELS = {
    "Raw COLMAP": ROOT / "output" / "evaluation" / "pikaqiu_8view_raw_colmap",
    "Mask-filtered COLMAP": ROOT / "output" / "evaluation" / "pikaqiu_8view_mask_filtered",
    "Visual Hull": ROOT / "output" / "evaluation" / "pikaqiu_8view_visual_hull",
}


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def tensor(image):
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()


def main():
    test_ids = np.atleast_1d(np.loadtxt(DATA / "sparse_test.txt", dtype=np.int32))
    lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()
    results = {}
    for name, model_dir in MODELS.items():
        render_dir = model_dir / "test" / "ours_10000" / "renders"
        gt_dir = model_dir / "test" / "ours_10000" / "gt"
        foreground_psnr = []
        crop_lpips = []
        for output_index, image_id in enumerate(test_ids):
            render = load_rgb(render_dir / f"{output_index:05d}.png")
            gt = load_rgb(gt_dir / f"{output_index:05d}.png")
            mask = cv2.imread(str(DATA / "masks" / f"{image_id + 1:03d}.png"), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
            mse = np.mean((render[mask] - gt[mask]) ** 2)
            foreground_psnr.append(-10.0 * np.log10(max(mse, 1e-12)))

            ys, xs = np.where(mask)
            margin = 8
            x0, x1 = max(0, xs.min() - margin), min(render.shape[1], xs.max() + margin + 1)
            y0, y1 = max(0, ys.min() - margin), min(render.shape[0], ys.max() + margin + 1)
            pred_crop = cv2.resize(render[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_AREA)
            gt_crop = cv2.resize(gt[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_AREA)
            with torch.no_grad():
                crop_lpips.append(lpips_fn(tensor(pred_crop), tensor(gt_crop)).item())
        results[name] = {
            "foreground_PSNR": float(np.mean(foreground_psnr)),
            "foreground_crop_LPIPS": float(np.mean(crop_lpips)),
        }

    output = ROOT / "output" / "evaluation" / "foreground_metrics.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(output)


if __name__ == "__main__":
    main()
