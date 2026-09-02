import csv
import json
import sys
from pathlib import Path

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.image_utils import psnr
from utils.loss_utils import ssim


DATA = ROOT / "data" / "realcap" / "pikaqiu_video54_colmap_undistorted"
OUT = ROOT / "output" / "evaluation"
MODELS = {
    "8-view mask-filtered": {
        "path": OUT / "pikaqiu_8view_mask_filtered",
        "split": DATA / "sparse_test.txt",
    },
    "40-view mask-filtered": {
        "path": OUT / "pikaqiu_40view_mask_filtered",
        "split": DATA / "sparse_test_40.txt",
    },
}
COMMON_IDS = [40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 52, 53]


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def as_tensor(image):
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()


def load_render(model, image_id):
    split = np.atleast_1d(np.loadtxt(model["split"], dtype=np.int32)).tolist()
    output_index = split.index(image_id)
    base = model["path"] / "test" / "ours_10000"
    return (
        load_rgb(base / "renders" / f"{output_index:05d}.png"),
        load_rgb(base / "gt" / f"{output_index:05d}.png"),
    )


def foreground_mask(image_id, shape):
    mask = cv2.imread(str(DATA / "masks" / f"{image_id + 1:03d}.png"), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 127


def evaluate_common_subset(lpips_fn):
    results = {}
    for name, model in MODELS.items():
        values = {"PSNR": [], "SSIM": [], "LPIPS": [], "foreground_PSNR": [], "foreground_crop_LPIPS": []}
        for image_id in COMMON_IDS:
            render, gt = load_render(model, image_id)
            render_t, gt_t = as_tensor(render), as_tensor(gt)
            with torch.no_grad():
                values["PSNR"].append(psnr(render_t, gt_t).item())
                values["SSIM"].append(ssim(render_t, gt_t).item())
                values["LPIPS"].append(lpips_fn(render_t, gt_t).item())

            mask = foreground_mask(image_id, render.shape)
            mse = np.mean((render[mask] - gt[mask]) ** 2)
            values["foreground_PSNR"].append(-10.0 * np.log10(max(mse, 1e-12)))
            ys, xs = np.where(mask)
            margin = 8
            x0, x1 = max(0, xs.min() - margin), min(render.shape[1], xs.max() + margin + 1)
            y0, y1 = max(0, ys.min() - margin), min(render.shape[0], ys.max() + margin + 1)
            pred_crop = cv2.resize(render[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_AREA)
            gt_crop = cv2.resize(gt[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_AREA)
            with torch.no_grad():
                values["foreground_crop_LPIPS"].append(lpips_fn(as_tensor(pred_crop), as_tensor(gt_crop)).item())
        results[name] = {key: float(np.mean(value)) for key, value in values.items()}
    return results


def save_metrics(results):
    json_path = OUT / "common_subset_metrics.json"
    json_path.write_text(json.dumps({"image_ids": COMMON_IDS, "metrics": results}, indent=2), encoding="utf-8")
    with (OUT / "common_subset_metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "PSNR", "SSIM", "LPIPS", "Foreground PSNR", "Foreground Crop LPIPS"])
        for name, value in results.items():
            writer.writerow([name, value["PSNR"], value["SSIM"], value["LPIPS"], value["foreground_PSNR"],
                             value["foreground_crop_LPIPS"]])


def save_chart(results):
    names = list(results)
    colors = ["#377eb8", "#e6a93d"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [("PSNR", "PSNR ↑"), ("SSIM", "SSIM ↑"), ("LPIPS", "LPIPS ↓")]
    for axis, (key, title) in zip(axes, metrics):
        values = [results[name][key] for name in names]
        bars = axis.bar(names, values, color=colors, width=0.62)
        axis.set_title(title, fontweight="bold")
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom")
    fig.suptitle("Common Unseen Top-View Subset (12 views)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "common_subset_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def add_label(image, label):
    canvas = Image.new("RGB", (image.width, image.height + 34), "white")
    canvas.paste(image, (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), label, fill="black", font=ImageFont.load_default())
    return canvas


def save_qualitative():
    selected_ids = [40, 43, 47, 53]
    rows = []
    for image_id in selected_ids:
        render_8, gt = load_render(MODELS["8-view mask-filtered"], image_id)
        render_40, _ = load_render(MODELS["40-view mask-filtered"], image_id)
        images = [
            add_label(Image.fromarray((gt * 255).astype(np.uint8)), f"GT | ID {image_id}"),
            add_label(Image.fromarray((render_8 * 255).astype(np.uint8)), "8-view"),
            add_label(Image.fromarray((render_40 * 255).astype(np.uint8)), "40-view"),
        ]
        row = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)), "white")
        x = 0
        for image in images:
            row.paste(image, (x, 0))
            x += image.width
        rows.append(row)
    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(OUT / "common_subset_qualitative.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()
    results = evaluate_common_subset(lpips_fn)
    save_metrics(results)
    save_chart(results)
    save_qualitative()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
