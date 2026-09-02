import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "evaluation"
MODELS = {
    "Raw COLMAP": ("pikaqiu_8view_raw_colmap", 5727),
    "Mask-filtered": ("pikaqiu_8view_mask_filtered", 285),
    "Visual Hull": ("pikaqiu_8view_visual_hull", 7787),
}


def metrics():
    result = {}
    for name, (folder, initial_points) in MODELS.items():
        path = OUT / folder / "test" / "ours_10000" / "results.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["initial_points"] = initial_points
        result[name] = value
    (OUT / "initialization_ablation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def chart(result):
    names = list(result)
    colors = ["#777777", "#e6a93d", "#377eb8"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    settings = [
        ("initial_points", "Initial points ↓", "{:.0f}"),
        ("PSNR", "PSNR ↑", "{:.3f}"),
        ("SSIM", "SSIM ↑", "{:.4f}"),
        ("LPIPS", "LPIPS ↓", "{:.4f}"),
    ]
    for axis, (key, title, pattern) in zip(axes, settings):
        values = [result[name][key] for name in names]
        bars = axis.bar(names, values, color=colors, width=0.65)
        axis.set_title(title, fontweight="bold")
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), pattern.format(value),
                      ha="center", va="bottom")
    fig.suptitle("8-view Initialization Ablation (46 held-out views)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "initialization_ablation_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def labeled(image, text):
    canvas = Image.new("RGB", (image.width, image.height + 30), "white")
    canvas.paste(image, (0, 30))
    ImageDraw.Draw(canvas).text((8, 7), text, fill="black", font=ImageFont.load_default())
    return canvas


def qualitative():
    # Output indices correspond to held-out image IDs 40, 43, 47, and 53.
    output_indices = [34, 37, 41, 45]
    rows = []
    for index in output_indices:
        columns = []
        gt_path = OUT / "pikaqiu_8view_raw_colmap" / "test" / "ours_10000" / "gt" / f"{index:05d}.png"
        columns.append(labeled(Image.open(gt_path).convert("RGB"), "Ground truth"))
        for name, (folder, _) in MODELS.items():
            path = OUT / folder / "test" / "ours_10000" / "renders" / f"{index:05d}.png"
            columns.append(labeled(Image.open(path).convert("RGB"), name))
        row = Image.new("RGB", (sum(image.width for image in columns), max(image.height for image in columns)), "white")
        x = 0
        for image in columns:
            row.paste(image, (x, 0))
            x += image.width
        rows.append(row)
    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(OUT / "initialization_ablation_qualitative.png")


def main():
    result = metrics()
    chart(result)
    qualitative()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
