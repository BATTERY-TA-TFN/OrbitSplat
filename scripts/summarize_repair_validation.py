import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "evaluation"
RUNS = {
    "Coarse 3DGS": OUT / "pikaqiu_40view_mask_filtered" / "train" / "ours_None",
    "Gaussian Repair": OUT / "pikaqiu_8view_mask_filtered" / "train" / "ours_None",
}


def labeled(image, text):
    canvas = Image.new("RGB", (image.width, image.height + 30), "white")
    canvas.paste(image, (0, 30))
    ImageDraw.Draw(canvas).text((8, 7), text, fill="black", font=ImageFont.load_default())
    return canvas


def main():
    results = {
        name: json.loads((path / "results.json").read_text(encoding="utf-8"))
        for name, path in RUNS.items()
    }
    payload = {
        "scope": "Engineering validation only: both models originate from a full-view HQ model.",
        "metrics": results,
    }
    (OUT / "repair_engineering_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    names = list(results)
    colors = ["#777777", "#d95f5f"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (key, title) in zip(axes, [("PSNR", "PSNR ↑"), ("SSIM", "SSIM ↑"), ("LPIPS", "LPIPS ↓")]):
        values = [results[name][key] for name in names]
        bars = axis.bar(names, values, color=colors, width=0.62)
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom")
    fig.suptitle("Gaussian Repair Engineering Validation (54 fitted views, not held-out)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "repair_engineering_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for index in [0, 14, 28, 43, 53]:
        columns = []
        gt = Image.open(RUNS["Coarse 3DGS"] / "gt" / f"{index:05d}.png").convert("RGB")
        columns.append(labeled(gt, f"Ground truth | ID {index}"))
        for name, path in RUNS.items():
            columns.append(labeled(Image.open(path / "renders" / f"{index:05d}.png").convert("RGB"), name))
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
    sheet.save(OUT / "repair_engineering_qualitative.png")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
