from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "data" / "realcap" / "pikaqiu_video40_clean"
OUTPUT = ROOT / "output" / "figures" / "mask_pipeline_innovation.png"

W, H = 2400, 1350
NAVY = "#17355c"
BLUE = "#2f80ed"
CYAN = "#eaf5ff"
ORANGE = "#f2994a"
GREEN = "#27ae60"
GRAY = "#5f6b7a"
LIGHT = "#f5f8fc"
WHITE = "#ffffff"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def rounded(draw, box, radius=24, fill=WHITE, outline=None, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, color=BLUE, width=12, head=22):
    draw.line([start, end], fill=color, width=width)
    x2, y2 = end
    draw.polygon([(x2, y2), (x2 - head, y2 - head // 2), (x2 - head, y2 + head // 2)], fill=color)


def fit_image(path: Path, size):
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.53))


def draw_centered(draw, text, center_x, y, fnt, fill):
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((center_x - (box[2] - box[0]) / 2, y), text, font=fnt, fill=fill)


def main():
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)

    title_font = font(56, True)
    subtitle_font = font(27)
    stage_font = font(31, True)
    small_font = font(23)
    module_font = font(28, True)
    emphasis_font = font(32, True)

    draw_centered(draw, "由粗到精的物体掩膜生成", W // 2, 42, title_font, NAVY)
    draw_centered(
        draw,
        "颜色先验定位目标，SAM 精细恢复轮廓，后处理保证几何一致性",
        W // 2,
        116,
        subtitle_font,
        GRAY,
    )

    paths = [
        SCENE / "images" / "001.jpg",
        SCENE / "mask_previews_color" / "001.png",
        SCENE / "sam_mask_previews" / "001.png",
        SCENE / "masks" / "001.png",
    ]
    labels = ["原始图像", "颜色先验粗分割", "SAM 精细分割", "最终二值掩膜"]
    notes = ["真实输入", "快速定位，存在边界外溢", "提示框引导，恢复精细轮廓", "连通域保留与形态学清理"]

    card_w, card_h = 475, 655
    image_h = 505
    gap = 95
    left = (W - (card_w * 4 + gap * 3)) // 2
    top = 200

    for i, (path, label, note) in enumerate(zip(paths, labels, notes)):
        x = left + i * (card_w + gap)
        rounded(draw, (x, top, x + card_w, top + card_h), 26, WHITE, "#d8e3f0", 3)
        image = fit_image(path, (card_w - 24, image_h))
        canvas.paste(image, (x + 12, top + 12))
        draw_centered(draw, label, x + card_w // 2, top + image_h + 31, stage_font, NAVY)
        draw_centered(draw, note, x + card_w // 2, top + image_h + 82, small_font, GRAY)

        if i < 3:
            arrow(draw, (x + card_w + 20, top + 265), (x + card_w + gap - 20, top + 265))

    # Step badges above the transitions.
    badge_y = top + 345
    badges = [
        ("生成目标区域\n与提示框", left + card_w + gap // 2),
        ("最大连通域保留", left + 2 * card_w + gap + gap // 2),
        ("孔洞填充\n形态学清理", left + 3 * card_w + 2 * gap + gap // 2),
    ]
    for text, cx in badges:
        rounded(draw, (cx - 72, badge_y, cx + 72, badge_y + 76), 18, CYAN, BLUE, 2)
        lines = text.split("\n")
        for j, line in enumerate(lines):
            draw_centered(draw, line, cx, badge_y + 10 + j * 29, font(19, True), BLUE)

    # Downstream dependency panel.
    panel_top = 930
    rounded(draw, (95, panel_top, W - 95, H - 55), 32, LIGHT, "#d8e3f0", 3)
    draw_centered(draw, "高质量掩膜是二维观测连接三维结构优化的共同基础", W // 2, panel_top + 30, module_font, NAVY)

    final_center = left + 3 * (card_w + gap) + card_w // 2
    draw.line([(final_center, top + card_h), (final_center, panel_top - 8)], fill=BLUE, width=9)
    draw.polygon(
        [(final_center, panel_top + 8), (final_center - 18, panel_top - 18), (final_center + 18, panel_top - 18)],
        fill=BLUE,
    )

    modules = [
        ("Visual Hull", "约束初始几何包络", BLUE),
        ("轮廓损失", "监督渲染轮廓一致性", ORANGE),
        ("点云过滤", "去除背景与漂浮结构", GREEN),
    ]
    module_w, module_h, module_gap = 550, 150, 100
    module_left = (W - (module_w * 3 + module_gap * 2)) // 2
    module_y = panel_top + 100
    for i, (name, desc, color) in enumerate(modules):
        x = module_left + i * (module_w + module_gap)
        rounded(draw, (x, module_y, x + module_w, module_y + module_h), 24, WHITE, color, 4)
        draw.ellipse((x + 28, module_y + 35, x + 108, module_y + 115), fill=color)
        draw_centered(draw, str(i + 1), x + 68, module_y + 48, font(30, True), WHITE)
        draw.text((x + 135, module_y + 30), name, font=module_font, fill=NAVY)
        draw.text((x + 135, module_y + 82), desc, font=small_font, fill=GRAY)

    draw_centered(
        draw,
        "掩膜误检会引入冗余几何，目标缺失会造成表面不完整",
        W // 2,
        H - 112,
        emphasis_font,
        "#c85c16",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=96)
    print(OUTPUT)


if __name__ == "__main__":
    main()
