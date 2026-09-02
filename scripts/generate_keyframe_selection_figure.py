from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "data" / "realcap" / "pikaqiu_video"
FRAME_DIR = ROOT / "output" / "keyframe_selection_frames"
OUTPUT = ROOT / "output" / "keyframe_selection_figure.png"
FFMPEG = ROOT / ".conda-pkgs" / "ffmpeg-4.3.1-ha925a31_0" / "Library" / "bin" / "ffmpeg.exe"
FFPROBE = FFMPEG.with_name("ffprobe.exe")

W, H = 2400, 1350
BG, TEXT, MUTED = "#F7F9FC", "#172033", "#667085"
BLUE, GREEN, RED = "#356AE6", "#159455", "#D92D20"


def font(size, bold=False):
    for candidate in [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def video_duration(path):
    command = [
        str(FFPROBE), "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    return float(subprocess.check_output(command, text=True).strip())


def extract_frame(video, timestamp, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(FFMPEG), "-loglevel", "error", "-y", "-ss", f"{timestamp:.3f}",
            "-i", str(video), "-frames:v", "1", "-q:v", "2", str(output),
        ],
        check=True,
    )


def sharpness(path):
    gray = np.asarray(Image.open(path).convert("L").resize((360, 640)), dtype=np.float32)
    center = -4 * gray[1:-1, 1:-1]
    lap = center + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def prepare_candidates():
    videos = [
        ("正面视频", VIDEO_DIR / "正面.mp4", 0.42),
        ("俯拍旋转视频", VIDEO_DIR / "俯拍旋转.mp4", 0.52),
        ("侧面俯拍视频", VIDEO_DIR / "侧面俯拍.mp4", 0.58),
    ]
    groups = []
    for group_index, (label, video, ratio) in enumerate(videos, 1):
        duration = video_duration(video)
        center = duration * ratio
        timestamps = [max(0.0, center - 0.25), center, min(duration - 0.05, center + 0.25)]
        paths = []
        scores = []
        for candidate_index, timestamp in enumerate(timestamps, 1):
            output = FRAME_DIR / f"window_{group_index}_candidate_{candidate_index}.jpg"
            extract_frame(video, timestamp, output)
            paths.append(output)
            scores.append(sharpness(output))
        best = int(np.argmax(scores))
        groups.append({"label": label, "paths": paths, "scores": scores, "best": best})
    return groups


def fit_image(path, size, dim=False):
    image = ImageOps.fit(Image.open(path).convert("RGB"), size, method=Image.Resampling.LANCZOS)
    if dim:
        image = image.filter(ImageFilter.GaussianBlur(2))
        image = Image.blend(image, Image.new("RGB", image.size, "#FFFFFF"), 0.32)
    return image


def centered(draw, center_x, y, text, text_font, fill=TEXT):
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((center_x - (box[2] - box[0]) / 2, y), text, font=text_font, fill=fill)


def card(draw, xy, outline="#D7DFEA", width=2):
    draw.rounded_rectangle(xy, 22, fill="#FFFFFF", outline=outline, width=width)


def arrow(draw, x, y1, y2):
    draw.line([(x, y1), (x, y2)], fill=GREEN, width=8)
    draw.polygon([(x, y2), (x - 16, y2 - 23), (x + 16, y2 - 23)], fill=GREEN)


def main():
    groups = prepare_candidates()
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((110, 55), "视频关键帧选择过程", font=font(58, True), fill=TEXT)
    draw.text((112, 135), "每个时间窗口内提取真实候选帧，并选择清晰度最高的一帧", font=font(27), fill=MUTED)

    draw.line([(150, 238), (2250, 238)], fill="#AFC0DA", width=6)
    starts = [110, 875, 1640]
    for x in [430, 1200, 1970]:
        draw.ellipse((x - 13, 225, x + 13, 251), fill=BLUE)

    card_y, card_w, card_h = 285, 650, 410
    thumb_w, thumb_h, gap = 175, 240, 25
    for group_i, (group, x0) in enumerate(zip(groups, starts), 1):
        card(draw, (x0, card_y, x0 + card_w, card_y + card_h))
        draw.text((x0 + 28, card_y + 20), f"候选窗口 {group_i} · {group['label']}", font=font(27, True), fill=TEXT)
        for i, path in enumerate(group["paths"]):
            px, py = x0 + 25 + i * (thumb_w + gap), card_y + 75
            best = i == group["best"]
            canvas.paste(fit_image(path, (thumb_w, thumb_h), dim=not best), (px, py))
            draw.rounded_rectangle((px - 4, py - 4, px + thumb_w + 4, py + thumb_h + 4), 12, outline=GREEN if best else "#B8C2D1", width=7 if best else 3)
            draw.rounded_rectangle((px + 8, py + 8, px + 95, py + 43), 14, fill=GREEN if best else RED)
            centered(draw, px + 51, py + 11, "✓ 入选" if best else "× 淘汰", font(18, True), "#FFFFFF")
            centered(draw, px + thumb_w / 2, py + thumb_h + 15, f"清晰度 {group['scores'][i]:.0f}", font(18, best), GREEN if best else MUTED)
        centered(draw, x0 + card_w / 2, card_y + card_h - 45, "窗口内最高分帧被选为关键帧", font(20, True), GREEN)
        arrow(draw, x0 + card_w / 2, card_y + card_h + 15, 800)

    draw.text((110, 795), "最终选择的关键帧", font=font(36, True), fill=TEXT)
    draw.text((495, 805), "下方图片与上方绿色入选帧严格对应", font=font(23), fill=MUTED)
    result_y, result_w, result_h = 865, 500, 350
    result_starts = [200, 950, 1700]
    for group, x0 in zip(groups, result_starts):
        card(draw, (x0, result_y, x0 + result_w, result_y + result_h), outline="#B7DEC9", width=3)
        selected_path = group["paths"][group["best"]]
        canvas.paste(fit_image(selected_path, (result_w - 28, result_h - 75)), (x0 + 14, result_y + 14))
        centered(draw, x0 + result_w / 2, result_y + result_h - 47, group["label"] + "关键帧", font(24, True), GREEN)

    draw.rounded_rectangle((110, 1260, 2290, 1320), 18, fill="#EAF6EF")
    centered(draw, 1200, 1272, "核心作用：减少运动模糊输入，提高后续特征匹配、位姿估计与重建稳定性", font(25, True), GREEN)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=96)
    print(OUTPUT)


if __name__ == "__main__":
    main()
