"""Build a report-ready comparison figure for confirmed SHA duplicate pairs."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "results" / "dataset_registry" / "perceptual_hash_review_sample.csv"
OUTPUT = ROOT / "results" / "dataset_review" / "sha_duplicate_demo.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
    ) if bold else (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def main() -> None:
    with REVIEW.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("manual_decision") == "confirmed_exact_duplicate"
        ]
    if len(rows) != 4:
        raise ValueError(f"expected 4 confirmed pairs, got {len(rows)}")

    width, height = 2400, 2520
    canvas = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(canvas)
    title_font = font(58, bold=True)
    subtitle_font = font(30)
    pair_font = font(34, bold=True)
    label_font = font(28, bold=True)
    note_font = font(30)

    draw.text((80, 52), "SHA-256 完全重复图片示例", fill="#17202a", font=title_font)
    draw.text(
        (82, 132),
        "抽查确认的 4 对图片：画面与文件字节均一致，未删除，仅归入同一重复图片组",
        fill="#52606d",
        font=subtitle_font,
    )

    margin_x = 80
    gap = 44
    image_w = (width - margin_x * 2 - gap) // 2
    image_h = 425
    row_h = 545
    top = 218
    border = "#d9dee5"
    green = "#16825d"

    for index, row in enumerate(rows):
        y = top + index * row_h
        train_path = Path(row["train_image"])
        val_path = Path(row["val_image"])
        pair_no = row.get("review_sample_id", f"PH-{index + 1:03d}")
        draw.text((margin_x, y), f"{pair_no}  确认重复", fill="#17202a", font=pair_font)

        image_y = y + 55
        for x, path, split in (
            (margin_x, train_path, "原 train 图片"),
            (margin_x + image_w + gap, val_path, "原 val 图片"),
        ):
            frame = Image.new("RGB", (image_w, image_h), "#ffffff")
            preview = fit_image(path, (image_w - 18, image_h - 18))
            frame.paste(preview, ((image_w - preview.width) // 2, (image_h - preview.height) // 2))
            canvas.paste(frame, (x, image_y))
            draw.rectangle((x, image_y, x + image_w - 1, image_y + image_h - 1), outline=border, width=3)
            draw.text((x + 18, image_y + image_h - 48), split, fill="#17202a", font=label_font)

        status_y = image_y + image_h + 16
        draw.text((margin_x, status_y), "SHA-256：一致", fill=green, font=note_font)
        draw.text((margin_x + 310, status_y), "dHash 距离：0", fill=green, font=note_font)
        draw.text((margin_x + 610, status_y), "处理：同组，不跨数据集", fill=green, font=note_font)

    footer_y = height - 75
    draw.text((80, footer_y), "说明：重复组处理用于防止同一文件同时出现在 train 和 val；不代表删除原始图片。", fill="#52606d", font=note_font)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, format="PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
