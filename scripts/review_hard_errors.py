"""Generate auditable visual review sheets for hard polygon coordinate errors.

The script is read-only with respect to the source dataset.  It groups the
coordinate-level CSV by annotation line, draws the affected polygon over the
source image, and writes contact sheets plus a review-template CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = (".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png")
SHEET_SIZE = (2400, 1800)
PANEL_SIZE = (800, 450)
PANELS_PER_SHEET = 12


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


TITLE_FONT = _font(25, bold=True)
BODY_FONT = _font(20)
SMALL_FONT = _font(17)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _image_for(images_root: Path, split: str, label_path: Path) -> Path:
    stem = label_path.stem
    for suffix in IMAGE_SUFFIXES:
        candidate = images_root / split / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = [path for path in (images_root / split).glob(f"{stem}.*") if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot locate source image for {label_path}")


def _parse_polygon(label_path: Path, line_number: int) -> tuple[int, list[tuple[float, float]]]:
    lines = label_path.read_text(encoding="utf-8").splitlines()
    parts = lines[line_number - 1].split()
    class_id = int(float(parts[0]))
    coords = [float(value) for value in parts[1:]]
    return class_id, list(zip(coords[::2], coords[1::2], strict=True))


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(image, size, Image.Resampling.LANCZOS)


def _draw_polygon(image: Image.Image, points: list[tuple[float, float]]) -> None:
    width, height = image.size
    pixels = [
        (round(min(1.0, max(0.0, x)) * (width - 1)), round(min(1.0, max(0.0, y)) * (height - 1)))
        for x, y in points
    ]
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if len(pixels) >= 3:
        draw.polygon(pixels, fill=(255, 0, 170, 55), outline=(255, 0, 170, 255), width=max(8, width // 550))
    for index, (x, y) in enumerate(pixels, start=1):
        radius = max(8, width // 700)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 255, 255, 255))
        draw.text((x + radius + 2, y - radius), str(index), font=SMALL_FONT, fill=(255, 255, 0, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    image.alpha_composite(overlay)


def _polygon_crop(image: Image.Image, points: list[tuple[float, float]]) -> Image.Image:
    width, height = image.size
    xs = [min(1.0, max(0.0, x)) * width for x, _ in points]
    ys = [min(1.0, max(0.0, y)) * height for _, y in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad_x = max(180, (x1 - x0) * 0.35)
    pad_y = max(140, (y1 - y0) * 0.35)
    left = max(0, int(x0 - pad_x))
    top = max(0, int(y0 - pad_y))
    right = min(width, int(x1 + pad_x))
    bottom = min(height, int(y1 + pad_y))
    if right - left < 32 or bottom - top < 32:
        return image.copy()
    return image.crop((left, top, right, bottom))


def _panel(
    index: int,
    image_path: Path,
    points: list[tuple[float, float]],
    rows: list[dict[str, str]],
) -> Image.Image:
    source = Image.open(image_path).convert("RGBA")
    annotated = source.copy()
    _draw_polygon(annotated, points)
    crop = _polygon_crop(annotated, points)

    panel = Image.new("RGB", PANEL_SIZE, "#f4f6f8")
    draw = ImageDraw.Draw(panel)
    overview = _fit(annotated.convert("RGB"), (380, 305))
    detail = _fit(crop.convert("RGB"), (380, 305))
    panel.paste(overview, (10 + (380 - overview.width) // 2, 102 + (305 - overview.height) // 2))
    panel.paste(detail, (410 + (380 - detail.width) // 2, 102 + (305 - detail.height) // 2))
    draw.rectangle((8, 100, 392, 409), outline="#778899", width=2)
    draw.rectangle((408, 100, 792, 409), outline="#778899", width=2)

    first = rows[0]
    maximum = max(float(row["overflow_amount"]) for row in rows)
    values = ", ".join(
        f"{row['coordinate_position']}={float(row['original_value']):.6f}"
        for row in rows
    )
    title = f"#{index:03d}  {image_path.name}"
    info = f"class {first['class_id']} {first['class_name']} | line {first['line']} | max {maximum:.6f}"
    draw.text((12, 8), title, font=TITLE_FONT, fill="#102a43")
    draw.text((12, 43), info, font=BODY_FONT, fill="#243b53")
    draw.text((12, 72), values[:105], font=SMALL_FONT, fill="#9b1c31")
    draw.text((15, 414), "左：原图概览    右：异常多边形局部放大（洋红轮廓，青色顶点）", font=SMALL_FONT, fill="#334e68")
    return panel


def build_review_assets(
    hard_csv: Path,
    images_root: Path,
    output_dir: Path,
    decision_config: Path | None = None,
) -> None:
    rows = _read_rows(hard_csv)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["file"], row["line"])].append(row)
    cases = sorted(grouped.items())
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir = output_dir / "review_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    review_rule: dict[str, object] = {}
    if decision_config is not None:
        loaded = json.loads(decision_config.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or loaded.get("version") != 1:
            raise ValueError("hard error review config must be a version: 1 JSON object")
        review_rule = loaded
    exceptions = {
        int(item["case_id"]): item
        for item in review_rule.get("exceptions", [])
        if isinstance(item, dict) and "case_id" in item
    }

    review_rows: list[dict[str, str | int | float]] = []
    sheets: list[Image.Image] = []
    current = Image.new("RGB", SHEET_SIZE, "white")
    for index, ((split, label_file, line), issue_rows) in enumerate(cases, start=1):
        label_path = Path(label_file)
        image_path = _image_for(images_root, split, label_path)
        class_id, points = _parse_polygon(label_path, int(line))
        row = (index - 1) % PANELS_PER_SHEET
        if row == 0 and index > 1:
            sheets.append(current)
            current = Image.new("RGB", SHEET_SIZE, "white")
        column = row % 3
        panel_row = row // 3
        current.paste(_panel(index, image_path, points, issue_rows), (column * 800, panel_row * 450))
        maximum = max(float(item["overflow_amount"]) for item in issue_rows)
        exception = exceptions.get(index, {})
        decision = str(exception.get("review_decision", review_rule.get("default_decision", "")))
        reason = str(exception.get("review_reason", review_rule.get("default_reason", "")))
        review_time = str(exception.get("review_time", review_rule.get("review_time", "")))
        review_rows.append(
            {
                "case_id": index,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "split": split,
                "line_number": int(line),
                "class_id": class_id,
                "class_name": issue_rows[0]["class_name"],
                "hard_coordinate_count": len(issue_rows),
                "max_overflow": maximum,
                "out_of_bounds_values": "; ".join(
                    f"{item['coordinate_position']}={item['original_value']}"
                    for item in issue_rows
                ),
                "review_decision": decision,
                "review_reason": reason,
                "review_time": review_time,
            }
        )
    if cases:
        sheets.append(current)

    for sheet_index, sheet in enumerate(sheets, start=1):
        sheet.save(sheets_dir / f"hard_error_review_{sheet_index:02d}.jpg", quality=91, optimize=True)

    fields = list(review_rows[0]) if review_rows else [
        "case_id", "image_path", "label_path", "split", "line_number", "class_id",
        "class_name", "hard_coordinate_count", "max_overflow", "out_of_bounds_values",
        "review_decision", "review_reason", "review_time",
    ]
    with (output_dir / "hard_error_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"cases={len(cases)} sheets={len(sheets)} output={output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-csv", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-config", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_review_assets(args.hard_csv, args.images, args.output, args.decision_config)
