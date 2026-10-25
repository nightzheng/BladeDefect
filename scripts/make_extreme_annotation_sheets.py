#!/usr/bin/env python
"""极端标注复核图版：把面积 > 阈值或长宽比 > 阈值的实例裁出来，叠加标注轮廓。

用途：判断这类超大/超长标注是否"过标"（边界过宽、把正常区域画进缺陷）。
每格：左=整图缩略（红框标出标注范围），右=该实例的放大裁剪（黄线为标注多边形）。

用法：
  python scripts/make_extreme_annotation_sheets.py --data-yaml <原 data.yaml> \
    --class-id 13 --area-threshold 500000 --aspect-threshold 5 \
    --out-dir /data/backup/round3/extreme_annotations
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blade_defect.data.indexed_splits import load_indexed_split  # noqa: E402


def _shoelace(pts):
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def main() -> int:
    from PIL import Image, ImageDraw

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--class-id", type=int, default=13)
    parser.add_argument("--area-threshold", type=float, default=500_000)
    parser.add_argument("--aspect-threshold", type=float, default=5.0)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    sheets_dir = out / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    found = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        for sample in load_indexed_split(args.data_yaml, split):
            label_path = sample.label_path
            if not label_path.is_file():
                continue
            lines = [ln.split() for ln in label_path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
            with Image.open(sample.image_path) as image:
                width, height = image.size
            for index, parts in enumerate(lines):
                if len(parts) < 7 or int(float(parts[0])) != args.class_id:
                    continue
                pts = [(float(parts[i]) * width, float(parts[i + 1]) * height) for i in range(1, len(parts) - 1, 2)]
                if len(pts) < 3:
                    continue
                area = _shoelace(pts)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bw = max(max(xs) - min(xs), 1e-6)
                bh = max(max(ys) - min(ys), 1e-6)
                aspect = max(bw, bh) / min(bw, bh)
                if area > args.area_threshold or aspect > args.aspect_threshold:
                    found.append(
                        {
                            "split": split,
                            "image": str(sample.image_path),
                            "instance_index": index,
                            "area_px": round(area, 1),
                            "area_ratio": round(area / (width * height), 4),
                            "aspect": round(aspect, 2),
                            "fill_ratio": round(area / (bw * bh), 3),
                            "bbox": f"{min(xs):.0f},{min(ys):.0f},{max(xs):.0f},{max(ys):.0f}",
                            "label_path": str(label_path),
                        }
                    )

    print(f"命中极端标注: {len(found)} 条")
    found.sort(key=lambda r: -r["area_px"])

    cols, cell_w, cell_h, cap_h = 2, 640, 340, 26
    cols = 2
    per_sheet = 4
    for start in range(0, len(found), per_sheet):
        chunk = found[start : start + per_sheet]
        sheet = Image.new("RGB", (cols * cell_w, (cell_h + cap_h) * 2), (246, 246, 246))
        for slot, row in enumerate(chunk):
            ox = (slot % cols) * cell_w
            oy = (slot // cols) * (cell_h + cap_h)
            with Image.open(row["image"]) as image:
                full = image.convert("RGB")
                width, height = full.size
                thumb = full.copy()
                thumb.thumbnail((cell_w // 2 - 6, cell_h - 6))
                sx, sy = thumb.width / width, thumb.height / height
                draw = ImageDraw.Draw(thumb)
                xs = [float(v) for v in row["bbox"].split(",")]
                draw.rectangle(
                    [int(xs[0] * sx), int(xs[1] * sy), int(xs[2] * sx), int(xs[3] * sy)],
                    outline=(230, 30, 30),
                    width=2,
                )
                sheet.paste(thumb, (ox + 2, oy + 2))

                # 放大裁剪 + 黄线描出标注多边形
                pad_x = max((xs[2] - xs[0]) * 0.15, 60)
                pad_y = max((xs[3] - xs[1]) * 0.15, 60)
                box = (
                    max(int(xs[0] - pad_x), 0),
                    max(int(xs[1] - pad_y), 0),
                    min(int(xs[2] + pad_x), width),
                    min(int(xs[3] + pad_y), height),
                )
                zoom = full.crop(box).convert("RGB")
                zoom.thumbnail((cell_w // 2 - 6, cell_h - 6))
                zx = zoom.width / (box[2] - box[0])
                zy = zoom.height / (box[3] - box[1])
                zdraw = ImageDraw.Draw(zoom)
                gx = ox + cell_w // 2 + 2
                gy = oy + 2
                sheet.paste(zoom, (gx, gy))
                lines = [ln.split() for ln in Path(row["label_path"]).read_text(encoding="utf-8-sig").splitlines()]
                if row["instance_index"] < len(lines):
                    parts = lines[row["instance_index"]]
                    pts = [
                        ((float(parts[i]) * width - box[0]) * zx, (float(parts[i + 1]) * height - box[1]) * zy)
                        for i in range(1, len(parts) - 1, 2)
                    ]
                    if len(pts) >= 3:
                        zdraw.polygon([(gx + p[0], gy + p[1]) for p in pts], outline=(255, 220, 0), width=3)
                ImageDraw.Draw(sheet).text(
                    (ox + 4, oy + cell_h + 4),
                    f"{Path(row['image']).name[:38]} | {row['split']} | 面积 {row['area_px']:,.0f} "
                    f"({row['area_ratio']*100:.1f}% 画面) | 长宽比 {row['aspect']} | 填充 {row['fill_ratio']}",
                    fill=(20, 20, 20),
                )
        sheet.save(sheets_dir / f"extreme_annotations_{start // per_sheet + 1:02d}.jpg", quality=90)

    index_csv = out / "extreme_annotations_index.csv"
    with index_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(found[0].keys()) if found else ["image"])
        writer.writeheader()
        writer.writerows(found)
    print(f"sheets: {len(list(sheets_dir.glob('*.jpg')))} | index: {index_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
