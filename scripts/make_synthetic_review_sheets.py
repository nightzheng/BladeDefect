#!/usr/bin/env python
"""合成图人工复核图版：每格左=整图（红框标出粘贴位置），右=粘贴处放大。

为什么要双面板：只看裁剪块无法判断"贴到了哪里"（叶面 / 天空 / 地面 / 已贴到边界外），
也看不出与周围纹理是否协调、比例是否离谱。

用法：
  python scripts/make_synthetic_review_sheets.py \
    --derived-dir /data/datasets/derived/blade-v3-grouped-202608-b1cp13 \
    --out-dir /data/backup/round3/synthetic_visual_review \
    --count 24 --cols 4
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blade_defect.data.class_copy_paste import polygon_bbox, read_label_file  # noqa: E402

CELL_W = 300
CELL_H = 220
CAPTION = 24


def main() -> int:
    from PIL import Image, ImageDraw

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--derived-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out_dir = Path(args.out_dir)
    sheets_dir = out_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    images = sorted((derived / "images" / "train").glob("cp_*.jpg"))
    if not images:
        print("未找到合成图", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)
    picked = rng.sample(images, min(args.count, len(images)))

    rows = []
    rows_per_sheet = args.cols * 2
    for sheet_index in range(0, len(picked), rows_per_sheet):
        chunk = picked[sheet_index : sheet_index + rows_per_sheet]
        sheet = Image.new("RGB", (args.cols * CELL_W, ((CELL_H + CAPTION) * len(chunk) + args.cols - 1) // args.cols), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        for slot, image_path in enumerate(chunk):
            label_path = derived / "labels" / "train" / f"{image_path.stem}.txt"
            instances = read_label_file(label_path)
            if not instances:
                continue
            # 约定：写入时「目标图原标注在前、贴入实例在最后一行」
            pasted = instances[-1]
            x1, y1, x2, y2 = polygon_bbox(pasted.polygon)

            column = slot % args.cols
            line = slot // args.cols
            origin_x = column * CELL_W
            origin_y = line * (CELL_H + CAPTION)

            with Image.open(image_path) as image:
                full = image.convert("RGB")
                width, height = full.size
                # 左：整图缩略 + 红框
                thumb = full.copy()
                thumb.thumbnail((CELL_W - 4, CELL_H - 4))
                scale_x = thumb.width / width
                scale_y = thumb.height / height
                draw.rectangle(
                    [
                        origin_x + int(x1 * width * scale_x),
                        origin_y + int(y1 * height * scale_y),
                        origin_x + int(x2 * width * scale_x),
                        origin_y + int(y2 * height * scale_y),
                    ],
                    outline=(230, 30, 30),
                    width=2,
                )
                sheet.paste(thumb, (origin_x + 2, origin_y + 2))

                # 右：粘贴处放大（贴在整图右侧空白处，若放不下就叠在整图上）
                pad_x = max((x2 - x1) * width * 0.6, 40)
                pad_y = max((y2 - y1) * height * 0.6, 40)
                box = (
                    max(int(x1 * width - pad_x), 0),
                    max(int(y1 * height - pad_y), 0),
                    min(int(x2 * width + pad_x), width),
                    min(int(y2 * height + pad_y), height),
                )
                zoom = full.crop(box)
                zoom.thumbnail((CELL_W // 2 - 6, CELL_H // 2 - 6))
                sheet.paste(zoom, (origin_x + CELL_W - zoom.width - 4, origin_y + CELL_H - zoom.height - 4))

            caption = f"#{sheet_index + slot} {pasted.class_id} area={pasted.polygon.shape[0]}pt box=({x1:.2f},{y1:.2f})-({x2:.2f},{y2:.2f})"
            draw.text((origin_x + 4, origin_y + CELL_H + 4), caption[:52], fill=(20, 20, 20))
            rows.append(
                {
                    "index": sheet_index + slot,
                    "image": str(image_path),
                    "label": str(label_path),
                    "instances_in_label": len(instances),
                    "pasted_class": pasted.class_id,
                    "bbox_norm": f"{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}",
                    "touches_border": x1 <= 0.001 or y1 <= 0.001 or x2 >= 0.999 or y2 >= 0.999,
                    "review_verdict": "",
                    "note": "",
                }
            )
        path = sheets_dir / f"synthetic_review_{sheet_index // rows_per_sheet + 1:02d}.jpg"
        sheet.save(path, quality=90)

    index_path = out_dir / "index.csv"
    fields = list(rows[0].keys())
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"sheets: {len(list(sheets_dir.glob('*.jpg')))} | samples: {len(rows)} | index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
