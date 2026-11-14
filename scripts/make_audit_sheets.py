#!/usr/bin/env python
"""把待核对实例拼成"审计图版"（contact sheet），供人工或 AI 逐格判读。

为什么需要它：类 10/13 的待审实例是 91 个，一张张打开看图效率极低。
本脚本按 label_audit 的优先级排序，把每个实例裁出来（带上下文），
按 4x5 拼成一版，左上角标注序号 / 类别 / 面积 / flag。

产物：
  sheets/sheet_01.jpg ...     拼版图（红框=priority>=5，橙框>=3，绿框<3）
  sheets/index.csv            每格对应的 split/class/image/instance/flags/priority

用法：
  python scripts/make_audit_sheets.py \
    --data-yaml /data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml \
    --classes 10,13 --splits train,val \
    --out-dir /data/backup/round3/audit_sheets \
    --top 120 --cols 4 --per-sheet 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blade_defect.data.class_copy_paste import polygon_bbox, read_label_file
from blade_defect.data.indexed_splits import load_index_config, load_indexed_split
from blade_defect.data.label_audit import AuditRow, audit_split

CELL = 260
CAPTION = 22


def _parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in raw.replace(" ", "").split(",") if item]


def _parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _crop_for_row(row: AuditRow, pad_ratio: float) -> tuple[Path, tuple[int, int, int, int]] | None:
    from PIL import Image

    image_path = Path(row.image_path)
    if not image_path.is_file():
        return None
    instances = read_label_file(row.label_path)
    if row.instance_index < 0 or row.instance_index >= len(instances):
        return None
    polygon = instances[row.instance_index].polygon
    x1, y1, x2, y2 = polygon_bbox(polygon)
    with Image.open(image_path) as image:
        width, height = image.size
    bw = (x2 - x1) * width
    bh = (y2 - y1) * height
    pad = max(bw, bh) * pad_ratio
    left = max(int(x1 * width - pad), 0)
    top = max(int(y1 * height - pad), 0)
    right = min(int(x2 * width + pad), width)
    bottom = min(int(y2 * height + pad), height)
    if right - left < 8 or bottom - top < 8:
        return None
    return image_path, (left, top, right, bottom)


def _border_color(priority: float) -> tuple[int, int, int]:
    if priority >= 5:
        return (220, 40, 40)
    if priority >= 3:
        return (230, 140, 20)
    return (40, 150, 60)


def build_sheets(
    rows: list[AuditRow],
    out_dir: Path,
    *,
    cols: int = 4,
    per_sheet: int = 5,
    pad_ratio: float = 0.35,
    quality: int = 88,
) -> dict[str, object]:
    from PIL import Image, ImageDraw

    sheets_dir = out_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    usable = [row for row in rows if row.instance_index >= 0]
    written: list[dict[str, object]] = []
    sheet_count = 0

    for start in range(0, len(usable), cols * per_sheet):
        chunk = usable[start : start + cols * per_sheet]
        sheet = Image.new("RGB", (cols * CELL, per_sheet * (CELL + CAPTION)), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        for slot, row in enumerate(chunk):
            column = slot % cols
            line = slot // cols
            x = column * CELL
            y = line * (CELL + CAPTION)
            prepared = _crop_for_row(row, pad_ratio)
            if prepared is None:
                draw.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(180, 180, 180))
                draw.text((x + 4, y + 4), f"#{start + slot} MISSING", fill=(120, 120, 120))
            else:
                image_path, box = prepared
                with Image.open(image_path) as image:
                    tile = image.convert("RGB").crop(box)
                tile.thumbnail((CELL - 4, CELL - 4))
                offset = ((CELL - tile.width) // 2, (CELL - tile.height) // 2)
                sheet.paste(tile, (x + offset[0], y + offset[1]))
                draw.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=_border_color(row.priority), width=3)
            caption = (
                f"#{start + slot} c{row.class_id} {int(row.area_px)}px {row.bbox_w_px:.0f}x{row.bbox_h_px:.0f}"
                f" {','.join(row.flags) or 'ok'}"
            )
            draw.text((x + 4, y + CELL + 4), caption[:44], fill=(20, 20, 20))
            written.append(
                {
                    "sheet": sheet_count + 1,
                    "slot": slot,
                    "index": start + slot,
                    "split": row.split,
                    "class_id": row.class_id,
                    "class_name": row.class_name,
                    "image_path": row.image_path,
                    "label_path": row.label_path,
                    "instance_index": row.instance_index,
                    "area_px": row.area_px,
                    "flags": ";".join(row.flags),
                    "priority": row.priority,
                    "review_verdict": "",
                    "reviewer_note": "",
                }
            )
        sheet_path = sheets_dir / f"sheet_{sheet_count + 1:02d}.jpg"
        sheet.save(sheet_path, quality=quality)
        sheet_count += 1

    index_path = sheets_dir / "index.csv"
    fields = list(written[0].keys()) if written else ["sheet", "slot", "index"]
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(written)
    return {"sheets": sheet_count, "cells": len(written), "index": str(index_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top", type=int, default=120)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--per-sheet", type=int, default=5)
    parser.add_argument("--pad", type=float, default=0.35)
    args = parser.parse_args()

    _, data_config = load_index_config(args.data_yaml)
    raw_names = data_config.get("names") or {}
    names = (
        {int(key): str(value) for key, value in raw_names.items()}
        if isinstance(raw_names, dict)
        else {index: str(value) for index, value in enumerate(raw_names)}
    )
    target_classes = _parse_int_list(args.classes)

    rows: list[AuditRow] = []
    for split in _parse_str_list(args.splits):
        rows.extend(audit_split(load_indexed_split(args.data_yaml, split), target_classes, names, split=split))
    rows.sort(key=lambda row: (-row.priority, row.class_id, row.image_path, row.instance_index))
    rows = rows[: args.top]

    result = build_sheets(rows, Path(args.out_dir), cols=args.cols, per_sheet=args.per_sheet, pad_ratio=args.pad)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
