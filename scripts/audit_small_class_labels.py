#!/usr/bin/env python
"""小样本类标注审计入口：产出待人工核对的优先级清单（CSV）与结构画像（JSON）。

用法：
  python scripts/audit_small_class_labels.py \
      --data-yaml /data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml \
      --classes 10,13,14 \
      --out-csv results/v3_round3/annotation_audit.csv \
      --out-json results/v3_round3/annotation_audit_summary.json \
      --top 120

产物：
  CSV  一实例一行，按 priority 降序；末尾两列 review_verdict / reviewer_note 供人工填写
       （verdict 建议取值：ok / fix_label / wrong_class / not_defect / need_engineer）
  JSON 每类实例数、命中各 flag 的数量、被标记比例

口径提醒：本脚本不判断"算不算缺陷"，只把可疑标注排到前面；
类 10/13/14 的最终结论必须由人工核对 + 企业工程师裁定，脚本结论不能替代。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blade_defect.data.indexed_splits import load_index_config, load_indexed_split
from blade_defect.data.label_audit import audit_split, summarize
from blade_defect.utils.files import save_json

CSV_FIELDS = [
    "split",
    "class_id",
    "class_name",
    "image_path",
    "label_path",
    "instance_index",
    "vertices",
    "area_px",
    "bbox_w_px",
    "bbox_h_px",
    "aspect",
    "co_occurring",
    "flags",
    "priority",
    "review_verdict",
    "reviewer_note",
]


def _parse_classes(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--classes 不能为空，例如 10,13,14")
    return [int(item) for item in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--classes", type=_parse_classes, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--top", type=int, default=None, help="只导出前 N 行（默认全部）")
    parser.add_argument(
        "--geometric-only",
        action="store_true",
        help="只做几何类检查（零面积/越界/重复/极小/过细），关闭 CONFUSABLE 共现判断。"
        "核对合成集时应使用：合成目标图本就可能含易混类，CONFUSABLE 会把正常样本标脏。",
    )
    args = parser.parse_args()

    _, data_config = load_index_config(args.data_yaml)
    raw_names = data_config.get("names") or {}
    if isinstance(raw_names, dict):
        names = {int(key): str(value) for key, value in raw_names.items()}
    else:
        names = {index: str(value) for index, value in enumerate(raw_names)}

    confusable = {} if args.geometric_only else None
    rows = []
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        samples = load_indexed_split(args.data_yaml, split)
        rows.extend(audit_split(samples, args.classes, names, split=split, confusable=confusable))
    rows.sort(key=lambda row: (-row.priority, row.class_id, row.image_path, row.instance_index))
    exported = rows[: args.top] if args.top else rows

    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in exported:
            payload = row.__dict__.copy()
            payload["flags"] = ";".join(row.flags)
            writer.writerow({key: payload[key] for key in CSV_FIELDS})

    summary = summarize(rows)
    summary.update(
        {
            "data_yaml": args.data_yaml,
            "target_classes": args.classes,
            "splits": args.splits,
            "geometric_only": bool(args.geometric_only),
            "confusable_enabled": not args.geometric_only,
            "exported_rows": len(exported),
            "csv": str(csv_path),
            "verdict_values": ["ok", "fix_label", "wrong_class", "not_defect", "need_engineer"],
        }
    )
    save_json(summary, Path(args.out_json))
    print(json.dumps({key: summary[key] for key in ("total_rows", "flagged_rows", "exported_rows", "by_class")}, ensure_ascii=False, indent=2))
    print(f"csv:  {csv_path}")
    print(f"json: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
