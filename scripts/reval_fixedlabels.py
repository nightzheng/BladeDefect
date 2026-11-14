#!/usr/bin/env python
"""在修复后的标签上复评已训练模型（B-1 的同口径基线）。

背景：2026-10-23 删除了类 13 的 3 对重复标注（val GT 25 → 23）。
Round 2 的 A0 指标是在旧标注上算的，不能与新标注直接比较；
本脚本用 A0 的 best.pt 在新 val 上重跑一次 validate，产出可比的基线。

用法：
  python scripts/reval_fixedlabels.py \
    --weights /data/runs/v3_seg_a0_res960_ft_e25/weights/best.pt \
    --data-yaml /data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml \
    --imgsz 960 --batch 4 --name v3_seg_a0_reval_fixedlabels \
    --archive /data/backup/round3/a0_reval_fixedlabels_metrics.json

说明：只用 val；脚本不读 test（红线：test 锁定）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--project", default="/data/runs")
    parser.add_argument("--name", default="reval_fixedlabels")
    parser.add_argument("--archive", default=None, help="把 metrics.json 复制到该路径")
    args = parser.parse_args()

    from ultralytics import YOLO

    from blade_defect.utils.files import resolved_data_yaml

    model = YOLO(args.weights)
    # 原始 data.yaml 的 path 可能是相对路径，直接从别的工作目录调用 Ultralytics 会解析错；
    # 项目统一用 resolved_data_yaml 把 path 绝对化后再交给 Ultralytics（与训练入口一致）。
    with resolved_data_yaml(Path(args.data_yaml)) as normalized_data_yaml:
        result = model.val(
            data=normalized_data_yaml,
            imgsz=args.imgsz,
            batch=args.batch,
            split="val",
            project=args.project,
            name=args.name,
            exist_ok=True,
        )

    # seg.p / seg.r 是逐类数组，不能直接 float()；整体值取 results_dict（标量），
    # 逐类值按 ap_class_index 展开，判据要用的是类 13 那一行。
    def _scalar(value: object) -> float | None:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    overall = {key: _scalar(value) for key, value in (result.results_dict or {}).items()}
    per_class: dict[str, dict[str, float]] = {}
    seg = result.seg
    index = getattr(seg, "ap_class_index", None)
    if index is not None:
        for position, class_id in enumerate(index):
            per_class[str(result.names[int(class_id)])] = {
                "precision": float(seg.p[position]),
                "recall": float(seg.r[position]),
                "ap50": float(seg.ap50[position]),
                "ap50_95": float(seg.ap[position]),
            }

    target = next((row for name, row in per_class.items() if "结构损伤" in name), None)
    summary = {
        "weights": args.weights,
        "data_yaml": args.data_yaml,
        "split": "val",
        "imgsz": args.imgsz,
        "overall": overall,
        "target_class_13": target,
        "per_class": per_class,
        "test_used": False,
    }
    print(json.dumps({k: summary[k] for k in ("imgsz", "target_class_13")}, ensure_ascii=False, indent=2))

    if args.archive:
        archive_path = Path(args.archive)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"archived: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
