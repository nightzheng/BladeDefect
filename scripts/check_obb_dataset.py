"""校验 YOLO-OBB 数据集的训练集和验证集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blade_defect.data import DEFECT_CLASSES, check_obb_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--min-area", type=float, default=1e-8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    split_reports = {
        split: check_obb_dataset(
            dataset / "images" / split,
            dataset / "labels" / split,
            num_classes=len(DEFECT_CLASSES),
            min_area=args.min_area,
        ).to_dict()
        for split in ("train", "val")
    }
    payload = {
        "dataset": str(dataset),
        "valid": all(bool(report["valid"]) for report in split_reports.values()),
        "splits": split_reports,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
