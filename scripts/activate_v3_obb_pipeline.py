"""一键激活正式 v3 OBB 派生数据：只读预检 → 全量转换 → validator → 并排预览。

链路完全复用现有入口（convert_seg_to_obb / check_obb_dataset /
visualize_obb_labels），不重新实现转换算法；任一步骤失败即中止，
不绕过数据问题。test 只参与转换与完整性校验，不用于训练或样本选择。

用法：
    python scripts/activate_v3_obb_pipeline.py
    python scripts/activate_v3_obb_pipeline.py --previews-per-split 30 --seed 42
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # 直接以脚本方式执行时把仓库根目录加入 sys.path，使 scripts 包可导入。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.convert_seg_to_obb import convert_indexed_dataset, dry_run_index_check
from scripts.visualize_obb_labels import generate_previews

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATA = Path("configs/data.yaml")
DEFAULT_OUTPUT = Path("datasets/blade-v3-grouped-obb")
DEFAULT_RESULTS = Path("results/obb_v3")


def _run_validator(dataset: Path, report: Path) -> dict[str, Any]:
    """调用既有 validator CLI，返回其 JSON 报告；valid=false 时中止。"""
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "check_obb_dataset.py"),
            "--dataset",
            str(dataset),
            "--report",
            str(report),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if not report.is_file():
        raise RuntimeError(
            f"validator 未生成报告：{report}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    payload = json.loads(report.read_text(encoding="utf-8"))
    if completed.returncode != 0 or not payload.get("valid"):
        raise RuntimeError(f"OBB validator 未通过：{report}")
    return payload


def activate_pipeline(
    source_data: Path = DEFAULT_SOURCE_DATA,
    output: Path = DEFAULT_OUTPUT,
    results_root: Path = DEFAULT_RESULTS,
    *,
    previews_per_split: int = 30,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """按序执行激活链路并汇总各步骤真实结果。"""
    source_data = source_data.resolve()
    output = output.resolve()
    results_root = results_root.resolve()
    summary: dict[str, Any] = {"source_data": str(source_data), "output": str(output)}

    precheck = dry_run_index_check(source_data)
    summary["dry_run_index_check"] = {
        "splits": precheck["splits"],
        "source_instances": precheck["source_instances"],
        "converted_instances": precheck["converted_instances"],
        "membership_sha256": precheck["membership_sha256"],
        "valid": precheck["valid"],
    }
    if not precheck["valid"]:
        raise RuntimeError("只读预检未通过，先处理数据问题再执行正式转换")

    conversion = convert_indexed_dataset(
        source_data, output, results_root, overwrite=overwrite
    )
    summary["conversion"] = {
        "valid": conversion["valid"],
        "totals": conversion["totals"],
        "parent_membership_sha256": conversion["parent_membership_sha256"],
        "derived_membership_sha256": conversion["derived_membership_sha256"],
    }
    if not conversion["valid"]:
        raise RuntimeError("全量转换未通过，审计结果见 conversion_summary.json 与 invalid_obb_labels.csv")

    validation = _run_validator(output, results_root / "validation_report.json")
    summary["validation"] = {
        "valid": validation["valid"],
        "splits": {
            split: {
                "images": report["images"],
                "valid_instances": report["valid_instances"],
                "negative_images": report["negative_images"],
            }
            for split, report in validation["splits"].items()
        },
    }

    preview = generate_previews(
        source_data.parent,
        output,
        results_root / "conversion_preview",
        count_per_split=previews_per_split,
        seed=seed,
        acceptance=results_root / "preview_review.csv",
    )
    summary["previews"] = {
        "written": preview["written"],
        "splits": {split: item["written"] for split, item in preview["splits"].items()},
        "acceptance_csv": preview["acceptance_csv"],
    }
    summary["valid"] = True
    (results_root / "activation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--previews-per-split", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="删除并重建已有派生数据集")
    args = parser.parse_args()
    summary = activate_pipeline(
        args.source_data,
        args.output,
        args.results,
        previews_per_split=args.previews_per_split,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
