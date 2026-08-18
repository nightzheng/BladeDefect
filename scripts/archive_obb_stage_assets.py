"""生成 OBB 阶段只读交付包：资产清单、SHA-256、校验报告、阶段指标与交付说明。

用途：
- 将 runs/v3_yolo11s_obb_960_e50 五件套（metrics / per_class / environment /
  run_manifest / 代表性预测）、results/obb_v3 的转换与对比产物、90 张预览审核、
  数据集描述文件与官方权重来源整理为一份只读阶段交付；
- 现场重算 OBB 派生数据的 split identity 与 membership 指纹，核对与父级
  blade-v3-grouped-202608 精确一致，并核验 48,291 张图片确为硬链接；
- 校验 OBB-seg 对比口径（仅共有 Box 指标与速度，不含 Mask mAP），保留
  “自动派生 OBB 不等于人工旋转框真值”声明；
- 全部产物计算 SHA-256，防止阶段收口期间资产漂移。

用法：
    python scripts/archive_obb_stage_assets.py
    python scripts/archive_obb_stage_assets.py --output results/obb_stage_handoff_202608
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from blade_defect.data import load_indexed_splits, membership_hash
from blade_defect.experiment.metadata import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DATASET_ID = "blade-v3-grouped-obb"
EXPECTED_PARENT_ID = "blade-v3-grouped-202608"
EXPECTED_MEMBERSHIP = "02a0251aeca30a4e8b332967c2124848955886a2550f75933cc1e38146dfc37b"
EXPECTED_WEIGHT_SHA256 = "43fa63102922e0701501241b307420d24fc55e080816888b18bf8c6f96b1a45a"
EXPECTED_EXPERIMENT_ID = "v3_yolo11s_obb_960_e50"
ALLOWED_COMPARISON_METRICS = {"box_precision", "box_recall", "box_mAP50", "box_mAP50-95", "fps"}

INVENTORY_FIELDS = ("group", "root", "files", "bytes", "description")
ARTIFACT_FIELDS = ("group", "relative_path", "size_bytes", "sha256")

GROUP_DESCRIPTIONS = {
    "run_e50": "OBB e50 正式 baseline 运行目录（五件套、权重与周期 checkpoint、曲线、8 张代表性预测）",
    "obb_results": "results/obb_v3 转换/校验/预览/对比/校准/权重来源产物（含 90 张并排预览与 preview_review.csv）",
    "dataset_descriptors": "派生数据集描述文件（data.yaml、dataset_manifest.json、三个 split 索引 txt；不含图片与标签本体）",
    "official_weight": "官方 yolo11s-obb.pt 预训练权重（来源与 SHA-256 已核验）",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _iter_group_files(group: str, root: Path, files: list[Path] | None = None) -> list[dict[str, Any]]:
    selected = files if files is not None else sorted(p for p in root.rglob("*") if p.is_file())
    rows = []
    for path in selected:
        rows.append(
            {
                "group": group,
                "relative_path": path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "_abs": str(path),
            }
        )
    return rows


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def collect_stage_artifacts(
    run_dir: Path,
    obb_results_dir: Path,
    dataset_root: Path,
    weight_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows += _iter_group_files("run_e50", run_dir)
    rows += _iter_group_files("obb_results", obb_results_dir)
    descriptors = [
        dataset_root / name
        for name in ("data.yaml", "dataset_manifest.json", "train.txt", "val.txt", "test.txt")
        if (dataset_root / name).is_file()
    ]
    rows += _iter_group_files("dataset_descriptors", dataset_root, descriptors)
    rows += _iter_group_files("official_weight", weight_path.parent, [weight_path])
    return rows


def validate_stage(
    run_dir: Path,
    obb_results_dir: Path,
    dataset_root: Path,
    weight_path: Path,
) -> list[dict[str, Any]]:
    """按验收口径逐项校验，返回结构化检查项（全部通过则交付有效）。"""
    checks: list[dict[str, Any]] = []
    metrics = _load_json(run_dir / "metrics.json")
    manifest = _load_json(run_dir / "run_manifest.json")
    environment = _load_json(run_dir / "environment.json")
    dataset_manifest = _load_json(dataset_root / "dataset_manifest.json")

    pieces = {
        "metrics.json": run_dir / "metrics.json",
        "per_class_metrics.csv": run_dir / "per_class_metrics.csv",
        "environment.json": run_dir / "environment.json",
        "run_manifest.json": run_dir / "run_manifest.json",
    }
    missing = [name for name, path in pieces.items() if not path.is_file()]
    predictions = sorted((run_dir / "predictions").glob("*")) if (run_dir / "predictions").is_dir() else []
    _check(
        checks,
        "five_piece_complete",
        not missing and len(predictions) >= 1,
        "五件套齐备（metrics/per_class/environment/run_manifest + 代表性预测 "
        f"{len(predictions)} 张）" if not missing else f"缺失：{', '.join(missing)}",
    )
    _check(
        checks,
        "metrics_status_ok_and_test_locked",
        metrics.get("status") == "ok"
        and metrics.get("test_used") is False
        and manifest.get("test_used") is False
        and sorted(metrics.get("validated_splits") or []) == ["train", "val"],
        f"status={metrics.get('status')}，test_used={metrics.get('test_used')}，"
        f"validated_splits={metrics.get('validated_splits')}",
    )
    _check(
        checks,
        "dataset_identity_match",
        metrics.get("dataset_id") == EXPECTED_DATASET_ID
        and manifest.get("dataset_id") == EXPECTED_DATASET_ID
        and metrics.get("parent_dataset_id") == EXPECTED_PARENT_ID
        and metrics.get("name") == EXPECTED_EXPERIMENT_ID,
        f"dataset_id={metrics.get('dataset_id')}，parent={metrics.get('parent_dataset_id')}，"
        f"experiment={metrics.get('name')}",
    )
    env_missing = [
        field
        for field in ("python.version", "packages.torch", "packages.ultralytics", "cuda.available", "gpus")
        if _nested(environment, field) is None
    ]
    _check(
        checks,
        "environment_fields_present",
        not env_missing,
        "环境字段齐备" if not env_missing else f"缺失：{', '.join(env_missing)}",
    )

    with (run_dir / "per_class_metrics.csv").open(encoding="utf-8-sig") as handle:
        per_class_rows = list(csv.DictReader(handle))
    class_ids = {int(row["class_id"]) for row in per_class_rows}
    _check(
        checks,
        "per_class_metrics_15_classes",
        class_ids == set(range(15)),
        f"逐类指标覆盖 {len(class_ids)} 类",
    )

    indexed = load_indexed_splits(dataset_root / "data.yaml", ("train", "val", "test"))
    live_membership = membership_hash(indexed)
    _check(
        checks,
        "membership_matches_parent_v3",
        live_membership == EXPECTED_MEMBERSHIP
        == dataset_manifest.get("derived_membership_sha256")
        == dataset_manifest.get("parent_membership_sha256"),
        f"live={live_membership[:16]}…，derived={str(dataset_manifest.get('derived_membership_sha256'))[:16]}…，"
        f"parent={str(dataset_manifest.get('parent_membership_sha256'))[:16]}…",
    )
    identity = {
        split: _identity_hash([sample.sample_id for sample in indexed[split]])
        for split in indexed
    }
    parent_identity = dataset_manifest.get("parent_split_identity_sha256", {})
    derived_identity = dataset_manifest.get("derived_split_identity_sha256", {})
    equals = {
        split: identity[split] == parent_identity.get(split) == derived_identity.get(split)
        for split in identity
    }
    _check(
        checks,
        "split_identity_parent_equals_derived",
        all(equals.values()),
        json.dumps(equals, ensure_ascii=False),
    )

    hardlink_total = 0
    hardlink_linked = 0
    for samples in indexed.values():
        for sample in samples:
            try:
                stat = os.stat(sample.image_path)
            except OSError:
                continue
            hardlink_total += 1
            if stat.st_nlink > 1:
                hardlink_linked += 1
    _check(
        checks,
        "images_are_hardlinks",
        hardlink_total > 0 and hardlink_linked == hardlink_total,
        f"硬链接图片 {hardlink_linked}/{hardlink_total}（0 复制）",
    )
    _check(
        checks,
        "labels_derived_from_polygon_min_area_obb",
        dataset_manifest.get("conversion_type") == "seg_polygon_to_min_area_obb"
        and dataset_manifest.get("instances") == dataset_manifest.get("converted_instances"),
        f"conversion_type={dataset_manifest.get('conversion_type')}，"
        f"instances={dataset_manifest.get('instances')}，"
        f"converted={dataset_manifest.get('converted_instances')}（自动派生 OBB 不等于人工旋转框真值）",
    )

    provenance_path = obb_results_dir / "weight_provenance.json"
    provenance = _load_json(provenance_path)
    actual_weight_sha = sha256_file(weight_path) if weight_path.is_file() else None
    _check(
        checks,
        "official_weight_provenance_verified",
        provenance.get("verified") is True
        and provenance.get("sha256") == EXPECTED_WEIGHT_SHA256 == actual_weight_sha,
        f"verified={provenance.get('verified')}，sha256={str(actual_weight_sha)[:16]}…，"
        f"size={provenance.get('size_bytes')}",
    )

    conversion = _load_json(obb_results_dir / "conversion_summary.json")
    split_names = ("train", "val", "test")
    converted_ok = all(
        conversion["splits"][split]["converted_instances"] == conversion["splits"][split]["source_instances"]
        and conversion["splits"][split]["failed_instances"] == 0
        and conversion["splits"][split]["degenerate_boxes"] == 0
        and conversion["splits"][split]["clipped_points"] == 0
        for split in split_names
    )
    _check(
        checks,
        "conversion_zero_failure",
        conversion.get("valid") is True and converted_ok,
        "；".join(
            f"{split}: converted={conversion['splits'][split]['converted_instances']}/"
            f"{conversion['splits'][split]['source_instances']}，failed="
            f"{conversion['splits'][split]['failed_instances']}"
            for split in split_names
        ),
    )

    with (obb_results_dir / "preview_review.csv").open(encoding="utf-8-sig") as handle:
        preview_rows = list(csv.DictReader(handle))
    per_split = {split: sum(1 for row in preview_rows if row["split"] == split) for split in split_names}
    covered_classes = set()
    for row in preview_rows:
        covered_classes.update(int(value) for value in row["class_ids"].split())
    preview_images = [
        p for p in (obb_results_dir / "conversion_preview").rglob("*") if p.is_file() and p.suffix.lower() != ".json"
    ]
    _check(
        checks,
        "preview_audit_90_covers_15_classes",
        len(preview_rows) == 90
        and per_split == {"train": 30, "val": 30, "test": 30}
        and covered_classes == set(range(15))
        and len(preview_images) == 90
        and all(row["conversion_only_no_prediction"] == "True" for row in preview_rows),
        f"审核 {len(preview_rows)} 条（{per_split}），类别覆盖 {len(covered_classes)}，"
        f"预览图 {len(preview_images)} 张，均为仅转换预览",
    )

    comparison = _load_json(obb_results_dir / "obb_seg_comparison.json")
    compared = {item["metric"] for item in comparison.get("metrics", [])}
    scope = str(comparison.get("scope", ""))
    _check(
        checks,
        "comparison_common_box_only",
        compared == ALLOWED_COMPARISON_METRICS
        and "Mask mAP" in scope
        and "不等同于人工旋转框真值" in scope,
        f"对比指标={sorted(compared)}，scope 含 Mask mAP 排除与派生口径声明",
    )

    prior_report = _load_json(obb_results_dir / "validation_report.json")
    _check(
        checks,
        "prior_gate_report_valid_and_membership_same",
        prior_report.get("valid") is True
        and (prior_report.get("split_consistency") or {}).get("membership_sha256") == live_membership,
        f"prior valid={prior_report.get('valid')}，membership 与现场重算一致",
    )
    return checks


def _nested(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _identity_hash(sample_ids: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids):
        digest.update(f"{sample_id}\n".encode("utf-8"))
    return digest.hexdigest()


def write_stage_metrics(
    output_dir: Path,
    run_dir: Path,
    obb_results_dir: Path,
) -> list[dict[str, Any]]:
    """汇总阶段指标 CSV：总体共有 Box 指标 + 15 类逐类 AP（OBB vs seg）。"""
    metrics = _load_json(run_dir / "metrics.json")
    comparison = _load_json(obb_results_dir / "obb_seg_comparison.json")
    rows: list[dict[str, Any]] = []
    budget_note = "OBB e50(50 轮) vs seg e100(100 轮) 训练预算不同，仅共有 Box 口径，不做排名"
    for item in comparison["metrics"]:
        rows.append(
            {
                "section": "overall_common_box",
                "item": item["metric_label"],
                "obb": f"{item['obb']:.6f}",
                "seg": f"{item['seg']:.6f}",
                "delta_obb_minus_seg": f"{item['delta_obb_minus_seg']:.6f}",
                "source": "runs/v3_yolo11s_obb_960_e50/metrics.json + results/obb_v3/obb_seg_comparison.json",
                "note": budget_note,
            }
        )
    rows.append(
        {
            "section": "overall_common_box",
            "item": "Box F1",
            "obb": f"{metrics['box_f1']:.6f}",
            "seg": "",
            "delta_obb_minus_seg": "",
            "source": "runs/v3_yolo11s_obb_960_e50/metrics.json",
            "note": "对比产物未含 seg 侧 F1，仅列 OBB 实测值",
        }
    )
    for item in comparison["per_class"]:
        for branch, obb_key, seg_key, delta_key in (
            ("AP50", "obb_ap50", "seg_ap50", "delta_ap50"),
            ("AP50-95", "obb_ap50_95", "seg_ap50_95", "delta_ap50_95"),
        ):
            rows.append(
                {
                    "section": f"per_class_{branch}",
                    "item": f"{item['class_id']}_{item['class_name']}",
                    "obb": f"{item[obb_key]:.6f}",
                    "seg": f"{item[seg_key]:.6f}",
                    "delta_obb_minus_seg": f"{item[delta_key]:.6f}",
                    "source": "results/obb_v3/obb_seg_per_class_ap.csv",
                    "note": budget_note,
                }
            )
    fields = ("section", "item", "obb", "seg", "delta_obb_minus_seg", "source", "note")
    with (output_dir / "obb_stage_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_handoff_notes(
    output_dir: Path,
    inventory_rows: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    validation: dict[str, Any],
    run_dir: Path,
) -> None:
    metrics = _load_json(run_dir / "metrics.json")
    manifest = _load_json(run_dir / "run_manifest.json")
    lines = [
        "# OBB 阶段交付说明（2026.08 收口）",
        "",
        f"- 生成时间：{validation['generated_at']}",
        f"- 校验结论：**{'PASS' if validation['valid'] else 'FAIL'}**（{len(validation['checks'])} 项检查，明细见 handoff_validation.json）",
        "- 口径：OBB e50 为 50 轮正式 baseline；与 15 类 seg e100（100 轮）仅比较共有 Box 指标与速度，不比较 Mask mAP；",
        "  自动 OBB 来自 polygon 最小外接矩形，**不等同于人工旋转框真值**；test 全程锁定（test_used=false）。",
        "",
        "## 交付资产分组",
        "",
        "| group | 文件数 | 总字节 | 说明 |",
        "|---|---|---|---|",
    ]
    for row in inventory_rows:
        lines.append(f"| {row['group']} | {row['files']} | {row['bytes']} | {row['description']} |")
    lines += [
        "",
        f"逐文件 SHA-256 共 {len(artifact_rows)} 条，见 artifact_sha256.csv。",
        "",
        "## e50 关键信息（实读 metrics.json / run_manifest.json）",
        "",
        "| 项目 | 数据 |",
        "|---|---|",
        f"| experiment_id | {manifest['experiment_id']} |",
        f"| dataset_id / parent | {metrics['dataset_id']} / {metrics['parent_dataset_id']} |",
        f"| code_commit | {metrics['code_commit']} |",
        f"| 训练时段 | {manifest['started_at']} → {manifest['finished_at']}（{metrics['train_seconds']:.1f}s，{metrics['seconds_per_epoch']:.1f}s/epoch） |",
        f"| Box P / R / F1 | {metrics['box_precision']:.4f} / {metrics['box_recall']:.4f} / {metrics['box_f1']:.4f} |",
        f"| Box mAP50 / mAP50-95 | {metrics['box_mAP50']:.4f} / {metrics['box_mAP50-95']:.4f} |",
        f"| FPS | {metrics['fps']:.2f} |",
        f"| best/last 权重 SHA-256 | {manifest['artifacts']['best_weights_sha256'][:16]}… / {manifest['artifacts']['last_weights_sha256'][:16]}… |",
        "",
        "复核方式：`Get-FileHash <文件> -Algorithm SHA256` 与 artifact_sha256.csv 对比；",
        "membership 与 split 一致性由 archive 脚本现场重算（非引用历史结论）。",
    ]
    (output_dir / "handoff_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_stage_handoff(
    run_dir: Path,
    obb_results_dir: Path,
    dataset_root: Path,
    weight_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_rows = collect_stage_artifacts(run_dir, obb_results_dir, dataset_root, weight_path)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in artifact_rows:
        groups.setdefault(row["group"], []).append(row)
    inventory_rows = [
        {
            "group": group,
            "root": str(
                {
                    "run_e50": run_dir,
                    "obb_results": obb_results_dir,
                    "dataset_descriptors": dataset_root,
                    "official_weight": weight_path.parent,
                }[group]
            ),
            "files": len(rows),
            "bytes": sum(row["size_bytes"] for row in rows),
            "description": GROUP_DESCRIPTIONS[group],
        }
        for group, rows in sorted(groups.items())
    ]
    checks = validate_stage(run_dir, obb_results_dir, dataset_root, weight_path)
    validation: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "obb_results_dir": str(obb_results_dir),
        "dataset_root": str(dataset_root),
        "checks": checks,
        "valid": all(check["passed"] for check in checks),
    }
    with (output_dir / "artifact_inventory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory_rows)
    with (output_dir / "artifact_sha256.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ARTIFACT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in artifact_rows:
            writer.writerow({field: row[field] for field in ARTIFACT_FIELDS})
    (output_dir / "handoff_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_stage_metrics(output_dir, run_dir, obb_results_dir)
    write_handoff_notes(output_dir, inventory_rows, artifact_rows, validation, run_dir)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=PROJECT_ROOT / "runs" / "v3_yolo11s_obb_960_e50")
    parser.add_argument("--obb-results", type=Path, default=PROJECT_ROOT / "results" / "obb_v3")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "blade-v3-grouped-obb")
    parser.add_argument("--weight", type=Path, default=PROJECT_ROOT / "yolo11s-obb.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "obb_stage_handoff_202608")
    args = parser.parse_args()
    validation = build_stage_handoff(args.run, args.obb_results, args.dataset, args.weight, args.output)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
