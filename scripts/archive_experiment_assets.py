"""生成两项 v3 正式实验的只读归档清单、SHA-256 校验表与交付校验报告。

用途：
- 为 15 类 e100（label_level=fine）与 6 类 e50（label_level=coarse）两项正式实验
  生成 experiment_inventory.csv 与 artifact_sha256.csv，防止后续分析期间资产漂移；
- 按统一口径校验五件套（metrics.json / per_class_metrics.csv /
  validation_predictions.json / environment.json / run_manifest.json）、
  experiment_id、dataset_id、label_level、code_commit 与环境字段；
- 校验两数据集 membership 一致且 test 仍处于锁定状态；
- 生成 handoff_notes.md 交付说明，随五件套一并交付分析负责人。

用法：
    python scripts/archive_experiment_assets.py
    python scripts/archive_experiment_assets.py --output results/v3_experiment_handoff
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blade_defect.experiment.metadata import (
    load_json,
    missing_environment_fields,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIVE_PIECE_SET = (
    "metrics.json",
    "per_class_metrics.csv",
    "validation_predictions.json",
    "environment.json",
    "run_manifest.json",
)

# 两项正式实验的期望身份；dataset_id/label_level 不得漂移。
FORMAL_EXPERIMENTS: dict[str, dict[str, Any]] = {
    "v3_yolo11s_seg_960_e100": {
        "dataset_id": "blade-v3-grouped-202608",
        "label_level": "fine",
        "num_classes": 15,
        "val_samples": 7244,
    },
    "v3_hier_coarse_yolo11s_seg_960_e50": {
        "dataset_id": "blade-v3-grouped-202608-6class",
        "label_level": "coarse",
        "num_classes": 6,
        "val_samples": 7244,
    },
}

INVENTORY_FIELDS = (
    "experiment_id", "dataset_id", "label_level", "code_commit", "status",
    "model", "imgsz", "epochs", "batch", "seed",
    "box_precision", "box_recall", "box_mAP50", "box_mAP50-95", "fps",
    "mask_precision", "mask_recall", "mask_mAP50", "mask_mAP50-95",
    "val_prediction_samples", "started_at", "finished_at",
    "gpu_name", "torch_version", "ultralytics_version",
    "artifact_files", "artifact_bytes",
)

ARTIFACT_FIELDS = ("experiment_id", "relative_path", "size_bytes", "sha256")


def _load_json(path: Path) -> dict[str, Any]:
    return load_json(path)


def iter_artifact_rows(run_dir: Path, experiment_id: str) -> list[dict[str, Any]]:
    """枚举 run 目录内全部文件并计算 SHA-256（流式，权重安全）。"""
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rows.append(
            {
                "experiment_id": experiment_id,
                "relative_path": path.relative_to(run_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def load_experiment_record(run_dir: Path) -> dict[str, Any]:
    """读取单个 run 的五件套与关键身份字段。"""
    record: dict[str, Any] = {"run_dir": str(run_dir), "experiment_id": run_dir.name}
    for piece in FIVE_PIECE_SET:
        path = run_dir / piece
        record[f"has_{piece}"] = path.is_file()
    if record["has_metrics.json"]:
        record["metrics"] = _load_json(run_dir / "metrics.json")
    if record["has_run_manifest.json"]:
        record["manifest"] = _load_json(run_dir / "run_manifest.json")
    if record["has_environment.json"]:
        record["environment"] = _load_json(run_dir / "environment.json")
    if record["has_validation_predictions.json"]:
        predictions = _load_json(run_dir / "validation_predictions.json")
        record["prediction_experiment_id"] = predictions.get("experiment_id")
        record["prediction_num_samples"] = predictions.get("num_samples")
    if record["has_per_class_metrics.csv"]:
        with (run_dir / "per_class_metrics.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        record["per_class_rows"] = rows
    return record


def validate_experiment(
    record: dict[str, Any], expected: dict[str, Any]
) -> list[dict[str, Any]]:
    """按期望身份校验单个实验，返回结构化检查项。"""
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    missing = [piece for piece in FIVE_PIECE_SET if not record.get(f"has_{piece}")]
    check(
        "five_piece_complete",
        not missing,
        "五件套齐备" if not missing else f"缺失：{', '.join(missing)}",
    )
    metrics = record.get("metrics", {})
    manifest = record.get("manifest", {})
    environment = record.get("environment", {})

    check(
        "metrics_status_ok",
        metrics.get("status") == "ok",
        f"metrics.status={metrics.get('status')}",
    )
    check(
        "manifest_status_ok",
        manifest.get("status") == "ok",
        f"run_manifest.status={manifest.get('status')}",
    )
    check(
        "dataset_id_match",
        metrics.get("dataset_id") == expected["dataset_id"]
        and manifest.get("dataset_id") == expected["dataset_id"],
        f"期望 {expected['dataset_id']}，metrics={metrics.get('dataset_id')}，"
        f"manifest={manifest.get('dataset_id')}",
    )
    check(
        "label_level_match",
        (manifest.get("config") or {}).get("label_level") == expected["label_level"],
        f"期望 {expected['label_level']}，"
        f"manifest.config.label_level={(manifest.get('config') or {}).get('label_level')}",
    )
    check(
        "code_commit_consistent",
        bool(metrics.get("code_commit"))
        and metrics.get("code_commit") == manifest.get("code_commit"),
        f"metrics={metrics.get('code_commit')}，manifest={manifest.get('code_commit')}",
    )
    missing_env = missing_environment_fields(environment)
    check(
        "environment_fields_present",
        not missing_env,
        "环境字段齐备" if not missing_env else f"缺失：{', '.join(missing_env)}",
    )
    check(
        "val_predictions_match",
        record.get("prediction_experiment_id") == record["experiment_id"]
        and record.get("prediction_num_samples") == expected["val_samples"],
        f"experiment_id={record.get('prediction_experiment_id')}，"
        f"num_samples={record.get('prediction_num_samples')}（期望 {expected['val_samples']}，val 口径）",
    )
    rows = record.get("per_class_rows") or []
    branches: dict[str, set[int]] = {}
    for row in rows:
        branches.setdefault(row["metric_branch"], set()).add(int(row["class_id"]))
    check(
        "per_class_metrics_complete",
        branches.get("mask") == set(range(expected["num_classes"]))
        and branches.get("box") == set(range(expected["num_classes"])),
        f"分支类别数：{ {k: len(v) for k, v in sorted(branches.items())} }，"
        f"期望 {expected['num_classes']} 类",
    )
    check(
        "no_test_evaluation_artifacts",
        not any(
            key.startswith("test") or "_test" in key for key in metrics
        ),
        "metrics.json 不含 test 评估字段，test 保持锁定",
    )
    return checks


def validate_membership_and_lock(run_dirs: list[Path]) -> list[dict[str, Any]]:
    """跨实验校验：两数据集 membership 一致且 test 锁定。"""
    checks: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        manifest = _load_json(run_dir / "run_manifest.json")
        dataset_manifest_path = Path(manifest["dataset_root"]) / "dataset_manifest.json"
        manifests[run_dir.name] = _load_json(dataset_manifest_path)
    memberships = {
        name: payload.get("membership_sha256") for name, payload in manifests.items()
    }
    checks.append(
        {
            "check": "dataset_membership_identical",
            "passed": len(set(memberships.values())) == 1 and None not in memberships.values(),
            "detail": json.dumps(memberships, ensure_ascii=False),
        }
    )
    lock_states = {
        name: (payload.get("test_lock") or {}).get("automatic_unlock") is False
        and bool(
            (payload.get("test_lock") or {}).get("locked_split_sha256")
            or (payload.get("test_lock") or {}).get("policy")
        )
        for name, payload in manifests.items()
    }
    checks.append(
        {
            "check": "test_lock_intact",
            "passed": all(lock_states.values()),
            "detail": json.dumps(lock_states, ensure_ascii=False),
        }
    )
    return checks


def build_inventory_row(
    record: dict[str, Any], artifact_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """汇总单个实验的清单行（身份 + 关键指标 + 环境 + 资产统计）。"""
    metrics = record.get("metrics", {})
    manifest = record.get("manifest", {})
    environment = record.get("environment", {})
    gpus = environment.get("gpus") or [{}]
    packages = environment.get("packages") or {}
    total_bytes = sum(row["size_bytes"] for row in artifact_rows)
    return {
        "experiment_id": record["experiment_id"],
        "dataset_id": metrics.get("dataset_id"),
        "label_level": (manifest.get("config") or {}).get("label_level"),
        "code_commit": metrics.get("code_commit"),
        "status": metrics.get("status"),
        "model": metrics.get("model"),
        "imgsz": metrics.get("imgsz"),
        "epochs": metrics.get("epochs"),
        "batch": metrics.get("batch"),
        "seed": metrics.get("seed"),
        "box_precision": metrics.get("box_precision"),
        "box_recall": metrics.get("box_recall"),
        "box_mAP50": metrics.get("box_mAP50"),
        "box_mAP50-95": metrics.get("box_mAP50-95"),
        "fps": metrics.get("fps"),
        "mask_precision": metrics.get("mask_precision"),
        "mask_recall": metrics.get("mask_recall"),
        "mask_mAP50": metrics.get("mask_mAP50"),
        "mask_mAP50-95": metrics.get("mask_mAP50-95"),
        "val_prediction_samples": record.get("prediction_num_samples"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "gpu_name": gpus[0].get("name"),
        "torch_version": packages.get("torch"),
        "ultralytics_version": packages.get("ultralytics"),
        "artifact_files": len(artifact_rows),
        "artifact_bytes": total_bytes,
    }


def write_handoff_notes(
    output_dir: Path,
    inventory_rows: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    """生成面向分析负责人的交付说明（指标数值来自 metrics.json 实读）。"""
    lines = [
        "# v3 正式实验交付说明（结果归档）",
        "",
        f"- 生成时间：{validation['generated_at']}",
        f"- 校验结论：**{'PASS' if validation['valid'] else 'FAIL'}**（明细见 handoff_validation.json）",
        "- 口径：两项均为 blade-v3-grouped 无泄漏正式实验；label_level 分开（fine/coarse）；",
        "  旧泄漏 baseline 与历史抽样结果不在本交付包内，不得混用口径。",
        "- test 状态：两数据集 test_lock 完好，本周交付不含任何 test 评估指标。",
        "",
        "## 交付内容（五件套 × 2）",
        "",
        "| experiment_id | dataset_id | label_level | status | 文件数 | 总字节 |",
        "|---|---|---|---|---|---|",
    ]
    for row in inventory_rows:
        lines.append(
            f"| {row['experiment_id']} | {row['dataset_id']} | {row['label_level']} "
            f"| {row['status']} | {row['artifact_files']} | {row['artifact_bytes']} |"
        )
    lines += [
        "",
        "五件套：metrics.json、per_class_metrics.csv、validation_predictions.json、",
        "environment.json、run_manifest.json；归档另含 weights/（best/last/周期 checkpoint）、",
        "results.csv 与曲线图。全部文件 SHA-256 见 artifact_sha256.csv，复核方式：",
        "",
        "```powershell",
        "# 在仓库根目录逐行复核（示例）",
        "# Get-FileHash <run>/<relative_path> -Algorithm SHA256 与 artifact_sha256.csv 对比",
        "```",
        "",
        "## 环境与可追溯字段",
        "",
        "| experiment_id | code_commit | GPU | torch | ultralytics | started_at | finished_at |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in inventory_rows:
        lines.append(
            f"| {row['experiment_id']} | {row['code_commit']} | {row['gpu_name']} "
            f"| {row['torch_version']} | {row['ultralytics_version']} "
            f"| {row['started_at']} | {row['finished_at']} |"
        )
    lines += [
        "",
        "## 关键指标（val 口径，来自 metrics.json）",
        "",
        "| experiment_id | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | FPS |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in inventory_rows:
        lines.append(
            f"| {row['experiment_id']} | {row['box_precision']:.4f} | {row['box_recall']:.4f} "
            f"| {row['box_mAP50']:.4f} | {row['box_mAP50-95']:.4f} "
            f"| {row['mask_mAP50']:.4f} | {row['mask_mAP50-95']:.4f} | {row['fps']:.2f} |"
        )
    lines += [
        "",
        f"SHA-256 校验表共 {len(artifact_rows)} 个文件条目。",
    ]
    (output_dir / "handoff_notes.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def archive_experiments(
    run_dirs: list[Path],
    output_dir: Path,
    expectations: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """生成清单、校验表、校验报告与交付说明，返回校验负载。"""
    expectations = expectations if expectations is not None else FORMAL_EXPERIMENTS
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    experiment_checks: dict[str, list[dict[str, Any]]] = {}
    for run_dir in run_dirs:
        record = load_experiment_record(run_dir)
        rows = iter_artifact_rows(run_dir, record["experiment_id"])
        artifact_rows.extend(rows)
        inventory_rows.append(build_inventory_row(record, rows))
        expected = expectations.get(record["experiment_id"])
        if expected is None:
            experiment_checks[record["experiment_id"]] = [
                {
                    "check": "known_formal_experiment",
                    "passed": False,
                    "detail": f"{record['experiment_id']} 不在正式实验期望表内",
                }
            ]
        else:
            experiment_checks[record["experiment_id"]] = validate_experiment(record, expected)

    cross_checks = validate_membership_and_lock(run_dirs) if len(run_dirs) > 1 else []
    all_checks = [c for checks in experiment_checks.values() for c in checks] + cross_checks
    validation: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dirs": [str(path) for path in run_dirs],
        "experiments": experiment_checks,
        "cross_experiment": cross_checks,
        "valid": all(check["passed"] for check in all_checks),
    }

    with (output_dir / "experiment_inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory_rows)
    with (output_dir / "artifact_sha256.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ARTIFACT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(artifact_rows)
    (output_dir / "handoff_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_handoff_notes(output_dir, inventory_rows, artifact_rows, validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        default=[
            PROJECT_ROOT / "runs" / "v3_yolo11s_seg_960_e100",
            PROJECT_ROOT / "runs" / "v3_hier_coarse_yolo11s_seg_960_e50",
        ],
        help="正式实验 run 目录（默认两项 v3 正式实验）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "v3_experiment_handoff",
    )
    args = parser.parse_args()
    validation = archive_experiments(args.runs, args.output)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
