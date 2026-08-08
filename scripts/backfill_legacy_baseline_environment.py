"""为旧全量 50 轮 baseline 补录稳定环境文件并标记泄漏已知状态。

背景：runs/full_primary_yolo11s_seg_960 在旧全量 split（blade-v2，
preserve_source_train_val、未划分 test）上完成训练，当时尚未接入独立的
environment.json 记录功能。本脚本：

1. 依据 run_manifest.json/metrics.json 中已内嵌的 hardware 字段生成
   runs/full_primary_yolo11s_seg_960/environment.json；无法回溯的字段显式置空
   并在 notes 中说明；
2. 在 run_manifest.json 与 metrics.json 中补充 split_leakage_known=true 与
   旧全量 split 标记，明确该结果不得作为无泄漏 v3 test 上的正式指标；
3. 不改动任何指标数值与历史时间戳，仅追加字段。

用法：
    python scripts/backfill_legacy_baseline_environment.py
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from blade_defect.experiment.metadata import atomic_write_json

RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "full_primary_yolo11s_seg_960"
DATASET_MANIFEST = (
    Path(__file__).resolve().parent.parent / "datasets" / "blade-v2" / "dataset_manifest.json"
)

LEGACY_SPLIT_NOTE = (
    "旧全量 split（blade-v2 / blade-v2-full-frozen-48291）：split_policy="
    "preserve_source_train_val，未划分 test；该 split 与 blade-v3-grouped 的 "
    "val/test 存在样本重叠，本 run 的指标为泄漏已知的历史 baseline，"
    "不得作为无泄漏 v3 test 上的正式指标。"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _save(path: Path, payload: dict) -> None:
    atomic_write_json(path, payload)


def _backup_once(path: Path) -> None:
    """首次变更前保留原始文件副本；runs/ 不入 git，损坏不可恢复。"""
    backup = path.with_name(path.name + ".bak")
    if path.is_file() and not backup.exists():
        shutil.copy2(path, backup)


def main() -> None:
    manifest_path = RUN_DIR / "run_manifest.json"
    metrics_path = RUN_DIR / "metrics.json"
    environment_path = RUN_DIR / "environment.json"
    manifest = _load(manifest_path)
    if manifest.get("environment_backfilled_at"):
        print(f"已补录过（environment_backfilled_at={manifest['environment_backfilled_at']}），跳过。")
        return
    metrics = _load(metrics_path)
    dataset_manifest = _load(DATASET_MANIFEST)
    hardware = manifest.get("hardware") or metrics.get("hardware") or {}
    backfilled_at = _now_iso()

    environment = {
        "capture_method": "backfilled_from_run_manifest_hardware",
        "captured_at": backfilled_at,
        "original_training_window": {
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
        },
        "os": hardware.get("platform"),
        "python": {"version": hardware.get("python_version")},
        "packages": {
            "torch": hardware.get("torch_version"),
            "ultralytics": hardware.get("ultralytics_version"),
            "torchvision": None,
            "numpy": None,
            "opencv_python": None,
        },
        "cuda": {
            "available": hardware.get("cuda_available"),
            "version": hardware.get("cuda_version"),
            "cudnn_version": None,
        },
        "gpus": [
            {
                "index": 0,
                "name": hardware.get("gpu_name"),
                "total_memory_mb": hardware.get("gpu_memory_total_mb"),
                "compute_capability": None,
            }
        ]
        if hardware.get("gpu_name")
        else [],
        "code_commit": manifest.get("code_commit"),
        "dataset": {
            "dataset_id": manifest.get("dataset_id"),
            "dataset_version_alias": "blade-v2-full-frozen-48291",
            "dataset_root": manifest.get("dataset_root"),
            "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
            "split_policy": dataset_manifest.get("split_policy"),
            "samples": dataset_manifest.get("samples"),
            "instances": dataset_manifest.get("instances"),
            "sample_lists_sha256": dataset_manifest.get("sample_lists_sha256"),
            "frozen_labels_sha256": dataset_manifest.get("frozen_labels_sha256"),
            "split_leakage_known": True,
        },
        "notes": [
            "本文件为事后补录：OS/Python/PyTorch/CUDA/GPU/Ultralytics/代码提交/数据版本"
            "取自训练时写入 run_manifest.json 与 metrics.json 的 hardware 字段，"
            "与 2026-07-30/31 实际训练环境一致。",
            "torchvision、numpy、opencv-python 版本与 cuDNN 版本、GPU compute "
            "capability 训练时未记录，无法回溯，显式置空。",
            LEGACY_SPLIT_NOTE,
        ],
    }
    _backup_once(manifest_path)
    _backup_once(metrics_path)
    _save(environment_path, environment)

    leakage_fields = {
        "split_leakage_known": True,
        "dataset_split": "legacy_full_split",
        "dataset_version_alias": "blade-v2-full-frozen-48291",
        "split_note": LEGACY_SPLIT_NOTE,
        "environment_backfilled_at": backfilled_at,
    }
    manifest.update(leakage_fields)
    manifest["artifacts"] = {
        **(manifest.get("artifacts") or {}),
        "environment": str(environment_path),
    }
    _save(manifest_path, manifest)
    metrics.update(leakage_fields)
    _save(metrics_path, metrics)

    print(f"environment.json 已生成: {environment_path}")
    print("run_manifest.json / metrics.json 已补充 split_leakage_known=true")


if __name__ == "__main__":
    main()
