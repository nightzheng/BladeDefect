from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import backfill_legacy_baseline_environment as backfill


def _write_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run_dir = tmp_path / "runs" / "full_primary_yolo11s_seg_960"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "experiment_id": "full_primary_yolo11s_seg_960",
            "started_at": "2026-07-30T10:00:00+08:00",
            "finished_at": "2026-07-31T01:00:00+08:00",
            "dataset_id": "blade-v2",
            "dataset_root": "datasets/blade-v2",
            "dataset_manifest_sha256": "a" * 64,
            "code_commit": "abc123",
            "hardware": {
                "platform": "Windows",
                "python_version": "3.10.20",
                "torch_version": "2.7.1+cu118",
                "ultralytics_version": "8.4.90",
                "cuda_available": True,
                "cuda_version": "11.8",
                "gpu_name": "RTX 4060 Laptop",
                "gpu_memory_total_mb": 8188,
            },
        }),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps({"mAP50": 0.3454}), encoding="utf-8")
    dataset_manifest = tmp_path / "datasets" / "blade-v2" / "dataset_manifest.json"
    dataset_manifest.parent.mkdir(parents=True)
    dataset_manifest.write_text(
        json.dumps({
            "split_policy": "preserve_source_train_val",
            "samples": {"train": 38000, "val": 10291},
            "instances": 49000,
            "sample_lists_sha256": "b" * 64,
            "frozen_labels_sha256": "c" * 64,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(backfill, "RUN_DIR", run_dir)
    monkeypatch.setattr(backfill, "DATASET_MANIFEST", dataset_manifest)
    return run_dir


def test_backfill_writes_environment_and_marks_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _write_run(tmp_path, monkeypatch)

    backfill.main()

    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["capture_method"] == "backfilled_from_run_manifest_hardware"
    assert environment["dataset"]["split_leakage_known"] is True
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_leakage_known"] is True
    assert manifest["dataset_split"] == "legacy_full_split"
    assert manifest["environment_backfilled_at"]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["split_leakage_known"] is True
    assert metrics["mAP50"] == 0.3454


def test_backfill_keeps_original_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _write_run(tmp_path, monkeypatch)

    backfill.main()

    manifest_backup = json.loads(
        (run_dir / "run_manifest.json.bak").read_text(encoding="utf-8")
    )
    assert "split_leakage_known" not in manifest_backup
    metrics_backup = json.loads((run_dir / "metrics.json.bak").read_text(encoding="utf-8"))
    assert metrics_backup == {"mAP50": 0.3454}


def test_backfill_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _write_run(tmp_path, monkeypatch)

    backfill.main()
    first = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    backfill.main()
    second = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert second["environment_backfilled_at"] == first["environment_backfilled_at"]
