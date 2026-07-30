"""failure_cases 测试：validation_predictions.json 聚合、FN/FP/matched 分类与 error_only 过滤。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from blade_defect.experiment import export_failure_cases


def _write_predictions(runs_dir: Path, experiment_id: str, samples: list[dict[str, object]]) -> None:
    exp_dir = runs_dir / experiment_id
    exp_dir.mkdir(parents=True)
    payload = {"experiment_id": experiment_id, "samples": samples}
    (exp_dir / "validation_predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _sample() -> dict[str, object]:
    return {
        "image_path": "datasets/images/a.jpg",
        "split": "val",
        "true_classes": [0, 1],
        "predicted_classes": [1, 2],
        "predictions": [
            {"class_id": 1, "confidence": 0.876543},
            {"class_id": 2, "confidence": 0.4},
            {"class_id": 2, "confidence": 0.654321},
        ],
    }


def _read_rows(output: Path) -> list[dict[str, str]]:
    with output.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_failure_case_index(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_predictions(runs_dir, "exp001", [_sample()])
    output = export_failure_cases(runs_dir, tmp_path / "cases.csv")
    rows = _read_rows(output)
    by_type = {row["candidate_type"]: row for row in rows}
    assert set(by_type) == {"auto_fn", "auto_fp", "auto_matched"}
    assert by_type["auto_fn"]["true_class"] == "0"
    assert by_type["auto_fn"]["predicted_class"] == ""
    assert by_type["auto_fp"]["predicted_class"] == "2"
    assert by_type["auto_fp"]["confidence"] == "0.6543"
    assert by_type["auto_matched"]["true_class"] == "1"
    assert by_type["auto_matched"]["confidence"] == "0.8765"
    # error_type 保留给人工复核，自动导出时为空。
    assert all(row["error_type"] == "" for row in rows)
    assert all(row["review_status"] == "pending" for row in rows)
    assert all(row["experiment_id"] == "exp001" for row in rows)
    assert all(row["image_path"] == "datasets/images/a.jpg" for row in rows)


def test_export_error_only_filters_unreviewed_rows(tmp_path: Path) -> None:
    # 自动导出的候选行 error_type 均为空，error_only 应将其全部过滤。
    runs_dir = tmp_path / "runs"
    _write_predictions(runs_dir, "exp001", [_sample()])
    output = export_failure_cases(runs_dir, tmp_path / "cases.csv", error_only=True)
    rows = _read_rows(output)
    assert rows == []


def test_export_empty_runs_writes_header_only(tmp_path: Path) -> None:
    output = export_failure_cases(tmp_path / "runs", tmp_path / "cases.csv")
    rows = _read_rows(output)
    assert rows == []
