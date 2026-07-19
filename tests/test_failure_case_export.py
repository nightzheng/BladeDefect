import csv
from pathlib import Path

import pytest

from blade_defect.experiment import export_failure_cases


def _case() -> dict[str, object]:
    return {
        "image_path": "datasets/images/a.jpg", "split": "val", "true_class": "crack",
        "predicted_class": "background", "confidence": 0.2, "iou": 0.0,
        "error_type": "small_object_miss", "experiment_id": "exp001",
    }


def test_export_failure_case_index(tmp_path: Path) -> None:
    output = export_failure_cases([_case()], tmp_path / "cases.csv")
    with output.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["image_path"] == "datasets/images/a.jpg"
    assert row["error_type"] == "small_object_miss"


def test_export_rejects_unknown_error_type(tmp_path: Path) -> None:
    case = _case()
    case["error_type"] = "invented"
    with pytest.raises(ValueError, match="unknown error_type"):
        export_failure_cases([case], tmp_path / "cases.csv")


def test_export_rejects_invalid_probability(tmp_path: Path) -> None:
    case = _case()
    case["confidence"] = 1.1
    with pytest.raises(ValueError, match="outside"):
        export_failure_cases([case], tmp_path / "cases.csv")
