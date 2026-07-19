import csv
import json
from pathlib import Path

import pytest

from blade_defect.experiment import analyze_experiments, publish_analysis_assets


def test_analysis_generates_required_artifacts(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "experiment name,model,mAP50,mAP50-95,precision,recall,fps\n"
        "exp001_model_640,model.pt,0.8,0.5,0.75,0.6,100\n",
        encoding="utf-8",
    )
    run = tmp_path / "runs" / "exp001_model_640"
    run.mkdir(parents=True)
    run.joinpath("metrics.json").write_text(json.dumps({
        "per_class_metrics": [
            {"class": "crack", "precision": 0.8, "recall": 0.7, "AP50": 0.75, "AP50-95": 0.4}
        ]
    }), encoding="utf-8")
    run.joinpath("confusion_matrix.png").write_bytes(b"matrix")

    outputs = analyze_experiments(summary, tmp_path / "runs", tmp_path / "analysis")

    assert {path.name for path in outputs} == {
        "model_comparison.csv", "per_class_metrics.csv", "model_comparison.png",
        "input_size_map.png", "input_size_fps.png", "pr_curve.png", "f1_curve.png",
        "confusion_matrix.png",
    }
    with (tmp_path / "analysis" / "model_comparison.csv").open(encoding="utf-8-sig") as handle:
        comparison = next(csv.DictReader(handle))
    assert comparison["imgsz"] == "640.0"
    assert float(comparison["f1"]) > 0.66
    with (tmp_path / "analysis" / "per_class_metrics.csv").open(encoding="utf-8-sig") as handle:
        per_class = next(csv.DictReader(handle))
    assert per_class["class"] == "crack"
    assert (tmp_path / "analysis" / "confusion_matrix.png").read_bytes() == b"matrix"


def test_publish_report_facing_analysis_assets(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    for name in ("input_size_map.png", "input_size_fps.png", "model_comparison.png"):
        analysis.joinpath(name).write_bytes(name.encode())

    outputs = publish_analysis_assets(analysis, tmp_path / "docs-assets")

    assert [path.name for path in outputs] == [
        "input_size_map.png", "input_size_fps.png", "model_comparison.png",
    ]
    assert all(path.read_bytes() == path.name.encode() for path in outputs)


def test_publish_rejects_incomplete_analysis(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Run analyze_experiments"):
        publish_analysis_assets(tmp_path / "missing", tmp_path / "docs-assets")
