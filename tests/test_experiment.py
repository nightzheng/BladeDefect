import csv
import json
from pathlib import Path

from blade_defect.cli import build_parser
from blade_defect.experiment import EXPERIMENTS, analyze_experiments, export_summary


def test_registry_contains_expected_baselines() -> None:
    assert [experiment.name for experiment in EXPERIMENTS] == [
        "exp001_yolov8n_seg_640", "exp002_yolov8s_seg_640",
        "exp003_yolo11n_seg_640", "exp004_yolo11s_seg_640",
    ]
    assert all(experiment.epochs == 50 and experiment.seed == 42 for experiment in EXPERIMENTS)


def test_export_summary(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "runs" / EXPERIMENTS[0].name
    metrics_dir.mkdir(parents=True)
    metrics_dir.joinpath("metrics.json").write_text(json.dumps({
        "name": EXPERIMENTS[0].name, "model": "model.pt", "mAP50": 0.8,
        "mAP50-95": 0.5, "precision": 0.7, "recall": 0.6, "fps": 100, "status": "ok",
    }), encoding="utf-8")
    output = export_summary(tmp_path / "runs", tmp_path / "results" / "summary.csv")
    with output.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["experiment name"] == EXPERIMENTS[0].name
    assert rows[0]["mAP50"] == "0.8"


def test_experiment_cli_commands_parse() -> None:
    assert build_parser().parse_args(["experiment", "run-all"]).experiment_command == "run-all"
    assert build_parser().parse_args(["experiment", "summary"]).experiment_command == "summary"
    assert build_parser().parse_args(["experiment", "analyze"]).experiment_command == "analyze"


def test_analyze_experiments_generates_publication_plots(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    summary = results / "summary.csv"
    summary.write_text(
        "experiment name,model,mAP50,mAP50-95,precision,recall,fps\n"
        "exp001,yolo11n-seg.pt,0.8,0.5,0.7,0.6,100\n",
        encoding="utf-8",
    )
    run = tmp_path / "runs" / "exp001"
    run.mkdir(parents=True)
    run.joinpath("metrics.json").write_text(json.dumps({
        "curves": {"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.6],
                   "map50_curve": [0.4, 0.8], "map50_95_curve": [0.2, 0.5]},
        "class_distribution": {"crack": 12, "corrosion": 8},
    }), encoding="utf-8")

    outputs = analyze_experiments(summary, tmp_path / "runs", results / "analysis")

    assert {path.name for path in outputs} == {
        "summary_plot.png", "pr_curve.png", "class_distribution.png",
        "loss_curve_comparison.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)
