import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blade_defect.cli import build_parser
from blade_defect.experiment import EXPERIMENTS, analyze_experiments, export_summary
from blade_defect.experiment.config import ExperimentConfig
from blade_defect.experiment import runner as runner_module


def test_registry_contains_expected_baselines() -> None:
    assert [experiment.name for experiment in EXPERIMENTS] == [
        "exp001_yolov8n_seg_640", "exp002_yolov8s_seg_640",
        "exp003_yolo11n_seg_640", "exp004_yolo11s_seg_640",
        "exp005_yolov8n_seg_960", "exp006_yolov8s_seg_960",
        "exp007_yolo11n_seg_960", "exp008_yolo11s_seg_960",
        "exp009_yolov8n_seg_1024", "exp010_yolov8s_seg_1024",
        "exp011_yolo11n_seg_1024", "exp012_yolo11s_seg_1024",
        "exp013_yolov8n_seg_1280", "exp014_yolov8s_seg_1280",
        "exp015_yolo11n_seg_1280", "exp016_yolo11s_seg_1280",
    ]
    assert all(experiment.epochs == 50 and experiment.seed == 42 for experiment in EXPERIMENTS)
    assert [experiment.imgsz for experiment in EXPERIMENTS] == (
        [640] * 4 + [960] * 4 + [1024] * 4 + [1280] * 4
    )


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
    run_all = build_parser().parse_args(["experiment", "run-all"])
    assert run_all.experiment_command == "run-all"
    assert run_all.config.name == "train.yaml"
    assert build_parser().parse_args(["experiment", "summary"]).experiment_command == "summary"
    assert build_parser().parse_args(["experiment", "analyze"]).experiment_command == "analyze"


def test_analyze_experiments_generates_publication_plots(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    summary = results / "summary.csv"
    summary_rows = ["experiment name,model,mAP50,mAP50-95,precision,recall,fps"]
    summary_rows.extend(
        f"exp{index:03d},model{index}.pt,0.8,0.5,0.7,0.6,{100 - index}"
        for index in range(1, 9)
    )
    summary.write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
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


def test_run_all_uses_original_dataset_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data.yaml"
    data.write_text("train: images/train\nval: images/val\nnames: [defect]\n", encoding="utf-8")
    config = tmp_path / "train.yaml"
    config.write_text(f"data: {data.as_posix()}\nworkers: 2\n", encoding="utf-8")
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeTrainer:
        def __init__(self, model: object) -> None:
            calls.append(("init", {"model": model}))

        def train(self, **kwargs: object) -> object:
            calls.append(("train", kwargs))
            return SimpleNamespace(save_dir=tmp_path / "runs" / "exp_test")

        def validate(self, data: Path, **kwargs: object) -> object:
            calls.append(("validate", {"data": data, **kwargs}))
            metrics = SimpleNamespace(p=0.8, r=0.7, map50=0.6, map=0.5)
            return SimpleNamespace(seg=metrics, speed={"inference": 10.0})

    monkeypatch.setattr(runner_module, "SegmentationTrainer", FakeTrainer)
    experiment = ExperimentConfig("exp_test", "model.pt", epochs=1)
    runner_module.run_all_experiments(
        [experiment], config=config, runs_dir=tmp_path / "runs",
        results_file=tmp_path / "results" / "summary.csv", continue_on_error=False,
        skip_validation=True,
    )

    train_call = next(payload for kind, payload in calls if kind == "train")
    validate_call = next(payload for kind, payload in calls if kind == "validate")
    assert train_call["data"] == data.resolve()
    assert train_call["normalize_data_yaml"] is False
    assert train_call["workers"] == 2
    assert validate_call["data"] == data.resolve()
    assert validate_call["normalize_data_yaml"] is False


def test_run_all_validation_gate_stops_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    # 孤立 label 在训练门禁中应按缺失图片处理。
    (dataset / "labels" / "train" / "orphan.txt").write_text(
        "0 0 0 1 0 1 1\n", encoding="utf-8"
    )
    data = tmp_path / "data.yaml"
    data.write_text(
        f"path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\nnames: [defect]\n",
        encoding="utf-8",
    )
    config = tmp_path / "train.yaml"
    config.write_text(f"data: {data.as_posix()}\n", encoding="utf-8")

    class UnexpectedTrainer:
        def __init__(self, model: object) -> None:
            raise AssertionError("training must not be initialized after validation failure")

    monkeypatch.setattr(runner_module, "SegmentationTrainer", UnexpectedTrainer)
    with pytest.raises(runner_module.DatasetValidationError, match="missing_images=1"):
        runner_module.run_all_experiments(
            [ExperimentConfig("blocked", "model.pt")],
            config=config,
            runs_dir=tmp_path / "runs",
            results_file=tmp_path / "summary.csv",
        )


def test_run_all_cli_validation_is_enabled_by_default() -> None:
    args = build_parser().parse_args(["experiment", "run-all"])
    assert args.skip_validation is False
    skipped = build_parser().parse_args(["experiment", "run-all", "--skip-validation"])
    assert skipped.skip_validation is True


def test_run_all_filters_experiments_by_image_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized_models: list[object] = []

    class FakeTrainer:
        def __init__(self, model: object) -> None:
            initialized_models.append(model)

        def train(self, **kwargs: object) -> object:
            name = str(kwargs["name"])
            return SimpleNamespace(save_dir=tmp_path / "runs" / name)

        def validate(self, data: Path, **kwargs: object) -> object:
            metrics = SimpleNamespace(p=0.8, r=0.7, map50=0.6, map=0.5)
            return SimpleNamespace(seg=metrics, speed={"inference": 10.0})

    data = tmp_path / "data.yaml"
    data.write_text("train: images/train\nval: images/val\nnames: [defect]\n", encoding="utf-8")
    config = tmp_path / "train.yaml"
    config.write_text(f"data: {data.as_posix()}\n", encoding="utf-8")
    monkeypatch.setattr(runner_module, "SegmentationTrainer", FakeTrainer)

    records = runner_module.run_all_experiments(
        config=config,
        runs_dir=tmp_path / "runs",
        results_file=tmp_path / "summary.csv",
        skip_validation=True,
        imgsz=960,
    )

    assert len(records) == 4
    assert all(record["imgsz"] == 960 for record in records)
    # 每组实验会分别创建训练器和评估器，因此四个模型共初始化八次。
    assert len(initialized_models) == 8


def test_run_all_cli_accepts_image_size_filter() -> None:
    args = build_parser().parse_args(["experiment", "run-all", "--imgsz", "1024"])
    assert args.imgsz == 1024
