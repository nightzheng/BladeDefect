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


def test_analyze_experiments_generates_expanded_artifacts(tmp_path: Path) -> None:
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

    # 6ad866c 起 analyzer 收窄为 5 个图表工件；深度分析迁移至 v3_analysis 工具链。
    core_expected = {
        "model_comparison.png", "accuracy_speed_tradeoff.png", "f1_curve.png",
        "input_size_map.png", "input_size_fps.png",
    }
    assert core_expected.issubset({path.name for path in outputs})
    assert all(path.stat().st_size > 0 for path in outputs)


def _build_minimal_dataset(root: Path) -> Path:
    """构造一个能通过 strict 门禁的最小目录型数据集，返回 data.yaml 路径。"""
    dataset = root / "dataset"
    for split in ("train", "val"):
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        images_dir.joinpath(f"{split}_0.jpg").write_bytes(b"\xff\xd8\xff")
        labels_dir.joinpath(f"{split}_0.txt").write_text(
            "0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8"
        )
    data = root / "data.yaml"
    data.write_text(
        f"path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\nnames: [defect]\n",
        encoding="utf-8",
    )
    return data


@pytest.mark.parametrize("workers_override, expected_workers", [(None, 2), (8, 8)])
def test_run_all_uses_original_dataset_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    workers_override: int | None, expected_workers: int,
) -> None:
    data = _build_minimal_dataset(tmp_path)
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
        workers=workers_override,
    )

    train_call = next(payload for kind, payload in calls if kind == "train")
    validate_call = next(payload for kind, payload in calls if kind == "validate")
    assert train_call["data"] == data.resolve()
    assert train_call["normalize_data_yaml"] is False
    assert train_call["workers"] == expected_workers
    assert validate_call["data"] == data.resolve()
    assert validate_call["normalize_data_yaml"] is False
    run_dir = tmp_path / "runs" / "exp_test"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "exp_test"
    assert manifest["status"] == "completed"
    assert manifest["dataset_id"] == "dataset"
    assert manifest["git"]["tags_exact"] == []
    assert manifest["config"]["effective"]["workers"] == expected_workers
    assert environment["python"]["version"]


@pytest.mark.parametrize("batch_override, expected_batches", [(None, [8, 6]), (12, [12, 12])])
def test_run_all_batch_priority_for_all_targets(tmp_path, monkeypatch, caplog,
                                              batch_override, expected_batches):
    data = _build_minimal_dataset(tmp_path)
    config = tmp_path / "train.yaml"
    config.write_text(f"data: {data.as_posix()}\nbatch: 20\nworkers: 4\nseed: 99\n")
    calls = []

    class FakeTrainer:
        def __init__(self, model):
            pass

        def train(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(save_dir=Path(kwargs["project"]) / kwargs["name"])

        def validate(self, data, **kwargs):
            metrics = SimpleNamespace(p=0.8, r=0.7, map50=0.6, map=0.5)
            return SimpleNamespace(seg=metrics, speed={"inference": 10.0})

    monkeypatch.setattr(runner_module, "SegmentationTrainer", FakeTrainer)
    experiments = [ExperimentConfig("batch_a", "model_a.pt", imgsz=960, batch=8),
                   ExperimentConfig("batch_b", "model_b.pt", imgsz=1280, batch=6)]
    with caplog.at_level("INFO", logger=runner_module.__name__):
        records = runner_module.run_all_experiments(
            experiments, config=config, runs_dir=tmp_path / "runs", device="cpu",
            results_file=tmp_path / "summary.csv", continue_on_error=False,
            batch=batch_override, workers=8,
        )
    assert [call["batch"] for call in calls] == expected_batches
    assert [record["batch"] for record in records] == expected_batches
    assert [call["workers"] for call in calls] == [8, 8]
    assert [call["seed"] for call in calls] == [42, 42]
    assert [call["imgsz"] for call in calls] == [960, 1280]
    assert [exp.batch for exp in experiments] == [8, 6]
    for experiment, expected_batch in zip(experiments, expected_batches):
        manifest = json.loads((tmp_path / "runs" / experiment.name / "run_manifest.json").read_text())
        metrics = json.loads((tmp_path / "runs" / experiment.name / "metrics.json").read_text())
        assert manifest["batch"] == expected_batch
        assert manifest["config"]["effective"]["batch"] == expected_batch
        assert manifest["metrics"]["batch"] == expected_batch
        assert metrics["batch"] == expected_batch
        assert (f"{experiment.name} | effective batch={expected_batch} | effective workers=8 "
                f"| imgsz={experiment.imgsz} | device=cpu") in caplog.text


def test_run_all_cli_forwards_explicit_batch(monkeypatch):
    import sys
    from blade_defect import cli

    calls = []
    monkeypatch.setattr(cli, "run_all_experiments", lambda **kwargs: calls.append(kwargs) or [])
    monkeypatch.setattr(cli, "setup_logging", lambda: None)
    monkeypatch.setattr(sys, "argv", ["blade-defect", "experiment", "run-all", "--batch", "12"])
    cli.main()
    assert calls[0]["batch"] == 12
    assert build_parser().parse_args(["experiment", "run-all"]).batch is None
    for invalid in ("0", "-1", "0.5"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["experiment", "run-all", "--batch", invalid])


@pytest.mark.parametrize("batch", [0, -1, True, 0.5])
def test_run_all_api_rejects_invalid_batch_before_dataset_scan(tmp_path, monkeypatch, batch):
    config = tmp_path / "train.yaml"
    config.write_text("{}")
    monkeypatch.setattr(runner_module, "validate_dataset_gate",
                        lambda *_: pytest.fail("must validate batch before dataset scan"))
    with pytest.raises(ValueError, match="batch must be a positive integer"):
        runner_module.run_all_experiments(config=config, batch=batch)


def test_train_uses_yaml_batch_and_cli_workers(tmp_path, monkeypatch):
    from blade_defect import cli
    from blade_defect.utils.files import load_project_config

    source = tmp_path / "train.yaml"
    source.write_text("batch: 12\nworkers: 4\nimgsz: 960\nseed: 42\n")
    calls = []
    trainer = SimpleNamespace(train=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(cli, "SegmentationTrainer", SimpleNamespace(
        from_config=lambda path: (trainer, load_project_config(path)[0])))
    cli._train(source)
    cli._train(source, device="cpu", workers=8)
    assert calls[0]["batch"] == calls[1]["batch"] == 12
    assert calls[0]["workers"] == 4 and calls[1]["workers"] == 8


def test_run_manifest_records_training_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _build_minimal_dataset(tmp_path)
    config = tmp_path / "train.yaml"
    config.write_text(f"data: {data.as_posix()}\n", encoding="utf-8")

    class FailingTrainer:
        def __init__(self, model: object) -> None:
            raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(runner_module, "SegmentationTrainer", FailingTrainer)
    records = runner_module.run_all_experiments(
        [ExperimentConfig("exp_failed", "model.pt", epochs=1)],
        config=config,
        runs_dir=tmp_path / "runs",
        results_file=tmp_path / "results" / "summary.csv",
    )

    manifest = json.loads(
        (tmp_path / "runs" / "exp_failed" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert records[0]["status"] == "failed"
    assert manifest["status"] == "failed"
    assert manifest["finished_at"]
    assert manifest["error"] == {
        "type": "RuntimeError",
        "message": "synthetic training failure",
    }


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
    with pytest.raises(runner_module.DatasetValidationError, match="orphan_labels=1"):
        runner_module.run_all_experiments(
            [ExperimentConfig("blocked", "model.pt")],
            config=config,
            runs_dir=tmp_path / "runs",
            results_file=tmp_path / "summary.csv",
        )


def test_run_all_cli_has_no_skip_validation_flag() -> None:
    args = build_parser().parse_args(["experiment", "run-all"])
    assert not hasattr(args, "skip_validation")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["experiment", "run-all", "--skip-validation"])


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

    data = _build_minimal_dataset(tmp_path)
    config = tmp_path / "train.yaml"
    config.write_text(f"data: {data.as_posix()}\n", encoding="utf-8")
    monkeypatch.setattr(runner_module, "SegmentationTrainer", FakeTrainer)

    records = runner_module.run_all_experiments(
        config=config,
        runs_dir=tmp_path / "runs",
        results_file=tmp_path / "summary.csv",
        imgsz=960,
    )

    assert len(records) == 4
    assert all(record["imgsz"] == 960 for record in records)
    # 每组实验会分别创建训练器和评估器，因此四个模型共初始化八次。
    assert len(initialized_models) == 8


def test_run_all_cli_accepts_image_size_filter() -> None:
    args = build_parser().parse_args(["experiment", "run-all", "--imgsz", "1024"])
    assert args.imgsz == 1024


def test_run_all_selects_experiments_by_name_or_id() -> None:
    selected = runner_module._select_experiments(
        EXPERIMENTS,
        imgsz=None,
        selectors=["exp014_yolov8s_seg_1280", "16"],
    )
    assert [experiment.name for experiment in selected] == [
        "exp014_yolov8s_seg_1280",
        "exp016_yolo11s_seg_1280",
    ]


def test_run_all_rejects_unknown_experiment_selector() -> None:
    with pytest.raises(ValueError, match="未知实验名称或 ID"):
        runner_module._select_experiments(EXPERIMENTS, None, ["exp999"])


def test_run_all_cli_accepts_repeated_experiment_filters() -> None:
    args = build_parser().parse_args([
        "experiment", "run-all",
        "--experiment", "exp014",
        "--experiment", "16",
    ])
    assert args.experiments == ["exp014", "16"]
