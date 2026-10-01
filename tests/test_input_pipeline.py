from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

from blade_defect.cli import build_parser
from blade_defect.models.performance import TrainingPerformance, performance_callbacks
from blade_defect.models.trainer import SegmentationTrainer
from scripts.benchmark_input_pipeline import build_cases, readonly_preflight


class FakeYOLO:
    def __init__(self):
        self.callbacks = {}
        self.calls = []

    def add_callback(self, event, callback):
        self.callbacks.setdefault(event, []).append(callback)

    def train(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


def test_profile_is_project_only_and_callbacks_are_scoped(tmp_path):
    trainer = SegmentationTrainer.__new__(SegmentationTrainer)
    trainer.model = FakeYOLO()
    previous = lambda _: None
    trainer.model.add_callback("on_train_start", previous)
    result = trainer.train(data=tmp_path / "data.yaml", normalize_data_yaml=False,
                           device="cpu", workers=8, pipeline_profile=True, seed=42)
    assert "pipeline_profile" not in result
    assert result["workers"] == 8 and result["seed"] == 42
    assert result["task"] == "segment"
    assert trainer.model.callbacks["on_train_start"] == [previous]
    assert all(not callbacks for event, callbacks in trainer.model.callbacks.items()
               if event != "on_train_start")


def test_disabled_profile_has_no_callbacks():
    model = FakeYOLO()
    with performance_callbacks(model):
        assert model.callbacks == {}


def test_callback_cleanup_on_failure():
    model = FakeYOLO()
    with pytest.raises(RuntimeError), performance_callbacks(model, True):
        raise RuntimeError("failed train")
    assert all(not callbacks for callbacks in model.callbacks.values())


@pytest.mark.parametrize("value", ["true", 1, {}])
def test_profile_rejects_non_boolean(value):
    with pytest.raises(ValueError), performance_callbacks(FakeYOLO(), value):
        pass


def test_host_diagnostics_report_wait_and_exclude_final_eval(tmp_path):
    import json

    times = iter([0.0, 0.3, 0.8, 1.0, 1.6, 1.7, 2.0])
    recorder = TrainingPerformance(clock=lambda: next(times))
    state = SimpleNamespace(epoch=0, device=torch.device("cpu"), save_dir=tmp_path,
                            train_loader=SimpleNamespace(batch_size=8, dataset=list(range(16))))
    recorder.epoch(state)
    recorder.batch(state)
    recorder.batch_end(state)
    recorder.batch(state)
    recorder.batch_end(state)
    recorder.epoch_end(state)
    recorder.fit_end(state)
    recorder.fit_end(state)  # official final evaluation callback must not duplicate rows
    lines = (tmp_path / "pipeline_performance.jsonl").read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["fetch_gap_s"] == pytest.approx(0.5)
    assert row["host_step_s"] == pytest.approx(1.1)
    assert row["epoch_wall_s"] == 2.0
    assert row["train_wall_s"] == 1.7


@pytest.mark.parametrize("command", ["train", "experiment"])
def test_cli_workers_compatible(command):
    prefix = [command] if command == "train" else [command, "run-all"]
    assert build_parser().parse_args(prefix).workers is None
    assert build_parser().parse_args(prefix + ["--workers", "8"]).workers == 8
    with pytest.raises(SystemExit):
        build_parser().parse_args(prefix + ["--workers", "-1"])


def test_benchmark_preserves_algorithm_and_separates_batch12():
    base = {"model": "yolo11n-seg.pt", "data": "data.yaml", "epochs": 100,
            "batch": 8, "imgsz": 960, "seed": 42, "cache": False,
            "workers": 4, "mosaic": 1.0, "close_mosaic": 10, "lr0": 0.01}
    original = dict(base)
    cases = build_cases(base, [4, 8, 12, 16], 12)
    assert base == original
    for case in cases:
        for key in ("model", "data", "epochs", "seed", "mosaic", "close_mosaic", "lr0", "cache"):
            assert case["config"][key] == base[key]
        assert case["config"]["batch"] == (12 if case["name"].startswith("D_") else 8)
    assert cases[0]["config"]["workers"] == 4
    with pytest.raises(ValueError):
        build_cases({**base, "cache": "disk"}, [4])


def test_preflight_refuses_upstream_reencoding_without_writing(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image = images / "sample.jpg"
    Image.new("RGB", (32, 32)).save(image)
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "images", "val": "images"}))
    assert readonly_preflight(data) == 1
    image.write_bytes(image.read_bytes()[:-2])
    before = image.read_bytes()
    with pytest.raises(ValueError, match="rewritten"):
        readonly_preflight(data)
    assert image.read_bytes() == before


def test_preflight_detects_implicit_npy_even_cache_false(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "sample.png")
    (images / "sample.npy").write_bytes(b"stale")
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "images", "val": "images"}))
    with pytest.raises(ValueError, match="implicit"):
        readonly_preflight(data)


def test_generated_plan_preserves_relative_model_origin(tmp_path, monkeypatch):
    import sys
    from scripts.benchmark_input_pipeline import main

    configs = tmp_path / "configs"
    configs.mkdir()
    source = configs / "train.yaml"
    source.write_text(yaml.safe_dump({"project_root": "..", "model": "weights/custom.pt",
                                     "data": "data.yaml", "epochs": 100, "seed": 42, "cache": False}))
    output = tmp_path / "benchmark"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--config", str(source), "--output", str(output)])
    main()
    planned = yaml.safe_load((output / "A_current.yaml").read_text())
    assert Path(planned["model"]) == tmp_path / "weights/custom.pt"
    assert Path(planned["data"]) == tmp_path / "data.yaml"
    assert planned["seed"] == 42
