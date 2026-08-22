import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

from blade_defect.data import DEFECT_CLASSES
from scripts.convert_seg_to_obb import convert_indexed_dataset
from scripts.run_v3_obb_baseline import calibrate_batch, run_obb_baseline

SEG_LABEL = "12 0.1 0.1 0.8 0.2 0.7 0.8 0.2 0.7\n"


def _build_indexed_parent(tmp_path: Path) -> Path:
    frozen = tmp_path / "frozen"
    release = tmp_path / "blade-v3-grouped"
    release.mkdir()
    samples = {
        "train": frozen / "images" / "legacy_train" / "a.png",
        "val": frozen / "images" / "legacy_train" / "b.png",
        "test": frozen / "images" / "legacy_val" / "c.png",
    }
    for image_path in samples.values():
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 12), "white").save(image_path)
        relative = image_path.relative_to(frozen / "images")
        label_path = frozen / "labels" / relative.with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(SEG_LABEL, encoding="utf-8")

    (release / "train.txt").write_text("../frozen/images/legacy_train/a.png\n", encoding="utf-8")
    (release / "val.txt").write_text(str(samples["val"]) + "\n", encoding="utf-8")
    (release / "test.txt").write_text("../frozen/images/legacy_val/c.png\n", encoding="utf-8")
    (release / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".", "train": "train.txt", "val": "val.txt", "test": "test.txt",
                "names": DEFECT_CLASSES,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (release / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "blade-v3-grouped",
                "samples": {"train": 1, "val": 1, "test": 1},
                "instances": 3,
            }
        ),
        encoding="utf-8",
    )
    return release / "data.yaml"


def _build_obb_dataset(tmp_path: Path) -> Path:
    parent_data = _build_indexed_parent(tmp_path)
    output = tmp_path / "blade-v3-grouped-obb"
    convert_indexed_dataset(parent_data, output, tmp_path / "results", image_mode="copy")
    return output / "data.yaml"


def _write_baseline_config(tmp_path: Path, data_yaml: Path) -> Path:
    config = {
        "project_root": str(tmp_path),
        "name": "v3_yolo11s_obb_960_e50",
        "task": "obb",
        "model": "yolo11s-obb.pt",
        "data": str(data_yaml),
        "epochs": 50,
        "imgsz": 960,
        "batch": 8,
        "device": 0,
        "workers": 2,
        "seed": 42,
        "pretrained": True,
        "cache": False,
        "amp": True,
        "val": True,
        "save": True,
        "save_period": 5,
        "exist_ok": True,
        "project": "runs",
        "label_level": "fine",
    }
    config_path = tmp_path / "v3_yolo11s_obb_960_e50.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config_path


class _FakeBox:
    p = np.asarray([0.5, 0.6] + [0.0] * 13)
    r = np.asarray([0.4, 0.5] + [0.0] * 13)
    map50 = 0.45
    map = 0.2
    ap_class_index = np.asarray(list(range(15)))
    ap50 = np.asarray([0.5] * 15)
    ap = np.asarray([0.2] * 15)


def _fake_eval_result() -> SimpleNamespace:
    return SimpleNamespace(
        box=_FakeBox(),
        seg=None,
        speed={"inference": 4.0, "preprocess": 1.0, "postprocess": 1.0},
        names={index: name for index, name in DEFECT_CLASSES.items()},
    )


def _fake_train(kwargs: dict) -> SimpleNamespace:
    save_dir = Path(kwargs["project"]) / kwargs["name"]
    weights = save_dir / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "best.pt").write_bytes(b"best")
    (weights / "last.pt").write_bytes(b"last")
    (save_dir / "results.csv").write_text("epoch,box_map50\n1,0.1\n", encoding="utf-8")
    return SimpleNamespace(save_dir=str(save_dir))


def _fake_predict(**kwargs) -> None:
    output = Path(kwargs["project"]) / kwargs["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "prediction.jpg").write_bytes(b"jpg")


def test_obb_runner_records_environment_and_manifest(tmp_path: Path, monkeypatch) -> None:
    data_yaml = _build_obb_dataset(tmp_path)
    config_path = _write_baseline_config(tmp_path, data_yaml)
    weight = tmp_path / "yolo11s-obb.pt"
    weight.write_bytes(b"w" * 200_000)
    monkeypatch.setattr(
        "scripts.run_v3_obb_baseline.acquire_official_weight", lambda *args: weight
    )

    result = run_obb_baseline(
        config_path,
        train_fn=lambda **kwargs: _fake_train(kwargs),
        eval_fn=lambda **kwargs: _fake_eval_result(),
        predict_fn=_fake_predict,
    )

    run_dir = Path(result["run_dir"])
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    # environment.json 与 seg 正式实验同一 schema（collect_environment_metadata）。
    for dotted in (
        "python.version",
        "python.executable",
        "platform.system",
        "packages.torch",
        "packages.ultralytics",
        "cuda.available",
        "gpus",
    ):
        current = environment
        for part in dotted.split("."):
            assert part in current, f"environment.json 缺少字段 {dotted}"
            current = current[part]

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["task"] == "obb"
    assert manifest["label_level"] == "fine"
    assert manifest["dataset_id"] == "blade-v3-grouped-obb"
    assert manifest["test_used"] is False
    assert manifest["validated_splits"] == ["train", "val"]
    assert manifest["started_at"] and manifest["finished_at"]
    assert manifest["resume_supported"] is True
    hardware = manifest["hardware"]
    # hardware 块与 seg 正式实验 run_manifest/metrics 同一字段集（_environment_info）。
    for field in (
        "platform", "python_version", "torch_version", "cuda_available",
        "cuda_version", "ultralytics_version",
    ):
        assert field in hardware, f"hardware 缺少字段 {field}"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "ok"
    assert metrics["task"] == "obb"
    for field in ("box_precision", "box_recall", "box_mAP50", "box_mAP50-95", "fps"):
        assert field in metrics
    assert metrics["fps"] == 250.0
    assert not any(key.startswith("mask_") for key in metrics)
    assert metrics["test_used"] is False

    per_class = (run_dir / "per_class_metrics.csv").read_text(encoding="utf-8-sig")
    assert "box" in per_class and "mask" not in per_class.splitlines()[1]
    assert manifest["artifacts"]["best_weights_sha256"]
    assert manifest["artifacts"]["environment"].endswith("environment.json")


def test_obb_runner_failure_marks_manifest_failed(tmp_path: Path, monkeypatch) -> None:
    data_yaml = _build_obb_dataset(tmp_path)
    config_path = _write_baseline_config(tmp_path, data_yaml)
    weight = tmp_path / "yolo11s-obb.pt"
    weight.write_bytes(b"w" * 200_000)
    monkeypatch.setattr(
        "scripts.run_v3_obb_baseline.acquire_official_weight", lambda *args: weight
    )

    def _broken_train(**kwargs):
        raise RuntimeError("simulated training crash")

    try:
        run_obb_baseline(config_path, train_fn=_broken_train)
    except RuntimeError:
        pass
    else:
        raise AssertionError("训练异常应向外抛出")

    manifest = json.loads(
        (tmp_path / "runs" / "v3_yolo11s_obb_960_e50" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "failed"
    assert "simulated training crash" in manifest["error"]
    assert manifest["finished_at"]


def test_materialize_absolute_data_yaml_creates_missing_parent(tmp_path: Path) -> None:
    from scripts.run_v3_obb_baseline import _materialize_absolute_data_yaml

    data_yaml = _build_obb_dataset(tmp_path)
    output = tmp_path / "runs" / "obb" / "probe" / "normalized_data.yaml"
    assert not output.parent.is_dir()

    normalized = _materialize_absolute_data_yaml(data_yaml, output)

    payload = yaml.safe_load(normalized.read_text(encoding="utf-8"))
    assert "test" not in payload
    for split in ("train", "val"):
        list_path = Path(payload[split])
        assert list_path.is_absolute() and list_path.is_file()
        entries = list_path.read_text(encoding="utf-8").splitlines()
        assert entries and all(Path(entry).is_absolute() for entry in entries)


def test_calibrate_batch_rejects_invalid_fraction(tmp_path: Path) -> None:
    config_path = _write_baseline_config(tmp_path, tmp_path / "unused.yaml")
    for fraction in (0.0, -0.1, 1.5):
        try:
            calibrate_batch(config_path, batch=8, fraction=fraction)
        except ValueError:
            continue
        raise AssertionError(f"fraction={fraction} 应被拒绝")
