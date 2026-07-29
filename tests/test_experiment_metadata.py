import json
from pathlib import Path

from blade_defect.experiment import metadata


def test_git_metadata_records_exact_tags_describe_and_dirty_files(
    tmp_path: Path, monkeypatch,
) -> None:
    responses = {
        ("rev-parse", "HEAD"): "abc123",
        ("branch", "--show-current"): "main",
        ("tag", "--points-at", "HEAD"): "paper-v1\nbaseline-ready",
        ("describe", "--tags", "--always", "--dirty"): "paper-v1-2-gabc123-dirty",
        ("status", "--porcelain", "--untracked-files=all"): " M tracked.py\n?? note.txt",
    }
    monkeypatch.setattr(
        metadata,
        "_git_output",
        lambda root, *arguments: responses.get(arguments),
    )

    result = metadata.collect_git_metadata(tmp_path)

    assert result["commit"] == "abc123"
    assert result["branch"] == "main"
    assert result["tags_exact"] == ["baseline-ready", "paper-v1"]
    assert result["describe"] == "paper-v1-2-gabc123-dirty"
    assert result["dirty"] is True
    assert result["dirty_files"] == [" M tracked.py", "?? note.txt"]


def test_dataset_metadata_uses_frozen_manifest_hashes(tmp_path: Path) -> None:
    dataset = tmp_path / "blade-v3-grouped"
    dataset.mkdir()
    manifest = {
        "dataset_id": "blade-v3-grouped",
        "frozen_labels_sha256": "labels-hash",
        "sample_lists_sha256": "lists-hash",
        "filter_config": {"sha256": "filter-hash"},
    }
    (dataset / "dataset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\nnames: [defect]\n",
        encoding="utf-8",
    )

    result = metadata.collect_dataset_metadata(data_yaml)

    assert result["dataset_id"] == "blade-v3-grouped"
    assert result["manifest_available"] is True
    assert result["dataset_hash"]
    assert len(result["dataset_hash"]) == 64
    assert result["dataset_hash_source"] == "dataset_manifest_content"
    assert result["manifest"] == manifest


def test_dataset_metadata_falls_back_without_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "labels" / "train" / "sample.txt").write_text(
        "0 0 0 1 0 1 1\n", encoding="utf-8"
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\nnames: [defect]\n",
        encoding="utf-8",
    )

    result = metadata.collect_dataset_metadata(data_yaml)

    assert result["manifest_available"] is False
    assert result["dataset_hash_source"] == "labels_split_lists_and_data_yaml"
    assert len(result["dataset_hash"]) == 64


def test_atomic_json_write_replaces_previous_payload(tmp_path: Path) -> None:
    output = tmp_path / "run_manifest.json"
    metadata.atomic_write_json(output, {"status": "running", "path": tmp_path})
    metadata.atomic_write_json(output, {"status": "completed"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "completed"}
    assert not list(tmp_path.glob("*.tmp"))


def test_environment_metadata_has_reproducibility_fields() -> None:
    result = metadata.collect_environment_metadata()

    assert result["python"]["version"]
    assert "torch" in result["packages"]
    assert "ultralytics" in result["packages"]
    assert "available" in result["cuda"]
    assert isinstance(result["gpus"], list)


def test_run_manifest_records_requested_and_resolved_device(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("epochs: 1\n", encoding="utf-8")

    result = metadata.create_run_manifest(
        experiment={"name": "exp_test", "epochs": 1},
        effective_config={"epochs": 1},
        config_path=config,
        model="model.pt",
        requested_device="1",
        git={"commit": "abc123"},
        dataset={"dataset_id": "dataset", "dataset_hash": "hash"},
        environment={
            "gpus": [
                {"index": 0, "name": "GPU 0"},
                {"index": 1, "name": "GPU 1"},
            ]
        },
    )

    assert result["requested_device"] == "1"
    assert result["device"] == "GPU 1"
