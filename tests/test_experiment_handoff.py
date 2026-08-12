import csv
import json
from pathlib import Path

from scripts.archive_experiment_assets import (
    FIVE_PIECE_SET,
    archive_experiments,
    load_experiment_record,
    validate_experiment,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _build_formal_run(
    root: Path,
    name: str,
    dataset_id: str,
    label_level: str,
    num_classes: int,
    val_samples: int,
    membership: str,
) -> Path:
    dataset_root = root / dataset_id
    _write_json(
        dataset_root / "dataset_manifest.json",
        {
            "dataset_id": dataset_id,
            "membership_sha256": membership,
            "test_lock": {
                "policy": "test is readable only for owner-approved final evaluation",
                "automatic_unlock": False,
                "locked_split_sha256": "a" * 64,
            },
        },
    )
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True)
    commit = "c" * 40
    _write_json(
        run_dir / "metrics.json",
        {
            "name": name,
            "status": "ok",
            "dataset_id": dataset_id,
            "code_commit": commit,
            "model": "yolo11s-seg.pt",
            "imgsz": 960,
            "epochs": 100,
            "batch": 8,
            "seed": 42,
            "box_precision": 0.4,
            "box_recall": 0.3,
            "box_mAP50": 0.29,
            "box_mAP50-95": 0.16,
            "mask_precision": 0.39,
            "mask_recall": 0.32,
            "mask_mAP50": 0.27,
            "mask_mAP50-95": 0.11,
            "fps": 90.0,
        },
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "experiment_id": name,
            "status": "ok",
            "dataset_id": dataset_id,
            "dataset_root": str(dataset_root),
            "code_commit": commit,
            "config": {"label_level": label_level},
            "started_at": "2026-08-05T15:54:59+08:00",
            "finished_at": "2026-08-07T09:17:46+08:00",
        },
    )
    _write_json(
        run_dir / "environment.json",
        {
            "python": {"version": "3.10.20"},
            "packages": {"torch": "2.7.1+cu118", "ultralytics": "8.4.90"},
            "cuda": {"available": True},
            "gpus": [{"index": 0, "name": "GPU"}],
        },
    )
    _write_json(
        run_dir / "validation_predictions.json",
        {"experiment_id": name, "num_samples": val_samples, "samples": []},
    )
    with (run_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["metric_branch", "class_id", "class_name", "precision", "recall", "ap50", "ap50_95"]
        )
        for branch in ("mask", "box"):
            for class_id in range(num_classes):
                writer.writerow([branch, class_id, f"class_{class_id}", 0.5, 0.4, 0.45, 0.2])
    (run_dir / "weights").mkdir()
    (run_dir / "weights" / "best.pt").write_bytes(b"fake-weights")
    return run_dir


def _expectations() -> dict:
    return {
        "exp_fine": {
            "dataset_id": "blade-v3-grouped-202608",
            "label_level": "fine",
            "num_classes": 15,
            "val_samples": 7244,
        },
        "exp_coarse": {
            "dataset_id": "blade-v3-grouped-202608-6class",
            "label_level": "coarse",
            "num_classes": 6,
            "val_samples": 7244,
        },
    }


def test_archive_experiments_generates_handoff_package(tmp_path: Path) -> None:
    membership = "b" * 64
    run_a = _build_formal_run(
        tmp_path, "exp_fine", "blade-v3-grouped-202608", "fine", 15, 7244, membership
    )
    run_b = _build_formal_run(
        tmp_path, "exp_coarse", "blade-v3-grouped-202608-6class", "coarse", 6, 7244, membership
    )

    validation = archive_experiments([run_a, run_b], tmp_path / "handoff", _expectations())

    assert validation["valid"] is True
    output = tmp_path / "handoff"
    assert (output / "experiment_inventory.csv").is_file()
    assert (output / "artifact_sha256.csv").is_file()
    assert (output / "handoff_validation.json").is_file()
    assert (output / "handoff_notes.md").is_file()

    with (output / "experiment_inventory.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["experiment_id"] for row in rows} == {"exp_fine", "exp_coarse"}
    fine = next(row for row in rows if row["experiment_id"] == "exp_fine")
    assert fine["dataset_id"] == "blade-v3-grouped-202608"
    assert fine["label_level"] == "fine"

    with (output / "artifact_sha256.csv").open(encoding="utf-8-sig") as handle:
        sha_rows = list(csv.DictReader(handle))
    sha_paths = {(row["experiment_id"], row["relative_path"]) for row in sha_rows}
    for piece in FIVE_PIECE_SET:
        assert ("exp_fine", piece) in sha_paths
        assert ("exp_coarse", piece) in sha_paths
    assert all(len(row["sha256"]) == 64 for row in sha_rows)


def test_validate_experiment_rejects_label_level_drift(tmp_path: Path) -> None:
    run_dir = _build_formal_run(
        tmp_path, "exp_fine", "blade-v3-grouped-202608", "coarse", 15, 7244, "b" * 64
    )
    record = load_experiment_record(run_dir)
    checks = validate_experiment(record, _expectations()["exp_fine"])
    label_check = next(check for check in checks if check["check"] == "label_level_match")
    assert label_check["passed"] is False


def test_validate_experiment_rejects_missing_five_piece(tmp_path: Path) -> None:
    run_dir = _build_formal_run(
        tmp_path, "exp_fine", "blade-v3-grouped-202608", "fine", 15, 7244, "b" * 64
    )
    (run_dir / "validation_predictions.json").unlink()
    record = load_experiment_record(run_dir)
    checks = validate_experiment(record, _expectations()["exp_fine"])
    completeness = next(check for check in checks if check["check"] == "five_piece_complete")
    assert completeness["passed"] is False


def test_archive_detects_membership_mismatch(tmp_path: Path) -> None:
    run_a = _build_formal_run(
        tmp_path, "exp_fine", "blade-v3-grouped-202608", "fine", 15, 7244, "b" * 64
    )
    run_b = _build_formal_run(
        tmp_path, "exp_coarse", "blade-v3-grouped-202608-6class", "coarse", 6, 7244, "d" * 64
    )
    validation = archive_experiments([run_a, run_b], tmp_path / "handoff", _expectations())
    membership_check = next(
        check for check in validation["cross_experiment"] if check["check"] == "dataset_membership_identical"
    )
    assert membership_check["passed"] is False
    assert validation["valid"] is False
