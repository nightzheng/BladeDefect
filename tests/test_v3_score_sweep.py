import csv
import json
from pathlib import Path

import yaml

from scripts.run_v3_score_sweep import (
    _dataset_preflight,
    _foreign_absolute_path,
    analyze,
    build_plan,
    load_sweep,
)


def _fixture(tmp_path: Path) -> Path:
    tasks = {}
    for task, metric, epochs in (
        ("seg", "mask_mAP50-95", 110),
        ("obb", "box_mAP50-95", 90),
    ):
        baseline = tmp_path / "runs" / f"baseline_{task}"
        (baseline / "weights").mkdir(parents=True)
        (baseline / "weights" / "best.pt").write_bytes(b"best")
        (baseline / "run_manifest.json").write_text(
            json.dumps({"status": "stopped_early"}), encoding="utf-8"
        )
        (baseline / "yolo_analysis_metrics.json").write_text(
            json.dumps({metric: 0.1, "fps": 100}), encoding="utf-8"
        )
        column = "metrics/mAP50-95(M)" if task == "seg" else "metrics/mAP50-95(B)"
        (baseline / "results.csv").write_text(
            f"epoch,{column}\n1,0.09\n2,0.10\n", encoding="utf-8"
        )
        base_config = tmp_path / "configs" / f"{task}.yaml"
        base_config.parent.mkdir(exist_ok=True)
        base_config.write_text("model: model.pt\n", encoding="utf-8")
        tasks[task] = {
            "baseline_run": f"runs/baseline_{task}",
            "base_config": f"configs/{task}.yaml",
            "primary_metric": metric,
            "recommended_from_scratch_epochs": epochs,
            "experiments": [
                {
                    "id": f"trial_{task}",
                    "stage": 1,
                    "hypothesis": "test",
                    "overrides": {"epochs": 3},
                }
            ],
        }
    config = tmp_path / "sweep.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project_root": str(tmp_path),
                "output_dir": "results/sweep",
                "tasks": tasks,
                "decision_policy": {
                    "minimum_primary_gain": 0.002,
                    "maximum_fps_drop_ratio": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def test_plan_uses_sealed_best_checkpoint_and_recommended_budgets(tmp_path: Path) -> None:
    payload, root, _ = load_sweep(_fixture(tmp_path))
    plan = build_plan(payload, root, task="all", stage="all")
    assert {row["task"] for row in plan} == {"seg", "obb"}
    assert all(row["baseline_best"].endswith("best.pt") for row in plan)
    assert payload["tasks"]["seg"]["recommended_from_scratch_epochs"] == 110
    assert payload["tasks"]["obb"]["recommended_from_scratch_epochs"] == 90


def test_analysis_ranks_gain_and_preserves_human_review(tmp_path: Path) -> None:
    payload, root, output = load_sweep(_fixture(tmp_path))
    trial = root / "runs" / "trial_seg"
    trial.mkdir(parents=True)
    (trial / "run_manifest.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (trial / "metrics.json").write_text(
        json.dumps({"mask_mAP50-95": 0.104, "fps": 85}), encoding="utf-8"
    )
    artifacts = analyze(payload, root, output)
    with Path(artifacts["summary"]).open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["experiment_id"]: row for row in csv.DictReader(handle)}
    assert rows["trial_seg"]["auto_decision"] == "candidate_for_human_review"
    assert Path(artifacts["human_review"]).is_file()
    assert Path(artifacts["convergence"]).is_file()


def test_foreign_absolute_paths_are_detected_for_both_os_families() -> None:
    assert _foreign_absolute_path(r"D:\datasets\blade", target_os="posix") is True
    assert _foreign_absolute_path("/mnt/data/blade", target_os="nt") is True
    assert _foreign_absolute_path("datasets/blade", target_os="posix") is False
    assert _foreign_absolute_path(r"datasets\blade", target_os="nt") is False


def test_dataset_preflight_checks_train_val_but_keeps_test_locked(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for name in ("train.jpg", "val.jpg"):
        (images / name).write_bytes(b"image")
    (tmp_path / "train.txt").write_text("images/train.jpg\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("images/val.jpg\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text(
        "path: .\ntrain: train.txt\nval: val.txt\ntest: deliberately-missing.txt\n",
        encoding="utf-8",
    )

    report = _dataset_preflight(data, full_data_check=True)

    assert report["errors"] == []
    assert report["test_read"] is False
    assert report["splits"]["train"]["missing"] == 0
    assert report["splits"]["val"]["missing"] == 0
