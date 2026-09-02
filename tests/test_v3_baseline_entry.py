import json
from pathlib import Path

import yaml

from scripts.run_v3_baselines import (
    load_suite,
    plan_extension,
    task_status,
    version_report,
)


def _suite(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir()
    data = tmp_path / "data.yaml"
    data.write_text("path: .\ntrain: train.txt\nval: val.txt\n", encoding="utf-8")
    for task, name, epochs in (
        ("seg", "v3_yolo11s_seg_960_e100", 100),
        ("obb", "v3_yolo11s_obb_960_e50", 50),
    ):
        (configs / f"{task}.yaml").write_text(
            yaml.safe_dump(
                {
                    "project_root": str(tmp_path),
                    "model": f"model-{task}.pt",
                    "data": str(data),
                    "epochs": epochs,
                    "project": "runs",
                    "name": name,
                }
            ),
            encoding="utf-8",
        )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "project_root": str(tmp_path),
                "required_versions": {"python": "3.10", "ultralytics": "8.4.90"},
                "tasks": {
                    "seg": {
                        "config": "configs/seg.yaml",
                        "experiment_id": "v3_yolo11s_seg_960_e100",
                        "task": "segment",
                        "total_epochs": 100,
                    },
                    "obb": {
                        "config": "configs/obb.yaml",
                        "experiment_id": "v3_yolo11s_obb_960_e50",
                        "task": "obb",
                        "total_epochs": 50,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return suite


def _completed(run_dir: Path, task: str, total: int, stage: int = 0) -> None:
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best checkpoint")
    (weights / "last.pt").write_bytes(b"last checkpoint")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "continuation": {
                    "task": task,
                    "stage": stage,
                    "cumulative_target_epochs": total,
                },
            }
        ),
        encoding="utf-8",
    )


def test_status_reports_missing_or_available_data(tmp_path: Path) -> None:
    suite, root = load_suite(_suite(tmp_path))
    status = task_status(suite, root, "seg")
    assert status["status"] == "not_started"
    assert status["data_available"] is True
    assert status["resumable"] is False


def test_extension_creates_new_stage_without_overwriting_parent(tmp_path: Path) -> None:
    suite, root = load_suite(_suite(tmp_path))
    parent = tmp_path / "runs" / "v3_yolo11s_seg_960_e100"
    _completed(parent, "seg", 100)

    plan = plan_extension(suite, root, "seg", add_epochs=50)

    assert plan["run_name"] == "v3_yolo11s_seg_960_e150_ft1"
    assert plan["stage_epochs"] == 50
    assert plan["cumulative_target_epochs"] == 150
    assert plan["parent_run"] == str(parent)
    assert plan["initial_checkpoint"].endswith("best.pt")
    assert plan["initial_checkpoint_sha256"]
    assert not Path(plan["run_dir"]).exists()


def test_extension_chains_from_latest_completed_stage(tmp_path: Path) -> None:
    suite, root = load_suite(_suite(tmp_path))
    base = tmp_path / "runs" / "v3_yolo11s_obb_960_e50"
    stage1 = tmp_path / "runs" / "v3_yolo11s_obb_960_e100_ft1"
    _completed(base, "obb", 50)
    _completed(stage1, "obb", 100, stage=1)

    plan = plan_extension(suite, root, "obb", add_epochs=50, checkpoint_kind="last")

    assert plan["run_name"] == "v3_yolo11s_obb_960_e150_ft2"
    assert plan["parent_run"] == str(stage1)
    assert plan["initial_checkpoint"].endswith("last.pt")
    assert plan["optimizer_state_policy"] == "fresh_optimizer_after_completed_stage"


def test_version_report_is_structured(tmp_path: Path) -> None:
    suite, _ = load_suite(_suite(tmp_path))
    report = version_report(suite)
    assert set(report["checks"]) == {"python", "ultralytics"}
    assert all("required" in item and "installed" in item for item in report["checks"].values())


def test_v3_environment_and_docs_do_not_pin_a_local_conda_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = yaml.safe_load((project_root / "environment.yml").read_text(encoding="utf-8"))
    pip_dependencies = next(
        item["pip"] for item in environment["dependencies"] if isinstance(item, dict) and "pip" in item
    )
    assert "ultralytics==8.4.90" in pip_dependencies

    guide = (project_root / "docs" / "experiments" / "v3_unified_runner.md").read_text(
        encoding="utf-8"
    )
    assert "conda activate bladedefect" in guide
    assert "python scripts/run_v3_baselines.py" in guide
    assert "python.exe scripts/run_v3_baselines.py" not in guide
    assert "D:\\miniconda" not in guide
