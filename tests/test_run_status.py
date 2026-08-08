import json
from pathlib import Path

from blade_defect.experiment.run_status import (
    inspect_run,
    interrupted_runs,
    scan_run_status,
)


def _write_manifest(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _touch_last_pt(run_dir: Path) -> None:
    weights = run_dir / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "last.pt").write_bytes(b"ckpt")


def test_running_manifest_without_finished_at_is_interrupted_and_resumable(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "exp_a"
    _write_manifest(
        run_dir,
        {"experiment_id": "exp_a", "status": "running",
         "started_at": "2026-08-03T10:00:00+08:00", "finished_at": None},
    )
    _touch_last_pt(run_dir)

    record = inspect_run(run_dir)

    assert record["status"] == "running"
    assert record["interrupted"] is True
    assert record["resumable"] is True
    assert record["last_checkpoint"].endswith("last.pt")


def test_interrupted_run_without_last_pt_is_not_resumable(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "exp_b"
    _write_manifest(
        run_dir,
        {"experiment_id": "exp_b", "status": "trained_pending_eval",
         "started_at": "2026-08-03T10:00:00+08:00", "finished_at": None},
    )

    record = inspect_run(run_dir)

    assert record["interrupted"] is True
    assert record["resumable"] is False
    assert record["last_checkpoint"] is None


def test_finished_run_is_not_interrupted(tmp_path: Path) -> None:
    for status in ("ok", "completed", "failed"):
        run_dir = tmp_path / "runs" / f"exp_{status}"
        _write_manifest(
            run_dir,
            {"experiment_id": f"exp_{status}", "status": status,
             "started_at": "2026-08-03T10:00:00+08:00",
             "finished_at": "2026-08-03T11:00:00+08:00"},
        )
        _touch_last_pt(run_dir)

        record = inspect_run(run_dir)

        assert record["status"] == status
        assert record["interrupted"] is False


def test_scan_run_status_filters_interrupted_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_manifest(
        runs_dir / "ok_run",
        {"experiment_id": "ok_run", "status": "ok",
         "finished_at": "2026-08-03T11:00:00+08:00"},
    )
    _write_manifest(
        runs_dir / "dead_run",
        {"experiment_id": "dead_run", "status": "running", "finished_at": None},
    )

    records = scan_run_status(runs_dir)
    interrupted = interrupted_runs(runs_dir)

    assert {record["experiment_id"] for record in records} == {"ok_run", "dead_run"}
    assert [record["experiment_id"] for record in interrupted] == ["dead_run"]


def test_missing_or_corrupt_manifest_handled(tmp_path: Path) -> None:
    empty_run = tmp_path / "runs" / "empty"
    empty_run.mkdir(parents=True)
    corrupt_run = tmp_path / "runs" / "corrupt"
    corrupt_run.mkdir(parents=True)
    (corrupt_run / "run_manifest.json").write_text("{not-json", encoding="utf-8")

    assert inspect_run(empty_run)["status"] == "no_manifest"
    corrupt = inspect_run(corrupt_run)
    assert corrupt["status"] == "manifest_corrupt"
    assert corrupt["interrupted"] is True


def test_scan_on_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert scan_run_status(tmp_path / "does_not_exist") == []
