"""扫描 runs 目录的 run_manifest.json，识别运行状态与异常退出。

状态机（scripts/run_full_primary.py 与 experiment.runner 约定）：
- running：训练进行中；若进程已退出而 manifest 停留在此状态，即为异常中断；
- trained_pending_eval：训练完成、评估/导出进行中，同样可能是中断现场；
- failed：流程内捕获的显式失败（manifest 含 error 字段）；
- ok / completed：正常完成。

异常退出的 run 若存在 weights/last.pt，可通过断点续训恢复
（run_full_primary.py 自动检测 last.pt 并 resume）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INTERRUPTED_STATUSES = ("running", "trained_pending_eval")
TERMINAL_OK_STATUSES = ("ok", "completed")


def inspect_run(run_dir: str | Path) -> dict[str, Any]:
    """检查单个 run 目录的 manifest 状态与断点恢复可用性。"""
    run_path = Path(run_dir)
    manifest_path = run_path / "run_manifest.json"
    record: dict[str, Any] = {
        "run_dir": str(run_path),
        "experiment_id": run_path.name,
        "manifest": None,
        "status": "no_manifest",
        "interrupted": False,
        "resumable": False,
        "last_checkpoint": None,
        "started_at": None,
        "finished_at": None,
        "error": None,
    }
    if not manifest_path.is_file():
        return record
    record["manifest"] = str(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        record.update(status="manifest_corrupt", interrupted=True, error=str(exc))
        return record

    status = manifest.get("status")
    last_pt = run_path / "weights" / "last.pt"
    record.update(
        experiment_id=manifest.get("experiment_id") or run_path.name,
        status=status,
        started_at=manifest.get("started_at"),
        finished_at=manifest.get("finished_at"),
        error=manifest.get("error"),
        last_checkpoint=str(last_pt) if last_pt.is_file() else None,
    )
    if status in INTERRUPTED_STATUSES and not manifest.get("finished_at"):
        record["interrupted"] = True
        record["resumable"] = last_pt.is_file()
    return record


def scan_run_status(runs_dir: str | Path) -> list[dict[str, Any]]:
    """扫描 runs_dir 下所有含 run_manifest.json 的 run，按目录名排序返回。"""
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    records = [
        inspect_run(manifest.parent)
        for manifest in sorted(root.glob("*/run_manifest.json"))
    ]
    return records


def interrupted_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """仅返回异常中断（status 停留在 running/trained_pending_eval）的 run。"""
    return [record for record in scan_run_status(runs_dir) if record["interrupted"]]


__all__ = [
    "INTERRUPTED_STATUSES",
    "TERMINAL_OK_STATUSES",
    "inspect_run",
    "interrupted_runs",
    "scan_run_status",
]
