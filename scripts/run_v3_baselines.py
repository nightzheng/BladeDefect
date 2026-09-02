"""v3 baseline 统一入口：状态检查、首次训练、断点恢复与完成后分阶段扩轮。

扩轮不会覆盖已经完成的正式实验。它从上一阶段的 best.pt（或显式选择
last.pt）初始化一个新的微调阶段；意外中断时，新阶段自己的 last.pt 仍按
Ultralytics resume 语义恢复优化器、调度器和当前 epoch。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blade_defect.utils.files import load_project_config, load_yaml
from blade_defect.utils.paths import resolve_path
from scripts.run_full_primary import run_primary
from scripts.run_v3_obb_baseline import run_obb_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = PROJECT_ROOT / "configs" / "v3_baselines.yaml"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_suite(path: str | Path = DEFAULT_SUITE) -> tuple[dict[str, Any], Path]:
    suite_path = resolve_path(path)
    payload = load_yaml(suite_path)
    project_root = resolve_path(payload.get("project_root", ".."), suite_path.parent)
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or not {"seg", "obb"}.issubset(tasks):
        raise ValueError(f"统一配置必须包含 tasks.seg 与 tasks.obb：{suite_path}")
    return payload, project_root


def version_report(suite: dict[str, Any]) -> dict[str, Any]:
    required = suite.get("required_versions") or {}
    installed_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    try:
        installed_ultralytics = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        installed_ultralytics = None
    checks = {
        "python": {
            "required": str(required.get("python") or ""),
            "installed": installed_python,
        },
        "ultralytics": {
            "required": str(required.get("ultralytics") or ""),
            "installed": installed_ultralytics,
        },
    }
    for check in checks.values():
        check["match"] = bool(check["required"] and check["installed"] == check["required"])
    return {"ok": all(item["match"] for item in checks.values()), "checks": checks}


def _profile_context(
    suite: dict[str, Any], project_root: Path, task: str
) -> dict[str, Any]:
    profile = dict(suite["tasks"][task])
    config_path = resolve_path(profile["config"], project_root)
    train_config, train_root = load_project_config(config_path, path_fields=("data", "project"))
    run_dir = resolve_path(train_config.get("project", "runs"), train_root) / str(
        profile["experiment_id"]
    )
    return {
        "task": task,
        "profile": profile,
        "config_path": config_path,
        "train_config": train_config,
        "project_root": train_root,
        "run_dir": run_dir,
    }


def task_status(suite: dict[str, Any], project_root: Path, task: str) -> dict[str, Any]:
    context = _profile_context(suite, project_root, task)
    run_dir = context["run_dir"]
    manifest = _read_json(run_dir / "run_manifest.json")
    last_pt = run_dir / "weights" / "last.pt"
    data_path = Path(context["train_config"]["data"])
    return {
        "task": task,
        "experiment_id": context["profile"]["experiment_id"],
        "config": str(context["config_path"]),
        "data": str(data_path),
        "data_available": data_path.is_file(),
        "run_dir": str(run_dir),
        "status": manifest.get("status", "not_started"),
        "last_checkpoint": str(last_pt) if last_pt.is_file() else None,
        "resumable": last_pt.is_file() and manifest.get("status") not in {"ok", "completed"},
    }


def _require_version(suite: dict[str, Any], allow_mismatch: bool) -> None:
    report = version_report(suite)
    if report["ok"] or allow_mismatch:
        return
    details = ", ".join(
        f"{name}: 需要 {item['required']}，当前 {item['installed']}"
        for name, item in report["checks"].items()
        if not item["match"]
    )
    raise RuntimeError(f"v3 正式环境版本不匹配（{details}）。如仅做试验可显式加 --allow-version-mismatch")


def _invoke(
    context: dict[str, Any], *, epochs: int | None, run_name: str | None,
    device: str | None, initial_checkpoint: Path | None, continuation: dict[str, Any],
) -> dict[str, Any]:
    common = {
        "config": context["config_path"],
        "epochs": epochs,
        "run_name": run_name,
        "device": device,
        "initial_checkpoint": initial_checkpoint,
        "continuation": continuation,
    }
    if context["task"] == "seg":
        return run_primary(**common)
    return run_obb_baseline(**common)


def run_base(
    suite: dict[str, Any], project_root: Path, task: str, *, device: str | None = None
) -> dict[str, Any]:
    context = _profile_context(suite, project_root, task)
    manifest = _read_json(context["run_dir"] / "run_manifest.json")
    if manifest.get("status") in {"ok", "completed"}:
        return {"task": task, "status": "skipped_completed", "run_dir": str(context["run_dir"])}
    total = int(context["profile"]["total_epochs"])
    lineage = {
        "kind": "base",
        "task": task,
        "stage": 0,
        "stage_epochs": total,
        "cumulative_target_epochs": total,
        "optimizer_state_policy": "resume_on_interruption",
    }
    return _invoke(
        context, epochs=None, run_name=None, device=device,
        initial_checkpoint=None, continuation=lineage,
    )


def _completed_candidates(context: dict[str, Any]) -> list[tuple[int, int, Path, dict[str, Any]]]:
    runs_dir = context["run_dir"].parent
    base_name = str(context["profile"]["experiment_id"])
    candidates: list[tuple[int, int, Path, dict[str, Any]]] = []
    if not runs_dir.is_dir():
        return candidates
    for manifest_path in runs_dir.glob("*/run_manifest.json"):
        manifest = _read_json(manifest_path)
        if manifest.get("status") not in {"ok", "completed"}:
            continue
        lineage = manifest.get("continuation") or {}
        is_base = manifest_path.parent.name == base_name
        if not is_base and lineage.get("task") != context["task"]:
            continue
        total = int(
            lineage.get("cumulative_target_epochs")
            or (context["profile"]["total_epochs"] if is_base else 0)
        )
        stage = int(lineage.get("stage") or 0)
        if total:
            candidates.append((total, stage, manifest_path.parent, manifest))
    return sorted(candidates, key=lambda item: (item[0], item[1]))


def plan_extension(
    suite: dict[str, Any], project_root: Path, task: str, *, add_epochs: int,
    checkpoint_kind: str = "best", from_run: str | Path | None = None,
) -> dict[str, Any]:
    if add_epochs <= 0:
        raise ValueError("--add-epochs 必须大于 0")
    context = _profile_context(suite, project_root, task)
    if from_run is None:
        candidates = _completed_candidates(context)
        if not candidates:
            raise RuntimeError(f"{task} 没有带权重的已完成阶段，不能扩轮")
        parent_total, parent_stage, parent_dir, _ = candidates[-1]
    else:
        parent_dir = resolve_path(from_run, project_root)
        manifest = _read_json(parent_dir / "run_manifest.json")
        if manifest.get("status") not in {"ok", "completed"}:
            raise RuntimeError(f"指定父阶段尚未完成：{parent_dir}")
        lineage = manifest.get("continuation") or {}
        parent_total = int(
            lineage.get("cumulative_target_epochs")
            or context["profile"]["total_epochs"]
        )
        parent_stage = int(lineage.get("stage") or 0)
    checkpoint = parent_dir / "weights" / f"{checkpoint_kind}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"父阶段缺少 {checkpoint_kind}.pt：{checkpoint}")
    total = parent_total + add_epochs
    stem = re.sub(r"_e\d+(?:_ft\d+)?$", "", str(context["profile"]["experiment_id"]))
    stage = parent_stage + 1
    run_name = f"{stem}_e{total}_ft{stage}"
    return {
        "task": task,
        "run_name": run_name,
        "run_dir": str(context["run_dir"].parent / run_name),
        "parent_run": str(parent_dir),
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": _sha256(checkpoint),
        "stage": stage,
        "stage_epochs": add_epochs,
        "cumulative_target_epochs": total,
        "optimizer_state_policy": "fresh_optimizer_after_completed_stage",
    }


def extend(
    suite: dict[str, Any], project_root: Path, task: str, *, add_epochs: int,
    checkpoint_kind: str = "best", from_run: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    context = _profile_context(suite, project_root, task)
    plan = plan_extension(
        suite, project_root, task, add_epochs=add_epochs,
        checkpoint_kind=checkpoint_kind, from_run=from_run,
    )
    return _invoke(
        context,
        epochs=add_epochs,
        run_name=plan["run_name"],
        device=device,
        initial_checkpoint=Path(plan["initial_checkpoint"]),
        continuation={"kind": "fine_tune_extension", **plan},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--allow-version-mismatch", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="只读查看版本、数据和训练状态")
    run_parser = subparsers.add_parser("run", help="首次运行或自动恢复未完成阶段")
    run_parser.add_argument("--task", choices=("seg", "obb", "all"), required=True)
    run_parser.add_argument("--device", default=None)
    extend_parser = subparsers.add_parser("extend", help="从已完成阶段新增微调轮次")
    extend_parser.add_argument("--task", choices=("seg", "obb"), required=True)
    extend_parser.add_argument("--add-epochs", type=int, required=True)
    extend_parser.add_argument("--weights", choices=("best", "last"), default="best")
    extend_parser.add_argument("--from-run", type=Path, default=None)
    extend_parser.add_argument("--device", default=None)
    args = parser.parse_args()

    suite, project_root = load_suite(args.suite)
    if args.command == "status":
        payload = {
            "suite": str(resolve_path(args.suite)),
            "versions": version_report(suite),
            "tasks": [task_status(suite, project_root, task) for task in ("seg", "obb")],
        }
    else:
        _require_version(suite, args.allow_version_mismatch)
        if args.command == "run":
            tasks = ("seg", "obb") if args.task == "all" else (args.task,)
            payload = {"runs": [run_base(suite, project_root, task, device=args.device) for task in tasks]}
        else:
            payload = extend(
                suite, project_root, args.task, add_epochs=args.add_epochs,
                checkpoint_kind=args.weights, from_run=args.from_run, device=args.device,
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
