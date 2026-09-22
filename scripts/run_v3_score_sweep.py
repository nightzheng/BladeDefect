"""v3 平台期后的提分实验入口：计划、顺序训练、自动排名与人工复核表。

实验从已封存基线的 best.pt 新建微调阶段；不会恢复或改写原始 optimizer，
也不会读取 test。默认先运行 stage 1，stage 2 需显式指定。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blade_defect.utils.files import load_project_config, load_yaml, save_json
from blade_defect.utils.paths import resolve_path, user_path
from scripts.run_full_primary import run_primary
from scripts.run_v3_obb_baseline import run_obb_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v3_score_sweep.yaml"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_sweep(path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    config_path = resolve_path(path)
    payload = load_yaml(config_path)
    root = resolve_path(payload.get("project_root", ".."), config_path.parent)
    output = resolve_path(payload.get("output_dir", "results/v3_score_sweep"), root)
    return payload, root, output


def _foreign_absolute_path(value: str, target_os: str | None = None) -> bool:
    """Return whether *value* is absolute for the other operating-system family."""
    raw = value.strip()
    target = target_os or os.name
    if target == "nt":
        return PurePosixPath(raw).is_absolute() and not PureWindowsPath(raw).is_absolute()
    return PureWindowsPath(raw).is_absolute()


def _dataset_preflight(data_yaml: Path, *, full_data_check: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "available": data_yaml.is_file(),
        "test_read": False,
        "manifest_available": False,
        "index_sha256_verified": {},
        "splits": {},
        "errors": [],
    }
    if not data_yaml.is_file():
        report["errors"].append(f"数据配置不存在：{data_yaml}")
        return report
    data = load_yaml(data_yaml)
    manifest_path = data_yaml.parent / "dataset_manifest.json"
    manifest = _read_json(manifest_path)
    report["manifest_available"] = bool(manifest)
    expected_hashes = (
        manifest.get("derived_split_sha256")
        or manifest.get("release_split_sha256")
        or {}
    )
    raw_root = str(data.get("path", "."))
    if _foreign_absolute_path(raw_root):
        report["errors"].append(f"数据根目录是异平台绝对路径：{raw_root}")
        return report
    dataset_root = resolve_path(raw_root, data_yaml.parent)
    for split in ("train", "val"):
        entry = data.get(split)
        split_report: dict[str, Any] = {
            "entry": entry,
            "indexed": False,
            "references": 0,
            "checked": 0,
            "missing": 0,
            "missing_examples": [],
        }
        report["splits"][split] = split_report
        if not isinstance(entry, str) or not entry.strip():
            report["errors"].append(f"{data_yaml} 缺少 {split} 配置")
            continue
        if _foreign_absolute_path(entry):
            report["errors"].append(f"{split} 是异平台绝对路径：{entry}")
            continue
        split_path = resolve_path(entry, dataset_root)
        if not entry.lower().endswith(".txt"):
            split_report["checked"] = 1
            split_report["missing"] = 0 if split_path.exists() else 1
            if not split_path.exists():
                split_report["missing_examples"].append(str(split_path))
                report["errors"].append(f"{split} 路径不存在：{split_path}")
            continue
        split_report["indexed"] = True
        if not split_path.is_file():
            split_report["missing"] = 1
            split_report["missing_examples"].append(str(split_path))
            report["errors"].append(f"{split} 索引不存在：{split_path}")
            continue
        lines = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        split_report["references"] = len(lines)
        expected_count = (manifest.get("samples") or {}).get(split)
        if expected_count is not None and int(expected_count) != len(lines):
            report["errors"].append(
                f"{split} 清单数量 {len(lines)} 与 manifest {expected_count} 不一致"
            )
        expected_hash = expected_hashes.get(split)
        if expected_hash:
            digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
            verified = digest.casefold() == str(expected_hash).casefold()
            report["index_sha256_verified"][split] = verified
            if not verified:
                report["errors"].append(
                    f"{split} 索引 SHA-256 与 dataset_manifest.json 不一致"
                )
        candidates = lines if full_data_check else lines[: min(100, len(lines))]
        for raw in candidates:
            split_report["checked"] += 1
            if _foreign_absolute_path(raw):
                split_report["missing"] += 1
                if len(split_report["missing_examples"]) < 5:
                    split_report["missing_examples"].append(raw)
                continue
            path = user_path(raw)
            if not path.is_absolute():
                path = resolve_path(path, split_path.parent)
            if not path.is_file():
                split_report["missing"] += 1
                if len(split_report["missing_examples"]) < 5:
                    split_report["missing_examples"].append(str(path))
        if split_report["missing"]:
            report["errors"].append(
                f"{split} 有 {split_report['missing']}/{split_report['checked']} 个引用不可用"
            )
    return report


def preflight(
    payload: dict[str, Any], root: Path, output: Path, *, require_cuda: bool,
    full_data_check: bool,
) -> dict[str, Any]:
    """Validate a copied checkout before spending GPU time on a Windows/Linux node."""
    errors: list[str] = []
    warnings: list[str] = []
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    try:
        ultralytics_version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        ultralytics_version = None
    if python_version != "3.10":
        errors.append(f"需要 Python 3.10，当前为 {python_version}")
    if ultralytics_version != "8.4.90":
        errors.append(f"需要 Ultralytics 8.4.90，当前为 {ultralytics_version}")
    try:
        import torch

        torch_version: str | None = torch.__version__
        cuda_available: bool | None = bool(torch.cuda.is_available())
        cuda_devices = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception as exc:  # pragma: no cover - depends on the deployment image
        torch_version = None
        cuda_available = None
        cuda_devices = 0
        errors.append(f"PyTorch 导入失败：{type(exc).__name__}: {exc}")
    if require_cuda and not cuda_available:
        errors.append("要求 CUDA，但 torch.cuda.is_available() 不是 True")

    tasks: dict[str, Any] = {}
    for task_name, profile in payload["tasks"].items():
        baseline = resolve_path(profile["baseline_run"], root)
        best = baseline / "weights" / "best.pt"
        config_path = resolve_path(profile["base_config"], root)
        task_errors: list[str] = []
        if not best.is_file():
            task_errors.append(f"基线权重不存在：{best}")
        if not config_path.is_file():
            task_errors.append(f"基础配置不存在：{config_path}")
            dataset_report = {"errors": [f"基础配置不存在：{config_path}"]}
        else:
            config, _ = load_project_config(config_path, path_fields=("data", "project"))
            data_path = Path(config["data"])
            dataset_report = _dataset_preflight(
                data_path, full_data_check=full_data_check
            )
            task_errors.extend(dataset_report["errors"])
        for item in profile.get("experiments", []):
            if ":\\" in item["id"] or "/" in item["id"] or "\\" in item["id"]:
                task_errors.append(f"实验 ID 不是可移植目录名：{item['id']}")
        tasks[task_name] = {
            "baseline_best": str(best),
            "base_config": str(config_path),
            "dataset": dataset_report,
            "errors": task_errors,
        }
        errors.extend(f"{task_name}: {message}" for message in task_errors)

    if _foreign_absolute_path(str(payload.get("output_dir", ""))):
        errors.append(f"output_dir 是异平台绝对路径：{payload.get('output_dir')}")
    if not root.is_dir():
        errors.append(f"project_root 不存在：{root}")
    if output.drive and os.name != "nt":
        warnings.append(f"输出路径可能含 Windows 盘符：{output}")
    return {
        "ok": not errors,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": python_version,
            "executable": sys.executable,
            "path_separator": os.sep,
        },
        "packages": {
            "ultralytics": ultralytics_version,
            "torch": torch_version,
        },
        "cuda": {
            "required": require_cuda,
            "available": cuda_available,
            "device_count": cuda_devices,
        },
        "project_root": str(root),
        "output_dir": str(output),
        "full_data_check": full_data_check,
        "test_used": False,
        "tasks": tasks,
        "warnings": warnings,
        "errors": errors,
    }


def _selected(
    payload: dict[str, Any], task: str, stage: str, experiment: str | None
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    tasks = ("seg", "obb") if task == "all" else (task,)
    selected: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for task_name in tasks:
        profile = payload["tasks"][task_name]
        for item in profile.get("experiments", []):
            if experiment and item["id"] != experiment:
                continue
            if stage != "all" and int(item.get("stage", 1)) != int(stage):
                continue
            selected.append((task_name, profile, item))
    if experiment and not selected:
        raise ValueError(f"未找到实验：{experiment}")
    return selected


def build_plan(
    payload: dict[str, Any], root: Path, *, task: str, stage: str,
    experiment: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_name, profile, item in _selected(payload, task, stage, experiment):
        baseline = resolve_path(profile["baseline_run"], root)
        run_dir = root / "runs" / item["id"]
        status = _read_json(run_dir / "run_manifest.json").get("status", "not_started")
        rows.append(
            {
                "task": task_name,
                "experiment_id": item["id"],
                "stage": int(item.get("stage", 1)),
                "hypothesis": item.get("hypothesis"),
                "baseline_best": str(baseline / "weights" / "best.pt"),
                "run_dir": str(run_dir),
                "status": status,
                "overrides": item.get("overrides", {}),
                "test_used": False,
            }
        )
    return rows


def _materialize_config(
    root: Path, output: Path, profile: dict[str, Any], item: dict[str, Any]
) -> Path:
    base_path = resolve_path(profile["base_config"], root)
    config = load_yaml(base_path)
    config.update(item.get("overrides", {}))
    config.update(
        {
            "project_root": str(root),
            "project": "runs",
            "name": item["id"],
            "notes": (
                f"从已封存基线 best.pt 启动的独立微调实验；假设：{item.get('hypothesis', '')}。"
                "仅使用 train/val，test 锁定。"
            ),
        }
    )
    config_dir = output / "resolved_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    target = config_dir / f"{item['id']}.yaml"
    target.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    return target


def _convergence(run_dir: Path, task: str) -> dict[str, Any]:
    csv_path = run_dir / "results.csv"
    metric = "metrics/mAP50-95(M)" if task == "seg" else "metrics/mAP50-95(B)"
    if not csv_path.is_file():
        return {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [float(row[metric]) for row in rows if row.get(metric) not in {None, ""}]
    if not values:
        return {}
    best_index = max(range(len(values)), key=values.__getitem__)
    result: dict[str, Any] = {
        "completed_epochs": len(values),
        "best_epoch": best_index + 1,
        "best_metric": values[best_index],
        "last_metric": values[-1],
    }
    for window in (5, 10, 20):
        if len(values) >= window * 2:
            previous = values[-2 * window : -window]
            recent = values[-window:]
            result[f"last_{window}_mean_gain"] = sum(recent) / window - sum(previous) / window
            result[f"last_{window}_range"] = max(recent) - min(recent)
    return result


def _metric_payload(run_dir: Path, baseline: bool = False) -> dict[str, Any]:
    candidates = [run_dir / "metrics.json"]
    if baseline:
        candidates.insert(0, run_dir / "yolo_analysis_metrics.json")
    for path in candidates:
        payload = _read_json(path)
        if payload:
            return payload
    return {}


def _per_class(run_dir: Path, task: str, baseline: bool = False) -> dict[str, float]:
    candidates = [run_dir / "per_class_metrics.csv"]
    if baseline:
        candidates.insert(0, run_dir / "yolo_analysis_per_class_metrics.csv")
    branch = "mask" if task == "seg" else "box"
    for path in candidates:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {
            row["class_name"]: float(row["ap50_95"])
            for row in rows
            if row.get("metric_branch") == branch and row.get("ap50_95") not in {None, ""}
        }
    return {}


def analyze(payload: dict[str, Any], root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    policy = payload.get("decision_policy", {})
    min_gain = float(policy.get("minimum_primary_gain", 0.002))
    max_fps_drop = float(policy.get("maximum_fps_drop_ratio", 0.20))
    max_class_drop = float(policy.get("maximum_single_class_ap_drop", 0.02))
    rows: list[dict[str, Any]] = []
    convergence: dict[str, Any] = {}
    for task_name, profile in payload["tasks"].items():
        baseline_dir = resolve_path(profile["baseline_run"], root)
        primary = profile["primary_metric"]
        baseline_metrics = _metric_payload(baseline_dir, baseline=True)
        baseline_classes = _per_class(baseline_dir, task_name, baseline=True)
        baseline_value = baseline_metrics.get(primary)
        baseline_fps = baseline_metrics.get("fps")
        convergence[task_name] = _convergence(baseline_dir, task_name)
        rows.append(
            {
                "task": task_name,
                "experiment_id": baseline_dir.name,
                "kind": "baseline",
                "stage": 0,
                "status": _read_json(baseline_dir / "run_manifest.json").get("status"),
                "primary_metric": primary,
                "primary_value": baseline_value,
                "delta_vs_baseline": 0.0 if baseline_value is not None else None,
                "fps": baseline_fps,
                "fps_drop_ratio": 0.0 if baseline_fps is not None else None,
                "worst_class_delta": 0.0 if baseline_classes else None,
                "regressed_class_count": 0 if baseline_classes else None,
                "regressed_classes": "",
                "auto_decision": "reference",
                "run_dir": str(baseline_dir),
            }
        )
        for item in profile.get("experiments", []):
            run_dir = root / "runs" / item["id"]
            metrics = _metric_payload(run_dir)
            value = metrics.get(primary)
            fps = metrics.get("fps")
            class_metrics = _per_class(run_dir, task_name)
            class_deltas = {
                name: class_metrics[name] - baseline_classes[name]
                for name in sorted(class_metrics.keys() & baseline_classes.keys())
            }
            worst_class_delta = min(class_deltas.values()) if class_deltas else None
            regressed = [
                name for name, class_delta in class_deltas.items()
                if class_delta < -max_class_drop
            ]
            delta = value - baseline_value if value is not None and baseline_value is not None else None
            fps_drop = (
                (baseline_fps - fps) / baseline_fps
                if fps is not None and baseline_fps not in {None, 0}
                else None
            )
            if value is None:
                decision = "pending"
            elif (
                delta is not None
                and delta >= min_gain
                and worst_class_delta is not None
                and worst_class_delta < -max_class_drop
            ):
                decision = "per_class_regression_review"
            elif delta is not None and delta >= min_gain and (fps_drop is None or fps_drop <= max_fps_drop):
                decision = "candidate_for_human_review"
            elif delta is not None and delta >= min_gain:
                decision = "accuracy_gain_speed_tradeoff"
            else:
                decision = "reject_or_rework"
            rows.append(
                {
                    "task": task_name,
                    "experiment_id": item["id"],
                    "kind": "experiment",
                    "stage": int(item.get("stage", 1)),
                    "status": _read_json(run_dir / "run_manifest.json").get("status", "not_started"),
                    "primary_metric": primary,
                    "primary_value": value,
                    "delta_vs_baseline": delta,
                    "fps": fps,
                    "fps_drop_ratio": fps_drop,
                    "worst_class_delta": worst_class_delta,
                    "regressed_class_count": len(regressed) if class_deltas else None,
                    "regressed_classes": "; ".join(regressed),
                    "auto_decision": decision,
                    "run_dir": str(run_dir),
                }
            )
    rows.sort(
        key=lambda row: (
            row["task"], row["kind"] != "baseline",
            -(row["primary_value"] if row["primary_value"] is not None else -1),
        )
    )
    summary_path = output / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(convergence, output / "baseline_convergence.json")
    save_json({"generated_at": datetime.now().astimezone().isoformat(), "rows": rows}, output / "ranking.json")

    review_path = output / "human_review.csv"
    old_review: dict[str, dict[str, str]] = {}
    if review_path.is_file():
        with review_path.open(encoding="utf-8-sig", newline="") as handle:
            old_review = {row["experiment_id"]: row for row in csv.DictReader(handle)}
    review_fields = [
        "task", "experiment_id", "auto_decision", "primary_value", "delta_vs_baseline",
        "fps", "per_class_regression", "confusion_matrix_review", "failure_case_review",
        "accept", "reviewer_notes",
    ]
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for row in rows:
            if row["kind"] != "experiment":
                continue
            previous = old_review.get(row["experiment_id"], {})
            writer.writerow(
                {
                    **{field: previous.get(field, "") for field in review_fields},
                    "task": row["task"],
                    "experiment_id": row["experiment_id"],
                    "auto_decision": row["auto_decision"],
                    "primary_value": row["primary_value"],
                    "delta_vs_baseline": row["delta_vs_baseline"],
                    "fps": row["fps"],
                    "per_class_regression": (
                        row["regressed_classes"] or previous.get("per_class_regression", "")
                    ),
                }
            )
    lines = [
        "# v3 提分实验自动分析", "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}", "",
        f"自动候选门槛：主指标绝对提升 ≥ {min_gain:.4f}，FPS 降幅 ≤ {max_fps_drop:.0%}，单类 AP50-95 降幅 ≤ {max_class_drop:.3f}。", "",
        "自动结论只用于排序；定版前必须填写 `human_review.csv`，复核逐类退化、混淆矩阵和误检漏检样例。", "",
        "| task | experiment | stage | value | delta | FPS | decision |", "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        value = "—" if row["primary_value"] is None else f"{row['primary_value']:.5f}"
        delta = "—" if row["delta_vs_baseline"] is None else f"{row['delta_vs_baseline']:+.5f}"
        fps = "—" if row["fps"] is None else f"{row['fps']:.2f}"
        lines.append(
            f"| {row['task']} | {row['experiment_id']} | {row['stage']} | {value} | {delta} | {fps} | {row['auto_decision']} |"
        )
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summary": str(summary_path),
        "analysis": str(output / "analysis.md"),
        "human_review": str(review_path),
        "convergence": str(output / "baseline_convergence.json"),
    }


def run_experiments(
    payload: dict[str, Any], root: Path, output: Path, *, task: str, stage: str,
    experiment: str | None, device: str | None,
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for task_name, profile, item in _selected(payload, task, stage, experiment):
        if stage == "all" and int(item.get("stage", 1)) == 2:
            primary = profile["primary_metric"]
            baseline_dir = resolve_path(profile["baseline_run"], root)
            baseline_value = _metric_payload(baseline_dir, baseline=True).get(primary)
            minimum_gain = float(
                payload.get("decision_policy", {}).get("minimum_primary_gain", 0.002)
            )
            stage_one_passed = any(
                (
                    (value := _metric_payload(root / "runs" / candidate["id"]).get(primary))
                    is not None
                    and baseline_value is not None
                    and value - baseline_value >= minimum_gain
                )
                for candidate in profile.get("experiments", [])
                if int(candidate.get("stage", 1)) == 1
            )
            if not stage_one_passed:
                outcomes.append(
                    {
                        "experiment_id": item["id"],
                        "status": "skipped_stage_gate",
                        "reason": "stage 1 尚无实验达到主指标最小增益",
                    }
                )
                continue
        baseline = resolve_path(profile["baseline_run"], root)
        checkpoint = baseline / "weights" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"基线 best.pt 不存在：{checkpoint}")
        run_dir = root / "runs" / item["id"]
        manifest = _read_json(run_dir / "run_manifest.json")
        if manifest.get("status") in {"ok", "completed", "stopped_early"}:
            outcomes.append({"experiment_id": item["id"], "status": "skipped_completed"})
            continue
        config_path = _materialize_config(root, output, profile, item)
        continuation = {
            "kind": "score_sweep_finetune",
            "task": task_name,
            "stage": int(item.get("stage", 1)),
            "parent_run": str(baseline),
            "parent_checkpoint": str(checkpoint),
            "optimizer_state_policy": "fresh_optimizer_from_best_weights",
            "hypothesis": item.get("hypothesis"),
            "test_used": False,
        }
        runner = run_primary if task_name == "seg" else run_obb_baseline
        outcome = runner(
            config=config_path,
            run_name=item["id"],
            device=device,
            initial_checkpoint=checkpoint,
            continuation=continuation,
        )
        outcomes.append(outcome)
        analyze(payload, root, output)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "plan", "run", "analyze"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--task", choices=("seg", "obb", "all"), default="all")
    parser.add_argument("--stage", choices=("1", "2", "all"), default="1")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--sample-data-check", action="store_true",
        help="只检查每个 train/val 索引的前 100 个引用；默认全量检查",
    )
    args = parser.parse_args()
    payload, root, output = load_sweep(args.config)
    if args.command == "preflight":
        result = preflight(
            payload, root, output, require_cuda=args.require_cuda,
            full_data_check=not args.sample_data_check,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["ok"]:
            raise SystemExit(2)
        return
    if args.command == "plan":
        result: Any = build_plan(
            payload, root, task=args.task, stage=args.stage, experiment=args.experiment
        )
    elif args.command == "run":
        result = run_experiments(
            payload, root, output, task=args.task, stage=args.stage,
            experiment=args.experiment, device=args.device,
        )
    else:
        result = analyze(payload, root, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
