"""按注册顺序执行 baseline 实验。"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from blade_defect.data.validation import (
    DatasetGateError,
    DatasetGateReport,
    validate_dataset_gate,
)
from blade_defect.evaluation import (
    FPS_METHOD,
    extended_metrics_from_ultralytics,
    per_class_metrics_from_ultralytics,
)
from blade_defect.models import SegmentationTrainer
from blade_defect.utils.files import load_project_config, load_yaml, save_json
from blade_defect.utils.paths import resolve_model_reference, resolve_path
from .config import ExperimentConfig
from .exporter import export_summary
from .failure_cases import export_failure_cases
from .metadata import (
    atomic_write_json,
    collect_dataset_metadata,
    collect_environment_metadata,
    collect_git_metadata,
    create_run_manifest,
    sha256_file,
    utc_timestamp,
)
from .prediction_exporter import export_validation_predictions
from .registry import EXPERIMENTS

# 兼容旧名称：门禁错误统一由 blade_defect.data.validation 提供。
DatasetValidationError = DatasetGateError


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            )
    except ImportError:
        info["torch_version"] = None
    try:
        import ultralytics

        info["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        info["ultralytics_version"] = None
    return info


def _dataset_identity(data_path: Path, gate_report: DatasetGateReport) -> dict[str, Any]:
    # dataset_id/root 直接取门禁结果：门禁使用不解析 junction 的 _abspath，
    # 与 Path.resolve() 混用曾在 6 类数据集上导致图片-标签错配。
    manifest_path = Path(gate_report.dataset_root) / "dataset_manifest.json"
    return {
        "dataset_id": gate_report.dataset_id,
        "dataset_yaml": str(data_path),
        "dataset_root": gate_report.dataset_root,
        "dataset_manifest_sha256": _sha256_file(manifest_path),
    }


def _write_per_class_metrics(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric_branch", "class_id", "class_name", "precision", "recall", "ap50", "ap50_95"]
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _experiment_id(name: str) -> str:
    """从实验名称中提取规范化 ID，例如 exp014。"""
    return name.split("_", 1)[0].lower()


def _normalize_selector(selector: str) -> str:
    """将 14、014、exp014 等写法统一为 exp014。"""
    value = selector.strip().lower()
    if value.isdigit():
        return f"exp{int(value):03d}"
    return value


def _select_experiments(
    experiments: Iterable[ExperimentConfig],
    imgsz: int | None,
    selectors: Iterable[str] | None,
) -> list[ExperimentConfig]:
    """按可选尺寸及名称/ID筛选实验，并拒绝未知选择器。"""
    available = list(experiments)
    normalized = [_normalize_selector(selector) for selector in selectors or []]
    if normalized:
        known_names = {experiment.name.lower() for experiment in available}
        known_ids = {_experiment_id(experiment.name) for experiment in available}
        unknown = [value for value in normalized if value not in known_names | known_ids]
        if unknown:
            raise ValueError(f"未知实验名称或 ID：{', '.join(unknown)}")
    selected = [
        experiment
        for experiment in available
        if (imgsz is None or experiment.imgsz == imgsz)
        and (
            not normalized
            or experiment.name.lower() in normalized
            or _experiment_id(experiment.name) in normalized
        )
    ]
    if not selected:
        conditions = []
        if imgsz is not None:
            conditions.append(f"imgsz={imgsz}")
        if normalized:
            conditions.append(f"experiment={','.join(normalized)}")
        raise ValueError(f"没有匹配的注册实验：{'; '.join(conditions)}")
    return selected


def _fps_from_result(result: Any) -> float:
    """将 Ultralytics 的单图推理毫秒数换算为 FPS。"""
    milliseconds = float((getattr(result, "speed", {}) or {}).get("inference", 0.0) or 0.0)
    return 1000.0 / milliseconds if milliseconds > 0 else 0.0


def run_all_experiments(
    experiments: Iterable[ExperimentConfig] = EXPERIMENTS, *,
    config: str | Path = "configs/train.yaml", runs_dir: str | Path = "runs",
    results_file: str | Path = "results/summary.csv",
    device: str | int | None = "auto", continue_on_error: bool = True,
    imgsz: int | None = None,
    experiment_selectors: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """使用同一份训练配置依次训练并评估筛选后的 baseline。

    训练前 strict 数据集门禁始终启用（支持目录型与 txt 清单型数据集），
    不再提供 ``--skip-validation`` 旁路。
    """
    selected_experiments = _select_experiments(experiments, imgsz, experiment_selectors)
    config_path = resolve_path(config)
    base_config, project_root = load_project_config(config_path)
    base_config.pop("model", None)
    data_path = resolve_path(base_config.get("data", "configs/data.yaml"), project_root)
    dataset_config = load_yaml(data_path)
    missing = [field for field in ("train", "val", "names") if field not in dataset_config]
    if missing:
        raise ValueError(f"Dataset YAML {data_path} is missing required fields: {', '.join(missing)}")
    gate_report = validate_dataset_gate(data_path)
    dataset_identity = _dataset_identity(data_path, gate_report)
    code_commit = _git_commit(project_root)
    environment = _environment_info()
    output_root = resolve_path(runs_dir)
    results_path = resolve_path(results_file)
    git_metadata = collect_git_metadata(project_root)
    environment_metadata = collect_environment_metadata()
    dataset_metadata = collect_dataset_metadata(data_path)
    records: list[dict[str, Any]] = []
    for experiment in selected_experiments:
        experiment_dir = output_root / experiment.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        model = resolve_model_reference(experiment.model, project_root)
        # 所有实验继承同一份 train.yaml；注册表只覆盖模型对比参数和输出路径。
        train_kwargs = {
            **base_config,
            **experiment.training_kwargs(data_path, output_root, device),
        }
        manifest_path = experiment_dir / "run_manifest.json"
        environment_path = experiment_dir / "environment.json"
        started_at = _now_iso()
        # 重跑前归档已有 manifest，避免覆盖上一轮（尤其已完成）的 provenance 记录。
        if manifest_path.is_file():
            archive_stamp = started_at.replace(":", "-")
            manifest_path.replace(
                manifest_path.with_name(f"run_manifest.{archive_stamp}.json")
            )
        base_record = {
            **experiment.to_dict(),
            "dataset_id": dataset_identity["dataset_id"],
            "code_commit": code_commit,
            "hardware": environment,
            "fps_method": FPS_METHOD,
        }
        manifest = create_run_manifest(
            experiment=experiment.to_dict(),
            effective_config=train_kwargs,
            config_path=config_path,
            model=model,
            requested_device=device,
            git=git_metadata,
            dataset=dataset_metadata,
            environment=environment_metadata,
        )
        # 补充门禁侧数据集 identity 与断点续训字段；顶层 dataset_id/config
        # 等键沿用 create_run_manifest 的统一 schema，不覆盖。
        manifest.update({
            "dataset_yaml": dataset_identity["dataset_yaml"],
            "dataset_root": dataset_identity["dataset_root"],
            "dataset_manifest_sha256": dataset_identity["dataset_manifest_sha256"],
            "command": " ".join(sys.argv),
            "resume_supported": True,
        })
        atomic_write_json(environment_path, environment_metadata)
        atomic_write_json(manifest_path, manifest)
        provenance = {
            "run_id": manifest["run_id"],
            "dataset_id": dataset_metadata["dataset_id"],
            "dataset_hash": dataset_metadata["dataset_hash"],
            "commit": git_metadata["commit"],
            "tags": git_metadata["tags_exact"],
        }
        try:
            trainer = SegmentationTrainer(model)
            train_result = trainer.train(normalize_data_yaml=False, **train_kwargs)
            save_dir = Path(getattr(train_result, "save_dir", None) or experiment_dir)
            # 优先评估训练得到的 best.pt；测试替身或中断场景下回退到原模型。
            best_model = save_dir / "weights" / "best.pt"
            evaluator = SegmentationTrainer(best_model if best_model.exists() else model)
            raw_result = evaluator.validate(
                data_path, imgsz=experiment.imgsz, device=device, normalize_data_yaml=False,
            )
            extended = extended_metrics_from_ultralytics(raw_result)
            record = {
                **base_record,
                **extended,
                # 兼容字段：历史 summary/分析脚本读取的标量口径为 Mask 指标。
                "mAP50": extended["mask_mAP50"],
                "mAP50-95": extended["mask_mAP50-95"],
                "precision": extended["mask_precision"],
                "recall": extended["mask_recall"],
                "fps": _fps_from_result(raw_result),
                "status": "ok",
            }
            _write_per_class_metrics(
                per_class_metrics_from_ultralytics(raw_result),
                experiment_dir / "per_class_metrics.csv",
            )
        except Exception as exc:
            record = {**base_record, "status": "failed", "error": str(exc)}
            save_json(record, experiment_dir / "metrics.json")
            manifest.update({
                "status": "failed",
                "finished_at": utc_timestamp(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "artifacts": {"metrics": str(experiment_dir / "metrics.json")},
            })
            atomic_write_json(manifest_path, manifest)
            if not continue_on_error:
                raise
        else:
            record.update(provenance)
            save_json(record, experiment_dir / "metrics.json")
            prediction_error: str | None = None
            predictions_path: Path | None = None
            try:
                predictions_path = experiment_dir / "validation_predictions.json"
                export_validation_predictions(
                    model_path=best_model if best_model.exists() else model,
                    data_yaml=data_path,
                    output_path=predictions_path,
                    experiment_id=experiment.name,
                    imgsz=experiment.imgsz,
                    device=str(device) if device is not None else "cpu",
                )
            except Exception as exc:
                prediction_error = f"{type(exc).__name__}: {exc}"
                record["prediction_export_error"] = str(exc)
                save_json(record, experiment_dir / "metrics.json")
            artifacts: dict[str, Any] = {
                "run_directory": str(save_dir),
                "metrics": str(experiment_dir / "metrics.json"),
                "environment": str(environment_path),
                "best_weights": str(best_model) if best_model.exists() else None,
                "best_weights_sha256": sha256_file(best_model) if best_model.exists() else None,
                "validation_predictions": (
                    str(predictions_path) if predictions_path is not None and predictions_path.exists() else None
                ),
            }
            manifest.update({
                "status": "completed",
                "finished_at": utc_timestamp(),
                "metrics": record,
                "artifacts": artifacts,
                "warnings": ([{"stage": "prediction_export", "message": prediction_error}]
                             if prediction_error else []),
            })
            atomic_write_json(manifest_path, manifest)
        records.append(record)
    export_summary(output_root, results_path)
    try:
        export_failure_cases(output_root, results_path.parent / "failure_cases" / "cases.csv")
    except Exception:
        pass
    return records


__all__ = ["DatasetGateError", "DatasetValidationError", "run_all_experiments"]
