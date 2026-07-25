"""按注册顺序执行 baseline 实验。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from blade_defect.data import check_dataset
from blade_defect.evaluation import metrics_from_ultralytics
from blade_defect.models import SegmentationTrainer
from blade_defect.utils.files import load_dataset_config, load_project_config, load_yaml, save_json
from blade_defect.utils.paths import resolve_model_reference, resolve_path
from .config import ExperimentConfig
from .exporter import export_summary
from .failure_cases import export_failure_cases
from .prediction_exporter import export_validation_predictions
from .registry import EXPERIMENTS


class DatasetValidationError(RuntimeError):
    """数据集未通过训练前严格检查时抛出。"""


def _validate_dataset(data_path: Path) -> None:
    """在创建模型前对 train/val 执行 strict 数据检查门禁。"""
    dataset = load_dataset_config(data_path)
    names = dataset.get("names", {})
    num_classes = len(names) if isinstance(names, (dict, list)) else None
    failures: list[str] = []
    for split in ("train", "val"):
        images_dir = dataset.get(split)
        labels_dir = dataset["path"] / "labels" / split
        if not isinstance(images_dir, Path) or not images_dir.is_dir():
            failures.append(f"{split}: images directory not found: {images_dir}")
            continue
        if not labels_dir.is_dir():
            failures.append(f"{split}: labels directory not found: {labels_dir}")
            continue
        report = check_dataset(
            images_dir,
            labels_dir,
            num_classes=num_classes,
            polygon_mode="strict",
            dry_run=True,
        )
        # 孤立标注意味着对应图片缺失，因此计入 missing_images。
        if report.orphan_labels:
            failures.append(f"{split}: missing_images={len(report.orphan_labels)}")
        if report.missing_labels:
            failures.append(f"{split}: missing_labels={len(report.missing_labels)}")
        if report.issues:
            failures.append(f"{split}: hard_error={len(report.issues)}")
    if failures:
        raise DatasetValidationError(
            "Dataset validation failed; run-all stopped before training: " + "; ".join(failures)
        )


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
    skip_validation: bool = False,
    imgsz: int | None = None,
    experiment_selectors: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """使用同一份训练配置依次训练并评估筛选后的 baseline。"""
    selected_experiments = _select_experiments(experiments, imgsz, experiment_selectors)
    base_config, project_root = load_project_config(config)
    base_config.pop("model", None)
    data_path = resolve_path(base_config.get("data", "configs/data.yaml"), project_root)
    dataset_config = load_yaml(data_path)
    missing = [field for field in ("train", "val", "names") if field not in dataset_config]
    if missing:
        raise ValueError(f"Dataset YAML {data_path} is missing required fields: {', '.join(missing)}")
    if not skip_validation:
        _validate_dataset(data_path)
    output_root = resolve_path(runs_dir)
    records: list[dict[str, Any]] = []
    for experiment in selected_experiments:
        experiment_dir = output_root / experiment.name
        try:
            model = resolve_model_reference(experiment.model, project_root)
            trainer = SegmentationTrainer(model)
            # 所有实验继承同一份 train.yaml；注册表只覆盖模型对比参数和输出路径。
            train_kwargs = {
                **base_config,
                **experiment.training_kwargs(data_path, output_root, device),
            }
            train_result = trainer.train(normalize_data_yaml=False, **train_kwargs)
            save_dir = Path(getattr(train_result, "save_dir", None) or experiment_dir)
            # 优先评估训练得到的 best.pt；测试替身或中断场景下回退到原模型。
            best_model = save_dir / "weights" / "best.pt"
            evaluator = SegmentationTrainer(best_model if best_model.exists() else model)
            raw_result = evaluator.validate(
                data_path, imgsz=experiment.imgsz, device=device, normalize_data_yaml=False,
            )
            metrics = metrics_from_ultralytics(raw_result)
            record = {**experiment.to_dict(), "mAP50": metrics.map50,
                      "mAP50-95": metrics.map50_95, "precision": metrics.precision,
                      "recall": metrics.recall, "fps": _fps_from_result(raw_result), "status": "ok"}
        except Exception as exc:
            record = {**experiment.to_dict(), "status": "failed", "error": str(exc)}
            save_json(record, experiment_dir / "metrics.json")
            if not continue_on_error:
                raise
        else:
            save_json(record, experiment_dir / "metrics.json")
            try:
                predictions_path = experiment_dir / "validation_predictions.json"
                export_validation_predictions(
                    model_path=best_model if best_model.exists() else model,
                    data_yaml=data_path,
                    output_path=predictions_path,
                    experiment_id=experiment.name,
                    imgsz=experiment.imgsz,
                    device=device,
                )
            except Exception:
                pass
        records.append(record)
    export_summary(output_root, results_file)
    try:
        export_failure_cases(output_root, results_file.parent / "failure_cases" / "cases.csv")
    except Exception:
        pass
    return records


__all__ = ["DatasetValidationError", "run_all_experiments"]
