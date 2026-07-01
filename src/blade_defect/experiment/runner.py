"""Sequential execution of registered baseline experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from blade_defect.evaluation import metrics_from_ultralytics
from blade_defect.models import SegmentationTrainer
from blade_defect.utils.files import load_project_config, load_yaml, save_json
from blade_defect.utils.paths import resolve_model_reference, resolve_path
from .config import ExperimentConfig
from .exporter import export_summary
from .registry import EXPERIMENTS


def _fps_from_result(result: Any) -> float:
    """Convert Ultralytics per-image inference milliseconds to frames/second."""
    milliseconds = float((getattr(result, "speed", {}) or {}).get("inference", 0.0) or 0.0)
    return 1000.0 / milliseconds if milliseconds > 0 else 0.0


def run_all_experiments(
    experiments: Iterable[ExperimentConfig] = EXPERIMENTS, *,
    config: str | Path = "configs/train.yaml", runs_dir: str | Path = "runs",
    results_file: str | Path = "results/summary.csv",
    device: str | int | None = "auto", continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Train and evaluate all baselines using one shared training config."""
    base_config, project_root = load_project_config(config)
    base_config.pop("model", None)
    data_path = resolve_path(base_config.get("data", "configs/data.yaml"), project_root)
    dataset_config = load_yaml(data_path)
    missing = [field for field in ("train", "val", "names") if field not in dataset_config]
    if missing:
        raise ValueError(f"Dataset YAML {data_path} is missing required fields: {', '.join(missing)}")
    output_root = resolve_path(runs_dir)
    records: list[dict[str, Any]] = []
    for experiment in experiments:
        experiment_dir = output_root / experiment.name
        try:
            model = resolve_model_reference(experiment.model, project_root)
            trainer = SegmentationTrainer(model)
            # All experiments inherit the same train.yaml options. Registry
            # values only override model-comparison parameters and run paths.
            train_kwargs = {
                **base_config,
                **experiment.training_kwargs(data_path, output_root, device),
            }
            train_result = trainer.train(normalize_data_yaml=False, **train_kwargs)
            save_dir = Path(getattr(train_result, "save_dir", None) or experiment_dir)
            # Evaluate the learned checkpoint when available; model identifiers
            # remain a safe fallback for mocked or interrupted training outputs.
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
        records.append(record)
    export_summary(output_root, results_file)
    return records
