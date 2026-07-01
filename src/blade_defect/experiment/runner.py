"""Sequential execution of registered baseline experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from blade_defect.evaluation import metrics_from_ultralytics
from blade_defect.models import SegmentationTrainer
from blade_defect.utils.files import save_json
from blade_defect.utils.paths import resolve_path
from .config import ExperimentConfig
from .exporter import export_summary
from .registry import EXPERIMENTS


def _fps_from_result(result: Any) -> float:
    """Convert Ultralytics per-image inference milliseconds to frames/second."""
    milliseconds = float((getattr(result, "speed", {}) or {}).get("inference", 0.0) or 0.0)
    return 1000.0 / milliseconds if milliseconds > 0 else 0.0


def run_all_experiments(
    experiments: Iterable[ExperimentConfig] = EXPERIMENTS, *,
    data: str | Path = "configs/data.yaml", runs_dir: str | Path = "runs",
    results_file: str | Path = "results/summary.csv",
    device: str | int | None = "auto", continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Train and evaluate all registered baselines in registry order."""
    data_path, output_root = resolve_path(data), resolve_path(runs_dir)
    records: list[dict[str, Any]] = []
    for experiment in experiments:
        experiment_dir = output_root / experiment.name
        try:
            trainer = SegmentationTrainer(experiment.model)
            train_result = trainer.train(**experiment.training_kwargs(data_path, output_root, device))
            save_dir = Path(getattr(train_result, "save_dir", None) or experiment_dir)
            # Evaluate the learned checkpoint when available; model identifiers
            # remain a safe fallback for mocked or interrupted training outputs.
            best_model = save_dir / "weights" / "best.pt"
            evaluator = SegmentationTrainer(best_model if best_model.exists() else experiment.model)
            raw_result = evaluator.validate(data_path, imgsz=experiment.imgsz, device=device)
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
