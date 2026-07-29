"""自动化 baseline 实验管理。"""
from .config import ExperimentConfig
from .analyzer import analyze_experiments, publish_analysis_assets
from .exporter import export_summary
from .failure_cases import export_failure_cases, export_failure_cases_csv
from .metadata import collect_environment_metadata, collect_git_metadata
from .prediction_exporter import export_validation_predictions
from .registry import EXPERIMENTS
from .runner import run_all_experiments

__all__ = [
    "EXPERIMENTS", "ExperimentConfig", "analyze_experiments", "export_summary",
    "export_failure_cases", "export_failure_cases_csv",
    "export_validation_predictions", "publish_analysis_assets",
    "collect_environment_metadata", "collect_git_metadata",
    "run_all_experiments",
]
