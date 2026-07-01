"""Automated baseline experiment management."""
from .config import ExperimentConfig
from .analyzer import analyze_experiments
from .exporter import export_summary
from .registry import EXPERIMENTS
from .runner import run_all_experiments

__all__ = [
    "EXPERIMENTS", "ExperimentConfig", "analyze_experiments", "export_summary",
    "run_all_experiments",
]
