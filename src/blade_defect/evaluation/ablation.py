"""Config-driven automatic ablation experiment framework."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from blade_defect.utils.files import load_yaml, save_json

ExperimentFn = Callable[[str, dict[str, Any]], dict[str, Any]]


class AblationRunner:
    """Run named parameter overrides through an injectable experiment function."""

    def __init__(self, config_path: str | Path, output_dir: str | Path = "runs/ablation_summary") -> None:
        self.config = load_yaml(config_path)
        self.output_dir = Path(output_dir)

    def build_experiments(self) -> list[tuple[str, dict[str, Any]]]:
        base = self.config.get("base", {})
        experiments = []
        for item in self.config.get("experiments", []):
            params = deepcopy(base)
            params.update(item.get("overrides", {}))
            experiments.append((item["name"], params))
        return experiments

    def run(self, experiment_fn: ExperimentFn, continue_on_error: bool = True) -> list[dict[str, Any]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for name, params in self.build_experiments():
            try:
                metrics = experiment_fn(name, params)
                record = {"name": name, "status": "ok", **params, **metrics}
            except Exception as exc:
                if not continue_on_error:
                    raise
                record = {"name": name, "status": "failed", "error": str(exc), **params}
            records.append(record)
            save_json(record, self.output_dir / f"{name}.json")
        self._write_csv(records)
        return records

    def _write_csv(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        fields = sorted({key for record in records for key in record})
        with (self.output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
