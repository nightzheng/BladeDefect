"""将 runs/exp001—exp008 标记为 blade-v2-sampled-4987 阶段性实验并补齐元数据。

只新增 dataset_id / manifest 哈希 / 代码提交 / 硬件等元数据字段，
不改动任何已有指标数值；采样版数据集已被全量冻结版覆盖，
其 manifest 哈希无法回溯，以 null 记录并在 note 中说明。
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SAMPLED_DATASET_ID = "blade-v2-sampled-4987"


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _environment_info() -> dict:
    import platform

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "note": "补齐时本机环境；历史训练的 torch/CUDA 版本未逐组记录",
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import ultralytics

        info["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        pass
    return info


def backfill(runs_dir: Path, project_root: Path) -> list[str]:
    commit = _git_commit(project_root)
    hardware = _environment_info()
    backfilled: list[str] = []
    for experiment_dir in sorted(runs_dir.glob("exp*/")):
        metrics_path = experiment_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.setdefault("dataset_id", SAMPLED_DATASET_ID)
        metrics["experiment_stage"] = "sampled_phase"
        metrics["code_commit"] = metrics.get("code_commit") or None
        metrics["hardware"] = hardware
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "experiment_id": metrics.get("name", experiment_dir.name),
            "experiment_stage": "sampled_phase",
            "dataset_id": SAMPLED_DATASET_ID,
            "dataset_manifest_sha256": None,
            "dataset_note": (
                "训练使用 blade-v2-sampled-4987 采样版（train 4000/val 1000，seed=42）；"
                "该采样目录已被全量冻结版覆盖，manifest 哈希无法回溯。"
                "本组结果为阶段性模型筛选结果，不与全量正式结果混排。"
            ),
            "code_commit": commit,
            "code_commit_note": "补齐时的仓库 HEAD；历史训练时的提交未逐组记录",
            "hardware": hardware,
            "config": {key: metrics.get(key) for key in ("model", "imgsz", "epochs", "batch", "seed")},
            "status": metrics.get("status", "ok"),
            "backfilled_at": hardware["recorded_at"],
        }
        (experiment_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        backfilled.append(experiment_dir.name)
    return backfilled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", type=Path)
    parser.add_argument("--project-root", default=".", type=Path)
    args = parser.parse_args()
    print(json.dumps(backfill(args.runs_dir, args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
