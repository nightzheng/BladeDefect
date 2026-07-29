"""采集并持久化可复现实验所需的代码、数据和运行环境元数据。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from blade_defect.utils.files import load_dataset_config
from blade_defect.utils.paths import resolve_path


def utc_timestamp() -> str:
    """返回带 UTC 时区的秒级 ISO 时间戳。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA256，避免将权重等大文件一次性读入内存。"""
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    """将 Path 等运行时对象递归转换为 JSON 可序列化值。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """通过同目录临时文件和原子替换写入 JSON。"""
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_value(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _git_output(project_root: Path, *arguments: str) -> str | None:
    """执行只读 Git 查询；目录不是仓库或命令失败时返回空值。"""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_git_metadata(project_root: str | Path) -> dict[str, Any]:
    """采集 commit、分支、精确 Tag、最近 Tag 描述和工作区脏状态。"""
    root = resolve_path(project_root)
    commit = _git_output(root, "rev-parse", "HEAD")
    branch = _git_output(root, "branch", "--show-current")
    exact_tags_text = _git_output(root, "tag", "--points-at", "HEAD") or ""
    describe = _git_output(root, "describe", "--tags", "--always", "--dirty")
    status_text = _git_output(root, "status", "--porcelain", "--untracked-files=all") or ""
    dirty_files = [line for line in status_text.splitlines() if line.strip()]
    return {
        "repository_root": str(root),
        "commit": commit,
        "branch": branch,
        "tags_exact": sorted(tag for tag in exact_tags_text.splitlines() if tag),
        "describe": describe,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment_metadata() -> dict[str, Any]:
    """采集 Python、核心依赖、CUDA 和 GPU 环境信息。"""
    environment: dict[str, Any] = {
        "captured_at": utc_timestamp(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": {
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "ultralytics": _package_version("ultralytics"),
            "numpy": _package_version("numpy"),
            "opencv_python": _package_version("opencv-python"),
        },
        "cuda": {"available": False, "version": None, "cudnn_version": None},
        "gpus": [],
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        environment["cuda"] = {
            "available": cuda_available,
            "version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if cuda_available:
            environment["gpus"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
    except (ImportError, RuntimeError, OSError) as exc:
        environment["collection_warning"] = f"{type(exc).__name__}: {exc}"
    return environment


def _resolve_device_name(
    requested_device: str | int | None,
    environment: Mapping[str, Any] | None,
) -> Any:
    """根据请求的设备编号返回对应 GPU 名称，无法判断时保留原值。"""
    if str(requested_device).lower() == "cpu":
        return "cpu"
    if not environment:
        return requested_device
    gpus = environment.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        return requested_device

    gpu_by_index: dict[int, str] = {}
    for gpu in gpus:
        if not isinstance(gpu, Mapping):
            continue
        index = gpu.get("index")
        name = gpu.get("name")
        if isinstance(index, int) and isinstance(name, str):
            gpu_by_index[index] = name

    requested_text = str(requested_device).lower().removeprefix("cuda:")
    if requested_text in {"none", "auto", "cuda"}:
        requested_indexes = [min(gpu_by_index)] if gpu_by_index else []
    else:
        try:
            requested_indexes = [int(part.strip()) for part in requested_text.split(",")]
        except ValueError:
            requested_indexes = []
    names: list[str] = [gpu_by_index[index] for index in requested_indexes if index in gpu_by_index]
    return ", ".join(names) if names else requested_device


def _content_dataset_hash(manifest: Mapping[str, Any]) -> str | None:
    """优先使用数据内容摘要组合出稳定的数据集哈希。"""
    if manifest.get("dataset_hash"):
        return str(manifest["dataset_hash"])
    components = {
        key: manifest.get(key)
        for key in ("frozen_labels_sha256", "sample_lists_sha256")
        if manifest.get(key)
    }
    filter_config = manifest.get("filter_config")
    if isinstance(filter_config, Mapping) and filter_config.get("sha256"):
        components["filter_config_sha256"] = filter_config["sha256"]
    if not components:
        return None
    serialized = json.dumps(components, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _fallback_dataset_hash(dataset_root: Path, data_path: Path) -> str:
    """聚合标签、样本清单和 data.yaml，避免对大型图片重复计算哈希。"""
    digest = hashlib.sha256()
    candidates = [data_path]
    labels_root = dataset_root / "labels"
    if labels_root.is_dir():
        candidates.extend(
            sorted(path for path in labels_root.rglob("*") if path.is_file())
        )
    candidates.extend(
        path for path in (dataset_root / "train.txt", dataset_root / "val.txt")
        if path.is_file()
    )
    for path in candidates:
        try:
            relative = path.relative_to(dataset_root).as_posix()
        except ValueError:
            relative = path.name
        digest.update(relative.encode("utf-8"))
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def collect_dataset_metadata(data_yaml: str | Path) -> dict[str, Any]:
    """读取 data.yaml 及冻结数据 manifest，不重复扫描大型数据集。"""
    data_path = resolve_path(data_yaml)
    dataset = load_dataset_config(data_path)
    dataset_root = Path(dataset["path"])
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                manifest = loaded
        except (OSError, json.JSONDecodeError):
            manifest = None
    manifest_hash = _content_dataset_hash(manifest or {})
    return {
        "dataset_id": str((manifest or {}).get("dataset_id") or dataset_root.name),
        "dataset_root": str(dataset_root),
        "data_yaml": str(data_path),
        "data_yaml_sha256": sha256_file(data_path),
        "dataset_manifest": str(manifest_path) if manifest_path.is_file() else None,
        "dataset_manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "dataset_hash": manifest_hash or _fallback_dataset_hash(dataset_root, data_path),
        "dataset_hash_source": (
            "dataset_manifest_content" if manifest_hash else "labels_split_lists_and_data_yaml"
        ),
        "manifest_available": manifest is not None,
        "manifest": manifest,
    }


def create_run_manifest(
    *,
    experiment: Mapping[str, Any],
    effective_config: Mapping[str, Any],
    config_path: str | Path,
    model: str | Path,
    requested_device: str | int | None,
    git: Mapping[str, Any],
    dataset: Mapping[str, Any],
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """创建训练开始前的实验 manifest。"""
    config_file = resolve_path(config_path)
    started_at = utc_timestamp()
    commit = git.get("commit")
    commit_short = str(commit)[:8] if commit else "no-git"
    timestamp_id = started_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
    run_id = f"{experiment.get('name')}__{timestamp_id}__{commit_short}"
    resolved_device = _resolve_device_name(requested_device, environment)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": experiment.get("name"),
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "dataset_id": dataset.get("dataset_id"),
        "dataset_hash": dataset.get("dataset_hash"),
        "dataset": dict(dataset),
        "model": str(model),
        "imgsz": experiment.get("imgsz"),
        "epochs": experiment.get("epochs"),
        "batch": experiment.get("batch"),
        "seed": experiment.get("seed"),
        "device": resolved_device,
        "requested_device": requested_device,
        "git": dict(git),
        "config": {
            "path": str(config_file),
            "sha256": sha256_file(config_file),
            "effective": dict(effective_config),
        },
        "metrics": None,
        "artifacts": {},
        "warnings": [],
        "error": None,
    }


__all__ = [
    "atomic_write_json",
    "collect_dataset_metadata",
    "collect_environment_metadata",
    "collect_git_metadata",
    "create_run_manifest",
    "sha256_file",
    "utc_timestamp",
]
