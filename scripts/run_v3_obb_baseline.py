"""运行正式 v3 OBB baseline（YOLO11s-obb@960，50 epochs）并导出统一口径工件。

特性：
- 训练前 OBB 数据门禁（仅 train/val，声明的 test 永不加载）；
- 官方 yolo11s-obb.pt 权重溯源核验（与 smoke 共用 acquire_official_weight）；
- 自动检测上次中断的 last.pt 并断点续训（--no-resume 可关闭）；
- 每 epoch 末验证（Ultralytics 默认）+ 周期性 checkpoint（save_period）；
- 训练开始前写入 environment.json 与 run_manifest.json（status=running），
  字段与 seg 正式实验（run_full_primary.py）保持一致；
- 完成后导出 metrics.json（Box 口径）、per_class_metrics.csv 与代表性预测；
- --calibrate-only 模式按真实显存与迭代耗时校准 batch 并记录，不进入正式训练。

用法：
    python scripts/run_v3_obb_baseline.py --config configs/experiments/v3_yolo11s_obb_960_e50.yaml
    python scripts/run_v3_obb_baseline.py --config configs/experiments/v3_yolo11s_obb_960_e50.yaml --calibrate-only --calibrate-batch 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in {None, ""}:
    # 直接以脚本方式执行时把仓库根目录加入 sys.path，使 scripts/blade_defect 均可导入。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blade_defect.evaluation import FPS_METHOD, per_class_metrics_from_ultralytics
from blade_defect.experiment.metadata import (
    atomic_write_json,
    collect_dataset_metadata,
    collect_environment_metadata,
    sha256_file,
)
from blade_defect.experiment.runner import (
    _environment_info,
    _fps_from_result,
    _git_commit,
    _write_per_class_metrics,
)
from blade_defect.utils.device import resolve_device
from blade_defect.utils.files import load_project_config, load_yaml, save_json
from blade_defect.utils.paths import posix_path, resolve_model_reference, resolve_path, user_path
from scripts.run_obb_smoke import acquire_official_weight, validate_smoke_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVENANCE = Path("results/obb_v3/weight_provenance.json")
CALIBRATION_REPORT = Path("results/obb_v3/batch_calibration.json")
# 校准允许的最高显存占用比例（含上下文开销的安全余量）。
CALIBRATION_MEMORY_BUDGET = 0.92

PASSTHROUGH_CONFIG_KEYS = {
    "epochs", "imgsz", "batch", "device", "workers", "seed", "pretrained",
    "cache", "amp", "val", "save", "save_period", "exist_ok", "patience",
}
NON_TRAINING_KEYS = {
    "model", "notes", "label_level", "task", "name", "project",
    "weight_provenance", "prediction_count",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _mean(value: Any) -> float:
    array = np.asarray(value if value is not None else [], dtype=float)
    return float(array.mean()) if array.size else 0.0


def _box_metrics(result: Any) -> dict[str, float]:
    """从 OBB 验证结果提取 Box 口径指标（OBB 无 Mask 分支）。"""
    box = getattr(result, "box", None)
    if box is None:
        raise ValueError("验证结果不包含 box 指标，请确认使用 obb 模型和数据")
    precision = _mean(getattr(box, "p", None))
    recall = _mean(getattr(box, "r", None))
    return {
        "box_precision": precision,
        "box_recall": recall,
        "box_mAP50": _mean(getattr(box, "map50", None)),
        "box_mAP50-95": _mean(getattr(box, "map", None)),
        "box_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _select_representative_val_images(
    val_images: list[Path], count: int, num_classes: int = 15
) -> list[Path]:
    """按贪心类别覆盖选取代表性 val 图片（确定性，按路径排序）。"""
    from blade_defect.data.indexed_splits import label_path_for_image

    def _classes(image: Path) -> set[int]:
        label = label_path_for_image(image)
        classes: set[int] = set()
        if label.is_file():
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                tokens = line.split()
                if tokens:
                    try:
                        classes.add(int(float(tokens[0])))
                    except ValueError:
                        continue
        return classes

    remaining = sorted(val_images)
    class_map = {image: _classes(image) for image in remaining}
    selected: list[Path] = []
    covered: set[int] = set()
    while remaining and len(selected) < count and len(covered) < num_classes:
        best = max(remaining, key=lambda image: len(class_map[image] - covered))
        if not class_map[best] - covered:
            break
        selected.append(best)
        covered.update(class_map[best])
        remaining.remove(best)
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected[:count]


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("请先安装 ultralytics：python -m pip install -r requirements.txt") from exc
    return YOLO


def _materialize_absolute_data_yaml(data_path: Path, output: Path) -> Path:
    """生成绝对化训练用 data.yaml（稳定路径，断点续训可复用）。

    派生 data.yaml 的 path 为 "."，Ultralytics 按当前工作目录解析相对 path；
    这里与 seg 正式实验的 resolved_data_yaml 同一策略：path 绝对化、train/val
    清单条目绝对化。训练可见 yaml 不含 test 键（test 锁定，训练永不加载）。
    """
    import os

    import yaml

    payload = load_yaml(data_path)
    dataset_root = resolve_path(payload.get("path", "."), data_path.parent)
    payload["path"] = posix_path(dataset_root)
    payload.pop("test", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        entry = payload.get(split)
        if not isinstance(entry, str) or not entry.lower().endswith(".txt"):
            continue
        list_path = resolve_path(entry, dataset_root)
        lines: list[str] = []
        for raw_line in list_path.read_text(encoding="utf-8-sig").splitlines():
            item = raw_line.strip()
            if not item:
                continue
            item_path = user_path(item)
            if not item_path.is_absolute():
                item_path = Path(os.path.abspath(os.fspath(list_path.parent / item_path)))
            lines.append(posix_path(item_path))
        absolute_list = output.with_name(f"{output.stem}.{split}.txt")
        absolute_list.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        payload[split] = posix_path(absolute_list)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)
    return output


def calibrate_batch(
    config: Path,
    batch: int,
    fraction: float = 0.02,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """用真实训练迭代校准 batch：测量峰值显存与秒/epoch 估算并落盘记录。"""
    if not 0 < fraction <= 1:
        raise ValueError("fraction 必须在 (0, 1] 内")
    train_config, project_root = load_project_config(config, path_fields=("data", "project"))
    data_path = resolve_path(train_config["data"], project_root)
    provenance_path = resolve_path(
        train_config.get("weight_provenance", DEFAULT_PROVENANCE), project_root
    )
    model_ref = resolve_model_reference(str(train_config.get("model", "yolo11s-obb.pt")), project_root)
    official_weight = acquire_official_weight(model_ref, project_root, provenance_path)
    validation_info, _ = validate_smoke_dataset(data_path)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("校准需要 CUDA GPU，当前 torch.cuda.is_available()=False")
    device = resolve_device(train_config.get("device", 0))
    device_index = int(str(device).split(",")[0]) if str(device).isdigit() else 0
    total_memory = torch.cuda.get_device_properties(device_index).total_memory
    imgsz = int(train_config.get("imgsz", 960))

    torch.cuda.reset_peak_memory_stats(device_index)
    torch.cuda.empty_cache()
    normalized_data = _materialize_absolute_data_yaml(
        data_path,
        PROJECT_ROOT / "runs" / "obb" / "batch_calibration_probe" / "normalized_data.yaml",
    )
    YOLO = _load_yolo()
    model = YOLO(str(official_weight))
    started = time.perf_counter()
    model.train(
        data=str(normalized_data),
        task="obb",
        epochs=1,
        fraction=fraction,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=int(train_config.get("workers", 4)),
        seed=int(train_config.get("seed", 42)),
        pretrained=True,
        cache=False,
        amp=bool(train_config.get("amp", True)),
        project=str(PROJECT_ROOT / "runs" / "obb"),
        name="batch_calibration_probe",
        exist_ok=True,
        val=False,
        save=False,
        plots=False,
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device_index))
    free_memory, _ = torch.cuda.mem_get_info(device_index)
    sample_base = int(validation_info.get("train_samples") or 0)
    if sample_base <= 0:
        raise RuntimeError("数据门禁未返回 train 图片数，无法估算秒/epoch")
    iterations = max(1, int(np.ceil(sample_base * fraction / max(batch, 1))))
    seconds_per_iteration = elapsed / iterations if iterations else 0.0
    estimated_seconds_per_epoch = (
        seconds_per_iteration * np.ceil(sample_base / max(batch, 1)) if sample_base else None
    )
    fits = peak_allocated <= total_memory * CALIBRATION_MEMORY_BUDGET
    payload: dict[str, Any] = {
        "generated_at": _now_iso(),
        "config": str(Path(config).resolve()),
        "data": str(data_path),
        "method": (
            "real short training probe: epochs=1, fraction="
            f"{fraction}, task=obb, imgsz={imgsz}, batch={batch}, amp={bool(train_config.get('amp', True))}"
        ),
        "batch_candidate": batch,
        "fraction": fraction,
        "probe_iterations": iterations,
        "probe_seconds": round(elapsed, 1),
        "seconds_per_iteration": round(seconds_per_iteration, 4),
        "estimated_seconds_per_epoch": (
            round(float(estimated_seconds_per_epoch), 1) if estimated_seconds_per_epoch else None
        ),
        "estimated_hours_50_epochs": (
            round(float(estimated_seconds_per_epoch) * 50 / 3600, 2)
            if estimated_seconds_per_epoch
            else None
        ),
        "gpu": torch.cuda.get_device_name(device_index),
        "gpu_total_memory_mb": round(total_memory / 1024 / 1024),
        "peak_allocated_mb": round(peak_allocated / 1024 / 1024, 1),
        "memory_budget_ratio": CALIBRATION_MEMORY_BUDGET,
        "free_memory_after_mb": round(free_memory / 1024 / 1024),
        "batch_fits_budget": bool(fits),
        "train_samples": sample_base,
        "note": (
            "peak_allocated_mb 为 torch.cuda.max_memory_allocated 实测；"
            "正式 batch 只在 batch_fits_budget=true 时采用该候选值。"
        ),
    }
    output = resolve_path(report_path or (PROJECT_ROOT / CALIBRATION_REPORT))
    atomic_write_json(output, payload)
    return payload


def run_obb_baseline(
    config: str | Path,
    *,
    epochs: int | None = None,
    run_name: str | None = None,
    device: str | None = None,
    resume: bool = True,
    force: bool = False,
    train_fn: Callable[..., Any] | None = None,
    eval_fn: Callable[..., Any] | None = None,
    predict_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """运行正式 OBB baseline：门禁 → 训练 → 评估 → 导出，全程可追溯。"""
    config_file = Path(config).resolve()
    train_config, project_root = load_project_config(
        config_file, path_fields=("data", "project", "weight_provenance")
    )
    data_path = resolve_path(train_config["data"], project_root)
    name = run_name or str(train_config.get("name", "v3_yolo11s_obb_960_e50"))
    if epochs is not None:
        train_config["epochs"] = epochs
    if device is not None:
        train_config["device"] = device

    validation_info, val_images = validate_smoke_dataset(data_path)
    dataset_metadata = collect_dataset_metadata(data_path)
    dataset_manifest = dataset_metadata.get("manifest") or {}
    code_commit = _git_commit(project_root)
    environment = _environment_info()

    project_dir = resolve_path(train_config.get("project", "runs"), project_root)
    run_dir = project_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"

    model_ref = str(train_config.get("model", "yolo11s-obb.pt"))
    last_pt = run_dir / "weights" / "last.pt"
    resume_active = bool(resume and last_pt.is_file())
    # 安全防护：已完成的运行不静默续训/覆盖；--no-resume 遇到已有权重时需显式 --force。
    previous_status: str | None = None
    if manifest_path.is_file():
        try:
            previous_status = json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            previous_status = None
    if previous_status == "ok" and not force:
        raise SystemExit(
            f"{run_dir} 已有完成的运行（status=ok）；如需重跑请更换 --run-name 或加 --force"
        )
    if not resume_active and last_pt.is_file() and not force:
        raise SystemExit(
            f"检测到已有权重 {last_pt}，当前设置将从头训练并覆盖它；确认请加 --force"
        )

    provenance_path = resolve_path(
        train_config.get("weight_provenance", DEFAULT_PROVENANCE), project_root
    )
    if resume_active:
        model_source: str | Path = last_pt
    else:
        model_source = acquire_official_weight(
            resolve_model_reference(model_ref, project_root), project_root, provenance_path
        )

    environment_path = run_dir / "environment.json"
    atomic_write_json(environment_path, collect_environment_metadata())
    # 绝对化训练用 data.yaml：path="." 会被 Ultralytics 按 CWD 解析；
    # 稳定落盘在 run_dir 内，断点续训时 checkpoint 记录的同一路径仍然有效。
    normalized_data = _materialize_absolute_data_yaml(data_path, run_dir / "normalized_data.yaml")

    started_at = _now_iso()
    manifest: dict[str, Any] = {
        "experiment_id": name,
        "task": "obb",
        "label_level": train_config.get("label_level", "fine"),
        "dataset_id": dataset_metadata["dataset_id"],
        "dataset_yaml": dataset_metadata["data_yaml"],
        "dataset_root": dataset_metadata["dataset_root"],
        "dataset_manifest_sha256": dataset_metadata["dataset_manifest_sha256"],
        "normalized_data_yaml": str(normalized_data),
        "parent_dataset_id": dataset_manifest.get("parent_dataset_id"),
        "code_commit": code_commit,
        "hardware": environment,
        "fps_method": FPS_METHOD,
        "config": {key: str(value) for key, value in train_config.items()},
        "command": " ".join(sys.argv),
        "started_at": started_at,
        "status": "running",
        "resume_supported": True,
        "resumed_from": str(last_pt) if resume_active else None,
        "test_used": False,
        "validated_splits": validation_info.get("validated_splits", ["train", "val"]),
    }
    save_json(manifest, manifest_path)

    train_kwargs = {
        key: value
        for key, value in train_config.items()
        if key in PASSTHROUGH_CONFIG_KEYS and value is not None
    }
    # 强制保存目录与 run_dir 一致，避免权重、manifest 与断点续训脱钩。
    train_kwargs["name"] = name
    train_kwargs["project"] = str(project_dir)
    train_kwargs.setdefault("val", True)
    train_kwargs.setdefault("save", True)
    train_kwargs.setdefault("exist_ok", True)
    if resume_active:
        train_kwargs["resume"] = True
        train_kwargs.pop("pretrained", None)

    resolved_device = resolve_device(train_config.get("device", 0))
    if train_fn is None:
        YOLO = _load_yolo()
        model = YOLO(str(model_source))
        train_callable = model.train
    else:
        train_callable = train_fn
    try:
        train_started = time.perf_counter()
        train_result = train_callable(data=str(normalized_data), task="obb", **train_kwargs)
        train_seconds = time.perf_counter() - train_started
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc), "finished_at": _now_iso()})
        save_json(manifest, manifest_path)
        raise
    save_dir = Path(getattr(train_result, "save_dir", None) or run_dir)

    # 训练完成后先落中间态：后续评估/导出若失败，现场可与"训练中断"区分。
    manifest.update(
        {
            "status": "trained_pending_eval",
            "save_dir": str(save_dir),
            "train_seconds": round(train_seconds, 1),
        }
    )
    save_json(manifest, manifest_path)

    try:
        best_model = save_dir / "weights" / "best.pt"
        eval_source = best_model if best_model.is_file() else model_source
        imgsz = int(train_config.get("imgsz", 960))
        if eval_fn is None:
            YOLO = _load_yolo()
            evaluator = YOLO(str(eval_source))
            eval_callable = evaluator.val
        else:
            eval_callable = eval_fn
        raw_result = eval_callable(
            data=str(normalized_data),
            task="obb",
            imgsz=imgsz,
            device=resolved_device,
            workers=int(train_config.get("workers", 4)),
        )
        box = _box_metrics(raw_result)
        record: dict[str, Any] = {
            "name": name,
            "task": "obb",
            "label_level": train_config.get("label_level", "fine"),
            "model": model_ref,
            "imgsz": imgsz,
            "epochs": train_config.get("epochs"),
            "batch": train_config.get("batch"),
            "seed": train_config.get("seed"),
            "dataset_id": dataset_metadata["dataset_id"],
            "parent_dataset_id": dataset_manifest.get("parent_dataset_id"),
            "code_commit": code_commit,
            "hardware": environment,
            **box,
            "fps": _fps_from_result(raw_result),
            "fps_method": FPS_METHOD,
            "train_seconds": round(train_seconds, 1),
            "seconds_per_epoch": (
                round(train_seconds / int(train_config.get("epochs", 1)), 1)
                if int(train_config.get("epochs", 1))
                else None
            ),
            "validated_splits": validation_info.get("validated_splits", ["train", "val"]),
            "test_used": False,
            "status": "ok",
        }
        save_json(record, run_dir / "metrics.json")
        _write_per_class_metrics(
            per_class_metrics_from_ultralytics(raw_result), run_dir / "per_class_metrics.csv"
        )
        prediction_count = int(train_config.get("prediction_count", 8))
        prediction_dir = run_dir / "predictions"
        try:
            representatives = _select_representative_val_images(val_images, prediction_count)
            if predict_fn is None:
                YOLO = _load_yolo()
                predictor = YOLO(str(eval_source))
                predict_callable = predictor.predict
            else:
                predict_callable = predict_fn
            for image in representatives:
                predict_callable(
                    source=str(image),
                    task="obb",
                    imgsz=imgsz,
                    device=resolved_device,
                    save=True,
                    project=str(prediction_dir.parent),
                    name=prediction_dir.name,
                    exist_ok=True,
                )
            record["representative_predictions"] = [
                str(image) for image in representatives
            ]
            save_json(record, run_dir / "metrics.json")
        except Exception as exc:
            record["prediction_export_error"] = f"{type(exc).__name__}: {exc}"
            save_json(record, run_dir / "metrics.json")
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc), "finished_at": _now_iso()})
        save_json(manifest, manifest_path)
        raise

    last_weights = save_dir / "weights" / "last.pt"
    manifest.update(
        {
            "status": "ok",
            "finished_at": _now_iso(),
            "save_dir": str(save_dir),
            "metrics": record,
            "artifacts": {
                "run_directory": str(save_dir),
                "best_weights": str(best_model) if best_model.is_file() else None,
                "best_weights_sha256": sha256_file(best_model) if best_model.is_file() else None,
                "last_weights": str(last_weights) if last_weights.is_file() else None,
                "last_weights_sha256": sha256_file(last_weights) if last_weights.is_file() else None,
                "results_csv": str(save_dir / "results.csv"),
                "metrics": str(run_dir / "metrics.json"),
                "per_class_metrics": str(run_dir / "per_class_metrics.csv"),
                "environment": str(environment_path),
                "weight_provenance": str(provenance_path),
            },
        }
    )
    save_json(manifest, manifest_path)
    return {"run_dir": str(run_dir), "metrics": record, "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epochs")
    parser.add_argument("--run-name", default=None, help="覆盖输出目录名")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-resume", action="store_true", help="即使存在 last.pt 也从头训练")
    parser.add_argument("--force", action="store_true", help="允许覆盖已完成运行或已有权重")
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="只运行真实显存/吞吐校准探针并记录，不进入正式训练",
    )
    parser.add_argument("--calibrate-batch", type=int, default=8, help="校准候选 batch")
    parser.add_argument("--calibrate-fraction", type=float, default=0.02, help="校准数据比例")
    args = parser.parse_args()
    if args.calibrate_only:
        payload = calibrate_batch(
            args.config,
            batch=args.calibrate_batch,
            fraction=args.calibrate_fraction,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not payload["batch_fits_budget"]:
            raise SystemExit(1)
        return
    result = run_obb_baseline(
        config=args.config,
        epochs=args.epochs,
        run_name=args.run_name,
        device=args.device,
        resume=not args.no_resume,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
