"""启动全量主 baseline（或 6 类 smoke）实验：门禁 → 训练 → 评估 → 导出。

特性：
- 训练前 strict 数据门禁始终启用（支持 txt 清单型数据集）；
- 自动检测上次中断的 last.pt 并断点续训（--no-resume 可关闭）；
- 每 epoch 末验证（Ultralytics 默认）+ 周期性 checkpoint（save_period）；
- 训练开始前写入 run_manifest.json（status=running），结束后更新；
- 完成后导出 metrics.json（Mask/Box 分离口径）、per_class_metrics.csv
  与 validation_predictions.json（原始逐样本/逐实例字段）。

用法：
    python scripts/run_full_primary.py --config configs/experiments/full_yolo11s_seg_960.yaml
    python scripts/run_full_primary.py --config configs/experiments/hier_coarse_yolo11s_seg_960.yaml --epochs 3 --run-name hier_coarse_yolo11s_seg_960_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from blade_defect.data.validation import validate_dataset_gate
from blade_defect.evaluation import (
    FPS_METHOD,
    extended_metrics_from_ultralytics,
    per_class_metrics_from_ultralytics,
)
from blade_defect.experiment.metadata import (
    atomic_write_json,
    collect_environment_metadata,
)
from blade_defect.experiment.prediction_exporter import export_validation_predictions
from blade_defect.experiment.runner import (
    _dataset_identity,
    _environment_info,
    _fps_from_result,
    _git_commit,
    _write_per_class_metrics,
)
from blade_defect.models import SegmentationTrainer
from blade_defect.utils.device import resolve_device
from blade_defect.utils.files import load_project_config, save_json
from blade_defect.utils.paths import resolve_model_reference, resolve_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run_primary(
    config: Path,
    epochs: int | None = None,
    run_name: str | None = None,
    device: str | None = None,
    resume: bool = True,
    force: bool = False,
    initial_checkpoint: str | Path | None = None,
    continuation: dict | None = None,
) -> dict:
    train_config, project_root = load_project_config(config, path_fields=("data", "project"))
    data_path = resolve_path(train_config["data"], project_root)
    name = run_name or str(train_config.get("name", "full_primary"))
    if epochs is not None:
        train_config["epochs"] = epochs
    if device is not None:
        train_config["device"] = device

    gate_report = validate_dataset_gate(data_path)
    dataset_identity = _dataset_identity(data_path, gate_report)
    code_commit = _git_commit(project_root)
    environment = _environment_info()

    project_dir = resolve_path(train_config.get("project", "runs"), project_root)
    run_dir = project_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"

    model_ref = str(train_config.get("model", "yolo11s-seg.pt"))
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
    if resume_active:
        model_source: str | Path = last_pt
    elif initial_checkpoint is not None:
        model_source = resolve_path(initial_checkpoint, project_root)
        if not Path(model_source).is_file():
            raise FileNotFoundError(f"续训起始权重不存在：{model_source}")
    else:
        model_source = resolve_model_reference(model_ref, project_root)

    environment_path = run_dir / "environment.json"
    atomic_write_json(environment_path, collect_environment_metadata())

    started_at = _now_iso()
    manifest = {
        "experiment_id": name,
        **dataset_identity,
        "code_commit": code_commit,
        "hardware": environment,
        "fps_method": FPS_METHOD,
        "config": {key: str(value) for key, value in train_config.items()},
        "command": " ".join(sys.argv),
        "started_at": started_at,
        "status": "running",
        "resume_supported": True,
        "resumed_from": str(last_pt) if resume_active else None,
        "initial_checkpoint": (
            str(model_source) if initial_checkpoint is not None and not resume_active else None
        ),
        "continuation": continuation,
    }
    save_json(manifest, manifest_path)

    train_kwargs = {
        key: value
        for key, value in train_config.items()
        if key not in {"model", "notes", "label_level"}
    }
    # 强制保存目录与 run_dir 一致：--run-name 覆盖时 config 里的 name/project 会导致
    # 权重写到别的目录，使断点续训与 manifest/metrics 全部脱钩。
    train_kwargs["name"] = name
    train_kwargs["project"] = str(project_dir)
    train_kwargs.setdefault("val", True)
    train_kwargs.setdefault("save", True)
    train_kwargs.setdefault("exist_ok", True)
    if resume_active:
        train_kwargs["resume"] = True
        train_kwargs.pop("pretrained", None)

    trainer = SegmentationTrainer(model_source)
    stop_request_path = run_dir / "stop_request.json"

    def _stop_after_completed_epoch(ultralytics_trainer) -> None:
        if stop_request_path.is_file():
            ultralytics_trainer.stop = True

    # External `run_v3_baselines.py stop` requests are honored only at an epoch
    # boundary, after which Ultralytics still performs validation and saves last.pt.
    trainer.model.add_callback("on_train_epoch_end", _stop_after_completed_epoch)
    try:
        # normalize_data_yaml=True：将 data.yaml 的 path 绝对化后再交给 Ultralytics，
        # 避免相对 path 被解析到 DATASETS_DIR 之外。
        train_result = trainer.train(normalize_data_yaml=True, **train_kwargs)
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc), "finished_at": _now_iso()})
        save_json(manifest, manifest_path)
        raise
    save_dir = Path(getattr(train_result, "save_dir", None) or run_dir)

    # 训练已完成后先落中间态：后续评估/导出若失败，现场可与"训练中断"区分。
    manifest.update({"status": "trained_pending_eval", "save_dir": str(save_dir)})
    save_json(manifest, manifest_path)
    try:
        best_model = save_dir / "weights" / "best.pt"
        eval_source = best_model if best_model.is_file() else model_source
        evaluator = SegmentationTrainer(eval_source)
        imgsz = int(train_config.get("imgsz", 960))
        raw_result = evaluator.validate(
            data_path, imgsz=imgsz, device=train_config.get("device", "auto"),
            normalize_data_yaml=True, plots=bool(train_config.get("plots", True)),
            project=str(run_dir), name="yolo_analysis_val", exist_ok=True,
        )
        extended = extended_metrics_from_ultralytics(raw_result)
        record = {
            "name": name,
            "model": model_ref,
            "imgsz": imgsz,
            "epochs": train_config.get("epochs"),
            "batch": train_config.get("batch"),
            "seed": train_config.get("seed"),
            "dataset_id": dataset_identity["dataset_id"],
            "code_commit": code_commit,
            "hardware": environment,
            **extended,
            "mAP50": extended["mask_mAP50"],
            "mAP50-95": extended["mask_mAP50-95"],
            "precision": extended["mask_precision"],
            "recall": extended["mask_recall"],
            "fps": _fps_from_result(raw_result),
            "fps_method": FPS_METHOD,
            "status": "ok",
        }
        if continuation:
            record.update(
                stage_epochs=continuation.get("stage_epochs"),
                cumulative_target_epochs=continuation.get("cumulative_target_epochs"),
                continuation_from=continuation.get("parent_run"),
            )
        save_json(record, run_dir / "metrics.json")
        _write_per_class_metrics(
            per_class_metrics_from_ultralytics(raw_result), run_dir / "per_class_metrics.csv"
        )
        try:
            export_validation_predictions(
                model_path=eval_source,
                data_yaml=data_path,
                output_path=run_dir / "validation_predictions.json",
                experiment_id=name,
                imgsz=imgsz,
                device=str(resolve_device(train_config.get("device", "auto"))),
            )
        except Exception as exc:
            record["prediction_export_error"] = str(exc)
            save_json(record, run_dir / "metrics.json")
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc), "finished_at": _now_iso()})
        save_json(manifest, manifest_path)
        raise

    stop_request = (
        json.loads(stop_request_path.read_text(encoding="utf-8-sig"))
        if stop_request_path.is_file()
        else None
    )
    manifest.update({"status": "ok", "finished_at": _now_iso(), "save_dir": str(save_dir)})
    if stop_request:
        manifest.update(
            {
                "termination": "manual_early_stop",
                "stop_request": stop_request,
                "training_closed": True,
                "resume_supported": False,
            }
        )
    save_json(manifest, manifest_path)
    return {"run_dir": str(run_dir), "metrics": record, "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epochs（smoke 用）")
    parser.add_argument("--run-name", default=None, help="覆盖输出目录名")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-resume", action="store_true", help="即使存在 last.pt 也从头训练")
    parser.add_argument("--force", action="store_true", help="允许覆盖已完成运行或已有权重")
    parser.add_argument(
        "--initial-checkpoint", type=Path, default=None,
        help="从已完成实验的 best.pt/last.pt 开始新的微调阶段（不恢复优化器）",
    )
    args = parser.parse_args()
    result = run_primary(
        config=args.config,
        epochs=args.epochs,
        run_name=args.run_name,
        device=args.device,
        resume=not args.no_resume,
        force=args.force,
        initial_checkpoint=args.initial_checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
