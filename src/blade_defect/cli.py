"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blade_defect.data import check_dataset, clean_dataset, split_dataset
from blade_defect.evaluation import AblationRunner, metrics_from_ultralytics
from blade_defect.models import SegmentationPredictor, SegmentationTrainer
from blade_defect.utils import resolve_model_reference, resolve_path, setup_logging
from blade_defect.utils.files import save_json


def _train(config: str | Path, device: str | None = None) -> Any:
    trainer, kwargs = SegmentationTrainer.from_config(config)
    if device is not None:
        kwargs["device"] = device
    return trainer.train(**kwargs)


def _ablation_experiment(name: str, params: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)
    model_source = params.pop("model", "yolo11n-seg.pt")
    trainer = SegmentationTrainer(model_source)
    params["name"] = name
    train_result = trainer.train(**params)
    save_dir = getattr(train_result, "save_dir", None)
    best = Path(save_dir) / "weights" / "best.pt" if save_dir else None
    evaluator = SegmentationTrainer(str(best) if best and best.exists() else model_source)
    result = evaluator.validate(
        params["data"],
        imgsz=params.get("imgsz", 640),
        device=params.get("device", "auto"),
    )
    return metrics_from_ultralytics(result).to_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blade-defect")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-labels")
    check.add_argument("--images", default="datasets/raw/images", type=resolve_path)
    check.add_argument("--labels", default="datasets/raw/labels", type=resolve_path)
    check.add_argument("--num-classes", type=int)
    check.add_argument("--output", default="runs/data_check.json", type=resolve_path)
    check.add_argument("--dry-run", action="store_true", help="仅输出检查结果，不写入 JSON 报告")
    check.add_argument(
        "--fix-float",
        action="store_true",
        help="将 [-0.01, 1.01] 内的轻微坐标越界 clip 到 [0, 1]",
    )

    clean = subparsers.add_parser("clean")
    clean.add_argument("--images", required=True, type=resolve_path)
    clean.add_argument("--labels", required=True, type=resolve_path)
    clean.add_argument("--output", default="runs/data_cleaning.json", type=resolve_path)

    split = subparsers.add_parser("split")
    split.add_argument("--images", required=True, type=resolve_path)
    split.add_argument("--labels", required=True, type=resolve_path)
    split.add_argument("--output", required=True, type=resolve_path)
    split.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.2, 0.1))
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--move", action="store_true")

    train = subparsers.add_parser("train")
    train.add_argument("--config", default="configs/train.yaml", type=resolve_path)
    train.add_argument("--device", choices=("auto", "0", "cpu"), help="覆盖配置中的计算设备")

    predict = subparsers.add_parser("predict")
    predict.add_argument("--model", "--weights", dest="model", required=True, type=resolve_model_reference)
    predict.add_argument("--source", required=True, type=resolve_path)
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--device", default="auto", choices=("auto", "0", "cpu"))

    evaluate = subparsers.add_parser("evaluate", aliases=["eval"])
    evaluate.add_argument("--model", required=True, type=resolve_model_reference)
    evaluate.add_argument("--data", required=True, type=resolve_path)
    evaluate.add_argument("--output", default="runs/evaluation/metrics.json", type=resolve_path)
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--device", default="auto", choices=("auto", "0", "cpu"))

    ablation = subparsers.add_parser("ablation")
    ablation.add_argument("--config", default="configs/ablation.yaml", type=resolve_path)
    ablation.add_argument("--output", default="runs/ablation_summary", type=resolve_path)
    ablation.add_argument("--device", choices=("auto", "0", "cpu"), help="覆盖配置中的计算设备")
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    if args.command == "check-labels":
        report = check_dataset(
            args.images,
            args.labels,
            args.num_classes,
            fix_float=args.fix_float,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            save_json(report.to_dict(), args.output)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "clean":
        report = clean_dataset(args.images, args.labels)
        payload = {
            "scanned": report.scanned,
            "corrupt_images": report.corrupt_images,
            "duplicate_images": report.duplicate_images,
            "empty_labels": report.empty_labels,
        }
        save_json(payload, args.output)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "split":
        counts = split_dataset(
            args.images, args.labels, args.output, tuple(args.ratios), args.seed, copy=not args.move
        )
        print(json.dumps(counts, ensure_ascii=False, indent=2))
    elif args.command == "train":
        _train(args.config, args.device)
    elif args.command == "predict":
        SegmentationPredictor(args.model).predict(
            args.source, conf=args.conf, imgsz=args.imgsz, device=args.device, save=True
        )
    elif args.command in {"evaluate", "eval"}:
        trainer = SegmentationTrainer(args.model)
        raw_metrics = trainer.validate(args.data, imgsz=args.imgsz, device=args.device)
        metrics = metrics_from_ultralytics(raw_metrics)
        save_json(metrics.to_dict(), args.output)
        print(json.dumps(metrics.to_dict(), indent=2))
    elif args.command == "ablation":
        runner = AblationRunner(args.config, args.output, device=args.device)
        runner.run(_ablation_experiment)


if __name__ == "__main__":
    main()
