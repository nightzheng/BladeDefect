"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blade_defect.data import check_dataset, clean_dataset, split_dataset
from blade_defect.evaluation import AblationRunner, metrics_from_ultralytics
from blade_defect.models import SegmentationPredictor, SegmentationTrainer
from blade_defect.utils import setup_logging
from blade_defect.utils.files import save_json


def _train(config: str) -> Any:
    trainer, kwargs = SegmentationTrainer.from_config(config)
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
    result = evaluator.validate(params["data"], imgsz=params.get("imgsz", 640))
    return metrics_from_ultralytics(result).to_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blade-defect")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-labels")
    check.add_argument("--images", required=True)
    check.add_argument("--labels", required=True)
    check.add_argument("--num-classes", type=int)
    check.add_argument("--output", default="runs/data_check.json")

    clean = subparsers.add_parser("clean")
    clean.add_argument("--images", required=True)
    clean.add_argument("--labels", required=True)
    clean.add_argument("--output", default="runs/data_cleaning.json")

    split = subparsers.add_parser("split")
    split.add_argument("--images", required=True)
    split.add_argument("--labels", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.2, 0.1))
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--move", action="store_true")

    train = subparsers.add_parser("train")
    train.add_argument("--config", default="configs/train.yaml")

    predict = subparsers.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--source", required=True)
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--device", default="0")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--output", default="runs/evaluation/metrics.json")
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--device", default="0")

    ablation = subparsers.add_parser("ablation")
    ablation.add_argument("--config", default="configs/ablation.yaml")
    ablation.add_argument("--output", default="runs/ablation_summary")
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    if args.command == "check-labels":
        report = check_dataset(args.images, args.labels, args.num_classes)
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
        print(counts)
    elif args.command == "train":
        _train(args.config)
    elif args.command == "predict":
        SegmentationPredictor(args.model).predict(
            args.source, conf=args.conf, imgsz=args.imgsz, device=args.device, save=True
        )
    elif args.command == "evaluate":
        trainer = SegmentationTrainer(args.model)
        raw_metrics = trainer.validate(args.data, imgsz=args.imgsz, device=args.device)
        metrics = metrics_from_ultralytics(raw_metrics)
        save_json(metrics.to_dict(), args.output)
        print(json.dumps(metrics.to_dict(), indent=2))
    elif args.command == "ablation":
        runner = AblationRunner(args.config, args.output)
        runner.run(_ablation_experiment)


if __name__ == "__main__":
    main()
