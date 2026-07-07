"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blade_defect.data import check_dataset, clean_dataset, split_dataset
from blade_defect.evaluation import AblationRunner, metrics_from_ultralytics
from blade_defect.experiment import analyze_experiments, export_summary, run_all_experiments
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
    repair_mode = check.add_mutually_exclusive_group()
    repair_mode.add_argument(
        "--polygon-mode",
        choices=("strict", "soft", "auto-fix"),
        default="auto-fix",
        help="polygon 越界处理模式（默认：auto-fix）",
    )
    repair_mode.add_argument(
        "--fix-float",
        dest="polygon_mode",
        action="store_const",
        const="auto-fix",
        help="兼容旧参数；等同于 --polygon-mode auto-fix",
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
    experiment = subparsers.add_parser("experiment", help="运行、汇总和分析 baseline 实验")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    run_all = experiment_commands.add_parser("run-all", help="顺序运行全部注册实验")
    run_all.add_argument("--config", default="configs/train.yaml", type=resolve_path)
    run_all.add_argument("--runs-dir", default="runs", type=resolve_path)
    run_all.add_argument("--output", default="results/summary.csv", type=resolve_path)
    run_all.add_argument("--device", default="auto", choices=("auto", "0", "cpu"))
    run_all.add_argument(
        "--imgsz",
        type=int,
        choices=(640, 960, 1024, 1280),
        help="只运行指定输入尺寸的四个模型；默认运行全部尺寸",
    )
    run_all.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        metavar="NAME_OR_ID",
        help="按完整名称或 ID 选择实验；可重复传入，例如 exp014 或 16",
    )
    run_all.add_argument(
        "--skip-validation",
        action="store_true",
        help="显式跳过训练前 strict dataset validation gate",
    )
    summary = experiment_commands.add_parser("summary", help="重新生成实验汇总 CSV")
    summary.add_argument("--runs-dir", default="runs", type=resolve_path)
    summary.add_argument("--output", default="results/summary.csv", type=resolve_path)
    analyze = experiment_commands.add_parser("analyze", help="生成论文级统计图表")
    analyze.add_argument("--summary", default="results/summary.csv", type=resolve_path)
    analyze.add_argument("--runs-dir", default="runs", type=resolve_path)
    analyze.add_argument("--output-dir", default="results/analysis", type=resolve_path)
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    if args.command == "check-labels":
        report = check_dataset(
            args.images,
            args.labels,
            args.num_classes,
            polygon_mode=args.polygon_mode,
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
    elif args.command == "experiment":
        if args.experiment_command == "run-all":
            records = run_all_experiments(config=args.config, runs_dir=args.runs_dir,
                                          results_file=args.output, device=args.device,
                                          skip_validation=args.skip_validation, imgsz=args.imgsz,
                                          experiment_selectors=args.experiments)
            print(json.dumps(records, ensure_ascii=False, indent=2))
        elif args.experiment_command == "summary":
            print(export_summary(args.runs_dir, args.output))
        elif args.experiment_command == "analyze":
            outputs = analyze_experiments(args.summary, args.runs_dir, args.output_dir)
            print(json.dumps([str(path) for path in outputs], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
