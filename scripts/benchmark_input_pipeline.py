"""Plan or run isolated input-pipeline comparisons. Default: plan only, no training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import statistics
from datetime import datetime

import yaml

from blade_defect.utils.files import load_project_config, resolved_data_yaml
from blade_defect.utils.paths import resolve_model_reference


def build_cases(base, workers, production_workers=None):
    if base.get("cache", False) not in (False, None):
        raise ValueError("Benchmark requires cache: false; RAM/disk cache is not created by this tool")
    if base.get("resume") or base.get("time") or base.get("fraction", 1.0) != 1.0:
        raise ValueError("Benchmark requires fresh full-dataset training without resume/time limit")
    if any(w < 0 for w in workers) or (production_workers is not None and production_workers < 0):
        raise ValueError("workers must be >= 0")
    fixed = {**base, "batch": 8, "imgsz": 960, "pipeline_profile": True}
    cases = [{"name": "A_current", "config": dict(fixed), "environment": {}}]
    for w in dict.fromkeys(workers):
        cases.append({"name": f"B_workers_{w}", "config": {**fixed, "workers": w}, "environment": {}})
        cases.append({"name": f"C_workers_{w}_threads1", "config": {**fixed, "workers": w},
                      "environment": {k: "1" for k in
                                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}})
    if production_workers is not None:
        cases.append({"name": f"D_batch12_workers_{production_workers}",
                      "config": {**fixed, "batch": 12, "workers": production_workers},
                      "environment": {}})
    return cases


def readonly_preflight(data):
    """Reject JPEGs that upstream would re-encode, and implicit .npy cache consumption."""
    from PIL import Image

    checked = set()
    with resolved_data_yaml(data) as normalized:
        payload = yaml.safe_load(Path(normalized).read_text(encoding="utf-8"))
        root = Path(payload["path"])
        for split in ("train", "val"):
            entries = payload[split]
            for entry in entries if isinstance(entries, list) else [entries]:
                source = Path(entry)
                if not source.is_absolute():
                    source = root / source
                if source.is_dir():
                    paths = (p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in
                             {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
                else:
                    paths = (Path(line.strip().replace("./", str(source.parent) + os.sep, 1))
                             if line.strip().startswith("./") else Path(line.strip())
                             for line in source.read_text(encoding="utf-8").splitlines() if line.strip())
                for path in paths:
                    if path in checked:
                        continue
                    checked.add(path)
                    if path.with_suffix(".npy").exists():
                        raise ValueError(f"Existing implicit image cache: {path.with_suffix('.npy')}; "
                                         "verify/remove separately before JPEG-path benchmark")
                    with Image.open(path) as image:
                        jpeg = image.format == "JPEG"
                    if jpeg:
                        with path.open("rb") as image_file:
                            image_file.seek(-2, 2)
                            if image_file.read() != b"\xff\xd9":
                                raise ValueError(f"JPEG would be rewritten by upstream verification: {path}")
    return len(checked)


def run_case(config_path, stop_epochs, threads):
    if threads:
        import torch
        torch.set_num_threads(threads)
    from blade_defect.models import SegmentationTrainer

    trainer, config = SegmentationTrainer.from_config(config_path)
    planned = int(config.get("epochs", 100))
    if stop_epochs >= planned:
        raise ValueError("Stop epoch count must be less than planned training epochs")

    def stop_after_prefix(state):
        if state.epoch + 1 >= stop_epochs:
            state.stop = True

    trainer.model.add_callback("on_fit_epoch_end", stop_after_prefix)
    trainer.train(**config)


def summarize(output):
    """Compare recorded epochs after warmup; no source images are read."""
    rows = []
    for path in sorted(output.glob("training/*/pipeline_performance.jsonl")):
        epochs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        # 5-epoch runs have epochs 4/5 after default 3-epoch warmup; shorter runs
        # are explicitly labelled as only partially warmed comparisons.
        selected = [row for row in epochs if row["epoch"] > 3]
        fully_warmed = bool(selected)
        if not selected:
            selected = [row for row in epochs if row["epoch"] > 1]
        if not selected:
            continue
        first = selected[0]
        rows.append({"case": path.parent.name, "measured_epochs": len(selected),
                     "after_default_warmup": fully_warmed,
                     "workers_effective": first["workers_effective"],
                     "batch_effective": first["batch_effective"], "imgsz": first["imgsz"],
                     "train_wall_median_s": statistics.median(r["train_wall_s"] for r in selected),
                     "epoch_wall_median_s": statistics.median(r["epoch_wall_s"] for r in selected),
                     "train_images_per_s_median": statistics.median(r["train_images_per_s"] for r in selected),
                     "fetch_gap_fraction": sum(r["fetch_gap_s"] for r in selected) /
                                           max(sum(r["train_wall_s"] for r in selected), 1e-9)})
    if not rows:
        raise ValueError("No completed performance records found")
    target = output / "summary.csv"
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--epochs", type=int, choices=[3, 4, 5], default=5,
                        help="Stop after this many epochs; preserve original LR/augmentation schedule")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", nargs="+", help="Select generated case names")
    parser.add_argument("--batch12-workers", type=int, help="Add independent D case after choosing best workers")
    parser.add_argument("--threads", type=int, choices=[1], help="Thread budget for all selected cases (e.g. D)")
    parser.add_argument("--execute", action="store_true", help="Run planned cases on full dataset")
    parser.add_argument("--summarize", type=Path, help="Summarize an existing benchmark directory")
    parser.add_argument("--run-case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize.resolve())
        return
    if args.run_case:
        run_case(args.run_case, args.epochs, args.threads)
        return
    base, project_root = load_project_config(args.config)
    base["model"] = resolve_model_reference(base.get("model", "yolo11n-seg.pt"), project_root)
    base["device"] = args.device
    if int(base.get("epochs", 100)) <= args.epochs:
        parser.error("Original config epochs must exceed benchmark stop epochs")
    cases = build_cases(base, args.workers, args.batch12_workers)
    if args.only:
        unknown = set(args.only) - {case["name"] for case in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in args.only]
    if args.threads:
        for case in cases:
            case["environment"] = {k: "1" for k in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    output = (args.output or Path("runs/pipeline_benchmark") / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    # Refuse reuse to keep output/config provenance unambiguous.
    output.mkdir(parents=True, exist_ok=False)
    for case in cases:
        case["config"].update(project=str(output / "training"), name=case["name"], exist_ok=False)
        case["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in case["config"].items()}
        (output / f"{case['name']}.yaml").write_text(
            yaml.safe_dump(case["config"], allow_unicode=True, sort_keys=False), encoding="utf-8")
    plan = {"source_config": str(args.config.resolve()),
            "source_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "stop_after_epochs": args.epochs, "cases": cases}
    (output / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Plan: {output / 'plan.json'}", flush=True)
    if not args.execute:
        print("Plan only. No images decoded or cached; add --execute on the server.")
        return
    print(f"Read-only preflight: {readonly_preflight(base['data'])} images", flush=True)
    for case in cases:
        command = [sys.executable, str(Path(__file__).resolve()), "--run-case",
                   str(output / f"{case['name']}.yaml"), "--epochs", str(args.epochs)]
        if case["environment"]:
            command += ["--threads", "1"]
        with (output / f"{case['name']}.log").open("w", encoding="utf-8") as log:
            print(f"Running {case['name']}; log: {log.name}", flush=True)
            subprocess.run(command, env={**os.environ, **case["environment"]},
                           stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
