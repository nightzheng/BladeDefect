"""全量数据吞吐校准：短周期训练并输出耗时/显存/总时长估算。

在全量冻结数据上以正式训练相同的模型、输入尺寸、batch 与 workers 运行
少量 epochs，记录每 epoch 耗时与峰值显存，据此估算 50 epochs 主 baseline
的总时长。校准结果写入 results/training_calibration/。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from blade_defect.data.validation import validate_dataset_gate
from blade_defect.models import SegmentationTrainer
from blade_defect.utils.files import save_json
from blade_defect.utils.paths import resolve_path


def run_calibration(
    data: Path,
    model: str,
    imgsz: int,
    batch: int,
    workers: int,
    epochs: int,
    device: str,
    output_dir: Path,
    runs_dir: Path,
) -> dict:
    gate_report = validate_dataset_gate(data)
    train_images = gate_report.splits["train"].images

    output_dir = resolve_path(output_dir)
    runs_dir = resolve_path(runs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"throughput_calibration_{Path(model).stem}_{imgsz}"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    trainer = SegmentationTrainer(model)
    result = trainer.train(
        normalize_data_yaml=False,
        data=str(data),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        seed=42,
        device=device,
        project=str(runs_dir),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        cache=False,
        amp=True,
        val=True,
        save=True,
        verbose=True,
    )
    wall_time_s = time.perf_counter() - started
    save_dir = Path(getattr(result, "save_dir", runs_dir / run_name))

    results_csv = save_dir / "results.csv"
    epoch_rows: list[dict] = []
    if results_csv.is_file():
        with results_csv.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
            cumulative = [(int(row["epoch"]), float(row["time"])) for row in reader]
        previous = 0.0
        for epoch, elapsed in cumulative:
            # Ultralytics results.csv 的 time 列是训练开始的累计秒数，需差分得到单 epoch 耗时。
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "epoch_time_s": round(elapsed - previous, 1),
                    "train_images": train_images,
                    "batch": batch,
                    "imgsz": imgsz,
                }
            )
            previous = elapsed
    epoch_times = [row["epoch_time_s"] for row in epoch_rows]
    first_epoch_s = epoch_times[0] if epoch_times else None
    steady_epoch_s = sum(epoch_times[1:]) / len(epoch_times[1:]) if len(epoch_times) > 1 else first_epoch_s

    throughput_csv = output_dir / "throughput_calibration.csv"
    with throughput_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=["epoch", "epoch_time_s", "train_images", "batch", "imgsz"]
        )
        writer.writeheader()
        writer.writerows(epoch_rows)

    gpu_report = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
        if torch.cuda.is_available()
        else None,
        "peak_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024),
        "peak_memory_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 / 1024),
        "batch": batch,
        "imgsz": imgsz,
        "workers": workers,
        "amp": True,
        "cache": False,
        "torch_version": torch.__version__,
    }
    save_json(gpu_report, output_dir / "gpu_memory_report.json")

    target_epochs = 50
    estimated_total_h = None
    if steady_epoch_s:
        estimated_total_h = round(steady_epoch_s * target_epochs / 3600, 1)

    estimate_md = output_dir / "estimated_training_time.md"
    estimate_md.write_text(
        "\n".join(
            [
                "# 全量训练耗时估算",
                "",
                f"- 校准时间：{started_at}",
                f"- 数据集：{gate_report.dataset_id}（train={train_images} 张，batch={batch}，每 epoch {train_images // batch + 1} 次迭代）",
                f"- 校准配置：{model} @ imgsz={imgsz}，epochs={epochs}，workers={workers}，amp=True，cache=False",
                f"- 首 epoch 耗时：{first_epoch_s} s（含 warmup/CUDA 初始化）",
                f"- 稳态 epoch 耗时：{steady_epoch_s and round(steady_epoch_s, 1)} s（约 {steady_epoch_s and round(steady_epoch_s / 60, 1)} 分钟）",
                f"- 校准总墙钟时间：{round(wall_time_s, 1)} s（含每 epoch 末验证）",
                f"- 峰值显存：allocated {gpu_report['peak_memory_allocated_mb']} MB / reserved {gpu_report['peak_memory_reserved_mb']} MB（总 {gpu_report['gpu_memory_total_mb']} MB）",
                "",
                "## 50 epochs 主 baseline 估算",
                "",
                f"- 预计总训练时长 ≈ {estimated_total_h} 小时（稳态 epoch × 50，含 epoch 末验证，不含最终评估与预测导出）",
                "- 数据加载瓶颈观察：见下方说明。",
                "",
                "估算仅基于本机 RTX 4060 Laptop（8GB）实测，换机器需重新校准。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "run_dir": str(save_dir),
        "started_at": started_at,
        "epochs_completed": len(epoch_rows),
        "epoch_times_s": epoch_times,
        "steady_epoch_s": steady_epoch_s,
        "estimated_50epoch_hours": estimated_total_h,
        "gpu": gpu_report,
        "throughput_csv": str(throughput_csv),
        "estimate_md": str(estimate_md),
    }
    save_json(summary, output_dir / "calibration_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/data.yaml", type=Path)
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", default=Path("results/training_calibration"), type=Path)
    parser.add_argument("--runs-dir", default=Path("runs/calibration"), type=Path)
    args = parser.parse_args()
    summary = run_calibration(**vars(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
