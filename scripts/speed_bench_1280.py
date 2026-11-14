#!/usr/bin/env python
"""事项 3a：1280 部署测速（FP16 / TensorRT / 按图分档），产出 Go/No-Go 依据。

测量口径与 Round 2 一致：ultralytics val 的 inference 毫秒数取倒数（1000/ms），
在同一个 val split 上跑完整 val，取平均。

产出 JSON：
  measurements: 各配置的 FPS / inference_ms
  tiered_policy: 若只有 f 比例的图走 1280、其余走 960，等效 FPS = 1/(f/FPS1280+(1-f)/FPS960)
  go_no_go: 阈值 150 FPS 的判定

用法：
  python scripts/speed_bench_1280.py --data-yaml <val yaml> --device 1 \
    --out /data/backup/round3_logs/speed_3a.json
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

THRESHOLD_FPS = 150.0


def _normalized_data_yaml(source: str | Path) -> str:
    """把 data.yaml 的 path 绝对化后写到 /tmp，避免从别的 cwd 调用时解析错。

    原始 data.yaml 的 path 可能是相对路径；ultralytics 会相对 cwd 解析，
    在 /root/BladeDefect 下会找成 /root/BladeDefect/val.txt（2026-10-23 踩过）。
    """
    import yaml

    src = Path(source)
    config = yaml.safe_load(src.read_text(encoding="utf-8"))
    root = config.get("path", ".")
    base = Path(root) if Path(root).is_absolute() else (src.parent / root)
    config["path"] = str(base.resolve())
    target = Path("/tmp/b1_speed_val_normalized.yaml")
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(target)


def _measure(model_path: str, data_yaml: str, imgsz: int, half: bool, device: str, batch: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    result = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        half=half,
        device=device,
        split="val",
        plots=False,
        save_json=False,
        verbose=False,
    )
    inference_ms = float(result.speed.get("inference", 0.0))
    fps = 1000.0 / inference_ms if inference_ms else 0.0
    return {
        "weights": model_path,
        "imgsz": imgsz,
        "half": half,
        "inference_ms": round(inference_ms, 3),
        "fps": round(fps, 2),
        "mask_mAP50-95": round(float(result.seg.map), 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="/data/runs/v3_seg_a1_res1280_ft_e25/weights/best.pt")
    parser.add_argument("--weights-960", default="/data/runs/v3_seg_a0_res960_ft_e25/weights/best.pt")
    parser.add_argument("--data-yaml", default="/data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml")
    parser.add_argument("--device", default="1")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", default="/data/backup/round3_logs/speed_3a.json")
    parser.add_argument("--skip-engine", action="store_true", help="不尝试 TensorRT 导出")
    args = parser.parse_args()

    report: dict = {"threshold_fps": THRESHOLD_FPS, "measurements": {}, "errors": {}, "tiered_policy": {}}
    # 复用项目里已被验证过的归一化助手（val/train 条目会解析成绝对路径）：
    # 直接用原始 data.yaml 会因相对路径基准不同而报 "images not found"。
    from contextlib import ExitStack

    from blade_defect.utils.files import resolved_data_yaml

    stack = ExitStack()
    normalized_yaml = stack.enter_context(resolved_data_yaml(Path(args.data_yaml)))

    plan = [
        ("a1_1280_fp32", args.weights, 1280, False),
        ("a1_1280_fp16", args.weights, 1280, True),
        ("a0_960_fp32", args.weights_960, 960, False),
        ("a0_960_fp16", args.weights_960, 960, True),
    ]
    for name, weights, imgsz, half in plan:
        try:
            print(f"[bench] {name} ...", flush=True)
            report["measurements"][name] = _measure(weights, normalized_yaml, imgsz, half, args.device, args.batch)
            print(f"[bench] {name} -> {report['measurements'][name]}", flush=True)
        except Exception as exc:
            report["errors"][name] = f"{type(exc).__name__}: {exc}"
            print(f"[bench] {name} FAILED: {exc}", flush=True)
            traceback.print_exc()

    # TensorRT（装不上就跳过，不阻塞结论）
    if not args.skip_engine:
        try:
            from ultralytics import YOLO

            print("[bench] 尝试导出 TensorRT engine（1280 FP16）...", flush=True)
            engine_path = YOLO(args.weights).export(format="engine", imgsz=1280, half=True, device=args.device)
            report["measurements"]["a1_1280_trt"] = _measure(
                str(engine_path), normalized_yaml, 1280, True, args.device, args.batch
            )
            print(f"[bench] trt -> {report['measurements']['a1_1280_trt']}", flush=True)
        except Exception as exc:
            report["errors"]["trt"] = f"{type(exc).__name__}: {exc}"
            print(f"[bench] TensorRT 不可用，降级为 ONNX Runtime/FP16 口径说明: {exc}", flush=True)

    fps_1280 = report["measurements"].get("a1_1280_fp32", {}).get("fps")
    fps_960 = report["measurements"].get("a0_960_fp32", {}).get("fps")
    fps_1280_fp16 = report["measurements"].get("a1_1280_fp16", {}).get("fps")
    if fps_1280 and fps_960:
        for fraction in (0.1, 0.2, 0.3, 0.5, 1.0):
            for tag, hi in (("fp32", fps_1280), ("fp16", fps_1280_fp16 or fps_1280)):
                equivalent = 1.0 / (fraction / hi + (1 - fraction) / fps_960)
                report["tiered_policy"][f"{tag}_1280_share_{fraction}"] = round(equivalent, 2)

    best = max(
        [(name, item.get("fps", 0.0)) for name, item in report["measurements"].items()],
        key=lambda pair: pair[1],
        default=("none", 0.0),
    )
    report["best_config"] = {"name": best[0], "fps": best[1]}
    report["go_no_go"] = {
        "threshold": THRESHOLD_FPS,
        "best_fps": best[1],
        "verdict": "GO（进入 3b）" if best[1] >= THRESHOLD_FPS else "NO-GO（关闭 1280 生产路线，不提 3b）",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    stack.close()
    print(json.dumps({k: report[k] for k in ("measurements", "best_config", "go_no_go")}, ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
