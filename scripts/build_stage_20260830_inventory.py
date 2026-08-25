"""阶段收尾（2026.08.30）正式实验资产清单与三基线指标汇总。

只读汇聚本周冻结的阶段证据，生成 `results/stage_20260830/` 下两份 CSV：

- ``02_formal_experiment_inventory.csv``
  三项正式实验（15 类 seg e100 / 6 类 seg e50 / 15 类 OBB e50）的五件套、
  OBB 阶段交付包（159 项资产清单与逐文件 SHA-256 的引用）、v3 实验交接包、
  数据门禁缓存证据、官方权重来源、90 张预览审核与转换报告的映射清单。
  大权重、checkpoint 与大体积预测只在清单中按受控路径 + SHA-256 引用，
  不复制进入小型阶段包。

- ``03_three_baseline_metrics.csv``
  三基线正式 val 指标长表（逐实验、逐指标、含来源文件 SHA-256），
  并固定口径声明：OBB e50（50 轮）与 seg e100（100 轮）训练预算不同，
  只比较共有 Box 指标、不比较 Mask mAP、不做排名；正式 test 全程锁定。

本脚本不读取正式 test 指标、不启动任何训练，只读取已冻结的
metrics/environment/manifest/交付包文件并现场重算 SHA-256。

用法：
    python scripts/build_stage_20260830_inventory.py
    python scripts/build_stage_20260830_inventory.py --output results/stage_20260830
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from blade_defect.experiment.metadata import load_json, nested_get, sha256_file
from blade_defect.utils.paths import resolve_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "experiment_id": "v3_yolo11s_seg_960_e100",
        "task": "segment",
        "label_level": "fine",
        "dataset_id": "blade-v3-grouped-202608",
        "role": "15类seg e100 mask基线",
        "five_piece": [
            "metrics.json",
            "per_class_metrics.csv",
            "environment.json",
            "run_manifest.json",
            "results.csv",
        ],
        "large_artifacts": [
            "weights/best.pt",
            "weights/last.pt",
            "validation_predictions.json",
        ],
    },
    {
        "experiment_id": "v3_hier_coarse_yolo11s_seg_960_e50",
        "task": "segment",
        "label_level": "coarse",
        "dataset_id": "blade-v3-grouped-202608-6class",
        "role": "6类seg e50 层级诊断",
        "five_piece": [
            "metrics.json",
            "per_class_metrics.csv",
            "environment.json",
            "run_manifest.json",
            "results.csv",
        ],
        "large_artifacts": [
            "weights/best.pt",
            "weights/last.pt",
            "validation_predictions.json",
        ],
    },
    {
        "experiment_id": "v3_yolo11s_obb_960_e50",
        "task": "obb",
        "label_level": "fine",
        "dataset_id": "blade-v3-grouped-obb",
        "role": "15类OBB e50 下一阶段暂时检测主线",
        "five_piece": [
            "metrics.json",
            "per_class_metrics.csv",
            "environment.json",
            "run_manifest.json",
            "results.csv",
        ],
        "large_artifacts": [
            "weights/best.pt",
            "weights/last.pt",
        ],
    },
]

# (section, item, 相对项目根路径, 分组, 备注)
REFERENCE_FILES: list[tuple[str, str, str, str, str]] = [
    (
        "obb_stage_handoff_package",
        "obb_159_asset_inventory",
        "results/obb_stage_handoff_202608/artifact_inventory.csv",
        "OBB阶段交付包",
        "4组159文件清单（dataset_descriptors 5 / obb_results 104 / official_weight 1 / run_e50 49），大权重与预测仅引用",
    ),
    (
        "obb_stage_handoff_package",
        "obb_159_asset_per_file_sha256",
        "results/obb_stage_handoff_202608/artifact_sha256.csv",
        "OBB阶段交付包",
        "159条逐文件SHA-256，阶段包只引用本清单不复制资产",
    ),
    (
        "obb_stage_handoff_package",
        "obb_handoff_validation_14_checks",
        "results/obb_stage_handoff_202608/handoff_validation.json",
        "OBB阶段交付包",
        "14项检查valid=true（五件套/test锁定/身份/环境/15类逐类/指纹/硬链接/派生口径/权重来源/转换零失败/90预览/对比口径/基线一致）",
    ),
    (
        "obb_stage_handoff_package",
        "obb_stage_metrics",
        "results/obb_stage_handoff_202608/obb_stage_metrics.csv",
        "OBB阶段交付包",
        "36行：6项总体共有Box + 15类×AP50/AP50-95（OBB vs seg）",
    ),
    (
        "obb_stage_handoff_package",
        "obb_handoff_notes",
        "results/obb_stage_handoff_202608/handoff_notes.md",
        "OBB阶段交付包",
        "交付说明与口径声明",
    ),
    (
        "v3_experiment_handoff_package",
        "v3_experiment_inventory",
        "results/v3_experiment_handoff/experiment_inventory.csv",
        "v3实验交接包",
        "15类seg e100与6类seg e50正式实验清单",
    ),
    (
        "v3_experiment_handoff_package",
        "v3_experiment_per_file_sha256",
        "results/v3_experiment_handoff/artifact_sha256.csv",
        "v3实验交接包",
        "v3两正式实验逐文件SHA-256",
    ),
    (
        "v3_experiment_handoff_package",
        "v3_experiment_handoff_validation",
        "results/v3_experiment_handoff/handoff_validation.json",
        "v3实验交接包",
        "v3交接校验报告",
    ),
    (
        "obb_analysis",
        "obb_seg_common_box_metrics",
        "results/obb_v3/obb_seg_common_metrics.csv",
        "共同Box对比",
        "OBB e50 vs seg e100 共有Box 5项（P/R/mAP50/mAP50-95/FPS）",
    ),
    (
        "obb_analysis",
        "obb_seg_per_class_ap",
        "results/obb_v3/obb_seg_per_class_ap.csv",
        "共同Box对比",
        "15类逐类AP50/AP50-95对照",
    ),
    (
        "obb_analysis",
        "obb_seg_comparison_report",
        "results/obb_v3/obb_seg_comparison.json",
        "共同Box对比",
        "对比口径与声明（排除Mask mAP、派生口径）",
    ),
    (
        "obb_analysis",
        "obb_conversion_summary",
        "results/obb_v3/conversion_summary.json",
        "派生数据转换",
        "seg polygon→最小外接矩形转换报告：49,471实例零失败",
    ),
    (
        "obb_analysis",
        "obb_preview_review_90",
        "results/obb_v3/preview_review.csv",
        "派生数据转换",
        "90张并排预览审核（train/val/test各30，覆盖15类，仅转换预览）",
    ),
    (
        "obb_analysis",
        "official_weight_provenance",
        "results/obb_v3/weight_provenance.json",
        "权重来源",
        "官方yolo11s-obb.pt来源URL与SHA-256核验",
    ),
    (
        "obb_analysis",
        "obb_dataset_gate_baseline",
        "results/obb_v3/validation_report.json",
        "数据门禁",
        "OBB派生数据全量校验基线（valid=true）",
    ),
    (
        "data_gate_cache",
        "cache_policy",
        "results/data_gate_cache/cache_policy.md",
        "数据门禁缓存",
        "缓存五条规则备案",
    ),
    (
        "data_gate_cache",
        "cold_run_report",
        "results/data_gate_cache/cold_run_report.json",
        "数据门禁缓存",
        "冷启动真实运行：48,291张全解码3,342.3s",
    ),
    (
        "data_gate_cache",
        "cache_hit_report",
        "results/data_gate_cache/cache_hit_report.json",
        "数据门禁缓存",
        "缓存命中真实运行：解码0复用48,291共35.7s",
    ),
    (
        "data_gate_cache",
        "sampled_smoke_report",
        "results/data_gate_cache/sampled_smoke_report.json",
        "数据门禁缓存",
        "CI固定种子抽样：512张65.3s，不读写正式缓存",
    ),
    (
        "data_gate_cache",
        "cache_invalidation_report",
        "results/data_gate_cache/cache_invalidation_report.json",
        "数据门禁缓存",
        "命中/失效/复检四场景结论",
    ),
]

DATASET_DESCRIPTORS: list[tuple[str, str, str]] = [
    ("blade-v3-grouped-202608", "正式v3 15类", "datasets/blade-v3-grouped-202608"),
    ("blade-v3-grouped-202608-6class", "6类层级诊断", "datasets/blade-v3-grouped-202608-6class"),
    ("blade-v3-grouped-obb", "OBB派生（polygon最小外接矩形）", "datasets/blade-v3-grouped-obb"),
]

DATASET_FILES = ["data.yaml", "dataset_manifest.json", "train.txt", "val.txt", "test.txt"]

METRIC_ROWS: list[tuple[str, str, str]] = [
    ("box_precision", "Box Precision", "box"),
    ("box_recall", "Box Recall", "box"),
    ("box_f1", "Box F1", "box"),
    ("box_mAP50", "Box mAP50", "box"),
    ("box_mAP50-95", "Box mAP50-95", "box"),
    ("fps", "FPS", "speed"),
    ("mask_precision", "Mask Precision", "mask"),
    ("mask_recall", "Mask Recall", "mask"),
    ("mask_f1", "Mask F1", "mask"),
    ("mask_mAP50", "Mask mAP50", "mask"),
    ("mask_mAP50-95", "Mask mAP50-95", "mask"),
]

COMPARISON_DECLARATIONS: list[str] = [
    "OBB e50（50轮）与15类seg e100（100轮）训练预算不同，仅比较共有Box指标，不做排名",
    "OBB为polygon最小外接矩形自动派生，不等同于人工旋转框真值",
    "不比较Mask mAP：Mask指标仅在15类seg e100与6类seg e50两项分割实验之间可比",
    "6类seg e50为层级诊断实验，类别体系不同，不与15类实验直接比较逐类指标",
    "三项实验均只使用正式val结果；正式test全程锁定（test_used=false），本周未读取test指标",
    "FPS为ultralytics val推理速度口径（fps_method=ultralytics_val_speed_inference_ms），非同口径测量不可跨实验外推",
]

REPRODUCTION_SCRIPTS: list[tuple[str, str, str]] = [
    ("scripts/cache_dataset_validation.py", "数据门禁缓存校验", "命中/冷启动/抽样三模式；命中只读校验不训练不评估"),
    ("scripts/archive_obb_stage_assets.py", "OBB阶段交付校验", "现场重算14项检查：指纹/硬链接/权重SHA/预览双向比对/对比口径"),
    ("scripts/build_stage_20260830_inventory.py", "阶段清单与三基线指标复算", "生成本阶段02/03/07三份CSV（本脚本）"),
    ("scripts/export_obb_val_predictions.py", "OBB val预测只读导出", "冻结best.pt只读推理正式val；不训练不读test"),
    ("scripts/build_stage_chart_data.py", "图表数据源复算", "从冻结预测与results.csv复算四份图表CSV"),
    ("scripts/generate_v3_charts.py", "正式图表生成", "渲染PR/F1/训练曲线/置信度可靠性四张PNG并写chart_manifest.json"),
    ("scripts/compare_obb_seg_common_metrics.py", "OBB-seg共有Box对照复算", "仅共有Box指标与速度，排除Mask mAP"),
    ("scripts/check_obb_dataset.py", "OBB派生数据全量校验", "标签/路径/split一致性基线校验"),
    ("scripts/audit_test_lock.py", "test锁审计", "校验test成员不可变与未参与模型选择"),
]

REPRODUCTION_EVIDENCE: list[tuple[str, str, str]] = [
    ("results/obb_stage_handoff_202608/handoff_validation.json", "OBB交付14项检查", "valid=true（2026-08-18验收，本周可现场重算复核）"),
    ("results/v3_experiment_handoff/handoff_validation.json", "v3实验交接校验", "seg e100与6类e50交接验收"),
    ("results/data_gate_cache/cold_run_report.json", "门禁冷启动证据", "48,291张全解码3,342.3s"),
    ("results/data_gate_cache/cache_hit_report.json", "门禁命中证据", "复用48,291共35.7s"),
    ("results/data_gate_cache/sampled_smoke_report.json", "CI抽样证据", "512张65.3s，不写正式缓存"),
    ("results/data_gate_cache/cache_hit_report_stage_20260830.json", "阶段收尾命中校验", "本周执行：hit/34.96s/复用48,291/valid=true"),
    ("results/stage_20260830/figures/chart_manifest.json", "图表来源清单", "四张PNG的CSV来源SHA-256/experiment_id/坐标轴/口径/生成环境"),
    ("results/stage_20260830/figures/data/chart_data_summary.json", "图表数据复核", "两实验GT=7,413与正式val口径一致；ECE与匹配规则"),
    ("results/obb_v3/weight_provenance.json", "官方权重来源", "yolo11s-obb.pt来源URL+SHA-256已核验"),
    ("docs/data_validation_gate.md", "门禁规则文档", "五条缓存规则成文（已入Git）"),
]

KEY_COMMITS: list[tuple[str, str]] = [
    ("0f0d030", "feat(obb): v3 50轮baseline链路与smoke加固"),
    ("8bf71f2", "fix(obb): 校准探针写清单前创建父目录并补回归测试"),
    ("89f227f", "feat(gate): OBB阶段交付归档与数据门禁解码缓存"),
    ("4555c08", "fix(gate): 抽样兜底与--no-cache写回缺陷修复、原子写缓存、共享校验助手"),
]


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _file_row(
    section: str,
    item: str,
    path: Path,
    group: str,
    note: str,
    *,
    reference_only: bool = False,
) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "section": section,
        "item": item,
        "group": group,
        "path": _rel(path) if path.is_absolute() and PROJECT_ROOT in path.parents else str(path),
        "exists": str(exists).lower(),
        "bytes": path.stat().st_size if exists else "",
        "sha256": sha256_file(path) if exists else "",
        "reference_only": str(reference_only).lower(),
        "note": note,
    }


def build_inventory_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        run_dir = PROJECT_ROOT / "runs" / experiment["experiment_id"]
        metrics = load_json(run_dir / "metrics.json")
        manifest = load_json(run_dir / "run_manifest.json")
        environment = load_json(run_dir / "environment.json")
        base_note = (
            f"role={experiment['role']}；dataset={experiment['dataset_id']}；"
            f"epochs={metrics.get('epochs')}；imgsz={metrics.get('imgsz')}；"
            f"seed={metrics.get('seed')}；status={metrics.get('status')}；"
            f"code_commit={metrics.get('code_commit') or nested_get(manifest, 'git.commit')}；"
            f"python={nested_get(environment, 'python.version')}；"
            f"torch={nested_get(environment, 'packages.torch')}；"
            f"ultralytics={nested_get(environment, 'packages.ultralytics')}；"
            f"gpu={(nested_get(environment, 'gpus') or [{}])[0].get('name') if nested_get(environment, 'gpus') else None}"
        )
        for filename in experiment["five_piece"]:
            rows.append(
                _file_row(
                    "formal_experiment_five_piece",
                    f"{experiment['experiment_id']}/{filename}",
                    run_dir / filename,
                    f"正式实验五件套/{experiment['experiment_id']}",
                    base_note,
                )
            )
        for relative in experiment["large_artifacts"]:
            rows.append(
                _file_row(
                    "formal_experiment_large_reference",
                    f"{experiment['experiment_id']}/{relative}",
                    run_dir / relative,
                    f"大型资产仅引用/{experiment['experiment_id']}",
                    "权重/checkpoint/大体积预测不复制进阶段包，仅以受控路径+SHA-256引用；" + base_note,
                    reference_only=True,
                )
            )
    for section, item, relative, group, note in REFERENCE_FILES:
        rows.append(_file_row(section, item, PROJECT_ROOT / relative, group, note))
    for dataset_id, label, relative_root in DATASET_DESCRIPTORS:
        for filename in DATASET_FILES:
            candidate = PROJECT_ROOT / relative_root / filename
            if not candidate.is_file():
                continue
            rows.append(
                _file_row(
                    "dataset_descriptor",
                    f"{dataset_id}/{filename}",
                    candidate,
                    f"数据集描述文件/{dataset_id}",
                    f"{label}；仅描述文件与split索引txt（6类为目录型布局，无txt索引），不含图片与标签本体",
                )
            )
    return rows


def build_baseline_metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        run_dir = PROJECT_ROOT / "runs" / experiment["experiment_id"]
        metrics_path = run_dir / "metrics.json"
        metrics = load_json(metrics_path)
        metrics_sha = sha256_file(metrics_path)
        for key, label, scope in METRIC_ROWS:
            value = metrics.get(key)
            if value is None:
                if scope == "mask" and experiment["task"] == "obb":
                    note = "OBB无Mask指标；按口径声明不与分割实验比较Mask mAP"
                else:
                    note = "该实验未记录此指标"
            elif scope == "mask":
                note = "Mask指标仅在两项seg实验之间可比"
            elif experiment["task"] == "obb":
                note = "与seg e100预算不同（50 vs 100轮），仅共有Box口径，不做排名"
            else:
                note = "正式val实测"
            rows.append(
                {
                    "section": "baseline_metric",
                    "experiment_id": experiment["experiment_id"],
                    "task": experiment["task"],
                    "label_level": experiment["label_level"],
                    "dataset_id": experiment["dataset_id"],
                    "epochs": metrics.get("epochs"),
                    "imgsz": metrics.get("imgsz"),
                    "metric": label,
                    "scope": scope,
                    "value": "" if value is None else f"{float(value):.6f}",
                    "source": _rel(metrics_path),
                    "source_sha256": metrics_sha,
                    "note": note,
                }
            )
    common_path = PROJECT_ROOT / "results/obb_v3/obb_seg_common_metrics.csv"
    common_sha = sha256_file(common_path)
    with common_path.open(encoding="utf-8-sig", newline="") as file:
        for record in csv.DictReader(file):
            rows.append(
                {
                    "section": "obb_vs_seg_common_box",
                    "experiment_id": "v3_yolo11s_obb_960_e50 vs v3_yolo11s_seg_960_e100",
                    "task": "obb vs segment",
                    "label_level": "fine",
                    "dataset_id": "blade-v3-grouped-obb / blade-v3-grouped-202608",
                    "epochs": "50 vs 100",
                    "imgsz": 960,
                    "metric": record["metric_label"],
                    "scope": "common_box",
                    "value": (
                        f"obb={float(record['obb']):.6f};seg={float(record['seg']):.6f};"
                        f"delta={float(record['delta_obb_minus_seg']):.6f}"
                    ),
                    "source": _rel(common_path),
                    "source_sha256": common_sha,
                    "note": "训练预算不同，仅共有Box口径，不做排名",
                }
            )
    for declaration in COMPARISON_DECLARATIONS:
        rows.append(
            {
                "section": "comparison_declaration",
                "experiment_id": "",
                "task": "",
                "label_level": "",
                "dataset_id": "",
                "epochs": "",
                "imgsz": "",
                "metric": "declaration",
                "scope": "",
                "value": "",
                "source": "",
                "source_sha256": "",
                "note": declaration,
            }
        )
    return rows


def build_reproducibility_rows() -> list[dict[str, Any]]:
    """复现清单：脚本、证据、关键提交、环境与数据指纹，全部可独立追溯。"""
    import subprocess

    rows: list[dict[str, Any]] = []
    for relative, item, note in REPRODUCTION_SCRIPTS:
        path = PROJECT_ROOT / relative
        rows.append(
            {
                "section": "reproduction_script",
                "item": item,
                "path_or_ref": relative,
                "exists": str(path.is_file()).lower(),
                "sha256_or_version": sha256_file(path) if path.is_file() else "",
                "note": note,
            }
        )
    for relative, item, note in REPRODUCTION_EVIDENCE:
        path = PROJECT_ROOT / relative
        rows.append(
            {
                "section": "reproduction_evidence",
                "item": item,
                "path_or_ref": relative,
                "exists": str(path.is_file()).lower(),
                "sha256_or_version": sha256_file(path) if path.is_file() else "",
                "note": note,
            }
        )
    for commit, subject in KEY_COMMITS:
        try:
            result = subprocess.run(
                ["git", "cat-file", "-t", commit],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            exists = result.stdout.strip() == "commit"
        except (OSError, subprocess.CalledProcessError):
            exists = False
        rows.append(
            {
                "section": "key_commit",
                "item": subject,
                "path_or_ref": commit,
                "exists": str(exists).lower(),
                "sha256_or_version": commit,
                "note": "git cat-file 可追溯" if exists else "仓库中不可追溯，需要负责人处理",
            }
        )
    cache_report = PROJECT_ROOT / "results/data_gate_cache/cache_hit_report_stage_20260830.json"
    if cache_report.is_file():
        report = load_json(cache_report)
        rows.append(
            {
                "section": "data_fingerprint",
                "item": "membership_sha256（48,291张成员指纹）",
                "path_or_ref": "results/data_gate_cache/cache_hit_report_stage_20260830.json",
                "exists": "true",
                "sha256_or_version": str(report.get("membership_sha256")),
                "note": "与dataset_manifest及历次门禁运行一致",
            }
        )
        rows.append(
            {
                "section": "data_fingerprint",
                "item": "file_state_sha256（逐文件size+mtime_ns聚合）",
                "path_or_ref": "results/data_gate_cache/cache_hit_report_stage_20260830.json",
                "exists": "true",
                "sha256_or_version": str(report.get("file_state_sha256")),
                "note": "缓存键组成部分；残留缺口见07b",
            }
        )
    environment = load_json(PROJECT_ROOT / "runs/v3_yolo11s_obb_960_e50/environment.json")
    rows.append(
        {
            "section": "formal_environment",
            "item": "正式实验环境（conda env bladedefect）",
            "path_or_ref": str(nested_get(environment, "python.executable")),
            "exists": "true",
            "sha256_or_version": (
                f"python={nested_get(environment, 'python.version')};"
                f"torch={nested_get(environment, 'packages.torch')};"
                f"ultralytics={nested_get(environment, 'packages.ultralytics')};"
                f"numpy={nested_get(environment, 'packages.numpy')};"
                f"opencv={nested_get(environment, 'packages.opencv_python')};"
                f"cuda={nested_get(environment, 'cuda.version')}"
            ),
            "note": "三项正式实验共用同一环境；matplotlib 3.10.9 同环境可用（图表生成）",
        }
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--output", default="results/stage_20260830", help="阶段成果包目录")
    args = parser.parse_args()
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = resolve_path(output_dir)

    inventory_rows = build_inventory_rows()
    inventory_path = _write_csv(
        output_dir / "02_formal_experiment_inventory.csv",
        inventory_rows,
        ["section", "item", "group", "path", "exists", "bytes", "sha256", "reference_only", "note"],
    )
    metric_rows = build_baseline_metric_rows()
    metrics_path = _write_csv(
        output_dir / "03_three_baseline_metrics.csv",
        metric_rows,
        [
            "section",
            "experiment_id",
            "task",
            "label_level",
            "dataset_id",
            "epochs",
            "imgsz",
            "metric",
            "scope",
            "value",
            "source",
            "source_sha256",
            "note",
        ],
    )

    missing = [row for row in inventory_rows if row["exists"] != "true"]
    repro_rows = build_reproducibility_rows()
    repro_path = _write_csv(
        output_dir / "07_reproducibility_inventory.csv",
        repro_rows,
        ["section", "item", "path_or_ref", "exists", "sha256_or_version", "note"],
    )
    repro_missing = [row for row in repro_rows if row["exists"] != "true"]
    summary = {
        "inventory_rows": len(inventory_rows),
        "inventory_missing_files": len(missing),
        "metric_rows": len(metric_rows),
        "reproducibility_rows": len(repro_rows),
        "reproducibility_missing": len(repro_missing),
        "outputs": [_rel(inventory_path), _rel(metrics_path), _rel(repro_path)],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing:
        for row in missing:
            print(f"MISSING: {row['path']}")
    if repro_missing:
        for row in repro_missing:
            print(f"MISSING_REPRO: {row['path_or_ref']}")


if __name__ == "__main__":
    main()
