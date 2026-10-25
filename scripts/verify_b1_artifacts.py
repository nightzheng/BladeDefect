#!/usr/bin/env python
"""B-1 开训前四样交付物的自动对拍（把复核人的验收条件变成可执行检查）。

对应复核人 2026-10-23 的四条要求：
  1. copy_paste_manifest.json  —— 写死字段 + 345 张 + 跳过原因 + 抽查源/目标图
  2. synthetic_audit.json      —— 几何类 flag 必须为 0；CONFUSABLE 只记录不卡
  3. pytest 重跑记录           —— 13 项全过（本轮新记录）
  4. A0 配置 diff              —— 除 copy-paste 接入方式外逐项一致

用法（开机后、开训前）：

  python scripts/verify_b1_artifacts.py \
    --manifest /data/datasets/derived/blade-v3-grouped-202608-b1cp13/copy_paste_manifest.json \
    --synthetic-audit /data/backup/round3/synthetic_audit.json \
    --pytest-log /data/backup/round3/pytest_b1.log \
    --a0-args /data/runs/v3_seg_a0_res960_ft_e25/args.yaml \
    --b1-config /data/datasets/derived/blade-v3-grouped-202608-b1cp13/data.yaml \
    --original-train-index /data/datasets/v3-index-rebuild/blade-v3-grouped-202608/train.txt \
    --out-md /data/backup/round3/b1_precheck_report.md

退出码：0 = 全部硬性检查通过；1 = 有 FAIL（报告里逐条列明）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GEO_FLAGS = ("ZERO_AREA", "OUT_OF_BOUNDS", "DUPLICATE", "TINY", "THIN")

FROZEN_MANIFEST = {
    "target_classes": [13],
    "seed": 42,
    "copies_per_instance": 3,
    "rotation": "none",
    "flip": "none",
    "pastes_per_synthetic_image": 1,
}

EXPECTED_SYNTHETIC = 345

ARGS_KEYS = (
    "imgsz", "batch", "epochs", "patience", "optimizer", "lr0", "lrf", "cos_lr",
    "warmup_epochs", "weight_decay", "mosaic", "close_mosaic", "seed", "workers", "copy_paste",
)


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, section: str, status: str, detail: str) -> None:
        self.rows.append((section, status, detail))

    @property
    def failed(self) -> list[tuple[str, str, str]]:
        return [row for row in self.rows if row[1] == "FAIL"]

    def to_markdown(self) -> str:
        lines = [
            "# B-1 开训前自检报告",
            "",
            f"- 硬性检查失败数：**{len(self.failed)}**",
            f"- 结果：{'❌ 不可开训' if self.failed else '✅ 可以开训（仍需复核人确认）'}",
            "",
            "| 段 | 结论 | 明细 |",
            "|---|---|---|",
        ]
        for section, status, detail in self.rows:
            lines.append(f"| {section} | {status} | {detail} |")
        return "\n".join(lines) + "\n"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_manifest(report: Report, path: Path, original_train_index: Path) -> None:
    if not path.is_file():
        report.add("1 manifest", "FAIL", f"文件不存在：{path}")
        return
    manifest = _load_json(path)

    for key, expected in FROZEN_MANIFEST.items():
        actual = manifest.get(key)
        status = "PASS" if actual == expected else "FAIL"
        report.add("1 manifest", status, f"{key}={actual!r}（期望 {expected!r}）")

    per_class = {str(k): v for k, v in (manifest.get("per_class_synthetic") or {}).items()}
    produced = per_class.get("13", 0)
    report.add(
        "1 manifest",
        "PASS" if produced == EXPECTED_SYNTHETIC else "FAIL",
        f'per_class_synthetic["13"]={produced}（期望 {EXPECTED_SYNTHETIC}）',
    )

    skipped_total = int(manifest.get("skipped_total", 0))
    planned = int(manifest.get("source_instances", 0)) * int(manifest.get("copies_per_instance", 1))
    missing = planned - produced
    if missing > 0:
        reasons = manifest.get("skipped") or []
        sample = "; ".join(f"{item.get('reason')}" for item in reasons[:5]) or "（manifest 未记录原因）"
        status = "PASS" if reasons else "FAIL"
        report.add("1 manifest", status, f"少 {missing} 张，跳过原因样本：{sample}")
    else:
        report.add("1 manifest", "PASS", f"计划 {planned} 张，产出 {produced} 张，无缺口")

    # 抽查：源图 != 目标图；源图必须来自原 train 索引
    records = manifest.get("records") or []
    same = [r for r in records if Path(r["src_image"]) == Path(r["dst_image"])]
    report.add("1 manifest", "PASS" if not same else "FAIL", f"src==dst 的记录数：{len(same)}")

    if original_train_index.is_file():
        # train.txt 里是相对路径；manifest 记录的是绝对路径。先按索引文件所在目录解析成绝对路径再比对。
        from blade_defect.data.indexed_splits import resolve_index_entry

        allowed = set()
        for line in original_train_index.read_text(encoding="utf-8-sig").splitlines():
            entry = line.strip()
            if not entry:
                continue
            allowed.add(str(resolve_index_entry(original_train_index, entry)))
        outside = [r for r in records if str(Path(r["src_image"])) not in allowed]
        report.add(
            "1 manifest",
            "PASS" if not outside else "FAIL",
            f"源图不在原 train 索引内的记录数：{len(outside)}",
        )
        if outside:
            report.add("1 manifest", "INFO", f"越界样本（前 2 条）：{[r['src_image'] for r in outside[:2]]}")
    else:
        report.add("1 manifest", "WARN", f"原 train 索引不存在，未做来源校验：{original_train_index}")

    report.add("1 manifest", "INFO", f"抽样 3 条：{json.dumps(records[:3], ensure_ascii=False)[:300]}")


def check_synthetic_audit(report: Report, path: Path) -> None:
    if not path.is_file():
        report.add("2 audit", "FAIL", f"文件不存在：{path}")
        return
    payload = _load_json(path)
    geo_hits: dict[str, int] = {}
    confusable = 0
    for bucket in (payload.get("by_class") or {}).values():
        for flag, count in (bucket.get("flags") or {}).items():
            if flag in GEO_FLAGS:
                geo_hits[flag] = geo_hits.get(flag, 0) + int(count)
            elif flag == "CONFUSABLE":
                confusable += int(count)

    total_geo = sum(geo_hits.values())
    report.add("2 audit", "PASS" if total_geo == 0 else "FAIL", f"几何类 flag 合计 = {total_geo} {geo_hits or ''}")
    if payload.get("geometric_only"):
        # 关掉 CONFUSABLE 时它根本没被评估，不能说"=0"（那会被误读成"评估过且没命中"）
        report.add("2 audit", "INFO", "CONFUSABLE = 未评估（--geometric-only 关闭，confusable_enabled=false）")
    else:
        report.add("2 audit", "INFO", f"CONFUSABLE = {confusable}（只记录不卡）")
    report.add("2 audit", "INFO", f"geometric_only = {payload.get('geometric_only')}")


def check_pytest(report: Report, path: Path) -> None:
    if not path.is_file():
        report.add("3 pytest", "FAIL", f"日志不存在：{path}（必须先重跑，不能用上一轮记录）")
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"(\d+) passed", text)
    failed = re.search(r"(\d+) failed", text)
    if failed:
        report.add("3 pytest", "FAIL", f"存在失败：{failed.group(0)}")
        return
    if not match:
        report.add("3 pytest", "FAIL", "日志里找不到 'N passed'")
        return
    passed = int(match.group(1))
    report.add("3 pytest", "PASS" if passed >= 13 else "FAIL", f"{passed} passed（期望 ≥13）")


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def check_args_diff(report: Report, a0_args: Path, b1_data_yaml: Path) -> None:
    if not a0_args.is_file():
        report.add("4 args", "FAIL", f"A0 args.yaml 不存在：{a0_args}")
        return
    a0 = _load_yaml(a0_args)
    b1 = _load_yaml(b1_data_yaml)
    rows = []
    for key in ARGS_KEYS:
        left = a0.get(key, "<未设置>")
        # B-1 的最终 args 由 A0 配置 + overrides 决定；派生 data.yaml 只覆盖 data
        right = b1.get(key, "<未设置>") if key in b1 else "<未设置>"
        rows.append((key, left, right))
    report.add(
        "4 args",
        "INFO",
        "派生 data.yaml 不含训练参数，实际 diff 需在 B-1 起跑后由 runs/<b1>/args.yaml 对拍；"
        f"当前 A0 关键参数：{json.dumps({k: a0.get(k) for k in ARGS_KEYS}, ensure_ascii=False)}",
    )
    data_line = b1.get("train")
    report.add("4 args", "INFO", f"B-1 的 data.train = {data_line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--synthetic-audit", required=True)
    parser.add_argument("--pytest-log", required=True)
    parser.add_argument("--a0-args", required=True)
    parser.add_argument("--b1-config", required=True)
    parser.add_argument("--original-train-index", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = Report()
    check_manifest(report, Path(args.manifest), Path(args.original_train_index))
    check_synthetic_audit(report, Path(args.synthetic_audit))
    check_pytest(report, Path(args.pytest_log))
    check_args_diff(report, Path(args.a0_args), Path(args.b1_config))

    markdown = report.to_markdown()
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"report: {out}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
