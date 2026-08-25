"""生成阶段成果包逐文件 SHA-256 清单（stage_artifact_sha256.csv）。

对 `results/stage_20260830/` 内全部文件（清单自身除外）现场重算 SHA-256，
供阶段验收冻结与后续漂移检测。阶段包只含文档/表格/图表/清单等小型文件；
大权重与大体积预测不进入本包，仅在 02 清单与 04 引用文档中按路径+SHA 引用。

用法：
    python scripts/hash_stage_artifacts.py
    python scripts/hash_stage_artifacts.py --stage-dir results/stage_20260830
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from blade_defect.experiment.metadata import sha256_file
from blade_defect.utils.paths import resolve_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--stage-dir", default="results/stage_20260830", help="阶段成果包目录")
    args = parser.parse_args()
    stage_dir = Path(args.stage_dir)
    if not stage_dir.is_absolute():
        stage_dir = PROJECT_ROOT / stage_dir
    stage_dir = resolve_path(stage_dir)
    output = stage_dir / "stage_artifact_sha256.csv"

    files = sorted(
        path for path in stage_dir.rglob("*") if path.is_file() and path != output
    )
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "bytes", "sha256"])
        for path in files:
            writer.writerow(
                [path.relative_to(stage_dir).as_posix(), path.stat().st_size, sha256_file(path)]
            )
    print(f"hashed {len(files)} files -> {output}")


if __name__ == "__main__":
    main()
