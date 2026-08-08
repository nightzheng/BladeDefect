"""扫描 runs 目录的 run_manifest.json，报告运行状态与异常中断。

异常中断判定：manifest.status 停留在 running/trained_pending_eval 且无
finished_at（进程已退出但未走正常收尾）。存在 weights/last.pt 的中断 run
可直接断点续训（重新执行原命令即可自动 resume）。

用法：
    python scripts/check_run_status.py [--runs-dir runs] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from blade_defect.experiment.run_status import scan_run_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", type=Path)
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = parser.parse_args()

    records = scan_run_status(args.runs_dir)
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return

    if not records:
        print(f"{args.runs_dir} 下没有发现 run_manifest.json")
        return
    interrupted = 0
    for record in records:
        line = f"{record['experiment_id']}: status={record['status']}"
        if record["interrupted"]:
            interrupted += 1
            line += " [异常中断]"
            line += " 可断点续训(last.pt 存在)" if record["resumable"] else " 无 last.pt，无法续训"
        print(line)
    print(f"\n共 {len(records)} 个 run，异常中断 {interrupted} 个")
    if interrupted:
        sys.exit(1)


if __name__ == "__main__":
    main()
