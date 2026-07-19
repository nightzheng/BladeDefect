"""Generate experiment comparison tables and plots."""

import argparse

from blade_defect.experiment import analyze_experiments, publish_analysis_assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument(
        "--publish-docs",
        action="store_true",
        help="Copy the three report-facing plots to docs/assets/analysis",
    )
    args = parser.parse_args()
    for path in analyze_experiments(args.summary, args.runs_dir, args.output_dir):
        print(path)
    if args.publish_docs:
        for path in publish_analysis_assets(args.output_dir):
            print(path)


if __name__ == "__main__":
    main()
