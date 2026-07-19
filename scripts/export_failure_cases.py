"""Validate a prepared failure-case CSV and export the canonical index."""

import argparse

from blade_defect.experiment import export_failure_cases_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Prepared CSV containing prediction metadata")
    parser.add_argument("--output", default="results/failure_cases/cases.csv")
    args = parser.parse_args()
    print(export_failure_cases_csv(args.source, args.output))


if __name__ == "__main__":
    main()
