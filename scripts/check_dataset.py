"""Standalone dataset validation script."""

from __future__ import annotations

import argparse
import json

from blade_defect.data import check_dataset
from blade_defect.utils import resolve_path

parser = argparse.ArgumentParser()
parser.add_argument("--images", required=True, type=resolve_path)
parser.add_argument("--labels", required=True, type=resolve_path)
parser.add_argument("--num-classes", type=int)
args = parser.parse_args()

print(json.dumps(check_dataset(args.images, args.labels, args.num_classes).to_dict(), ensure_ascii=False, indent=2))
