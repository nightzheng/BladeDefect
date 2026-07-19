"""Standalone training script."""

from __future__ import annotations

import argparse

from blade_defect.models import SegmentationTrainer
from blade_defect.utils import resolve_path

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml", type=resolve_path)
args = parser.parse_args()

trainer, config = SegmentationTrainer.from_config(args.config)
trainer.train(**config)
