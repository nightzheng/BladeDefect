"""Opt-in host timing through official YOLO callbacks; never synchronize CUDA."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path


class TrainingPerformance:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.epoch_start = self.previous_end = self.batch_start = 0.0
        self.steps = self.wait = self.host_step = 0
        self.metadata = {}
        self.train_seconds = 0.0

    def start(self, trainer):
        import cv2
        import torch

        loader = trainer.train_loader
        self.metadata = {
            "ultralytics": version("ultralytics"), "torch": torch.__version__,
            "device": str(trainer.device), "cpu_count": os.cpu_count(),
            "batch_requested": trainer.args.batch, "batch_effective": loader.batch_size,
            "workers_requested": trainer.args.workers, "workers_effective": loader.num_workers,
            "imgsz": trainer.args.imgsz, "seed": trainer.args.seed,
            "cache": trainer.args.cache, "pin_memory": loader.pin_memory,
            "persistent_workers_flag": loader.persistent_workers,
            "loader_class": type(loader).__name__, "prefetch_factor": loader.prefetch_factor,
            "opencv_threads_main": cv2.getNumThreads(),
            "torch_threads_main": torch.get_num_threads(),
            "thread_environment": {k: os.getenv(k) for k in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "timing_note": "Host wall times, no CUDA sync. Fetch gap includes loop overhead; "
                           "host step includes preprocess/enqueue/backpressure, not GPU kernel time.",
        }

    def epoch(self, trainer):
        self.epoch_start = self.previous_end = self.clock()
        self.steps = 0
        self.wait = self.host_step = 0.0

    def batch(self, trainer):
        self.batch_start = self.clock()
        self.wait += self.batch_start - self.previous_end

    def batch_end(self, trainer):
        self.previous_end = self.clock()
        self.host_step += self.previous_end - self.batch_start
        self.steps += 1

    def epoch_end(self, trainer):
        self.train_seconds = self.clock() - self.epoch_start

    def fit_end(self, trainer):
        # BaseTrainer also calls this event during final best-weight evaluation.
        if self.steps == 0:
            return
        import torch

        row = {
            **self.metadata, "epoch": trainer.epoch + 1, "iterations": self.steps,
            "batch_effective": trainer.train_loader.batch_size,
            "train_wall_s": self.train_seconds,
            "epoch_wall_s": self.clock() - self.epoch_start,
            "fetch_gap_s": self.wait, "host_step_s": self.host_step,
            "iteration_host_mean_s": (self.wait + self.host_step) / self.steps,
            "train_images_per_s": len(trainer.train_loader.dataset) / max(self.train_seconds, 1e-9),
            "gpu_memory_allocated_bytes": (
                torch.cuda.memory_allocated(trainer.device) if trainer.device.type == "cuda" else 0),
            "gpu_memory_reserved_bytes": (
                torch.cuda.memory_reserved(trainer.device) if trainer.device.type == "cuda" else 0),
        }
        path = Path(trainer.save_dir) / "pipeline_performance.jsonl"
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.steps = 0


@contextmanager
def performance_callbacks(model, enabled=False):
    if not isinstance(enabled, bool):
        raise ValueError("pipeline_profile must be true or false")
    if not enabled:
        yield
        return
    recorder = TrainingPerformance()
    callbacks = {
        "on_train_start": recorder.start, "on_train_epoch_start": recorder.epoch,
        "on_train_batch_start": recorder.batch, "on_train_batch_end": recorder.batch_end,
        "on_train_epoch_end": recorder.epoch_end, "on_fit_epoch_end": recorder.fit_end,
    }
    try:
        for event, callback in callbacks.items():
            model.add_callback(event, callback)
        yield
    finally:
        for event, callback in callbacks.items():
            if callback in model.callbacks.get(event, []):
                model.callbacks[event].remove(callback)
