"""Training paketi — Aether Core eğitim + ablation + eval.

Bkz: upcoming/06_TRAINING_ABLATION.txt
"""

from __future__ import annotations

from .train import (
    train_baseline, train_dann_phase,
    create_optimizer, create_scheduler, create_optimizer_and_scheduler,
    save_checkpoint, load_checkpoint,
)
from .eval import (
    evaluate, compute_detection_metrics, compute_recognition_metrics,
    compute_identification_metrics,
)
from .ablation import ABLATION_VARIANTS, run_ablation


__all__ = [
    "train_baseline", "train_dann_phase",
    "create_optimizer", "create_scheduler", "create_optimizer_and_scheduler",
    "save_checkpoint", "load_checkpoint",
    "evaluate", "compute_detection_metrics",
    "compute_recognition_metrics", "compute_identification_metrics",
    "ABLATION_VARIANTS", "run_ablation",
]
