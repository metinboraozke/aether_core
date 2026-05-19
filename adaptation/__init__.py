"""Modül 5 — Sim-to-Real Adaptasyonu (The Bridge).

Bkz: README.md → Modül 5, upcoming/04_MODUL_5_ADAPTATION.txt
"""

from __future__ import annotations

from .dann import (
    GradientReversalFn, grad_reverse, grl_lambda_schedule,
    DomainDiscriminator, DANNWrapper,
)
from .residual_adapter import (
    ResidualAdapter,
    freeze_encoder, unfreeze_all, count_trainable,
    train_phase1, train_phase2,
)
from .distillation import (
    LightCSIEncoder, LightUWBEncoder, StudentFusedNet,
    distillation_loss, train_distillation,
)


__all__ = [
    # DANN
    "GradientReversalFn", "grad_reverse", "grl_lambda_schedule",
    "DomainDiscriminator", "DANNWrapper",
    # Adapter
    "ResidualAdapter", "freeze_encoder", "unfreeze_all", "count_trainable",
    "train_phase1", "train_phase2",
    # Distillation
    "LightCSIEncoder", "LightUWBEncoder", "StudentFusedNet",
    "distillation_loss", "train_distillation",
]
