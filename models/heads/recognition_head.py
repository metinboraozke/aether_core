"""Recognition Head — CSI tabanlı 4-class malzeme sınıflandırma (PDF Modül 4 Adım 3).

Sınıflar: Metal / Plastik / Ahşap / Karton (n_classes=4).
Sadece ROI içindeki (slot_labels=1) slotlarda CE loss uygulanır.

Veri etiket şeması (data/<preset>/material_labels.npy):
    0 = boş slot   (ignore, ROI mask=0)
    1 = metal
    2 = plastik
    3 = ahşap
    4 = karton

Model output 4 class — label = material_label - 1 (boş slotlar zaten masked).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecognitionHead(nn.Module):
    """Slot latent → 4-class material logits."""

    def __init__(
        self,
        in_dim: int = 256,
        n_classes: int = 4,
        hidden: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, slot_latent: torch.Tensor) -> torch.Tensor:
        """slot_latent: (B, n_slots, in_dim) → logits (B, n_slots, n_classes)"""
        return self.mlp(slot_latent)


def recognition_loss(
    logits: torch.Tensor,
    material_labels: torch.Tensor,
    slot_labels: torch.Tensor,
    n_classes: int = 4,
) -> torch.Tensor:
    """Masked cross-entropy — sadece ROI (slot_labels=1) içindeki slotlar.

    logits:           (B, n_slots, n_classes)
    material_labels:  (B, n_slots) — 0=boş, 1..4=metal/plastik/ahşap/karton
    slot_labels:      (B, n_slots) — 0/1 ROI mask
    """
    B, S, C = logits.shape

    # material_label 1..4 → class index 0..3
    labels_shifted = (material_labels.long() - 1).clamp(min=0, max=n_classes - 1)

    # CE per slot
    loss_per = F.cross_entropy(
        logits.reshape(-1, C),
        labels_shifted.reshape(-1),
        reduction='none',
    ).view(B, S)                                              # (B, S)

    # ROI mask
    mask = slot_labels.float()
    total_loss = (loss_per * mask).sum()
    n_valid = mask.sum().clamp_min(1.0)
    return total_loss / n_valid


__all__ = ["RecognitionHead", "recognition_loss"]
