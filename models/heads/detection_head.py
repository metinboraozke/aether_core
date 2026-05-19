"""Detection Head — Sparse ROI tespiti (PDF Modül 4 Adım 1).

UWB global latent → 6 slot doluluk olasılığı (sigmoid).
Threshold (0.5) üstündeki slotlar ROI → sonraki başlıklar sadece ROI'da çalışır
→ CPU/GPU yükü %80 azalır (PDF mantığı).

Loss: BCEWithLogitsLoss(logits, slot_labels.float())
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """UWB latent → 6 slot detection (sigmoid)."""

    def __init__(
        self,
        in_dim: int = 128,
        n_slots: int = 6,
        hidden: int = 64,
        threshold: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.threshold = threshold
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_slots),
        )

    def forward(self, uwb_global_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """uwb_global_latent: (B, in_dim) → dict {logits, prob, mask}"""
        logits = self.mlp(uwb_global_latent)        # (B, n_slots)
        prob = torch.sigmoid(logits)
        mask = (prob > self.threshold).float()
        return {"logits": logits, "prob": prob, "mask": mask}


def detection_loss(logits: torch.Tensor,
                    slot_labels: torch.Tensor) -> torch.Tensor:
    """BCE with logits.

    logits:      (B, n_slots)
    slot_labels: (B, n_slots)  binary 0/1
    """
    return F.binary_cross_entropy_with_logits(
        logits, slot_labels.float(), reduction='mean'
    )


__all__ = ["DetectionHead", "detection_loss"]
