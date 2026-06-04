"""Detection Head — Sparse ROI tespiti (PDF Modül 4 Adım 1).

UWB global latent → 6 slot doluluk olasılığı (sigmoid).
Threshold (0.5) üstündeki slotlar ROI → sonraki başlıklar sadece ROI'da çalışır
→ CPU/GPU yükü %80 azalır (PDF mantığı).

Loss: BCEWithLogitsLoss(logits, slot_labels.float())

E Paketi (2026-06-03) — DENEME + ROLLBACK kayıt:
  E1 (pos_weight=0.55) + E2 (threshold 0.55) smoke test'te birlikte denendi:
    det_precision: 0.635 → 0.712 (+0.077)  iyileşti
    det_recall:    0.957 → 0.674 (-0.283)  çok düştü
    det_f1:        0.763 → 0.690 (-0.073)  NET KÖTÜLEŞTİ
  Tanı: pos_weight + threshold etkisi birikti, model "boş" tarafına aşırı kaydı.
  Karar: E1 ve E2 ROLLBACK — original davranış (pos_weight=None, threshold=0.5).
         E4 (forward threshold override) KORUNDU — sıfır risk, API esnekliği.
  ID head E3 fix'leri korundu (id_bit +0.017, codeword +0.006 iyileşmesi gerçek).
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
        threshold: float = 0.5,         # E2 rollback: 0.55 → 0.5 (default)
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

    def forward(
        self,
        uwb_global_latent: torch.Tensor,
        threshold: float | None = None,    # E4 (KORUNDU): inference-time override
    ) -> dict[str, torch.Tensor]:
        """uwb_global_latent: (B, in_dim) → dict {logits, prob, mask}

        threshold None ise self.threshold; production'da deployment-spesifik
        override mümkün (model'i yeniden init etmeden, örn. /api/set_threshold).
        """
        thr = self.threshold if threshold is None else threshold
        logits = self.mlp(uwb_global_latent)        # (B, n_slots)
        prob = torch.sigmoid(logits)
        mask = (prob > thr).float()
        return {"logits": logits, "prob": prob, "mask": mask}


def detection_loss(
    logits: torch.Tensor,
    slot_labels: torch.Tensor,
    pos_weight: float | None = None,        # E1 rollback: 0.55 → None (eski davranış)
) -> torch.Tensor:
    """BCE with logits + opsiyonel pos_weight.

    logits:      (B, n_slots)
    slot_labels: (B, n_slots)  binary 0/1
    pos_weight:  positive class ağırlığı (None = standart BCE).
                 Production tune için opsiyonel kalır (API'de değişebilir).
                 NOT: E paketi smoke testinde pos_weight=0.55 + threshold=0.55
                 kombinasyonu F1'i düşürdü, default'tan kaldırıldı.
    """
    pw = None
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits, slot_labels.float(), pos_weight=pw, reduction='mean'
    )


__all__ = ["DetectionHead", "detection_loss"]
