"""Residual Adapter + iki fazlı fine-tune (PDF Modül 5 Adım 2).

Adapter formülü:
    çıktı = x + scale · adapter(x)
    scale_init = 0.1  (modelin bilgisi korunur, küçük perturbation)

Phase 1 (Etiketsiz Ön Eğitim):
    Gerçek odadan etiketsiz veri + sentetik.
    Loss: DANN + UWB-CSI contrastive (varsa).
    Encoder + adapter eğitilir, task head'ler dondurulur veya ignore.

Phase 2 (Etiketli İnce Ayar):
    Küçük etiketli gerçek veri.
    Encoder DONDURULUR (freeze), sadece adapter + task head'ler eğitilir.
    Loss: recognition + detection (PDF order).

freeze/unfreeze helpers: model'in hangi parametrelerinin gradient alacağını
kontrol eder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualAdapter(nn.Module):
    """Encoder içine sokulan hafif adapter katmanı.

    çıktı = x + scale * adapter(x)
    scale öğrenilebilir (nn.Parameter), scale_init=0.1 ile başlar.

    Adapter MLP bottleneck (dim → hidden → dim) — düşük parametre maliyeti.
    """

    def __init__(self, dim: int, hidden: int = 64, scale_init: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        # Adapter MLP'nin son katmanı zero-init (initial adapter ≈ identity)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.mlp(x)


# ── Freeze / unfreeze utilities ────────────────────────────


def freeze_encoder(model: nn.Module, keep_adapter: bool = True,
                   keep_heads: bool = True) -> int:
    """Encoder parametrelerini dondur (requires_grad=False).

    Phase 2 fine-tune için. Sadece 'adapter' ve 'head' isimli modül
    parametreleri eğitilir.

    Args:
        model: FusedCSIUWBNet veya benzeri
        keep_adapter: 'adapter' içeren modül parametreleri serbest
        keep_heads:   'head', 'tsdf', 'recognition', 'identification',
                      'detection' içeren parametreler serbest

    Returns:
        n_frozen: dondurulan parametre sayısı
    """
    HEAD_KEYWORDS = (
        "head", "tsdf", "recognition", "identification", "detection",
    )
    n_frozen = 0
    for name, p in model.named_parameters():
        name_lower = name.lower()
        is_adapter = "adapter" in name_lower
        is_head = any(k in name_lower for k in HEAD_KEYWORDS)
        free = (keep_adapter and is_adapter) or (keep_heads and is_head)
        if free:
            p.requires_grad = True
        else:
            p.requires_grad = False
            n_frozen += 1
    return n_frozen


def unfreeze_all(model: nn.Module) -> None:
    """Tüm parametreleri serbest bırak (Phase 1 öncesi reset)."""
    for p in model.parameters():
        p.requires_grad = True


def count_trainable(model: nn.Module) -> dict[str, int]:
    """Hangi parametre kaç tane trainable + frozen — debug için."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": trainable + frozen,
    }


# ── Phase loop'ları (training/train.py'da çağrılır) ──────


def train_phase1(
    model: nn.Module,
    dann_wrapper,
    sim_loader,
    real_unlabeled_loader,
    epochs: int,
    optimizer,
    feature_extractor_fn,
):
    """Phase 1 — etiketsiz ön eğitim (DANN + contrastive).

    Bu fonksiyon SHELL — eğitim script'inde (training/train.py) çağrılır.
    Gerçek implementasyon Modül 6 (training)'de doldurulacak.

    Args:
        feature_extractor_fn: callable(batch) → latent (DANN için)

    TODO: contrastive UWB-CSI alignment loss (Modül 6 implementasyonu)
    """
    raise NotImplementedError(
        "train_phase1: Modül 6 (training) implementasyonunda doldurulacak."
    )


def train_phase2(
    model: nn.Module,
    labeled_real_loader,
    epochs: int,
    optimizer,
    compute_loss_fn,
):
    """Phase 2 — etiketli ince ayar (recognition + detection).

    Encoder freeze, sadece adapter + heads eğitilir.
    """
    raise NotImplementedError(
        "train_phase2: Modül 6 (training) implementasyonunda doldurulacak."
    )


__all__ = [
    "ResidualAdapter",
    "freeze_encoder", "unfreeze_all", "count_trainable",
    "train_phase1", "train_phase2",
]
