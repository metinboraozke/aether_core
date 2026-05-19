"""DANN — Adversarial Domain Adaptation (PDF Modül 5 Adım 1).

Sionna sentetik vs gerçek ESP32 domain gap'i kapatma.

Bileşenler:
    GradientReversalFn  — forward=identity, backward=-λ·grad (autograd)
    grad_reverse(x, λ)  — fonksiyonel wrapper
    grl_lambda_schedule — REVİZE (R10): warmup + sigmoid ramp + cap
    DomainDiscriminator — latent → domain logit (0=sim, 1=real)
    DANNWrapper         — encoder + GRL + disc orchestrator

Loss entegrasyonu:
    L_total = L_task + L_domain
    GRL otomatik -λ·grad uygular → encoder discriminator'ı YANILTACAK
    feature öğrenir → domain-invariant latent.

configs/training.yaml > runtime > dann_warmup_epochs (3), dann_max_lambda (0.7)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Gradient Reversal Layer (custom autograd) ──────────────


class GradientReversalFn(torch.autograd.Function):
    """forward = identity, backward = -λ · grad

    DANN'ın çekirdeği. Encoder'a "discriminator'ı yanılt" sinyali iletir.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Functional wrapper. nn.Module dışında inline kullanım için."""
    return GradientReversalFn.apply(x, lambda_)


# ── λ Schedule ─────────────────────────────────────────────


def grl_lambda_schedule(
    progress: float,
    alpha: float = 5.0,
    warmup_p: float = 0.1,
    max_lambda: float = 0.7,
) -> float:
    """REVİZE — yumuşak sigmoid ramp + warmup + cap.

    Klasik DANN formülü: λ = 2/(1+exp(-10·p)) − 1, p∈[0,1]
    REVİZE riskleri (R10):
      - alpha 10 → 5  (daha yavaş ramp)
      - warmup_p ilk %10 λ=0  (recognition önce stabilize olsun)
      - max_lambda 1.0 → 0.7  (cap, encoder ezilmesin)

    Args:
        progress: eğitim ilerleme oranı, [0, 1] (epoch_idx / total_epochs)
        alpha:    sigmoid steepness (config > grl_schedule_alpha)
        warmup_p: bu progress oranına kadar λ=0
        max_lambda: üst sınır cap

    Returns:
        λ ∈ [0, max_lambda]
    """
    p = max(0.0, min(1.0, float(progress)))
    if p < warmup_p:
        return 0.0
    p_eff = (p - warmup_p) / (1.0 - warmup_p)
    raw = 2.0 / (1.0 + math.exp(-alpha * p_eff)) - 1.0
    return min(float(max_lambda), raw)


# ── Domain Discriminator ───────────────────────────────────


class DomainDiscriminator(nn.Module):
    """Latent → binary domain logit (0=Sionna sentetik, 1=gerçek ESP32).

    Slot-level latent (B, n_slots, slot_dim) veya global latent (B, dim)
    girişi destekler — slot-level durumunda mean pooling.
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))           # binary logit
        self.mlp = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: (B, in_dim) veya (B, n_slots, in_dim) → (B,)"""
        if latent.dim() == 3:
            latent = latent.mean(dim=1)              # slot pool
        return self.mlp(latent).squeeze(-1)


# ── DANN Wrapper (model + GRL + discriminator orchestration) ───


class DANNWrapper(nn.Module):
    """Encoder feature + GRL + DomainDiscriminator orchestratör.

    Kullanım:
        wrapper = DANNWrapper(encoder=fused_model.encoder, discriminator=...)
        latent, L_dom, lambda_used = wrapper(csi, uwb, link_geo,
                                              domain_labels, progress=0.42)
        # L_dom otomatik GRL ile encoder'a -λ·grad ileti
    """

    def __init__(
        self,
        discriminator: DomainDiscriminator,
        feature_key: str = "uwb_global_latent",
        max_lambda: float = 0.7,
        alpha: float = 5.0,
        warmup_p: float = 0.1,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.feature_key = feature_key
        self.max_lambda = max_lambda
        self.alpha = alpha
        self.warmup_p = warmup_p

    def compute_domain_loss(
        self,
        latent: torch.Tensor,
        domain_labels: torch.Tensor,
        progress: float,
    ) -> tuple[torch.Tensor, float]:
        """Latent'ten domain loss hesapla.

        latent:        (B, dim) veya (B, n_slots, dim)  — encoder çıktısı
        domain_labels: (B,) binary {0=sim, 1=real}
        progress:      [0, 1]

        Returns: (L_domain, lambda_used)
        """
        lambda_ = grl_lambda_schedule(
            progress,
            alpha=self.alpha,
            warmup_p=self.warmup_p,
            max_lambda=self.max_lambda,
        )
        # GRL — encoder'a -λ·grad
        reversed_latent = grad_reverse(latent, lambda_)
        domain_logits = self.discriminator(reversed_latent)
        L_dom = F.binary_cross_entropy_with_logits(
            domain_logits, domain_labels.float()
        )
        return L_dom, lambda_


__all__ = [
    "GradientReversalFn", "grad_reverse", "grl_lambda_schedule",
    "DomainDiscriminator", "DANNWrapper",
]
