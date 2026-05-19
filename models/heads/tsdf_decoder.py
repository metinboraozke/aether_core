"""Bayesian TSDF Decoder — Voxel grid μ + σ² (PDF Modül 4 Adım 2).

Her slot için 8×8×8 voxel grid başına:
    μ      — doluluk olasılığı (Bayesian mean)
    σ²     — belirsizlik (Bayesian variance)

Realist etki: gürültülü bölgelerde σ artar → dashboard'da "bulanık" görünür.

Loss: Gaussian NLL (heteroscedastic) veya basit MSE(μ, voxel_label) + KL prior.
Burada Gaussian NLL kullanıyoruz çünkü σ² öğrenilebilir.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianTSDFDecoder(nn.Module):
    """Slot başına 8×8×8 voxel μ + σ² (heteroscedastic)."""

    def __init__(
        self,
        in_dim: int = 256,
        n_slots: int = 6,
        voxel_grid: tuple[int, int, int] = (8, 8, 8),
        hidden: int = 256,
        dropout: float = 0.1,
        min_logvar: float = -6.0,
        max_logvar: float = 3.0,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.voxel_grid = tuple(voxel_grid)
        self.n_voxels = voxel_grid[0] * voxel_grid[1] * voxel_grid[2]
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        # Shared trunk → ayrı head (μ ve logvar)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(hidden, self.n_voxels)
        self.logvar_head = nn.Linear(hidden, self.n_voxels)

    def forward(self, slot_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """slot_latent: (B, n_slots, in_dim) → dict {mu, sigma2}"""
        h = self.trunk(slot_latent)                              # (B, 6, hidden)
        mu_raw = self.mu_head(h)                                 # (B, 6, n_voxels)
        logvar = self.logvar_head(h)
        # μ ∈ [0, 1] (sigmoid — voxel occupancy probability)
        mu = torch.sigmoid(mu_raw)
        # σ² stabilize için clamp + exp
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        sigma2 = torch.exp(logvar)

        # Reshape to voxel grid
        gx, gy, gz = self.voxel_grid
        mu = mu.view(-1, self.n_slots, gx, gy, gz)
        sigma2 = sigma2.view(-1, self.n_slots, gx, gy, gz)
        return {"mu": mu, "sigma2": sigma2}


def tsdf_loss(
    mu: torch.Tensor,
    sigma2: torch.Tensor,
    voxel_target: torch.Tensor | None = None,
    slot_labels: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """TSDF heteroscedastic loss.

    Eğer voxel_target verilmişse: Gaussian NLL (heteroscedastic).
        L = 0.5 * log(σ²) + 0.5 * (μ - y)² / σ²
    Eğer voxel_target YOK ama slot_labels varsa: basit slot-level proxy.
        Boş slot için μ → 0, dolu slot için μ → 1 (slot maskesi tile edilir).

    Sentetik veri tarafında detaylı voxel ground-truth yok; slot_labels ile
    proxy supervision uygulanır. Real fine-tune'da voxel scan eklenebilir.

    mu:           (B, 6, 8, 8, 8)
    sigma2:       (B, 6, 8, 8, 8)
    voxel_target: (B, 6, 8, 8, 8) veya None
    slot_labels:  (B, 6) — 0=boş, 1=dolu
    """
    if voxel_target is None:
        if slot_labels is None:
            raise ValueError("voxel_target veya slot_labels gerekli")
        # Slot label → tüm voxel'lere broadcast
        target = slot_labels.float().view(-1, mu.size(1), 1, 1, 1).expand_as(mu)
    else:
        target = voxel_target.float()

    # Heteroscedastic NLL
    log_sigma2 = torch.log(sigma2 + eps)
    nll = 0.5 * log_sigma2 + 0.5 * (mu - target) ** 2 / (sigma2 + eps)
    return nll.mean()


__all__ = ["BayesianTSDFDecoder", "tsdf_loss"]
