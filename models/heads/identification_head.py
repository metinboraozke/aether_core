"""Identification Head — UWB CIR tabanlı spektral barkod (PDF Modül 4 Adım 4).

REVİZE 2026-05-14: Çift çıkışlı multi-task:
  a) 64-d L2-normalize latent (cosine similarity ile SpectralBarcodeDB lookup)
  b) 7-bit Hamming(7,4) codeword tahmini (BCE, decode → 4-bit data → 16 ID)

Loss = TripletMargin(latent) + α · BCE(codeword)    α = 0.5

Inference akışı:
  raw CIR → encoder → identification_head
    → latent_64d  → cosine sim ile DB lookup → ID
    → codeword_7b → Hamming decode → 4-bit data → ID
  iki yol birbirini doğrular (validation)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentificationHead(nn.Module):
    """Slot latent → (64-d L2-norm latent, 7-bit codeword logit)."""

    def __init__(
        self,
        in_dim: int = 256,
        latent_dim: int = 64,
        codeword_bits: int = 7,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.codeword_bits = codeword_bits

        self.latent_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )
        self.codeword_proj = nn.Sequential(
            nn.Linear(in_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, codeword_bits),
        )

    def forward(self, slot_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """slot_latent: (B, n_slots, in_dim)
        → dict {latent_64d, codeword_logits}"""
        lat = self.latent_proj(slot_latent)
        lat = F.normalize(lat, p=2, dim=-1)                    # L2-norm
        cw = self.codeword_proj(slot_latent)
        return {"latent_64d": lat, "codeword_logits": cw}


def identification_loss(
    out: dict[str, torch.Tensor],
    codeword_labels: torch.Tensor,
    slot_labels: torch.Tensor,
    alpha_bce: float = 0.5,
    triplet_margin: float = 0.2,
) -> torch.Tensor:
    """Multi-task loss: triplet (latent) + BCE (codeword).

    out['latent_64d']:        (B, n_slots, 64)  L2-norm
    out['codeword_logits']:   (B, n_slots, 7)
    codeword_labels:          (B, n_slots, 7)  binary
    slot_labels:              (B, n_slots)     0/1 ROI

    Triplet mining (batch içinden):
      Anchor: tüm ROI slot latent'leri
      Positive: aynı codeword'lu başka slot (varsa)
      Negative: farklı codeword'lu slot
      Eğer pozitif yoksa sample atılır.
    """
    latent = out["latent_64d"]
    cw_logits = out["codeword_logits"]
    B, S, _ = latent.shape

    # BCE codeword (sadece ROI)
    bce_per = F.binary_cross_entropy_with_logits(
        cw_logits, codeword_labels.float(), reduction='none'
    ).mean(dim=-1)                                              # (B, S)
    mask = slot_labels.float()
    bce_total = (bce_per * mask).sum() / mask.sum().clamp_min(1.0)

    # Triplet mining
    triplet_loss = _triplet_mining_loss(
        latent, codeword_labels, slot_labels, margin=triplet_margin
    )

    return triplet_loss + alpha_bce * bce_total


def _triplet_mining_loss(
    latent: torch.Tensor,           # (B, S, D) L2-norm
    codewords: torch.Tensor,        # (B, S, 7)
    slot_labels: torch.Tensor,      # (B, S)
    margin: float = 0.2,
) -> torch.Tensor:
    """Basit batch içi triplet mining.

    Tüm ROI slot pair'leri arasında dağılır:
      pos pair: aynı codeword
      neg pair: farklı codeword
    Cosine distance = 1 - cosine_sim.
    L = max(0, d(a,p) - d(a,n) + margin)

    Hızlı approximation: tüm (B,S) slot'ları flatten, ROI olmayanları at,
    pairwise distance matrix oluştur.
    """
    B, S, D = latent.shape
    flat_lat = latent.view(-1, D)                                # (BS, D)
    flat_cw = codewords.view(-1, codewords.size(-1))             # (BS, 7)
    flat_mask = slot_labels.view(-1) > 0                          # (BS,)

    valid_lat = flat_lat[flat_mask]                               # (N, D)
    valid_cw = flat_cw[flat_mask]                                 # (N, 7)
    N = valid_lat.size(0)

    if N < 3:
        return torch.tensor(0.0, device=latent.device, requires_grad=True)

    # Cosine sim → distance
    sim = valid_lat @ valid_lat.T                                 # (N, N)
    dist = 1.0 - sim                                              # ∈ [0, 2]

    # Codeword eşitlik matrisi (aynı ID ↔ aynı 7-bit pattern)
    cw_eq = (valid_cw.unsqueeze(1) == valid_cw.unsqueeze(0)).all(dim=-1)
    diag = torch.eye(N, dtype=torch.bool, device=latent.device)
    pos_mask = cw_eq & ~diag                                      # aynı codeword, kendisi değil
    neg_mask = ~cw_eq                                              # farklı codeword

    # Eğer pozitif veya negatif yoksa loss = 0
    if not pos_mask.any() or not neg_mask.any():
        return torch.tensor(0.0, device=latent.device, requires_grad=True)

    # Her anchor için en zor pozitif (max dist) ve en kolay negatif (min dist)
    pos_dist = torch.where(pos_mask, dist, torch.full_like(dist, -1.0))
    hard_pos = pos_dist.max(dim=1).values                         # (N,) — -1 ise pos yok
    neg_dist = torch.where(neg_mask, dist, torch.full_like(dist, float('inf')))
    hard_neg = neg_dist.min(dim=1).values                         # (N,) — inf ise neg yok

    valid_triplet = (hard_pos >= 0) & torch.isfinite(hard_neg)
    if not valid_triplet.any():
        return torch.tensor(0.0, device=latent.device, requires_grad=True)

    losses = F.relu(hard_pos[valid_triplet] - hard_neg[valid_triplet] + margin)
    return losses.mean()


# ── Inference helper: cosine lookup ────────────────────────


def cosine_lookup(
    latent_64d: torch.Tensor,           # (B, S, 64) L2-norm
    db_signatures: torch.Tensor,         # (n_db, 64) L2-norm
    top_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine similarity ile DB lookup.

    Döner: (top_k_similarities, top_k_db_indices) shape (B, S, k)
    """
    sim = latent_64d @ db_signatures.T                            # (B, S, n_db)
    return sim.topk(top_k, dim=-1)


__all__ = [
    "IdentificationHead", "identification_loss",
    "cosine_lookup", "_triplet_mining_loss",
]
