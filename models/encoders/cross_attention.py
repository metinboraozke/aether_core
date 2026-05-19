"""Cross-Stream Alignment — UWB ↔ CSI füzyon + 6 slot embedding.

Modül 3 — Bölüm C (README.md). PDF mantığı:
    UWB mesafe bilgisi 'Sorgu' (Query); WiFi linkler 'Anahtar/Değer' (K/V).
    6 bölmeli raf için slot-specific learnable embedding'ler. Cross-attention
    sonrası her slot için context-aware fused latent.

Mimari:
    Input:
        uwb_latent:  (B, 128)
        csi_latent:  (B, 28, 256)
        slot_emb:    (B, 6, 128)  ← öğrenilebilir slot query'leri
    Output:
        fused_slot:  (B, 6, 256)  ← her slot için cross-attention çıktısı
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotEmbedding(nn.Module):
    """6 bölme için öğrenilebilir slot query embedding + FiLM modulation."""

    def __init__(self, n_slots: int = 6, embed_dim: int = 128):
        super().__init__()
        self.n_slots = n_slots
        self.embed_dim = embed_dim
        # nn.Embedding kullan — daha temiz
        self.slot_emb = nn.Embedding(n_slots, embed_dim)
        # FiLM (gamma, beta) için MLP
        self.film_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
        )

    def forward(self, batch_size: int,
                device: torch.device | None = None) -> torch.Tensor:
        """(B, n_slots, embed_dim) — slot query embedding'leri."""
        ids = torch.arange(self.n_slots,
                           device=device or self.slot_emb.weight.device)
        emb = self.slot_emb(ids)                            # (n_slots, embed_dim)
        return emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, n_slots, dim)

    def modulate(self, features: torch.Tensor,
                 slot_emb: torch.Tensor) -> torch.Tensor:
        """FiLM-style modulation: features * gamma + beta.

        features: (B, n_slots, dim)
        slot_emb: (B, n_slots, dim)
        """
        gamma_beta = self.film_mlp(slot_emb)                # (B, n_slots, 2*dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return features * gamma + beta


class UWBCSICrossAttention(nn.Module):
    """UWB(Q) × CSI(K, V) cross-attention.

    UWB net mesafe bilgisi her slot query'sine bias verir; CSI link
    feature'larından bu mesafe aralığındaki bilgiyi attend eder. PDF
    mantığı: "arka plan gürültüsünü eleyip sadece nesnenin olduğu
    mesafe aralığındaki WiFi değişimlerine odaklan".
    """

    def __init__(
        self,
        uwb_dim: int = 128,
        csi_dim: int = 256,
        slot_dim: int = 128,
        out_dim: int = 256,
        n_slots: int = 6,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.out_dim = out_dim
        # UWB global latent → slot bias
        self.uwb_proj = nn.Linear(uwb_dim, slot_dim)
        # Q (slot + uwb bias) → out_dim
        self.q_proj = nn.Linear(slot_dim, out_dim)
        # K, V csi_latent → out_dim
        self.k_proj = nn.Linear(csi_dim, out_dim)
        self.v_proj = nn.Linear(csi_dim, out_dim)
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        # Output normalize
        self.ln = nn.LayerNorm(out_dim)
        # FFN
        self.ff = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(
        self,
        uwb_latent: torch.Tensor,
        csi_latent: torch.Tensor,
        slot_emb: torch.Tensor,
    ) -> torch.Tensor:
        # uwb_latent: (B, uwb_dim)
        # csi_latent: (B, n_csi_links=28, csi_dim)
        # slot_emb:   (B, n_slots=6, slot_dim)
        B = uwb_latent.size(0)

        # UWB global'i slot dim'e indir, her slot'a broadcast
        uwb_bias = self.uwb_proj(uwb_latent).unsqueeze(1) \
                       .expand(B, self.n_slots, -1)         # (B, 6, slot_dim)
        q_in = slot_emb + uwb_bias                           # (B, 6, slot_dim)
        q = self.q_proj(q_in)                                # (B, 6, out_dim)
        k = self.k_proj(csi_latent)                          # (B, 28, out_dim)
        v = self.v_proj(csi_latent)                          # (B, 28, out_dim)

        # Cross-attention: Q'dan K/V'ye attend
        attn_out, _ = self.mha(q, k, v, need_weights=False)  # (B, 6, out_dim)
        x = self.ln(q + attn_out)                            # residual + norm
        x = self.ln2(x + self.ff(x))                          # FFN + residual + norm
        return x                                              # (B, 6, out_dim)


__all__ = ["SlotEmbedding", "UWBCSICrossAttention"]
