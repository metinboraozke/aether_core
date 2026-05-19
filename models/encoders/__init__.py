"""Modül 3 — Dual-Stream Encoder Mimarisi (The Eyes).

Bkz: README.md → Modül 3, upcoming/02_MODUL_3_ENCODERS.txt

DualStreamEncoder = CSI Transformer + UWB 3-Tier + Cross-Attention +
                    auxiliary head'ler (KD + path DML-AP)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .csi_encoder import (
    CSILinkEncoder, LinkGeometryEmbed, CSITransformerBlock,
    CSITransformerEncoder, CSIPathAuxHead,
)
from .uwb_encoder import (
    UWBSummaryMLP, UWBCIRBranch, UWBCFRBranch,
    UWBEncoder, UWBTeacherProjector, UWBPathAuxHead,
)
from .cross_attention import SlotEmbedding, UWBCSICrossAttention


__all__ = [
    # CSI
    "CSILinkEncoder", "LinkGeometryEmbed", "CSITransformerBlock",
    "CSITransformerEncoder", "CSIPathAuxHead",
    # UWB
    "UWBSummaryMLP", "UWBCIRBranch", "UWBCFRBranch",
    "UWBEncoder", "UWBTeacherProjector", "UWBPathAuxHead",
    # Cross
    "SlotEmbedding", "UWBCSICrossAttention",
    # Main
    "DualStreamEncoder",
]


class DualStreamEncoder(nn.Module):
    """CSI Transformer + UWB 3-Tier + Cross-Attention + Auxiliary.

    Input:
        csi:       (B, 28, 2, 108)   — preprocess_csi çıktısı
        uwb:       (B, 6, 2, 32)     — preprocess_uwb çıktısı
        link_geo:  (B, 28, 3)        — link midpoint koordinatları (m)
        return_aux: True ise auxiliary head'ler de döner

    Output dict:
        slot_latent: (B, 6, cross_out_dim)    — slot-aware fused latent
        uwb_global:  (B, uwb_out_dim)         — UWB global feature
        csi_latent:  (B, 28, csi_out_dim)     — raw CSI latent (heads için)
        [aux]
        csi_path_pred:   (B, 28, max_paths, 4)
        uwb_path_pred:   (B, 6, max_paths, 4)
        uwb_oracle_pred: (B, 6, n_taps, 2)
    """

    def __init__(
        self,
        n_csi_links: int = 28,
        n_subcarriers: int = 108,
        n_uwb_links: int = 6,
        n_taps: int = 32,
        csi_out_dim: int = 256,
        uwb_out_dim: int = 128,
        slot_dim: int = 128,
        n_slots: int = 6,
        cross_out_dim: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        cross_heads: int = 4,
        max_paths: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_csi_links = n_csi_links
        self.n_uwb_links = n_uwb_links
        self.n_slots = n_slots

        self.csi_enc = CSITransformerEncoder(
            n_links=n_csi_links, n_subcarriers=n_subcarriers,
            transformer_layers=transformer_layers, n_heads=transformer_heads,
            out_dim=csi_out_dim, dropout=dropout,
        )
        self.uwb_enc = UWBEncoder(
            n_links=n_uwb_links, n_taps=n_taps, out_dim=uwb_out_dim,
        )
        self.slot_embedding = SlotEmbedding(n_slots=n_slots, embed_dim=slot_dim)
        self.cross_attn = UWBCSICrossAttention(
            uwb_dim=uwb_out_dim, csi_dim=csi_out_dim, slot_dim=slot_dim,
            out_dim=cross_out_dim, n_slots=n_slots,
            n_heads=cross_heads, dropout=dropout,
        )

        # Auxiliary heads (training-time)
        self.csi_path_aux = CSIPathAuxHead(
            latent_dim=csi_out_dim, n_links=n_csi_links, max_paths=max_paths,
        )
        self.uwb_path_aux = UWBPathAuxHead(
            latent_dim=uwb_out_dim, n_links=n_uwb_links, max_paths=max_paths,
        )
        self.uwb_teacher_proj = UWBTeacherProjector(
            latent_dim=uwb_out_dim, n_slots=n_slots, n_taps=n_taps,
        )

    def forward(
        self,
        csi: torch.Tensor,
        uwb: torch.Tensor,
        link_geo: torch.Tensor,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        B = csi.size(0)

        csi_latent = self.csi_enc(csi, link_geo)                # (B, 28, 256)
        uwb_latent = self.uwb_enc(uwb)                           # (B, 128)
        slot_q = self.slot_embedding(B, device=csi.device)       # (B, 6, 128)
        fused = self.cross_attn(uwb_latent, csi_latent, slot_q)  # (B, 6, 256)

        out: dict[str, torch.Tensor] = {
            "slot_latent": fused,
            "uwb_global": uwb_latent,
            "csi_latent": csi_latent,
        }
        if return_aux:
            out["csi_path_pred"] = self.csi_path_aux(csi_latent)
            out["uwb_path_pred"] = self.uwb_path_aux(uwb_latent)
            out["uwb_oracle_pred"] = self.uwb_teacher_proj(uwb_latent)
        return out

    def count_params(self) -> dict[str, int]:
        """Parametre sayısı per submodule (debugging için)."""
        return {
            "csi_encoder": sum(p.numel() for p in self.csi_enc.parameters()),
            "uwb_encoder": sum(p.numel() for p in self.uwb_enc.parameters()),
            "slot_embedding": sum(p.numel() for p in self.slot_embedding.parameters()),
            "cross_attention": sum(p.numel() for p in self.cross_attn.parameters()),
            "csi_path_aux": sum(p.numel() for p in self.csi_path_aux.parameters()),
            "uwb_path_aux": sum(p.numel() for p in self.uwb_path_aux.parameters()),
            "uwb_teacher_proj": sum(p.numel() for p in self.uwb_teacher_proj.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
