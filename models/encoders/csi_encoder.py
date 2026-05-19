"""Stream 1 — CSI Transformer Encoder (Mekânsal Gözlemci).

Modül 3 — Bölüm A (README.md). REVİZE 2026-05-15:
    - Hibrit topoloji: 8 ESP32 → 28 link
    - Path-aware auxiliary head (CSIPathAuxHead) — DML-AP teacher signal

Akış:
    Input:
        csi:      (B, 28, 2, 108)   — preprocess_csi çıktısı (real/imag x subc)
        link_geo: (B, 28, 3)        — link midpoint koordinatları (m)
    Output:
        latent:   (B, 28, 256)

Mimari:
    Per-link CNN (108 subc → 128-d) +
    LinkGeometryEmbed (3-d → 64-d) →
    concat (192-d) →
    4 katmanlı, 8 kafalı Transformer →
    Linear proj (256-d)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── A: Per-link CNN ─────────────────────────────────────────


class CSILinkEncoder(nn.Module):
    """108 alt taşıyıcılı CSI'dan per-link 128-d feature.

    Her link bağımsız 1D-CNN ile işlenir; conv kernel'leri 7/5/3
    progressive azalıyor → küresel → lokal pattern hierarchy.
    """

    def __init__(self, in_channels: int = 2, out_dim: int = 128,
                 n_subcarriers: int = 108):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, out_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        # csi: (B, n_links, 2, 108) → per-link conv
        B, L, C, S = csi.shape
        x = csi.reshape(B * L, C, S)            # (B*L, 2, 108)
        x = F.relu(self.bn1(self.conv1(x)))     # (B*L, 32, 108)
        x = F.relu(self.bn2(self.conv2(x)))     # (B*L, 64, 108)
        x = F.max_pool1d(x, kernel_size=2)      # (B*L, 64, 54)
        x = F.relu(self.bn3(self.conv3(x)))     # (B*L, 128, 54)
        x = self.pool(x).squeeze(-1)            # (B*L, 128)
        return x.view(B, L, -1)                  # (B, L, 128)


# ── B: Link Geometry Embedding ──────────────────────────────


class LinkGeometryEmbed(nn.Module):
    """Anten koordinat ortası [x, y, z] → 64-d embedding.

    Modelin "hangi link nereden" sorusunu uzaysal olarak anlamasını sağlar.
    Sensor placement randomization ile birlikte deployment-invariant.
    """

    def __init__(self, in_dim: int = 3, hidden: int = 32, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, link_geo: torch.Tensor) -> torch.Tensor:
        # link_geo: (B, n_links, 3)
        return self.mlp(link_geo)               # (B, n_links, 64)


# ── C: Transformer Block ───────────────────────────────────


class CSITransformerBlock(nn.Module):
    """Standard pre-norm transformer block (Pre-LN).

    Self-attention: linkler arası ilişki (28 link birbiriyle attend eder).
    Scale dot-product: Attn(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, dim: int = 192, n_heads: int = 8,
                 ff_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = dim * 2
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_links, dim)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


# ── D: CSI Transformer Encoder (ana sınıf) ─────────────────


class CSITransformerEncoder(nn.Module):
    """CSI Stream 1 — Mekânsal Gözlemci."""

    def __init__(
        self,
        n_links: int = 28,
        n_subcarriers: int = 108,
        cnn_dim: int = 128,
        geo_dim: int = 64,
        transformer_layers: int = 4,
        n_heads: int = 8,
        out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_links = n_links
        self.link_encoder = CSILinkEncoder(in_channels=2, out_dim=cnn_dim,
                                            n_subcarriers=n_subcarriers)
        self.geo_embed = LinkGeometryEmbed(in_dim=3, out_dim=geo_dim)
        self.block_dim = cnn_dim + geo_dim
        self.transformer = nn.ModuleList([
            CSITransformerBlock(dim=self.block_dim, n_heads=n_heads,
                                dropout=dropout)
            for _ in range(transformer_layers)
        ])
        self.proj = nn.Linear(self.block_dim, out_dim)

    def forward(self, csi: torch.Tensor,
                link_geo: torch.Tensor) -> torch.Tensor:
        # csi:      (B, 28, 2, 108)
        # link_geo: (B, 28, 3)
        link_feat = self.link_encoder(csi)             # (B, 28, 128)
        geo_feat = self.geo_embed(link_geo)            # (B, 28, 64)
        x = torch.cat([link_feat, geo_feat], dim=-1)   # (B, 28, 192)
        for block in self.transformer:
            x = block(x)
        latent = self.proj(x)                           # (B, 28, 256)
        return latent


# ── E: Path-aware Auxiliary Head (DML-AP teacher) ──────────


class CSIPathAuxHead(nn.Module):
    """Per-link latent → path_params tahmin.

    Training-time supervision: MSE(predicted_paths, path_params_csi.npy).
    Validity mask kanalı ile padding'e loss uygulanmaz.
    Inference'ta detached (encoder zaten path-aware temsil çıkarmayı
    içselleştirmiş olur).
    """

    def __init__(self, latent_dim: int = 256, n_links: int = 28,
                 max_paths: int = 20, n_channels: int = 4,
                 hidden: int = 128):
        super().__init__()
        self.max_paths = max_paths
        self.n_channels = n_channels
        self.out_dim = max_paths * n_channels
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, csi_latent: torch.Tensor) -> torch.Tensor:
        # csi_latent: (B, n_links, latent_dim)
        out = self.mlp(csi_latent)
        return out.view(*csi_latent.shape[:-1], self.max_paths, self.n_channels)


__all__ = [
    "CSILinkEncoder", "LinkGeometryEmbed", "CSITransformerBlock",
    "CSITransformerEncoder", "CSIPathAuxHead",
]
