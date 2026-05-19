"""Stream 2 — 3-Tier UWB Encoder (Fiziksel Parmak İzi).

Modül 3 — Bölüm B (README.md). REVİZE 2026-05-15:
    - Hibrit topoloji: 4 UWB master, Round-Robin TDMA → 6 bistatic link
    - Input shape: (B, 6, 2, 32)  ← 6 bistatic link (4 anchor değil)
    - Auxiliary: TeacherProjector (KD) + PathAuxHead (DML-AP)

Mimari:
    Tier 1 — Summary MLP    : 5 stat → 32-d
    Tier 2 — CIR Time CNN   : 6 link × 32 tap, kernel=7 → 64-d
    Tier 3 — CFR Freq CNN   : FFT → 32 bin → 64-d (rezonatör çentikleri)
    Fuse                    : 32 + 64 + 64 → 128-d UWB latent
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Tier 1: Summary MLP ────────────────────────────────────


class UWBSummaryMLP(nn.Module):
    """5 fiziksel istatistik (range, RSSI, NLOS, ...) → 32-d.

    İstatistikler her bistatic link için CIR'dan TÜRETİLİR:
      1. mean_energy:  mean(|c|^2)
      2. peak_mag:     max(|c|)
      3. first_path_idx (range göstergesi) / n_taps  ∈ [0, 1]
      4. rms_delay_spread / n_taps                    ∈ [0, ~1]
      5. nlos_flag: peak/mean ratio < 4 ise 1, değilse 0
    """

    def __init__(self, n_stats: int = 5, hidden: int = 16,
                 out_dim: int = 32, n_links: int = 6, n_taps: int = 32):
        super().__init__()
        self.n_stats = n_stats
        self.n_links = n_links
        self.n_taps = n_taps
        self.mlp = nn.Sequential(
            nn.Linear(n_stats, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def _extract_stats(self, cir: torch.Tensor) -> torch.Tensor:
        """cir: (B, n_links, 2, n_taps) → stats: (B, n_links, 5)"""
        eps = 1e-12
        # Complex
        c = torch.complex(cir[..., 0, :], cir[..., 1, :])    # (B, L, T)
        mag = torch.abs(c)                                    # (B, L, T)
        power = mag ** 2                                      # (B, L, T)

        # 1. mean energy
        mean_energy = power.mean(dim=-1)                      # (B, L)

        # 2. peak amplitude
        peak = mag.max(dim=-1).values                         # (B, L)

        # 3. first-path tap (leading edge: amp > 0.1 * peak)
        threshold = 0.1 * peak.unsqueeze(-1)                  # (B, L, 1)
        above = mag > threshold                                # (B, L, T) bool
        # ilk True index (yoksa 0)
        any_above = above.any(dim=-1)                          # (B, L)
        # argmax(int) → ilk True'nun indeksi (False=0, True=1 → max ilk True'da)
        first_idx = above.float().argmax(dim=-1).float()      # (B, L)
        first_path_norm = first_idx / float(self.n_taps)      # [0, 1]

        # 4. RMS delay spread (normalized)
        taps = torch.arange(self.n_taps, device=cir.device,
                            dtype=mag.dtype)
        power_sum = power.sum(dim=-1) + eps
        mean_tap = (power * taps).sum(dim=-1) / power_sum      # (B, L)
        var_tap = (power * (taps - mean_tap.unsqueeze(-1)) ** 2).sum(dim=-1) / power_sum
        rms_spread = torch.sqrt(var_tap + eps) / float(self.n_taps)

        # 5. NLOS flag (peak/mean ratio < 4)
        peak_to_mean = peak / (mag.mean(dim=-1) + eps)
        nlos_flag = (peak_to_mean < 4.0).to(mag.dtype)

        stats = torch.stack([
            mean_energy, peak, first_path_norm, rms_spread, nlos_flag
        ], dim=-1)                                              # (B, L, 5)
        return stats

    def forward(self, cir: torch.Tensor) -> torch.Tensor:
        # cir: (B, 6, 2, 32)
        stats = self._extract_stats(cir)                       # (B, 6, 5)
        per_link = self.mlp(stats)                              # (B, 6, 32)
        # Global pool over links → global summary
        return per_link.mean(dim=1)                             # (B, 32)


# ── Tier 2: CIR Time-Domain Branch ────────────────────────


class UWBCIRBranch(nn.Module):
    """32 tap zaman domain CIR → per-link 64-d → global 64-d.

    Geniş bantlı yansımaları yakalamak için kernel_size=7 (PDF).
    """

    def __init__(self, in_channels: int = 2, out_dim: int = 64,
                 n_links: int = 6, n_taps: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, out_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, cir: torch.Tensor) -> torch.Tensor:
        # cir: (B, 6, 2, 32)
        B, L, C, T = cir.shape
        x = cir.reshape(B * L, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool1d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)                            # (B*L, out_dim)
        per_link = x.view(B, L, -1)
        return per_link.mean(dim=1)                              # (B, out_dim)


# ── Tier 3: CFR Frequency-Domain Branch (rezonatör çentikleri) ───


class UWBCFRBranch(nn.Module):
    """CIR → FFT → 32 bin CFR → per-link 64-d → global 64-d.

    Lorentzian rezonatör çentikleri burada görünür. CSITransformer'a
    benzer mimari ama frekans domeninde.
    """

    def __init__(self, in_channels: int = 2, out_dim: int = 64,
                 n_links: int = 6, n_taps: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, out_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, cir: torch.Tensor) -> torch.Tensor:
        # cir: (B, 6, 2, 32) — time domain
        # FFT zaman ekseninde → spektrum
        c = torch.complex(cir[..., 0, :], cir[..., 1, :])      # (B, 6, 32)
        cfr = torch.fft.fft(c, dim=-1)                          # (B, 6, 32) complex
        # Real/imag yeniden ayır → (B, 6, 2, 32)
        cfr_ri = torch.stack([cfr.real, cfr.imag], dim=-2)
        B, L, C, F_dim = cfr_ri.shape
        x = cfr_ri.reshape(B * L, C, F_dim)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool1d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        per_link = x.view(B, L, -1)
        return per_link.mean(dim=1)                              # (B, out_dim)


# ── UWBEncoder (3-Tier fuse) ──────────────────────────────


class UWBEncoder(nn.Module):
    """3-Tier UWB Encoder: Summary + CIR + CFR → 128-d global latent."""

    def __init__(self, n_links: int = 6, n_taps: int = 32,
                 summary_dim: int = 32, cir_dim: int = 64,
                 cfr_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.tier1 = UWBSummaryMLP(out_dim=summary_dim, n_links=n_links,
                                    n_taps=n_taps)
        self.tier2 = UWBCIRBranch(out_dim=cir_dim, n_links=n_links,
                                   n_taps=n_taps)
        self.tier3 = UWBCFRBranch(out_dim=cfr_dim, n_links=n_links,
                                   n_taps=n_taps)
        self.fuse = nn.Sequential(
            nn.Linear(summary_dim + cir_dim + cfr_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, cir: torch.Tensor) -> torch.Tensor:
        # cir: (B, 6, 2, 32)
        s1 = self.tier1(cir)                                    # (B, 32)
        s2 = self.tier2(cir)                                    # (B, 64)
        s3 = self.tier3(cir)                                    # (B, 64)
        x = torch.cat([s1, s2, s3], dim=-1)                     # (B, 160)
        return self.fuse(x)                                     # (B, 128)


# ── Auxiliary: Teacher Projector (KD) ──────────────────────


class UWBTeacherProjector(nn.Module):
    """Anchor latent → Oracle CIR reconstruction.

    Training-time loss: MSE(reconstruct, uwb_cir_oracle.npy)
    Student encoder anchor input'tan oracle slot CIR'ını tahmin etmeye
    zorlanır → Knowledge Distillation.
    """

    def __init__(self, latent_dim: int = 128, n_slots: int = 6,
                 n_taps: int = 32, hidden: int = 256):
        super().__init__()
        self.n_slots = n_slots
        self.n_taps = n_taps
        self.out_dim = n_slots * n_taps * 2
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, uwb_latent: torch.Tensor) -> torch.Tensor:
        # uwb_latent: (B, 128) → (B, 6, 32, 2)
        out = self.mlp(uwb_latent)
        return out.view(-1, self.n_slots, self.n_taps, 2)


# ── Auxiliary: Path Aux Head (DML-AP) ──────────────────────


class UWBPathAuxHead(nn.Module):
    """UWB latent → 6 bistatic link path_params tahmin."""

    def __init__(self, latent_dim: int = 128, n_links: int = 6,
                 max_paths: int = 20, n_channels: int = 4,
                 hidden: int = 256):
        super().__init__()
        self.n_links = n_links
        self.max_paths = max_paths
        self.n_channels = n_channels
        self.out_dim = n_links * max_paths * n_channels
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, uwb_latent: torch.Tensor) -> torch.Tensor:
        # uwb_latent: (B, 128) → (B, 6, 20, 4)
        out = self.mlp(uwb_latent)
        return out.view(-1, self.n_links, self.max_paths, self.n_channels)


__all__ = [
    "UWBSummaryMLP", "UWBCIRBranch", "UWBCFRBranch",
    "UWBEncoder", "UWBTeacherProjector", "UWBPathAuxHead",
]
