"""FusedCSIUWBNet — Modül 3 (Encoder) + Modül 4 (Heads) ana orkestratör.

Bkz: README.md → "Modül 4", upcoming/03_MODUL_4_BRAIN.txt

Pipeline:
    Input: csi (B, 28, 2, 108) + uwb (B, 6, 2, 32) + link_geo (B, 28, 3)
    1. DualStreamEncoder → slot_latent (B,6,256) + uwb_global (B,128)
                          + (opsiyonel) csi_path_pred / uwb_path_pred / oracle_pred
    2. Heads:
       Detection (UWB global)       → mask (B, 6)
       TSDF (slot latent)           → μ, σ² (B, 6, 8, 8, 8)
       Recognition (slot latent)    → material logits (B, 6, 4)
       Identification (slot latent) → latent_64d + codeword_logits

NOT: Tracker AYRI yönetilir (sliding window state'i var) — backend/inference_server
     içinde TrackingEngine ile çağrılır. fused_model.forward'a dahil DEĞİL.

compute_loss çok-task'lı: detection + recognition + identification + tsdf
                          + auxiliary (path_csi, path_uwb, oracle_kd)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders import DualStreamEncoder
from models.heads import (
    DetectionHead, BayesianTSDFDecoder,
    RecognitionHead, IdentificationHead,
    detection_loss, tsdf_loss, recognition_loss, identification_loss,
)


# Default loss weights (configs/training.yaml > lambdas)
# D Paketi (2026-06-03): recognition 1.0 → 0.0 (bypass)
#   Material classification production'da DB lookup ile yapılır (configs/resonator_db.yaml +
#   backend/product_db.py). Recognition head model'de kalır (refactor riski yok) ama loss'a
#   girmez, gradient akmaz. Sebep: rec_acc 0.255 (random) sentetik veride (ε_r overlap +
#   bağımsız material_id/data_id random), tek kaynak DB lookup mimari karar.
#   Detay: README "Recognition Strategy" + "ID Lifecycle Management" bölümleri.
DEFAULT_LAMBDAS = {
    "detection": 1.0,
    "recognition": 0.0,         # D Paketi: bypass (DB lookup primer)
    "identification": 1.0,
    "tsdf": 0.3,
    "csi_path": 0.3,           # auxiliary
    "uwb_path": 0.3,           # auxiliary
    "kd_oracle": 0.5,          # auxiliary
}


class FusedCSIUWBNet(nn.Module):
    """Modül 3 (Encoder) + Modül 4 (Heads) ana model."""

    def __init__(
        self,
        n_slots: int = 6,
        n_materials: int = 4,
        codeword_bits: int = 7,
        barcode_latent_dim: int = 64,
        voxel_grid: tuple[int, int, int] = (8, 8, 8),
        # Encoder dim'leri
        n_csi_links: int = 28,
        n_subcarriers: int = 108,
        n_uwb_links: int = 6,
        n_taps: int = 32,
        csi_out_dim: int = 256,
        uwb_out_dim: int = 128,
        slot_dim: int = 128,
        cross_out_dim: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        cross_heads: int = 4,
        max_paths: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.n_materials = n_materials

        # Encoder
        self.encoder = DualStreamEncoder(
            n_csi_links=n_csi_links,
            n_subcarriers=n_subcarriers,
            n_uwb_links=n_uwb_links,
            n_taps=n_taps,
            csi_out_dim=csi_out_dim,
            uwb_out_dim=uwb_out_dim,
            slot_dim=slot_dim,
            n_slots=n_slots,
            cross_out_dim=cross_out_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            cross_heads=cross_heads,
            max_paths=max_paths,
            dropout=dropout,
        )

        # Heads (slot-level: cross_out_dim, global: uwb_out_dim)
        self.detection = DetectionHead(in_dim=uwb_out_dim, n_slots=n_slots,
                                        dropout=dropout)
        self.tsdf = BayesianTSDFDecoder(in_dim=cross_out_dim, n_slots=n_slots,
                                         voxel_grid=voxel_grid, dropout=dropout)
        self.recognition = RecognitionHead(in_dim=cross_out_dim, n_classes=n_materials,
                                            dropout=dropout)
        self.identification = IdentificationHead(in_dim=cross_out_dim,
                                                   latent_dim=barcode_latent_dim,
                                                   codeword_bits=codeword_bits,
                                                   dropout=dropout)

    def forward(
        self,
        csi: torch.Tensor,
        uwb: torch.Tensor,
        link_geo: torch.Tensor,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        # 1. Encoder
        enc_out = self.encoder(csi, uwb, link_geo, return_aux=return_aux)
        slot_latent = enc_out["slot_latent"]                # (B, 6, 256)
        uwb_global = enc_out["uwb_global"]                   # (B, 128)

        # 2. Detection (UWB global)
        det = self.detection(uwb_global)                     # dict {logits, prob, mask}

        # 3. TSDF (slot latent)
        tsdf = self.tsdf(slot_latent)                        # dict {mu, sigma2}

        # 4. Recognition (slot latent)
        rec_logits = self.recognition(slot_latent)           # (B, 6, 4)

        # 5. Identification (slot latent)
        ident = self.identification(slot_latent)             # dict {latent_64d, codeword_logits}

        out: dict[str, torch.Tensor] = {
            "detection_logits": det["logits"],
            "detection_prob": det["prob"],
            "detection_mask": det["mask"],
            "tsdf_mu": tsdf["mu"],
            "tsdf_sigma2": tsdf["sigma2"],
            "material_logits": rec_logits,
            "barcode_latent": ident["latent_64d"],
            "barcode_codeword": ident["codeword_logits"],
        }
        if return_aux:
            out["csi_path_pred"] = enc_out["csi_path_pred"]
            out["uwb_path_pred"] = enc_out["uwb_path_pred"]
            out["uwb_oracle_pred"] = enc_out["uwb_oracle_pred"]
            # encoder internal feature'lar (DANN için potansiyel)
            out["uwb_global_latent"] = uwb_global
            out["slot_latent"] = slot_latent
        return out

    def count_params(self) -> dict[str, int]:
        return {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "detection": sum(p.numel() for p in self.detection.parameters()),
            "tsdf": sum(p.numel() for p in self.tsdf.parameters()),
            "recognition": sum(p.numel() for p in self.recognition.parameters()),
            "identification": sum(p.numel() for p in self.identification.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


# =====================================================================
# COMPUTE LOSS — multi-task
# =====================================================================


def compute_loss(
    out: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    aux_targets: dict[str, torch.Tensor] | None = None,
    lambdas: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Ana multi-task loss.

    Args:
        out: FusedCSIUWBNet.forward(...) çıktısı (return_aux=True olabilir)
        labels: dict {
            'slot':     (B, 6) binary,
            'material': (B, 6) 0..4,
            'codeword': (B, 6, 7) binary,
        }
        aux_targets: opsiyonel dict {
            'path_csi':    (B, 28, 20, 4) — preprocess_path_params edilmiş
            'path_uwb':    (B, 6, 20, 4)
            'oracle_cir':  (B, 6, 32, 2) — uwb_cir_oracle.npy
        }
        lambdas: ağırlıklar (None ise DEFAULT_LAMBDAS)

    Returns:
        dict {
            'total': scalar loss,
            'detection', 'recognition', 'identification', 'tsdf': scalar
            (opsiyonel) 'csi_path', 'uwb_path', 'kd_oracle': scalar
        }
    """
    if lambdas is None:
        lambdas = DEFAULT_LAMBDAS

    losses: dict[str, torch.Tensor] = {}

    # Detection
    L_det = detection_loss(out["detection_logits"], labels["slot"])
    losses["detection"] = L_det

    # Recognition (masked CE)
    L_rec = recognition_loss(
        out["material_logits"], labels["material"], labels["slot"]
    )
    losses["recognition"] = L_rec

    # Identification (triplet + BCE codeword)
    ident_out = {
        "latent_64d": out["barcode_latent"],
        "codeword_logits": out["barcode_codeword"],
    }
    L_id = identification_loss(
        ident_out, labels["codeword"], labels["slot"]
    )
    losses["identification"] = L_id

    # TSDF (heteroscedastic NLL, slot-level proxy supervision)
    L_tsdf = tsdf_loss(
        out["tsdf_mu"], out["tsdf_sigma2"],
        voxel_target=None, slot_labels=labels["slot"]
    )
    losses["tsdf"] = L_tsdf

    # Ana toplam
    total = (
        lambdas["detection"] * L_det
        + lambdas["recognition"] * L_rec
        + lambdas["identification"] * L_id
        + lambdas["tsdf"] * L_tsdf
    )

    # Auxiliary
    if aux_targets is not None:
        if "path_csi" in aux_targets and "csi_path_pred" in out:
            L_csi_path = _masked_path_mse(
                out["csi_path_pred"], aux_targets["path_csi"]
            )
            losses["csi_path"] = L_csi_path
            total = total + lambdas.get("csi_path", 0.3) * L_csi_path

        if "path_uwb" in aux_targets and "uwb_path_pred" in out:
            L_uwb_path = _masked_path_mse(
                out["uwb_path_pred"], aux_targets["path_uwb"]
            )
            losses["uwb_path"] = L_uwb_path
            total = total + lambdas.get("uwb_path", 0.3) * L_uwb_path

        if "oracle_cir" in aux_targets and "uwb_oracle_pred" in out:
            import torch.nn.functional as F
            L_kd = F.mse_loss(out["uwb_oracle_pred"], aux_targets["oracle_cir"])
            losses["kd_oracle"] = L_kd
            total = total + lambdas.get("kd_oracle", 0.5) * L_kd

    losses["total"] = total
    return losses


def _masked_path_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Path params MSE — validity mask (kanal 3) ile padding'e loss uygulanmaz.

    pred, target: shape (B, ..., max_paths, 4)
        kanal 0,1: real(a), imag(a)
        kanal 2:   tau (normalize edilmiş)
        kanal 3:   validity (0/1)
    """
    valid = target[..., 3:4]                                       # (..., max_paths, 1)
    # Sadece amplitude + tau kanallarına loss
    diff = (pred[..., :3] - target[..., :3]) ** 2                  # (..., max_paths, 3)
    masked = diff * valid                                           # broadcast
    n_valid = valid.sum().clamp_min(1.0)
    return masked.sum() / (n_valid * 3.0)


__all__ = ["FusedCSIUWBNet", "compute_loss", "DEFAULT_LAMBDAS"]
