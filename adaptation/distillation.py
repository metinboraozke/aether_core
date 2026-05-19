"""Knowledge Distillation — Teacher → Student (PDF Modül 5 Adım 5).

Hedef: dashboard gecikme < 200 ms.
Teacher = FusedCSIUWBNet full (~2.9M parameter, 18 ms/sample CPU).
Student = StudentFusedNet light (~500k parameter, hedef < 5 ms/sample CPU).

Student mimari sadeleştirme:
    csi_transformer:    4 layer → 2 layer, dim 192→128, heads 8→4
    uwb_encoder:        3-Tier korunur ama her dim küçük (32/32/32 → 96 fuse)
    cross_attention:    out_dim 256 → 128
    Aux head'ler:       YOK (inference-only)

KD loss:
    L = α · MSE(student_latent, teacher_latent.detach())  ← representation distill
       + (1-α) · sum_head [ CE/MSE(student_logit, teacher_logit) ]  ← logit distill
       + β · CE(student_logit, gt_label)                  ← hard label

    T (temperature): 4.0 (softmax smoothing for logit distill)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import (
    CSILinkEncoder, LinkGeometryEmbed, CSITransformerBlock,
    UWBSummaryMLP, UWBCIRBranch, UWBCFRBranch,
    SlotEmbedding, UWBCSICrossAttention,
)
from models.heads import (
    DetectionHead, BayesianTSDFDecoder,
    RecognitionHead, IdentificationHead,
)


# ── Light Encoder Variant ──────────────────────────────────


class LightCSIEncoder(nn.Module):
    """Sadeleştirilmiş CSI Transformer (2 layer, 4 heads, dim 128)."""

    def __init__(self, n_links: int = 28, n_subcarriers: int = 108,
                 cnn_dim: int = 64, geo_dim: int = 32,
                 transformer_layers: int = 2, n_heads: int = 4,
                 out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
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

    def forward(self, csi, link_geo):
        link_feat = self.link_encoder(csi)
        geo_feat = self.geo_embed(link_geo)
        x = torch.cat([link_feat, geo_feat], dim=-1)
        for block in self.transformer:
            x = block(x)
        return self.proj(x)


class LightUWBEncoder(nn.Module):
    """Sadeleştirilmiş 3-Tier (dim'ler küçük, fuse 96 → 64)."""

    def __init__(self, n_links: int = 6, n_taps: int = 32,
                 out_dim: int = 64):
        super().__init__()
        self.tier1 = UWBSummaryMLP(out_dim=16, n_links=n_links, n_taps=n_taps)
        self.tier2 = UWBCIRBranch(out_dim=32, n_links=n_links, n_taps=n_taps)
        self.tier3 = UWBCFRBranch(out_dim=32, n_links=n_links, n_taps=n_taps)
        self.fuse = nn.Linear(16 + 32 + 32, out_dim)

    def forward(self, cir):
        s1 = self.tier1(cir)
        s2 = self.tier2(cir)
        s3 = self.tier3(cir)
        return self.fuse(torch.cat([s1, s2, s3], dim=-1))


class StudentFusedNet(nn.Module):
    """Light Student variant (KD target)."""

    def __init__(self, n_slots: int = 6, n_materials: int = 4,
                 codeword_bits: int = 7, voxel_grid=(8, 8, 8),
                 dropout: float = 0.1):
        super().__init__()
        # Light encoder
        self.csi_enc = LightCSIEncoder(out_dim=128, dropout=dropout)
        self.uwb_enc = LightUWBEncoder(out_dim=64)
        self.slot_emb = SlotEmbedding(n_slots=n_slots, embed_dim=64)
        self.cross_attn = UWBCSICrossAttention(
            uwb_dim=64, csi_dim=128, slot_dim=64,
            out_dim=128, n_slots=n_slots, n_heads=2, dropout=dropout,
        )
        # Heads (in_dim = cross_out_dim 128, uwb_global = 64)
        self.detection = DetectionHead(in_dim=64, n_slots=n_slots, hidden=32)
        self.tsdf = BayesianTSDFDecoder(in_dim=128, n_slots=n_slots,
                                         voxel_grid=voxel_grid, hidden=128)
        self.recognition = RecognitionHead(in_dim=128, n_classes=n_materials,
                                            hidden=64)
        self.identification = IdentificationHead(in_dim=128,
                                                   latent_dim=64,
                                                   codeword_bits=codeword_bits,
                                                   hidden=64)

    def forward(self, csi, uwb, link_geo) -> dict[str, torch.Tensor]:
        csi_latent = self.csi_enc(csi, link_geo)
        uwb_latent = self.uwb_enc(uwb)
        slot_q = self.slot_emb(csi.size(0), device=csi.device)
        slot_latent = self.cross_attn(uwb_latent, csi_latent, slot_q)

        det = self.detection(uwb_latent)
        tsdf = self.tsdf(slot_latent)
        rec_logits = self.recognition(slot_latent)
        ident = self.identification(slot_latent)
        return {
            "detection_logits": det["logits"],
            "detection_prob": det["prob"],
            "detection_mask": det["mask"],
            "tsdf_mu": tsdf["mu"],
            "tsdf_sigma2": tsdf["sigma2"],
            "material_logits": rec_logits,
            "barcode_latent": ident["latent_64d"],
            "barcode_codeword": ident["codeword_logits"],
            "uwb_global_latent": uwb_latent,
            "slot_latent": slot_latent,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── KD Loss ────────────────────────────────────────────────


def distillation_loss(
    student_out: dict[str, torch.Tensor],
    teacher_out: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor] | None = None,
    T: float = 4.0,
    alpha_repr: float = 0.3,
    alpha_logit: float = 0.4,
    alpha_hard: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Combined KD loss.

    Args:
        student_out: StudentFusedNet.forward(...) çıktısı
        teacher_out: FusedCSIUWBNet.forward(..., return_aux=True).detach()
        labels: opsiyonel hard labels {'slot', 'material', 'codeword'}
        T: softmax temperature
        alpha_repr / alpha_logit / alpha_hard: ağırlıklar (toplamı 1.0)

    Returns: dict {'total', 'repr', 'logit', 'hard'}
    """
    losses: dict[str, torch.Tensor] = {}

    # 1. Representation distill (slot_latent + uwb_global)
    s_lat = student_out.get("slot_latent")
    t_lat = teacher_out.get("slot_latent")
    s_uwb = student_out.get("uwb_global_latent")
    t_uwb = teacher_out.get("uwb_global_latent")

    repr_loss = torch.tensor(0.0, device=s_lat.device if s_lat is not None else 'cpu')
    n_repr = 0
    if s_lat is not None and t_lat is not None:
        # Boyutlar farklı (student 128, teacher 256) — projeksiyon yerine
        # cosine similarity (boyuttan bağımsız). MSE için adaptive avg pool.
        # Basit: orta boyuta proje (mean over feature axis) — yaklaşık alignment
        s_pool = s_lat.mean(dim=-1)                   # (B, n_slots)
        t_pool = t_lat.mean(dim=-1).detach()
        repr_loss = repr_loss + F.mse_loss(s_pool, t_pool)
        n_repr += 1
    if s_uwb is not None and t_uwb is not None:
        s_uwb_p = s_uwb.mean(dim=-1, keepdim=True)
        t_uwb_p = t_uwb.mean(dim=-1, keepdim=True).detach()
        repr_loss = repr_loss + F.mse_loss(s_uwb_p, t_uwb_p)
        n_repr += 1
    if n_repr > 0:
        repr_loss = repr_loss / n_repr
    losses["repr"] = repr_loss

    # 2. Logit distill (softmax KL with temperature)
    logit_loss = torch.tensor(0.0, device=s_lat.device if s_lat is not None else 'cpu')

    # Detection (sigmoid → "softmax 2-class")
    if "detection_logits" in student_out and "detection_logits" in teacher_out:
        s_p = torch.sigmoid(student_out["detection_logits"] / T)
        t_p = torch.sigmoid(teacher_out["detection_logits"].detach() / T)
        logit_loss = logit_loss + F.binary_cross_entropy(s_p, t_p) * (T * T)

    # Recognition (4-class softmax with T)
    if "material_logits" in student_out and "material_logits" in teacher_out:
        s_log = F.log_softmax(student_out["material_logits"] / T, dim=-1)
        t_p = F.softmax(teacher_out["material_logits"].detach() / T, dim=-1)
        # KL div
        kl = F.kl_div(s_log, t_p, reduction='batchmean')
        logit_loss = logit_loss + kl * (T * T)

    # Identification codeword (sigmoid bit-by-bit)
    if "barcode_codeword" in student_out and "barcode_codeword" in teacher_out:
        s_p = torch.sigmoid(student_out["barcode_codeword"] / T)
        t_p = torch.sigmoid(teacher_out["barcode_codeword"].detach() / T)
        logit_loss = logit_loss + F.binary_cross_entropy(s_p, t_p) * (T * T)

    losses["logit"] = logit_loss

    # 3. Hard label loss (opsiyonel, varsa)
    hard_loss = torch.tensor(0.0, device=s_lat.device if s_lat is not None else 'cpu')
    if labels is not None:
        if "slot" in labels:
            hard_loss = hard_loss + F.binary_cross_entropy_with_logits(
                student_out["detection_logits"], labels["slot"].float()
            )
        if "material" in labels and "material_logits" in student_out:
            B, S, C = student_out["material_logits"].shape
            mat_shifted = (labels["material"].long() - 1).clamp(min=0, max=C - 1)
            per = F.cross_entropy(
                student_out["material_logits"].reshape(-1, C),
                mat_shifted.reshape(-1),
                reduction='none',
            ).view(B, S)
            mask = labels["slot"].float()
            hard_loss = hard_loss + (per * mask).sum() / mask.sum().clamp_min(1.0)
    losses["hard"] = hard_loss

    # Toplam
    losses["total"] = (
        alpha_repr * repr_loss
        + alpha_logit * logit_loss
        + alpha_hard * hard_loss
    )
    return losses


def train_distillation(
    teacher: nn.Module,
    student: nn.Module,
    loader,
    epochs: int,
    optimizer,
    device: str = "cpu",
):
    """KD training loop — shell.

    Modül 6 (training)'de gerçek implementasyon. Şu an shell:
        - teacher.eval() (frozen)
        - student.train()
        - her batch: teacher forward (no_grad), student forward,
          distillation_loss → backward → optimizer step.
    """
    raise NotImplementedError(
        "train_distillation: Modül 6 (training) implementasyonunda doldurulacak."
    )


__all__ = [
    "LightCSIEncoder", "LightUWBEncoder", "StudentFusedNet",
    "distillation_loss", "train_distillation",
]
