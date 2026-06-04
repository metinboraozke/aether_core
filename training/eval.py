"""Eval metrics — Detection / Recognition / Identification / TSDF.

Mini implementasyon (production'da sklearn.metrics ile değiştirilebilir).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ── Detection (slot dolu/boş, 6 slot) ───────────────────


def compute_detection_metrics(
    logits: torch.Tensor,
    slot_labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Per-slot accuracy + macro F1 + precision/recall.

    logits:      (B, 6)
    slot_labels: (B, 6) binary
    """
    pred = (torch.sigmoid(logits) > threshold).long()
    gt = slot_labels.long()

    tp = ((pred == 1) & (gt == 1)).sum().item()
    fp = ((pred == 1) & (gt == 0)).sum().item()
    fn = ((pred == 0) & (gt == 1)).sum().item()
    tn = ((pred == 0) & (gt == 0)).sum().item()

    total = tp + fp + fn + tn
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "det_accuracy": acc,
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
    }


# ── Recognition (4-class, sadece ROI) ───────────────────


def compute_recognition_metrics(
    logits: torch.Tensor,
    material_labels: torch.Tensor,
    slot_labels: torch.Tensor,
    n_classes: int = 4,
) -> dict[str, float]:
    """Masked accuracy + per-class precision (sadece slot=1 olanlar).

    logits:          (B, 6, 4)
    material_labels: (B, 6) — 0=boş, 1..4=metal/plastik/ahşap/karton
    slot_labels:     (B, 6) binary
    """
    pred = logits.argmax(dim=-1)                            # (B, 6) ∈ 0..3
    gt = (material_labels.long() - 1).clamp(min=0, max=n_classes - 1)
    mask = slot_labels.bool()

    if mask.sum() == 0:
        return {"rec_accuracy": 0.0, "rec_n_samples": 0}

    correct = ((pred == gt) & mask).sum().item()
    total = mask.sum().item()
    return {
        "rec_accuracy": correct / total,
        "rec_n_samples": total,
    }


# ── Identification (codeword bit accuracy + ID lookup) ──


def compute_identification_metrics(
    codeword_logits: torch.Tensor,
    barcode_latent: torch.Tensor,
    codeword_labels: torch.Tensor,
    slot_labels: torch.Tensor,
    db_signatures: torch.Tensor | None = None,
) -> dict[str, float]:
    """Codeword bit accuracy + (varsa) cosine lookup top-1 ID accuracy.

    codeword_logits:  (B, 6, 7)
    barcode_latent:   (B, 6, 64)  L2-norm
    codeword_labels:  (B, 6, 7) binary
    slot_labels:      (B, 6) binary
    db_signatures:    (n_db, 64) — varsa cosine top-1 hesap

    Hamming decode + 4-bit ID recovery için resonator_inject.hamming_decode_7_4
    kullanılabilir (training'de zorunlu değil).
    """
    pred_bits = (torch.sigmoid(codeword_logits) > 0.5).long()
    gt_bits = codeword_labels.long()
    mask = slot_labels.bool().unsqueeze(-1)                  # (B, 6, 1)

    correct_bits = ((pred_bits == gt_bits) & mask).sum().item()
    total_bits = mask.sum().item() * 7
    bit_acc = correct_bits / max(total_bits, 1)

    # Tam codeword eşleşmesi (7/7 bit doğru)
    bit_match = (pred_bits == gt_bits).all(dim=-1) & slot_labels.bool()
    full_acc = bit_match.sum().item() / max(slot_labels.bool().sum().item(), 1)

    out = {
        "id_bit_accuracy": bit_acc,
        "id_full_codeword_accuracy": full_acc,
    }

    if db_signatures is not None and slot_labels.bool().any():
        sim = barcode_latent @ db_signatures.T               # (B, 6, n_db)
        top1 = sim.argmax(dim=-1)
        # Ground truth ID için: codeword_labels Hamming → 4-bit → int
        # Şimdilik basit: latent cosine ile self-similarity
        out["id_cosine_top1_max_sim"] = sim.max(dim=-1).values[
            slot_labels.bool()
        ].mean().item()

    return out


# ── Master eval loop ────────────────────────────────────


def evaluate(
    model: torch.nn.Module,
    loader,
    device: str = "cpu",
    return_aux: bool = False,
    skip_recognition: bool = False,
) -> dict[str, float]:
    """Loader üzerinde tüm metrikleri topla.

    D Paketi (2026-06-03): skip_recognition=True ise rec_* metrikleri hesaplanmaz.
    Recognition head λ=0 ile bypass edildiğinde rec_accuracy 0.25 random kalır;
    sentetik veride material_id ↔ data_id bağımsız random olduğu için anlamsız.
    Production raporlarında True önerilir.
    """
    model.eval()

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    with torch.no_grad():
        for batch in loader:
            csi = batch["csi"].to(device)
            uwb = batch["uwb"].to(device)
            link_geo = batch["link_geo"].to(device)
            slot_lbl = batch["labels"]["slot"].to(device)
            cw_lbl = batch["labels"]["codeword"].to(device)

            out = model(csi, uwb, link_geo, return_aux=return_aux)

            metrics = {}
            metrics.update(compute_detection_metrics(
                out["detection_logits"], slot_lbl
            ))
            if not skip_recognition:
                mat_lbl = batch["labels"]["material"].to(device)
                metrics.update(compute_recognition_metrics(
                    out["material_logits"], mat_lbl, slot_lbl
                ))
            metrics.update(compute_identification_metrics(
                out["barcode_codeword"], out["barcode_latent"],
                cw_lbl, slot_lbl
            ))

            for k, v in metrics.items():
                if k.endswith("_n_samples"):
                    continue
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1

    return {k: sums[k] / max(counts[k], 1) for k in sums}


__all__ = [
    "compute_detection_metrics", "compute_recognition_metrics",
    "compute_identification_metrics", "evaluate",
]
