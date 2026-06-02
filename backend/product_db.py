"""C Paketi 2026-06-03 — Spektral Barkod Ürün Veritabanı (lokal YAML).

Aether Core jüri sunum hazırlığı için: identification head'in 7-bit codeword
çıktısı → Hamming(7,4) decode → 4-bit ID → 16 ürün lookup.

Recognition head dead (rec_acc 0.25 random) → material classification için
DB lookup primer, model rec head "çift doğrulama" olarak yan tarafta.

Reuse edilen mevcut fonksiyonlar:
    - hamming_decode_7_4 (data_synthesis/resonator_inject.py L69-94)
    - data_bits_to_int   (data_synthesis/resonator_inject.py L108-112)
    - cosine_lookup      (models/heads/identification_head.py L159-169) — fallback

Yeni kod sadece glue: YAML loader + slot başına paket field builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from data_synthesis.resonator_inject import (
    hamming_decode_7_4,
    data_bits_to_int,
)


MATERIAL_PALETTE_DEFAULT = {
    "metal":   "#999999",
    "plastik": "#3498DB",
    "ahşap":   "#8B4513",
    "karton":  "#F5DEB3",
    "unknown": "#404040",
}


class ProductDB:
    """Lokal YAML tabanlı 16-ürün spektral barkod veritabanı.

    Kullanım:
        db = ProductDB()  # configs/resonator_db.yaml'dan yükler
        info = db.lookup(8)               # ID → dict
        pid, corrected = db.decode_codeword(codeword_7bit)   # logit/binary → ID
        products = db.build_packet_products(...)             # WS paket field
    """

    def __init__(self, yaml_path: str | Path = "configs/resonator_db.yaml"):
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"DB YAML yok: {self.yaml_path}")
        with open(self.yaml_path, encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

        products = self._raw.get("products", []) or []
        # ID-indexed lookup dict
        self._by_id: dict[int, dict[str, Any]] = {}
        for p in products:
            pid = int(p["id"])
            self._by_id[pid] = p

        self.material_palette = {
            **MATERIAL_PALETTE_DEFAULT,
            **(self._raw.get("material_palette", {}) or {}),
        }

    # ── Sorgu API ────────────────────────────────────────────

    def all(self) -> list[dict[str, Any]]:
        """Tüm ürünleri sıralı liste olarak döndür (HTTP endpoint için)."""
        return [self._by_id[i] for i in sorted(self._by_id.keys())]

    def lookup(self, product_id: int) -> dict[str, Any]:
        """ID → ürün dict. Bilinmiyorsa 'unknown' fallback."""
        pid = int(product_id)
        if pid in self._by_id:
            return self._by_id[pid]
        return {
            "id": pid,
            "material": "unknown",
            "name": f"Bilinmeyen ürün (ID#{pid})",
            "color_hex": self.material_palette.get("unknown", "#404040"),
            "description": "Veritabanında kayıt yok",
        }

    def color_for_material(self, material: str) -> str:
        """Material adı → fallback color_hex (DB'de ID yoksa)."""
        return self.material_palette.get(material, self.material_palette["unknown"])

    # ── Codeword decode (Hamming(7,4) + ID dönüşüm) ──────────

    def decode_codeword(
        self,
        codeword_logits_or_bits: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[int, bool]:
        """7-bit codeword (logit veya binary) → (id, error_corrected_flag).

        Input shape (7,) — tek slot, batch yok.
        Logit → sigmoid > threshold → binary; eğer zaten 0/1 ise direkt geçer.
        Hamming(7,4) decode tek bit hata düzeltir.
        """
        arr = np.asarray(codeword_logits_or_bits, dtype=np.float32).reshape(7)
        # Eğer logit gibi görünüyorsa (değerler 0/1 dışında) sigmoid threshold
        if arr.max() > 1.0 or arr.min() < 0.0:
            arr = 1.0 / (1.0 + np.exp(-arr))
        bits = (arr > threshold).astype(np.uint8)
        data_4bit, corrected_flag = hamming_decode_7_4(bits)
        pid = int(data_bits_to_int(data_4bit))
        return pid, bool(corrected_flag)

    # ── WebSocket paket field builder ────────────────────────

    def build_packet_products(
        self,
        detection_mask: np.ndarray,            # (6,) binary
        codeword_logits: np.ndarray,           # (6, 7) logits
        material_logits: np.ndarray | None = None,   # (6, 4) — opsiyonel, model rec head
    ) -> list[dict[str, Any]]:
        """Slot başına paket dict üretir (dashboard'ın okuyacağı format).

        Her slot için:
          {slot, empty?, id?, material_db?, material_model?, name?, color_hex?,
           match?, hamming_corrected?, description?}
        """
        det = np.asarray(detection_mask).astype(int).reshape(-1)
        cw = np.asarray(codeword_logits).reshape(det.size, -1)
        ml = None
        if material_logits is not None:
            ml = np.asarray(material_logits).reshape(det.size, -1)

        material_names = ["metal", "plastik", "ahşap", "karton"]
        out = []
        for slot_idx in range(det.size):
            if det[slot_idx] == 0:
                out.append({"slot": int(slot_idx), "empty": True})
                continue
            # DB lookup
            pid, hamming_corrected = self.decode_codeword(cw[slot_idx])
            info = self.lookup(pid)
            # Model rec head tahmini (varsa)
            model_mat = None
            if ml is not None:
                model_mat_idx = int(np.argmax(ml[slot_idx]))
                if 0 <= model_mat_idx < len(material_names):
                    model_mat = material_names[model_mat_idx]
            match = (model_mat is not None and model_mat == info.get("material"))
            out.append({
                "slot": int(slot_idx),
                "empty": False,
                "id": pid,
                "material_db": info.get("material", "unknown"),
                "material_model": model_mat,
                "match": bool(match),
                "name": info.get("name", "?"),
                "color_hex": info.get("color_hex",
                                       self.color_for_material(info.get("material", "unknown"))),
                "description": info.get("description", ""),
                "hamming_corrected": hamming_corrected,
            })
        return out


__all__ = ["ProductDB", "MATERIAL_PALETTE_DEFAULT"]
