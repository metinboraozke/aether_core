"""Adım 3: Diferansiyel Kanal Çıkarımı (CSI_Δ) + Adaptive Baseline.

Modül 2 — Adım 3 (README.md). Hiyerarşinin "Aether Core" ruhu.

    CSI_Δ = CSI_anlık − CSI_ref

Statik duvarlar yerine sadece raftaki "yeni" nesneye bakmamızı sağlar.

REVİZE 2026-05-15: AdaptiveBaseline class — B + A combo strategy.

    Strategy A (sürekli yumuşak EMA):
        Her frame'de çok düşük α ile reference güncellenir.
        α_continuous = 1e-4 → temperature/nem drift'ini emer ama hareketli
        nesneye karşı dayanıklı.

    Strategy B (motion-gated hızlı EMA):
        Detection head "tüm slot boş" derse ve quiet_frames yeterince
        doluysa, daha yüksek α ile reference hızlı toparlanır.
        α_quiet = 1e-2 → boş oda fırsatlarında daha agresif güncelleme.

    Combo: A her zaman aktif (drift için bekçi), B sadece "quiet" zamanlarda
    devreye girer. İki strateji aynı reference üzerinde sırayla uygulanır.

Inference'ta:
    Server boot → AdaptiveBaseline().load_reference(path) veya init=None
    Her tick:
        baseline.update(current_csi, last_detection_mask)
        csi_delta = baseline.compute_delta(current_csi)
    Periyodik:
        baseline.save_reference(path)   # disk persistance
"""

from __future__ import annotations

import numpy as np


class AdaptiveBaseline:
    """Drift'e dayanıklı CSI referans yönetimi.

    Inference path'inde kullanılır. Sentetik veri üretiminde (Sionna)
    gerek yok çünkü orada her sample içinde boş oda referansı zaten
    aynı sample'da hesaplanır (drift yok).
    """

    def __init__(
        self,
        alpha_continuous: float = 1e-4,
        alpha_quiet: float = 1e-2,
        quiet_frames_threshold: int = 100,
    ):
        self.ref: np.ndarray | None = None
        self.alpha_continuous = float(alpha_continuous)
        self.alpha_quiet = float(alpha_quiet)
        self.quiet_frames_threshold = int(quiet_frames_threshold)
        self.quiet_count = 0

    def update(
        self,
        current_csi: np.ndarray,
        detection_mask: np.ndarray | None = None,
    ) -> None:
        """Reference'ı güncelle.

        current_csi: shape (28, 108, 2) veya (28, 108) complex / float
        detection_mask: shape (6,) — 1 = slot dolu, 0 = boş. None ise
                        motion-gated güncelleme atlanır.
        """
        if self.ref is None:
            self.ref = current_csi.copy()
            return

        # Strategy A: her zaman çalışır (drift için bekçi)
        self.ref = (1.0 - self.alpha_continuous) * self.ref \
                   + self.alpha_continuous * current_csi

        # Strategy B: motion yoksa daha agresif EMA
        if detection_mask is not None:
            if int(np.asarray(detection_mask).sum()) == 0:
                self.quiet_count += 1
                if self.quiet_count >= self.quiet_frames_threshold:
                    self.ref = (1.0 - self.alpha_quiet) * self.ref \
                               + self.alpha_quiet * current_csi
            else:
                self.quiet_count = 0

    def compute_delta(self, current_csi: np.ndarray) -> np.ndarray:
        """CSI_Δ = current − ref. Reference yoksa current ile init eder."""
        if self.ref is None:
            self.ref = current_csi.copy()
        return current_csi - self.ref

    def reset(self, csi: np.ndarray | None = None) -> None:
        """Reference'ı sıfırla veya verilen CSI ile başlat."""
        self.ref = None if csi is None else csi.copy()
        self.quiet_count = 0

    def save_reference(self, path: str) -> None:
        """Disk'e kaydet (server reboot sonrası tekrar yüklemek için)."""
        if self.ref is None:
            raise ValueError("Reference henüz init edilmedi")
        np.save(path, self.ref)

    def load_reference(self, path: str) -> None:
        """Disk'ten yükle. Yoksa init=None kalır."""
        import os
        if os.path.exists(path):
            self.ref = np.load(path)
            self.quiet_count = 0

    def __repr__(self) -> str:
        ref_shape = self.ref.shape if self.ref is not None else "None"
        return (f"AdaptiveBaseline(ref_shape={ref_shape}, "
                f"alpha_continuous={self.alpha_continuous}, "
                f"alpha_quiet={self.alpha_quiet}, "
                f"quiet_count={self.quiet_count})")


# ── Stateless yardımcı (eski PDF API'si için) ──────────────


def compute_delta(csi: np.ndarray, csi_ref: np.ndarray) -> np.ndarray:
    """CSI_Δ = CSI_anlık − CSI_ref. Tek seferlik kullanım için."""
    return csi - csi_ref


def update_baseline(
    csi_ref: np.ndarray,
    recent_window: np.ndarray,
    alpha: float = 1e-4,
) -> np.ndarray:
    """Stateless EMA güncellemesi. AdaptiveBaseline kullanmadan önce
    eski PDF API'si ile uyum için."""
    return (1.0 - alpha) * csi_ref + alpha * recent_window.mean(axis=0)
