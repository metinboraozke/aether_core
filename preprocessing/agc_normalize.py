"""Adım 2: Genlik yönetimi + AGC telafisi + outlier clip + dB.

Modül 2 — Adım 2 (README.md). PDF gerekçesi:
    Donanımın AGC'si sinyalin gerçek attenuation bilgisini bozar.
    Per-link RMS normalize → linkler arası güç dengesizliği giderilir.
    μ±3σ clip → spike'lar tıraşlanır, gradyan patlaması önlenir.
    dB scale → küçük dielektrik değişimleri belirginleşir.

NOT: noise_models.apply_agc'den FARKLI — burası deterministic
(RNG yok). Inference path'inde kullanılır.
"""

from __future__ import annotations

import numpy as np


def agc_normalize(csi: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Her link için RMS=1'e normalize (per-sample, per-link).

    csi:    shape (..., n_links, n_subc, 2)  veya (..., n_subc, 2)
    döner:  aynı shape, RMS norm
    """
    arr = np.asarray(csi, dtype=np.float32)
    # RMS, link bazında (son 2 axis: n_subc, 2)
    sq = arr ** 2
    rms = np.sqrt(np.mean(sq, axis=(-2, -1), keepdims=True))
    rms = np.maximum(rms, eps)
    return (arr / rms).astype(np.float32)


def clip_outliers(csi: np.ndarray, k: float = 3.0) -> np.ndarray:
    """μ ± k·σ dışındaki değerleri kırp (per-link).

    csi:    shape (..., n_links, n_subc, 2)
    k:      sigma çarpanı (default 3)
    döner:  aynı shape
    """
    arr = np.asarray(csi, dtype=np.float32)
    mu = arr.mean(axis=(-2, -1), keepdims=True)
    sigma = arr.std(axis=(-2, -1), keepdims=True)
    lo = mu - k * sigma
    hi = mu + k * sigma
    return np.clip(arr, lo, hi).astype(np.float32)


def to_db(csi: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Genliği dB ölçeğine çevir (faz korunur, real/imag tekrar yazılır).

    csi: shape (..., n_subc, 2)
    döner: aynı shape ama magnitude dB'de.

    Math:
        mag_db = 20 * log10(|csi| + eps)
        new_real = mag_db * cos(phase)
        new_imag = mag_db * sin(phase)

    NOT: dB negatif değerler de olur (zayıf sinyaller); model bunu öğrenir.
    """
    arr = np.asarray(csi, dtype=np.float32)
    c = arr[..., 0] + 1j * arr[..., 1]
    mag = np.abs(c)
    phase = np.angle(c)
    mag_db = 20.0 * np.log10(mag + eps)
    return np.stack([
        mag_db * np.cos(phase),
        mag_db * np.sin(phase),
    ], axis=-1).astype(np.float32)
