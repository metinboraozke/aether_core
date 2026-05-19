"""Adım 1: Ham Faz Sanitizasyonu (Phase Sanitization).

Modül 2 — Adım 1 (README.md). PDF gerekçesi:
    2.4 GHz bandında ESP32'den gelen faz verisi donanımsal saat kaymaları
    (CFO) yüzünden "çöp" haldedir. Phase unwrapping + linear phase removal
    + static offset correction ile temizlenir.

Pipeline:
    csi (real/imag) → complex → unwrap angle → linear fit & remove
                     → static offset (csi_ref farkı) → magnitude * exp(1j * phase_clean)
                     → tekrar real/imag
"""

from __future__ import annotations

import numpy as np


# ── Yardımcı: real/imag <-> complex dönüşümleri ─────────────


def _to_complex(csi_ri: np.ndarray) -> np.ndarray:
    """(..., 2) → (...) complex64. Son axis real/imag."""
    return (csi_ri[..., 0] + 1j * csi_ri[..., 1]).astype(np.complex64)


def _to_realimag(csi_c: np.ndarray) -> np.ndarray:
    """(...) complex → (..., 2) float32. Son axis real/imag."""
    return np.stack([csi_c.real, csi_c.imag], axis=-1).astype(np.float32)


# ── Adım 1.1: Phase Unwrapping ──────────────────────────────


def unwrap_phase(csi: np.ndarray) -> np.ndarray:
    """Subcarrier ekseninde -π ↔ +π sıçramalarını sürekli çizgiye çevir.

    csi: shape (..., n_subcarriers, 2)  real/imag
    döner: aynı shape, ama faz unwrap edilmiş (magnitude korunur)
    """
    c = _to_complex(csi)                # (..., n_subc) complex
    mag = np.abs(c)
    phase = np.angle(c)
    phase_unwrapped = np.unwrap(phase, axis=-1)
    c_new = mag * np.exp(1j * phase_unwrapped)
    return _to_realimag(c_new)


# ── Adım 1.2: Linear Phase Removal (CFO compensation) ──────


def remove_linear_phase(csi: np.ndarray) -> np.ndarray:
    """Subcarrier index'e karşı lineer faz eğimini (CFO) least-squares ile çıkar.

    Her bağımsız link için:
        phi(k) ≈ a*k + b   → çıkar
        Yeni faz: phi(k) - (â*k + b̂)
    Magnitude değişmez.

    csi: shape (..., n_subc, 2)
    """
    c = _to_complex(csi)
    mag = np.abs(c)
    phase = np.unwrap(np.angle(c), axis=-1)
    n_subc = phase.shape[-1]
    k = np.arange(n_subc, dtype=np.float64)
    # Per-link linear fit: ax + b
    # X = [k, 1], coef = (X^T X)^-1 X^T y
    # Vectorized: en küçük kare doğrudan formülle
    k_mean = k.mean()
    y_mean = phase.mean(axis=-1, keepdims=True)
    num = ((k - k_mean) * (phase - y_mean)).sum(axis=-1, keepdims=True)
    den = ((k - k_mean) ** 2).sum()
    a = num / den                       # slope (..., 1)
    b = y_mean - a * k_mean             # intercept
    phase_lin = a * k + b               # broadcast (..., n_subc)
    phase_clean = phase - phase_lin
    c_new = mag * np.exp(1j * phase_clean)
    return _to_realimag(c_new)


# ── Adım 1.3: Static Phase Offset Correction ──────────────


def apply_static_offset(
    csi: np.ndarray,
    csi_ref: np.ndarray,
) -> np.ndarray:
    """Linkler arası sabit faz offset'ini csi_ref'e göre normalize et.

    Her link için referansın ortalama faz offset'ini current'tan çıkar.
    Magnitude değişmez.

    csi:     (..., n_links, n_subc, 2)
    csi_ref: (n_links, n_subc, 2)   veya broadcast-uyumlu
    """
    c_now = _to_complex(csi)
    c_ref = _to_complex(csi_ref)

    # Referansın ortalama faz offset'i (per link)
    ref_phase_mean = np.angle(c_ref).mean(axis=-1, keepdims=True)   # (..., n_links, 1)

    mag = np.abs(c_now)
    phase = np.angle(c_now)
    phase_corrected = phase - ref_phase_mean
    c_new = mag * np.exp(1j * phase_corrected)
    return _to_realimag(c_new)
