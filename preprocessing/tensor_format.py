"""Adım 6: Tensor Formatting + Normalize.

Modül 2 — Adım 6 (README.md). PDF gerekçesi:
    Modelin beklediği şekle çevir + [-1, 1] aralığına sıkıştır
    (CSITransformerEncoder gradyan akışı için).

Final input formatları (PyTorch model'in beklediği):
    CSI:  (B, 28, 2, 108)   ← (link, channel=real/imag, subcarrier)
    UWB:  (B, 6, 2, 32)     ← (bistatic_link, channel, tap)
"""

from __future__ import annotations

import numpy as np


def to_model_shape_csi(csi: np.ndarray) -> np.ndarray:
    """CSI: (..., n_links, n_subc, 2) → (..., n_links, 2, n_subc).

    Sionna çıktısı (28, 108, 2) → model input (28, 2, 108) — axis swap.
    """
    arr = np.asarray(csi, dtype=np.float32)
    # Son iki axis swap
    return np.swapaxes(arr, -1, -2).copy()


def to_model_shape_uwb(cir: np.ndarray) -> np.ndarray:
    """UWB: (..., n_links, n_taps, 2) → (..., n_links, 2, n_taps).

    Sionna çıktısı (6, 32, 2) → model input (6, 2, 32) — axis swap.
    """
    arr = np.asarray(cir, dtype=np.float32)
    return np.swapaxes(arr, -1, -2).copy()


def normalize_minus1_1(
    x: np.ndarray,
    per_link: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Tensörü [-1, 1] aralığına min-max normalize.

    per_link=True (default):
        Her link kendi min/max'ı ile normalize (relative scale kaybedilir
        ama train stability artar).
    per_link=False:
        Global min/max ile normalize (relative scale korunur).

    x: shape (..., n_links, ...) veya generic
       per_link mode için son 2-3 axis link ölçüm boyutu varsayılır.
    """
    arr = np.asarray(x, dtype=np.float32)
    if per_link:
        # Per-link: son 2 axis (subc/tap, channel) üzerinde min/max
        reduce_axes = tuple(range(-2, 0))
        mn = arr.min(axis=reduce_axes, keepdims=True)
        mx = arr.max(axis=reduce_axes, keepdims=True)
    else:
        mn = arr.min()
        mx = arr.max()
    rng = np.maximum(mx - mn, eps)
    return (2.0 * (arr - mn) / rng - 1.0).astype(np.float32)


def normalize_path_params_tau(
    path_params: np.ndarray,
    max_window_ns: float = 64.0,
) -> np.ndarray:
    """path_params'in tau (delay) kanalını saniyeden normalize ns'e çevir.

    Risk R3 (memory/project_decisions): tau saniye mertebesinde 10⁻⁹,
    float32 hassasiyet sınırına yakın. Modeli eğitmeden önce normalize:
        tau_normalized = tau_seconds * 1e9 / max_window_ns   ∈ [0, ~1]

    path_params: shape (..., 20, 4) — kanal: [real(a), imag(a), tau, valid]
    Sadece kanal 2 (tau) etkilenir.
    """
    arr = np.asarray(path_params, dtype=np.float32).copy()
    arr[..., 2] = arr[..., 2] * 1e9 / max_window_ns
    return arr
