"""Adım 5: UWB CIR Sinc-Resampling + First-Path Alignment + Windowing.

Modül 2 — Adım 5 (README.md). PDF gerekçesi:
    - Sinc resampling: ham CIR'ı 2 ns standart grid'ine oturt.
    - First-path alignment: ilk path tepesini bul, hizala.
    - Windowing: ilk 64 ns (32 tap) pencereye kırp.

Sionna sentetik veri için MOST FUNCTIONS NO-OP:
    - Sionna zaten 2 ns grid'inde compute_paths yapıyor → sinc_resample atlanır
    - Sionna ilk path normalize ediliyor (normalize_delays=True) → align hafif
    - Window zaten 32 tap → no-op

Real ESP32+UWB verisi geldiğinde gerçek değer kazanır.
"""

from __future__ import annotations

import numpy as np


def sinc_resample(
    cir: np.ndarray,
    src_dt_ns: float,
    tgt_dt_ns: float = 2.0,
    n_taps_out: int = 32,
) -> np.ndarray:
    """Ham CIR'ı standart zaman grid'ine sinc interpolation ile oturt.

    cir: shape (..., n_taps_in, 2)  real/imag
    src_dt_ns: kaynak tap spacing (ns)
    tgt_dt_ns: hedef tap spacing (ns), default 2 ns
    n_taps_out: çıktı tap sayısı

    Sentetik veri için src_dt_ns == tgt_dt_ns == 2 → no-op (trim/pad).
    """
    arr = np.asarray(cir, dtype=np.float32)
    n_in = arr.shape[-2]

    if abs(src_dt_ns - tgt_dt_ns) < 1e-9 and n_in == n_taps_out:
        return arr  # no-op

    # Hedef zaman noktaları
    t_src = np.arange(n_in) * src_dt_ns
    t_tgt = np.arange(n_taps_out) * tgt_dt_ns

    orig_shape = arr.shape
    flat = arr.reshape(-1, n_in, 2)
    out = np.zeros((flat.shape[0], n_taps_out, 2), dtype=np.float32)

    # Sinc interpolation: x_out[m] = sum_n x_in[n] * sinc((t_tgt[m] - t_src[n]) / src_dt_ns)
    for i in range(flat.shape[0]):
        for ch in range(2):
            for m in range(n_taps_out):
                weights = np.sinc((t_tgt[m] - t_src) / src_dt_ns)
                out[i, m, ch] = (flat[i, :, ch] * weights).sum()

    return out.reshape(*orig_shape[:-2], n_taps_out, 2).astype(np.float32)


def align_first_path(
    cir: np.ndarray,
    threshold_ratio: float = 0.1,
) -> np.ndarray:
    """İlk path tepesini bul, CIR'ı bu noktaya göre sola kaydır.

    cir: shape (..., n_taps, 2)
    threshold_ratio: max amplitude'in bu oranı first-path eşiği

    Sentetik veride Sionna normalize_delays=True ile zaten ilk path
    yaklaşık tap 0'da → bu fonksiyon ~no-op davranır.
    """
    arr = np.asarray(cir, dtype=np.float32)
    orig_shape = arr.shape
    flat = arr.reshape(-1, orig_shape[-2], orig_shape[-1])
    out = np.zeros_like(flat)

    for i in range(flat.shape[0]):
        c = flat[i, :, 0] + 1j * flat[i, :, 1]
        mag = np.abs(c)
        if mag.max() < 1e-12:
            out[i] = flat[i]
            continue
        threshold = threshold_ratio * mag.max()
        # İlk eşik geçişi (leading edge)
        first_idx = int(np.argmax(mag > threshold))
        # Sola kaydır
        if first_idx > 0:
            out[i, :-first_idx, :] = flat[i, first_idx:, :]
        else:
            out[i] = flat[i]
    return out.reshape(orig_shape).astype(np.float32)


def window_cir(
    cir: np.ndarray,
    window_ns: float = 64.0,
    tap_dt_ns: float = 2.0,
) -> np.ndarray:
    """İlk window_ns kadar pencereyi al, gerisini at.

    cir: shape (..., n_taps, 2)
    Default: 64 ns / 2 ns = 32 tap → sentetik için no-op.
    """
    arr = np.asarray(cir, dtype=np.float32)
    n_keep = int(round(window_ns / tap_dt_ns))
    n_taps = arr.shape[-2]
    if n_keep >= n_taps:
        return arr
    return arr[..., :n_keep, :].astype(np.float32)
