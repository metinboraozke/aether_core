"""Adım 4: DWT (Ayrık Dalgacık Dönüşümü) ile gürültü temizleme.

Modül 2 — Adım 4 (README.md). PDF gerekçesi:
    İşlem yükünü uçurmadan gürültüyü frekans-zaman uzayında temizle.
    db4 (Daubechies-4) dalgacığı, 3-4 seviye decomposition,
    yüksek frekans katsayılarına soft-thresholding.

PyWavelets (pywt) kullanır. Complex CSI için real/imag bağımsız denoise.
"""

from __future__ import annotations

import numpy as np

try:
    import pywt
except ImportError:
    pywt = None


def _check_pywt():
    if pywt is None:
        raise ImportError(
            "PyWavelets (pywt) gerekli. Kur: pip install pywavelets"
        )


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Soft thresholding: sign(x) * max(|x| - thresh, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _denoise_1d(signal: np.ndarray, wavelet: str, level: int,
                threshold_mode: str = 'soft') -> np.ndarray:
    """Tek bir 1D sinyalde DWT denoise."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Sigma estimation: MAD (Median Absolute Deviation) on finest detail
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745 + 1e-12
    # Universal threshold (VisuShrink): sigma * sqrt(2*log(N))
    N = signal.size
    threshold = sigma * np.sqrt(2.0 * np.log(N + 1))

    # Tüm detail (yüksek frekans) coefficient'lere uygula (approx korunur)
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        if threshold_mode == 'soft':
            new_coeffs.append(soft_threshold(c, threshold))
        else:
            new_coeffs.append(np.where(np.abs(c) > threshold, c, 0.0))

    rec = pywt.waverec(new_coeffs, wavelet)
    # waverec bazen 1 sample fazla döner (odd-length signal)
    return rec[: signal.size]


def dwt_denoise(
    csi: np.ndarray,
    wavelet: str = 'db4',
    level: int = 3,
    threshold_mode: str = 'soft',
) -> np.ndarray:
    """CSI'ya per-link DWT denoise (real ve imag bağımsız).

    csi: shape (..., n_subc, 2)  — son axis real/imag
    döner: aynı shape
    """
    _check_pywt()
    arr = np.asarray(csi, dtype=np.float32)
    orig_shape = arr.shape
    # Reshape: (..., n_subc, 2) → (-1, n_subc, 2)
    flat = arr.reshape(-1, orig_shape[-2], orig_shape[-1])
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        for ch in range(2):  # real, imag
            out[i, :, ch] = _denoise_1d(flat[i, :, ch], wavelet, level, threshold_mode)
    return out.reshape(orig_shape).astype(np.float32)
