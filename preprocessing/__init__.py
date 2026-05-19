"""Modül 2 — Çok Kademeli Ön İşleme Pipeline.

Bkz. README.md → "Modül 2" bölümü.

End-to-end wrapper'lar:
    preprocess_csi(...)    — WiFi CSI için 6 adımlı pipeline
    preprocess_uwb(...)    — UWB CIR için pipeline
    preprocess_path_params(...) — auxiliary path params normalize
"""

from __future__ import annotations

import numpy as np

from .phase_sanitize import unwrap_phase, remove_linear_phase, apply_static_offset
from .agc_normalize import agc_normalize, clip_outliers, to_db
from .differential_csi import AdaptiveBaseline, compute_delta, update_baseline
from .dwt_denoise import dwt_denoise, soft_threshold
from .uwb_resample import sinc_resample, align_first_path, window_cir
from .tensor_format import (
    to_model_shape_csi, to_model_shape_uwb,
    normalize_minus1_1, normalize_path_params_tau,
)


__all__ = [
    # Phase
    "unwrap_phase", "remove_linear_phase", "apply_static_offset",
    # AGC
    "agc_normalize", "clip_outliers", "to_db",
    # Differential
    "AdaptiveBaseline", "compute_delta", "update_baseline",
    # DWT
    "dwt_denoise", "soft_threshold",
    # UWB resample
    "sinc_resample", "align_first_path", "window_cir",
    # Tensor format
    "to_model_shape_csi", "to_model_shape_uwb",
    "normalize_minus1_1", "normalize_path_params_tau",
    # Pipeline wrappers
    "preprocess_csi", "preprocess_uwb", "preprocess_path_params",
]


def preprocess_csi(
    csi_raw: np.ndarray,
    baseline: AdaptiveBaseline | None = None,
    detection_mask: np.ndarray | None = None,
    apply_phase_clean: bool = True,
    apply_dwt: bool = True,
    apply_log_scale: bool = False,
    csi_ref: np.ndarray | None = None,
) -> np.ndarray:
    """WiFi CSI için end-to-end preprocessing pipeline.

    Input shape:  (..., 28, 108, 2)  — Sionna çıktısı (csivec_delta)
    Output shape: (..., 28, 2, 108)  — model input formatı, [-1, 1] normalize

    Adımlar:
      1. Phase sanitize (unwrap + linear removal + static offset)
      2. AGC (per-link RMS normalize + outlier clip)
      3. (opsiyonel) dB log-scale
      4. (opsiyonel) Adaptive baseline ile diferansiyel CSI
      5. (opsiyonel) DWT denoise
      6. Tensor format ([-1, 1] normalize + axis swap)
    """
    arr = np.asarray(csi_raw, dtype=np.float32)

    # 1. Phase sanitize
    if apply_phase_clean:
        arr = unwrap_phase(arr)
        arr = remove_linear_phase(arr)
        if csi_ref is not None:
            arr = apply_static_offset(arr, csi_ref)

    # 2. AGC + clip
    arr = agc_normalize(arr)
    arr = clip_outliers(arr, k=3.0)

    # 3. dB (opsiyonel)
    if apply_log_scale:
        arr = to_db(arr)

    # 4. Adaptive baseline (inference path için, sentetik verisi NO-OP)
    if baseline is not None:
        baseline.update(arr, detection_mask=detection_mask)
        arr = baseline.compute_delta(arr)

    # 5. DWT denoise
    if apply_dwt:
        arr = dwt_denoise(arr, wavelet='db4', level=3, threshold_mode='soft')

    # 6. Tensor format
    arr = normalize_minus1_1(arr, per_link=True)
    arr = to_model_shape_csi(arr)
    return arr


def preprocess_uwb(
    cir_raw: np.ndarray,
    apply_align: bool = True,
    apply_window: bool = True,
    src_dt_ns: float = 2.0,
    window_ns: float = 64.0,
) -> np.ndarray:
    """UWB CIR için end-to-end preprocessing pipeline.

    Input shape:  (..., 6, 32, 2)   — Sionna çıktısı
    Output shape: (..., 6, 2, 32)   — model input formatı, [-1, 1] normalize

    Sentetik veri için çoğu fonksiyon NO-OP davranışında (Sionna zaten
    2 ns grid'te + normalize delays). Gerçek ESP32 UWB verisi için aktif.
    """
    arr = np.asarray(cir_raw, dtype=np.float32)

    # 1. Sinc resample (sentetik için no-op)
    arr = sinc_resample(arr, src_dt_ns=src_dt_ns, tgt_dt_ns=2.0,
                         n_taps_out=arr.shape[-2])

    # 2. First-path alignment
    if apply_align:
        arr = align_first_path(arr, threshold_ratio=0.1)

    # 3. Windowing
    if apply_window:
        arr = window_cir(arr, window_ns=window_ns, tap_dt_ns=2.0)

    # 4. Tensor format
    arr = normalize_minus1_1(arr, per_link=True)
    arr = to_model_shape_uwb(arr)
    return arr


def preprocess_path_params(
    path_params: np.ndarray,
    max_window_ns: float = 64.0,
) -> np.ndarray:
    """Auxiliary path params için tau normalize.

    Risk R3: tau saniye (10⁻⁹) → ns/window normalize ([0, ~1]).
    Real/imag amplitude kanalları olduğu gibi kalır (zaten makul ölçek).
    Validity mask 0/1 olduğu gibi kalır.

    Input/Output shape:  (..., 20, 4)
    """
    return normalize_path_params_tau(path_params, max_window_ns=max_window_ns)
