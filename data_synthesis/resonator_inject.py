"""Sentetik Rezonatör ID — Hamming(7,4) + S(f) Lorentzian enjeksiyonu.

Modül 1 — Adım 3 (README.md). REVİZE 2026-05-14:
    4-bit data ID (0-15) → Hamming(7,4) encode → 7-bit codeword
    7 spektral bant, codeword[i]=1 olan bantta Lorentzian çentik

Formül:
    S(f) = ∏_{i=1..7} [ 1 - bit_i · A_i / (1 + Q_i² · (f/f_i - f_i/f)²) ]

    f_i = bant merkezi (UWB BW = 500 MHz, merkez 6.5 GHz, 7 eşit bant)
    Q_i ∈ [100, 150] stochastic — karbon-katkılı PLA dielektrik blok
    A_i ∈ [0.6, 0.9] stochastic — sönümleme derinliği
"""

from __future__ import annotations

import numpy as np

# ── Hamming(7,4) standart parity check matris ────────────────
# Encoding: codeword = (G @ data_4bit) mod 2
# Sıralama: [p1, p2, d1, p3, d2, d3, d4]
HAMMING_G = np.array(
    [
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.uint8,
)

# Parity check matris (decode için)
HAMMING_H = np.array(
    [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)

# Syndrome → bit pozisyonu lookup (1-based, sırayla bit 1..7)
# syndrome (s2 s1 s0 binary) = bit pozisyonu
_SYNDROME_TO_POS = {
    (0, 0, 0): None,  # hata yok
    (1, 0, 0): 0,
    (0, 1, 0): 1,
    (1, 1, 0): 2,
    (0, 0, 1): 3,
    (1, 0, 1): 4,
    (0, 1, 1): 5,
    (1, 1, 1): 6,
}


def hamming_encode_7_4(data_4bit: np.ndarray) -> np.ndarray:
    """4-bit data → 7-bit codeword (Hamming(7,4)).

    data_4bit: shape (..., 4), values in {0, 1}
    döner: shape (..., 7)
    """
    data = np.asarray(data_4bit, dtype=np.uint8)
    return np.mod(data @ HAMMING_G.T, 2).astype(np.uint8)


def hamming_decode_7_4(codeword_7bit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """7-bit codeword → (4-bit data, error_corrected_flag).

    Tek-bit hatayı düzeltir. Çift-bit hatasını her zaman saptayamaz.
    """
    codeword = np.asarray(codeword_7bit, dtype=np.uint8).copy()
    syndrome = np.mod(codeword @ HAMMING_H.T, 2)  # (..., 3)
    if codeword.ndim == 1:
        pos = _SYNDROME_TO_POS[tuple(int(s) for s in syndrome)]
        if pos is not None:
            codeword[pos] ^= 1
        # data bitleri 7-bit'in [2, 4, 5, 6] indexlerinde (sıralama p1 p2 d1 p3 d2 d3 d4)
        data = codeword[[2, 4, 5, 6]]
        return data, np.array(pos is not None)
    # Vektörel (batch) hali
    out = codeword.copy()
    corrected = np.zeros(codeword.shape[:-1], dtype=bool)
    flat_syn = syndrome.reshape(-1, 3)
    flat_out = out.reshape(-1, 7)
    for i, s in enumerate(flat_syn):
        pos = _SYNDROME_TO_POS[tuple(int(x) for x in s)]
        if pos is not None:
            flat_out[i, pos] ^= 1
            corrected.reshape(-1)[i] = True
    data = out[..., [2, 4, 5, 6]]
    return data, corrected


def int_to_data_bits(ids: np.ndarray) -> np.ndarray:
    """0-15 arası tamsayı ID → 4-bit binary vektör (MSB first).

    ids: shape (...), uint8
    döner: shape (..., 4)
    """
    ids = np.asarray(ids, dtype=np.uint8)
    out = np.unpackbits(ids[..., None], axis=-1)[..., -4:]
    return out


def data_bits_to_int(data_bits: np.ndarray) -> np.ndarray:
    """4-bit binary vektör → 0-15 tamsayı."""
    data = np.asarray(data_bits, dtype=np.uint8)
    weights = np.array([8, 4, 2, 1], dtype=np.uint8)
    return np.sum(data * weights, axis=-1).astype(np.uint8)


# ── Lorentzian çentik enjeksiyonu ────────────────────────────


def _lorentzian_notch(
    freqs_hz: np.ndarray,
    f0: float,
    A: float,
    Q: float,
) -> np.ndarray:
    """Tek bir Lorentzian çentik şekli.

    Çıkış: 1 - A / (1 + Q² · (f/f0 - f0/f)²)
    Değer 0..1 arasında; çentik merkezde minimum.
    """
    ratio = freqs_hz / f0 - f0 / freqs_hz
    return 1.0 - A / (1.0 + (Q * ratio) ** 2)


def codeword_to_spectral_envelope(
    codeword_7bit: np.ndarray,
    n_freq_bins: int,
    band_center_hz: float = 6.5e9,
    band_width_hz: float = 500e6,
    q_range: tuple[float, float] = (100.0, 150.0),
    amplitude_range: tuple[float, float] = (0.6, 0.9),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """7-bit codeword → çentikler enjekte edilmiş spektral zarf [n_freq_bins].

    Spektrum band_width_hz aralığında 7 eşit alt-banda bölünür; codeword[i]=1
    olan bantta Lorentzian çentik üretilir, codeword[i]=0 olan bantta düz 1.0.

    Q ve A her bant için stochastic atanır (üretim çeşitliliği için).
    """
    if rng is None:
        rng = np.random.default_rng()

    f_lo = band_center_hz - band_width_hz / 2.0
    f_hi = band_center_hz + band_width_hz / 2.0
    freqs = np.linspace(f_lo, f_hi, n_freq_bins, dtype=np.float64)

    n_bands = len(codeword_7bit)
    band_edges = np.linspace(f_lo, f_hi, n_bands + 1)
    band_centers = 0.5 * (band_edges[:-1] + band_edges[1:])

    envelope = np.ones(n_freq_bins, dtype=np.float64)
    for i, bit in enumerate(codeword_7bit):
        if bit == 0:
            continue
        Q = float(rng.uniform(*q_range))
        A = float(rng.uniform(*amplitude_range))
        envelope *= _lorentzian_notch(freqs, band_centers[i], A, Q)

    return envelope


def inject_resonator_into_cir(
    cir_complex: np.ndarray,
    codeword_7bit: np.ndarray,
    sample_rate_hz: float,
    band_center_hz: float = 6.5e9,
    band_width_hz: float = 500e6,
    q_range: tuple[float, float] = (100.0, 150.0),
    amplitude_range: tuple[float, float] = (0.6, 0.9),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """CIR (kompleks zaman domeni) → CFR'da çentik enjekte → CIR geri.

    cir_complex: shape (..., n_taps), complex64/128
    codeword_7bit: shape (7,), uint8
    sample_rate_hz: CIR örnekleme hızı (örn. 500e6 → 2 ns tap)

    Akış: CIR → FFT (CFR) → spectral envelope ile çarp → IFFT → CIR'.
    """
    n_taps = cir_complex.shape[-1]
    cfr = np.fft.fftshift(np.fft.fft(cir_complex, axis=-1), axes=-1)

    envelope = codeword_to_spectral_envelope(
        codeword_7bit=codeword_7bit,
        n_freq_bins=n_taps,
        band_center_hz=band_center_hz,
        band_width_hz=band_width_hz,
        q_range=q_range,
        amplitude_range=amplitude_range,
        rng=rng,
    )

    cfr_modulated = cfr * envelope.astype(cfr.dtype)
    cir_modulated = np.fft.ifft(np.fft.ifftshift(cfr_modulated, axes=-1), axis=-1)
    return cir_modulated.astype(cir_complex.dtype)


# ── Yardımcı: kompleks → real/imag tensör ────────────────────


def complex_to_realimag(arr: np.ndarray) -> np.ndarray:
    """(..., n_taps) complex → (..., n_taps, 2) float."""
    return np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)
