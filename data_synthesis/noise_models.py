"""Student CIR için gürültü ve donanım kusuru modelleri.

Modül 1 — Adım 4 (README.md). Sionna'nın ideal CIR'sini gerçek dünya
"çöplüğüne" dönüştürür. Sadece ANCHOR (student) CIR'a uygulanır;
Teacher (oracle, slot merkezli) CIR temiz kalır → KD hedefi.

Uygulanan gürültüler:
    1. Path-loss attenuation (mesafeye göre düşüş)
    2. AGC — link bazlı rastgele genlik normalizasyonu
    3. CFO — ±π aralığında lineer faz rampası
    4. WiFi narrow-band interference — rastgele tonlar
    5. Bluetooth bursts — ardışık bin'lerde dropout
    6. AWGN — termal gürültü
"""

from __future__ import annotations

import numpy as np

C_LIGHT = 2.998e8  # m/s


# ── 1. Path-loss attenuation ────────────────────────────────


def apply_path_loss(
    cir_complex: np.ndarray,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    freq_hz: float = 6.5e9,
) -> np.ndarray:
    """Free-space path loss (Friis): A = (λ / 4πd)².

    cir_complex: shape (..., n_taps), complex
    tx_pos, rx_pos: shape (3,), m
    """
    d = float(np.linalg.norm(np.asarray(tx_pos) - np.asarray(rx_pos)))
    d = max(d, 1e-3)  # numerical guard
    wavelength = C_LIGHT / freq_hz
    amplitude_factor = wavelength / (4.0 * np.pi * d)
    return cir_complex * amplitude_factor


# ── 2. AGC — link bazlı genlik normalizasyonu ───────────────


def apply_agc(
    cir_complex: np.ndarray,
    gain_db_range: tuple[float, float] = (-6.0, 6.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Her bağımsız link için RMS=1'e normalize + rastgele dB kazanç.

    Sionna RT amplitudeleri çok küçük (10^-3 - 10^-4); bunu gerçek AGC
    davranışına benzeterek önce RMS=1'e çekeriz, sonra ±6 dB stochastic
    kazanç uygularız. Aksi halde sonraki noise katmanları (WiFi interferer,
    AWGN) sinyali tamamen ezer.

    cir_complex: shape (n_links, n_taps), complex
    """
    if rng is None:
        rng = np.random.default_rng()
    n_links = cir_complex.shape[0]
    # 1) Her link için RMS normalize
    rms = np.sqrt(np.mean(np.abs(cir_complex) ** 2, axis=-1, keepdims=True))
    rms = np.maximum(rms, 1e-12)
    cir_norm = cir_complex / rms
    # 2) Stochastic ±6 dB
    gains_db = rng.uniform(*gain_db_range, size=n_links)
    gains_lin = (10.0 ** (gains_db / 20.0)).astype(cir_complex.real.dtype)
    return cir_norm * gains_lin[:, None]


# ── 3. CFO — Carrier Frequency Offset ──────────────────────


def apply_cfo(
    cir_complex: np.ndarray,
    cfo_phase_max_rad: float = np.pi,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Her link için bağımsız lineer faz rampası (CFO).

    Faz birikimi: phi(t) = 2π·Δf·t — burada Δf rastgele.
    cir_complex: shape (n_links, n_taps), complex
    """
    if rng is None:
        rng = np.random.default_rng()
    n_links, n_taps = cir_complex.shape[-2], cir_complex.shape[-1]
    end_phases = rng.uniform(-cfo_phase_max_rad, cfo_phase_max_rad, size=n_links)
    t_norm = np.linspace(0.0, 1.0, n_taps, dtype=np.float64)
    # Faz rampası: 0 → end_phase
    ramps = np.outer(end_phases, t_norm)
    rotators = np.exp(1j * ramps).astype(cir_complex.dtype)
    return cir_complex * rotators


# ── 4. WiFi dar bantlı interferans ──────────────────────────


def apply_wifi_interference(
    cfr_complex: np.ndarray,
    n_interferers_max: int = 3,
    interferer_power_range: tuple[float, float] = (0.005, 0.02),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Frekans domeninde rastgele bin'lere narrow-band interferer ekler.

    cfr_complex: shape (..., n_bins), complex (FFT alınmış halde)
    """
    if rng is None:
        rng = np.random.default_rng()
    out = cfr_complex.copy()
    n_bins = cfr_complex.shape[-1]
    n_interferers = rng.integers(0, n_interferers_max + 1)
    for _ in range(n_interferers):
        bin_idx = rng.integers(0, n_bins)
        power = rng.uniform(*interferer_power_range)
        phase = rng.uniform(0, 2 * np.pi)
        out[..., bin_idx] += power * np.exp(1j * phase).astype(cfr_complex.dtype)
    return out


# ── 5. Bluetooth bursts — ardışık bin dropout ───────────────


def apply_bluetooth_bursts(
    cfr_complex: np.ndarray,
    burst_prob: float = 0.05,
    burst_width_bins: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Frekans domeninde rastgele bin'lerden başlayarak ardışık dropout."""
    if rng is None:
        rng = np.random.default_rng()
    out = cfr_complex.copy()
    n_bins = cfr_complex.shape[-1]
    for start in range(n_bins):
        if rng.random() < burst_prob:
            end = min(start + burst_width_bins, n_bins)
            out[..., start:end] = 0.0 + 0.0j
    return out


# ── 6. Termal gürültü (AWGN) ────────────────────────────────


def apply_awgn(
    cir_complex: np.ndarray,
    snr_db: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Verili SNR'da kompleks Gaussian gürültü ekler."""
    if rng is None:
        rng = np.random.default_rng()
    sig_power = float(np.mean(np.abs(cir_complex) ** 2))
    if sig_power < 1e-30:
        return cir_complex
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    noise = (rng.standard_normal(cir_complex.shape) + 1j * rng.standard_normal(cir_complex.shape)) * sigma
    return cir_complex + noise.astype(cir_complex.dtype)


# ── Birleşik uygulayıcı: tüm pipeline ──────────────────────


def apply_all_anchor_noise(
    cir_anchor_complex: np.ndarray,
    rng: np.random.Generator | None = None,
    snr_db: float = 30.0,
    gain_db_range: tuple[float, float] = (-6.0, 6.0),
    cfo_phase_max_rad: float = np.pi,
    wifi_interferer_max: int = 3,
    bt_burst_prob: float = 0.05,
) -> np.ndarray:
    """Anchor (student) CIR'a tüm gürültüleri sırayla uygular.

    Sıra: AGC → CFO → (FFT → WiFi+BT → IFFT) → AWGN.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Time-domain: AGC + CFO
    cir = apply_agc(cir_anchor_complex, gain_db_range=gain_db_range, rng=rng)
    cir = apply_cfo(cir, cfo_phase_max_rad=cfo_phase_max_rad, rng=rng)

    # Frequency-domain: WiFi interference + BT bursts
    cfr = np.fft.fftshift(np.fft.fft(cir, axis=-1), axes=-1)
    cfr = apply_wifi_interference(cfr, n_interferers_max=wifi_interferer_max, rng=rng)
    cfr = apply_bluetooth_bursts(cfr, burst_prob=bt_burst_prob, rng=rng)
    cir = np.fft.ifft(np.fft.ifftshift(cfr, axes=-1), axis=-1)

    # Final: AWGN
    cir = apply_awgn(cir, snr_db=snr_db, rng=rng)

    return cir.astype(cir_anchor_complex.dtype)
