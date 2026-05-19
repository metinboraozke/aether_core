"""Dynamic Tracking — Mikro-Doppler + Sliding Window real-time.

Modül 4 — Adım 5 (README.md). REVİZE 2026-05-15:
    - PDF "T=100 ölçüm (1 sn)" pencere boyu.
    - PDF "gecikme < 200 ms" hedefi.
    - ÇELİŞKİYİ ÇÖZÜM: SLIDING WINDOW.

Akış:
    RING BUFFER (continuous, 1 sn = 100 örnek):
        [t-99, t-98, ..., t-1, t]   FIFO

    INFERENCE TICK (her master_tick_ms = 50 ms):
        Son 100 örnek = ring buffer kopyası (kayan pencere)
        → DC sub + FFT (Doppler spektrogramı)
        → Presence + Class + Velocity tahmin

    COLD START: ilk 100 örnek dolana kadar tahmin "WARMING_UP".
    NET LATENCY: 50 ms (yeni örnek geldikten sonra tahmin çıkışına kadar).

Çıktılar:
    - Presence: hareket var mı? (BCE Loss)
    - Class: İnsan / Forklift / Statik (Cross-Entropy)
    - Velocity: vx, vy, vz (MSE Loss)
"""

from __future__ import annotations

from typing import Any

import numpy as np


# =====================================================================
# RING BUFFER (numpy-only, framework-agnostic)
# =====================================================================


class DopplerRingBuffer:
    """Tek CSI stream için 100-örnekli ring buffer (FIFO).

    Pre-allocated numpy array; push O(1). Inference tick'inde son N
    örneği kopya olarak döndürür (chronological order).
    """

    def __init__(
        self,
        window_size: int = 100,
        csi_shape: tuple[int, ...] = (28, 108, 2),
        dtype: np.dtype = np.float32,
    ):
        self.window_size = int(window_size)
        self.csi_shape = tuple(csi_shape)
        self.buffer = np.zeros((self.window_size, *self.csi_shape), dtype=dtype)
        self.write_idx = 0
        self.fill_count = 0

    def push(self, csi_sample: np.ndarray) -> None:
        """Yeni CSI örneği push et. Shape: csi_shape ile uyumlu."""
        if csi_sample.shape != self.csi_shape:
            raise ValueError(
                f"csi_sample shape {csi_sample.shape} != beklenen {self.csi_shape}"
            )
        self.buffer[self.write_idx] = csi_sample
        self.write_idx = (self.write_idx + 1) % self.window_size
        if self.fill_count < self.window_size:
            self.fill_count += 1

    def is_ready(self) -> bool:
        """Buffer 100 örnek doldu mu? (cold start kontrolü)"""
        return self.fill_count >= self.window_size

    def fill_ratio(self) -> float:
        """0.0-1.0 arası doluluk oranı (warming up göstergesi)."""
        return self.fill_count / self.window_size

    def get_window(self) -> np.ndarray:
        """Son window_size örneği kronolojik sırada döndür.

        Shape: (window_size, *csi_shape)
        Buffer dolu değilse ValueError. Önce is_ready() ile kontrol et.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Buffer henüz dolu değil ({self.fill_count}/{self.window_size})"
            )
        # write_idx en eski örneğin indeksi (overwrite edilen ilk slot)
        return np.concatenate([
            self.buffer[self.write_idx:],
            self.buffer[:self.write_idx],
        ], axis=0)

    def reset(self) -> None:
        self.buffer.fill(0)
        self.write_idx = 0
        self.fill_count = 0


# =====================================================================
# SLIDING TRACKER (PyTorch model — Modül 4 implementasyonu ✓ 2026-05-15)
# =====================================================================

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore


class SlidingTracker(nn.Module):
    """Modül 4 Adım 5 — Mikro-Doppler analizi + sliding window.

    Pipeline (her inference tick):
      Input: csi_window (B, T=100, 28, 108, 2)  ← DopplerRingBuffer.get_window()
      1) DC subtract zaman ortalaması
      2) FFT zaman ekseninde (real+imag → complex → FFT → magnitude)
      3) 2D CNN: input (B, 28, T, 108) — link × Doppler × subcarrier
      4) Global avg pool → latent
      5) 3 head:
         presence:  BCE   (1)
         class:     CE    (n_motion_classes: insan/forklift/statik)
         velocity:  MSE   (vx, vy, vz)
    """

    def __init__(
        self,
        window_size: int = 100,
        n_links: int = 28,
        n_subcarriers: int = 108,
        n_motion_classes: int = 3,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.n_links = n_links
        self.n_subc = n_subcarriers
        self.n_motion = n_motion_classes
        self.latent_dim = latent_dim

        # 2D CNN — input shape: (B, n_links=28, T=100, n_subc=108)
        # n_links channel olarak işlenir
        self.cnn = nn.Sequential(
            nn.Conv2d(n_links, 64, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                       # (B, 64, 50, 54)
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                       # (B, 128, 25, 27)
            nn.Conv2d(128, latent_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(latent_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                          # (B, 64, 1, 1)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)

        # 3 multi-task head
        self.presence_head = nn.Linear(latent_dim, 1)
        self.class_head = nn.Linear(latent_dim, n_motion_classes)
        self.velocity_head = nn.Linear(latent_dim, 3)

    def _doppler_spectrogram(self, csi_window: torch.Tensor) -> torch.Tensor:
        """csi_window: (B, T, 28, 108, 2) → Doppler magnitude (B, 28, T, 108)

        1) DC sub (zaman ortalaması çıkar)
        2) Complex (real + j*imag)
        3) FFT zaman ekseninde
        4) Magnitude
        5) Axis permute: (B, T, L, S) → (B, L, T, S)
        """
        # DC sub
        x = csi_window - csi_window.mean(dim=1, keepdim=True)
        # Complex (T axis = 1)
        c = torch.complex(x[..., 0], x[..., 1])                    # (B, T, L, S)
        # Doppler FFT (zaman ekseninde)
        fft = torch.fft.fft(c, dim=1)
        mag = torch.abs(fft)                                        # (B, T, L, S)
        # Permute: (B, L, T, S)
        return mag.permute(0, 2, 1, 3).contiguous()

    def forward(self, csi_window: torch.Tensor) -> dict[str, torch.Tensor]:
        # csi_window: (B, T=100, 28, 108, 2)
        spec = self._doppler_spectrogram(csi_window)               # (B, 28, T, 108)
        feat = self.cnn(spec)                                       # (B, latent_dim, 1, 1)
        latent = self.dropout(self.flatten(feat))                   # (B, latent_dim)
        return {
            "presence": self.presence_head(latent),                 # (B, 1)
            "class": self.class_head(latent),                        # (B, n_motion)
            "velocity": self.velocity_head(latent),                  # (B, 3)
        }


def tracker_loss(
    out: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    alpha_class: float = 0.5,
    alpha_velocity: float = 0.3,
) -> torch.Tensor:
    """Multi-task tracker loss.

    out:     {'presence', 'class', 'velocity'}
    targets: {'presence': (B,), 'class': (B,), 'velocity': (B, 3)}
    """
    L_pres = F.binary_cross_entropy_with_logits(
        out["presence"].squeeze(-1), targets["presence"].float()
    )
    L_cls = F.cross_entropy(out["class"], targets["class"].long())
    L_vel = F.mse_loss(out["velocity"], targets["velocity"].float())
    return L_pres + alpha_class * L_cls + alpha_velocity * L_vel


# =====================================================================
# TRACKING ENGINE (real-time wrapper, backend'den çağrılır)
# =====================================================================


class TrackingEngine:
    """Real-time tracker wrapper.

    backend/inference_server.py'den her master_tick_ms (50 ms) çağrılır.
    İçinde ring buffer + (sonradan) SlidingTracker model var.

    Cold start: ilk 100 push'ta WARMING_UP döner, tahmin yok.
    Hazır olduğunda her tick'te tahmin döner.
    """

    def __init__(
        self,
        model: Any = None,         # SlidingTracker instance (None = stub mode)
        window_size: int = 100,
        csi_shape: tuple[int, ...] = (28, 108, 2),
        master_tick_ms: int = 50,
    ):
        self.buffer = DopplerRingBuffer(window_size=window_size, csi_shape=csi_shape)
        self.model = model
        self.master_tick_ms = int(master_tick_ms)
        self.last_pred: dict[str, Any] | None = None

    def step(self, csi_sample: np.ndarray) -> dict[str, Any]:
        """Tek master tick. Yeni örneği push et, hazırsa tahmin yap.

        csi_sample: shape (28, 108, 2)
        Döner:
            {'status': 'WARMING_UP', 'progress': float}
            VEYA
            {'status': 'OK', 'presence': ..., 'class': ..., 'velocity': ...}
        """
        self.buffer.push(csi_sample)

        if not self.buffer.is_ready():
            return {
                "status": "WARMING_UP",
                "progress": self.buffer.fill_ratio(),
            }

        window = self.buffer.get_window()  # (100, 28, 108, 2)

        if self.model is None:
            # Stub mode — model henüz implement edilmedi
            self.last_pred = {
                "status": "OK_STUB",
                "presence": 0.0,
                "class": 0,
                "velocity": [0.0, 0.0, 0.0],
            }
            return self.last_pred

        # Gerçek inference (Modül 4 implementasyonu sonrası)
        try:
            import torch  # type: ignore
            with torch.no_grad():
                batch = torch.from_numpy(window).unsqueeze(0)  # (1, 100, ...)
                out = self.model(batch)
            self.last_pred = {
                "status": "OK",
                "presence": float(out["presence"].sigmoid().item()),
                "class": int(out["class"].argmax(dim=-1).item()),
                "velocity": out["velocity"][0].tolist(),
            }
        except Exception as e:
            self.last_pred = {"status": "ERROR", "error": str(e)}

        return self.last_pred

    def reset(self) -> None:
        self.buffer.reset()
        self.last_pred = None

    def __repr__(self) -> str:
        ms_state = "OK" if self.buffer.is_ready() else f"WARMING ({int(self.buffer.fill_ratio()*100)}%)"
        return f"TrackingEngine(window={self.buffer.window_size}, tick={self.master_tick_ms}ms, state={ms_state})"
