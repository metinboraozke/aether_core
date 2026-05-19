"""Multi-Sensor Timestamp Alignment (Sync Buffer).

REVİZE 2026-05-15: Sensör senkronizasyonu için ring buffer + tolerance
window matching. Inference path'inde 8 ESP32 + 4 UWB master cihazından
gelen async paketleri timestamp-aligned tensora dönüştürür.

Gerekçe (sim2real bridge):
    Gerçek ESP32 ~100-1000 Hz, DW1000/DW3000 ~100-200 Hz. Bağımsız
    çalışan cihazlar farklı timestamp'larda paket atar. Cross-Attention
    eş zamanlı snapshot bekler. Bu modül her master_tick'te (50 ms)
    ±sync_tolerance içindeki en yakın sample'ları seçip senkronize
    {wifi: [28,108,2], uwb: [6,32,2]} frame oluşturur.

Sionna sentetik veriye etkisi YOK — sentetik tarafta her sample zaten
"ideal anlık snapshot".

Kullanım (backend/inference_server.py içinde):
    sync = MultiSensorSyncBuffer(wifi_link_count=28, uwb_link_count=6,
                                  sync_tolerance_ms=5, master_tick_ms=50)
    # WS endpoint'inden paket gelince:
    sync.push_wifi(link_id, t_ns, csi_payload)
    sync.push_uwb(link_id, t_ns, cir_payload)
    # Master tick (50 ms timer'da):
    frame = sync.aligned_frame(target_t_ns=now_ns)
    if frame is not None:
        model.predict(**frame)
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


# =====================================================================
# TEK SENSÖR RING BUFFER
# =====================================================================


class SensorRingBuffer:
    """Tek bir sensör (link veya bistatic) için timestamp'li ring buffer.

    Her entry: (t_capture_ns, payload). Sorted insertion ya da push +
    binary search nearest. Şimdilik basit deque ile linear search
    (capacity ≤ 200, hızı kabul edilebilir).
    """

    def __init__(self, capacity: int = 200, payload_dtype: Any = np.complex64):
        self.capacity = int(capacity)
        self.payload_dtype = payload_dtype
        # deque: hızlı append + len + sıralı iteration (push timestamp'lı)
        self._timestamps: deque[int] = deque(maxlen=self.capacity)
        self._payloads: deque[np.ndarray] = deque(maxlen=self.capacity)

    def push(self, t_capture_ns: int, payload: np.ndarray) -> None:
        """Yeni paket ekle. ts artar varsayımı (sensor monoton)."""
        self._timestamps.append(int(t_capture_ns))
        self._payloads.append(payload)

    def find_nearest(
        self,
        target_t_ns: int,
        tolerance_ns: int,
    ) -> np.ndarray | None:
        """target_t_ns'e en yakın paket, ±tolerance içindeyse döner.

        Yoksa None (tick atlanır). Linear search — capacity küçük.
        """
        if not self._timestamps:
            return None
        best_idx, best_diff = -1, tolerance_ns + 1
        for i, ts in enumerate(self._timestamps):
            d = abs(ts - target_t_ns)
            if d < best_diff:
                best_diff = d
                best_idx = i
        if best_idx < 0 or best_diff > tolerance_ns:
            return None
        return self._payloads[best_idx]

    def __len__(self) -> int:
        return len(self._timestamps)

    def clear(self) -> None:
        self._timestamps.clear()
        self._payloads.clear()


# =====================================================================
# MULTI-SENSOR SYNC BUFFER
# =====================================================================


class MultiSensorSyncBuffer:
    """28 WiFi + 6 UWB bistatic link için senkronize frame oluşturucu.

    Master tick (50 ms) atıldığında, her buffer'dan ±sync_tolerance içindeki
    en yakın paket alınır. Eksik link varsa o tick atılır (None döner).
    """

    def __init__(
        self,
        wifi_link_count: int = 28,
        uwb_link_count: int = 6,
        sync_tolerance_ms: int = 5,
        master_tick_ms: int = 50,
        ring_capacity: int = 200,
        wifi_payload_shape: tuple[int, ...] = (108, 2),
        uwb_payload_shape: tuple[int, ...] = (32, 2),
    ):
        self.wifi_buffers = [
            SensorRingBuffer(capacity=ring_capacity, payload_dtype=np.float32)
            for _ in range(wifi_link_count)
        ]
        self.uwb_buffers = [
            SensorRingBuffer(capacity=ring_capacity, payload_dtype=np.float32)
            for _ in range(uwb_link_count)
        ]
        self.wifi_link_count = wifi_link_count
        self.uwb_link_count = uwb_link_count
        self.sync_tolerance_ns = int(sync_tolerance_ms) * 1_000_000
        self.master_tick_ns = int(master_tick_ms) * 1_000_000
        self.wifi_payload_shape = tuple(wifi_payload_shape)
        self.uwb_payload_shape = tuple(uwb_payload_shape)

        # Telemetri
        self.dropped_ticks = 0
        self.aligned_ticks = 0

    def push_wifi(self, link_id: int, t_ns: int, csi: np.ndarray) -> None:
        """WiFi link paketini buffer'a yaz.

        link_id: 0..wifi_link_count-1
        csi: shape wifi_payload_shape (örn 108×2)
        """
        if not (0 <= link_id < self.wifi_link_count):
            raise ValueError(f"link_id {link_id} aralık dışı")
        self.wifi_buffers[link_id].push(t_ns, csi)

    def push_uwb(self, link_id: int, t_ns: int, cir: np.ndarray) -> None:
        """UWB bistatic link paketini buffer'a yaz.

        link_id: 0..uwb_link_count-1 (C(4,2)=6 link sırası)
        cir: shape uwb_payload_shape (örn 32×2)
        """
        if not (0 <= link_id < self.uwb_link_count):
            raise ValueError(f"link_id {link_id} aralık dışı")
        self.uwb_buffers[link_id].push(t_ns, cir)

    def aligned_frame(self, target_t_ns: int) -> dict[str, Any] | None:
        """Tüm sensörlerden ±tolerance içindeki sample'ları topla.

        Eksik link varsa None döner (tick atlanır). Drop telemetrisi
        self.dropped_ticks'te tutulur.
        """
        wifi_stack = []
        for buf in self.wifi_buffers:
            s = buf.find_nearest(target_t_ns, self.sync_tolerance_ns)
            if s is None:
                self.dropped_ticks += 1
                return None
            wifi_stack.append(s)

        uwb_stack = []
        for buf in self.uwb_buffers:
            s = buf.find_nearest(target_t_ns, self.sync_tolerance_ns)
            if s is None:
                self.dropped_ticks += 1
                return None
            uwb_stack.append(s)

        self.aligned_ticks += 1
        return {
            "wifi": np.stack(wifi_stack),    # (28, 108, 2)
            "uwb": np.stack(uwb_stack),      # (6, 32, 2)
            "t_master_ns": int(target_t_ns),
        }

    def telemetry(self) -> dict[str, int]:
        return {
            "aligned_ticks": self.aligned_ticks,
            "dropped_ticks": self.dropped_ticks,
            "drop_ratio": (self.dropped_ticks
                           / max(1, self.aligned_ticks + self.dropped_ticks)),
        }

    def __repr__(self) -> str:
        return (f"MultiSensorSyncBuffer(wifi={self.wifi_link_count} link, "
                f"uwb={self.uwb_link_count} link, "
                f"tol={self.sync_tolerance_ns/1e6:.1f}ms, "
                f"tick={self.master_tick_ns/1e6:.0f}ms, "
                f"aligned={self.aligned_ticks}, dropped={self.dropped_ticks})")
