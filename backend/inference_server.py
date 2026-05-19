"""InferenceEngine — Master tick loop + sync_buffer + model + tracker + broadcast.

PDF Modül 5 Adım 3+5 (README.md). REVİZE 2026-05-15.

Akış (her master_tick_ms = 50 ms, 20 Hz):
    1. sync_buffer.aligned_frame() → senkronize WiFi/UWB tensor
    2. AdaptiveBaseline.update + compute_delta (CSI drift düzeltme)
    3. preprocess_csi + preprocess_uwb (Modül 2 pipeline)
    4. model.forward (FusedCSIUWBNet veya StudentFusedNet)
    5. tracker.step (sliding window mikro-Doppler, ayrı path)
    6. WS broadcast (subscribers'a JSON paket)

Telemetri:
    - inference_ms (model forward)
    - sync drop_ratio
    - tick lateness

MOCK mode: model=None → random output, end-to-end loop test için.
"""

from __future__ import annotations

import asyncio
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

try:
    import torch
except ImportError:
    torch = None

from backend.sync_buffer import MultiSensorSyncBuffer
from preprocessing import (
    AdaptiveBaseline, preprocess_csi, preprocess_uwb,
)
from models.heads import TrackingEngine, DopplerRingBuffer


# ── Link geometry helper ───────────────────────────────────


def compute_link_geo(scene_yaml_path: str | Path) -> np.ndarray:
    """scene.yaml'dan ESP32 düğüm konumları → link midpoint koordinatları.

    Returns: shape (28, 3) — C(8,2) = 28 link, her link için XYZ midpoint
    """
    with open(scene_yaml_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    esp32 = [n['position_m'] for n in cfg['nodes']
             if 'esp32' in n.get('sensors', [])]
    esp32 = np.array(esp32, dtype=np.float32)        # (8, 3)
    pairs = list(combinations(range(len(esp32)), 2))  # 28
    midpoints = np.array([
        (esp32[i] + esp32[j]) / 2.0 for i, j in pairs
    ], dtype=np.float32)                              # (28, 3)
    return midpoints


# ── InferenceEngine ────────────────────────────────────────


class InferenceEngine:
    """Real-time inference loop + WebSocket broadcast.

    Mock mode (model=None): random output üretir, end-to-end loop test için.
    Production mode: model=FusedCSIUWBNet veya StudentFusedNet checkpoint yüklü.
    """

    def __init__(
        self,
        model: Any = None,                # nn.Module veya None (mock)
        tracker_model: Any = None,        # SlidingTracker veya None
        scene_yaml_path: str = "configs/scene.yaml",
        master_tick_ms: int = 50,
        sync_tolerance_ms: int = 5,
        ring_capacity: int = 200,
        baseline_cfg: dict | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.tracker_model = tracker_model
        self.device = device
        self.master_tick_ms = int(master_tick_ms)
        self.master_tick_s = master_tick_ms / 1000.0

        # Sync buffer (28 WiFi + 6 UWB bistatic)
        self.sync_buffer = MultiSensorSyncBuffer(
            wifi_link_count=28,
            uwb_link_count=6,
            sync_tolerance_ms=sync_tolerance_ms,
            master_tick_ms=master_tick_ms,
            ring_capacity=ring_capacity,
        )

        # Adaptive baseline (drift)
        baseline_cfg = baseline_cfg or {
            "alpha_continuous": 1e-4,
            "alpha_quiet": 1e-2,
            "quiet_frames_threshold": 100,
        }
        self.baseline = AdaptiveBaseline(**baseline_cfg)

        # Sliding tracker (mikro-Doppler)
        self.tracker = TrackingEngine(
            model=tracker_model,
            window_size=100,
            csi_shape=(28, 108, 2),
            master_tick_ms=master_tick_ms,
        )

        # Link geometry (cache, çoğunlukla sabit)
        self.link_geo_np = compute_link_geo(scene_yaml_path)        # (28, 3)
        if torch is not None:
            self.link_geo_t = torch.from_numpy(self.link_geo_np).float() \
                                   .unsqueeze(0).to(device)         # (1, 28, 3)

        # Lifecycle
        self.running = False
        self.task: asyncio.Task | None = None
        self.subscribers: set[Any] = set()
        self.last_detection_mask = np.zeros(6, dtype=np.uint8)

        # Telemetri
        self.tick_count = 0
        self.inference_total_ms = 0.0
        self.last_inference_ms = 0.0
        self.last_packet: dict[str, Any] | None = None

    # ── Subscriber yönetimi ────────────────────────────────

    def add_subscriber(self, ws: Any) -> None:
        self.subscribers.add(ws)

    def remove_subscriber(self, ws: Any) -> None:
        self.subscribers.discard(ws)

    # ── Lifecycle ──────────────────────────────────────────

    def start(self) -> None:
        """Async predict_loop'u arka planda başlat."""
        if self.running:
            return
        self.running = True
        try:
            loop = asyncio.get_running_loop()
            self.task = loop.create_task(self._predict_loop())
        except RuntimeError:
            # asyncio loop yoksa (testte) sadece flag set et
            self.task = None

    def stop(self) -> None:
        self.running = False
        if self.task is not None:
            self.task.cancel()
            self.task = None

    # ── Ana predict loop ───────────────────────────────────

    async def _predict_loop(self) -> None:
        """Master tick (50 ms) loop."""
        while self.running:
            await asyncio.sleep(self.master_tick_s)
            try:
                packet = self.step_one_tick()
                if packet is not None:
                    await self._broadcast(packet)
            except Exception as e:
                # Loop dökülmesin
                print(f"[InferenceEngine] tick error: {e}")

    def step_one_tick(self, target_t_ns: int | None = None
                      ) -> dict[str, Any] | None:
        """Tek bir master tick — sync + preprocess + model + tracker.

        Async loop dışından (test/manual) çağrılabilir.
        """
        target_t_ns = target_t_ns or time.time_ns()
        frame = self.sync_buffer.aligned_frame(target_t_ns)
        if frame is None:
            return None

        wifi_raw = frame["wifi"]                                    # (28, 108, 2)
        uwb_raw = frame["uwb"]                                       # (6, 32, 2)

        # 1) Adaptive baseline (CSI drift)
        self.baseline.update(wifi_raw, detection_mask=self.last_detection_mask)
        csi_delta = self.baseline.compute_delta(wifi_raw)

        # 2) Preprocess (batch dim ekle)
        csi_proc = preprocess_csi(csi_delta[None, ...],
                                   apply_dwt=False)                  # (1,28,2,108)
        uwb_proc = preprocess_uwb(uwb_raw[None, ...])                # (1,6,2,32)

        # 3) Model inference
        t0 = time.perf_counter()
        if self.model is None:
            # Mock mode — random output
            model_out = self._mock_model_output()
        else:
            if torch is None:
                raise RuntimeError("torch import yok — model var ama torch yok")
            csi_t = torch.from_numpy(csi_proc).float().to(self.device)
            uwb_t = torch.from_numpy(uwb_proc).float().to(self.device)
            with torch.no_grad():
                model_out = self.model(csi_t, uwb_t, self.link_geo_t)
            # Tensor → numpy/list
            model_out = self._tensorize_out(model_out)

        inf_ms = (time.perf_counter() - t0) * 1000.0
        self.last_inference_ms = inf_ms
        self.inference_total_ms += inf_ms
        self.tick_count += 1

        # 4) Tracker (sliding window CSI)
        track = self.tracker.step(csi_delta.astype(np.float32))

        # 5) Detection mask update (baseline gating için)
        det_mask = np.array(model_out["detection_mask"], dtype=np.uint8)
        self.last_detection_mask = det_mask.flatten()[:6]

        # 6) Output paketi
        packet = {
            "t_master_ns": frame["t_master_ns"],
            "detection_mask": model_out["detection_mask"],
            "materials": model_out["materials"],
            "barcode_sparse": model_out["barcode_sparse"],
            "uncertainty": model_out["uncertainty"],
            "tracking": track,
            "telemetry": {
                "inference_ms": round(inf_ms, 3),
                "tick_count": self.tick_count,
                "sync": self.sync_buffer.telemetry(),
            },
        }
        self.last_packet = packet
        return packet

    # ── Mock + tensor helpers ──────────────────────────────

    def _mock_model_output(self) -> dict[str, Any]:
        """Random output (mock mode, model yok)."""
        rng = np.random.default_rng()
        return {
            "detection_mask": rng.integers(0, 2, size=6).tolist(),
            "materials": rng.integers(0, 4, size=6).tolist(),
            "barcode_sparse": rng.standard_normal(64).tolist(),
            "uncertainty": rng.uniform(0.01, 0.5, size=(6, 8, 8, 8)).tolist(),
        }

    def _tensorize_out(self, raw: dict[str, Any]) -> dict[str, Any]:
        """FusedCSIUWBNet çıktısı → JSON-serializable."""
        import torch as _torch
        det_mask = raw["detection_mask"][0].cpu().numpy().astype(int).tolist()
        materials = raw["material_logits"][0].argmax(-1).cpu().numpy().tolist()
        # Identification: 64-d latent (cosine için DB lookup ileride)
        barcode = raw["barcode_latent"][0].mean(dim=0).cpu().numpy().tolist()
        # Uncertainty: σ² voxel
        uncertainty = raw["tsdf_sigma2"][0].cpu().numpy().tolist()
        return {
            "detection_mask": det_mask,
            "materials": materials,
            "barcode_sparse": barcode,
            "uncertainty": uncertainty,
        }

    # ── Broadcast ──────────────────────────────────────────

    async def _broadcast(self, packet: dict[str, Any]) -> None:
        if not self.subscribers:
            return
        dead = []
        for ws in list(self.subscribers):
            try:
                await ws.send_json(packet)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.subscribers.discard(ws)

    # ── Manual push helpers (test + WS endpoint kullanır) ───

    def push_wifi(self, link_id: int, t_ns: int, csi: np.ndarray) -> None:
        self.sync_buffer.push_wifi(link_id, t_ns, csi)

    def push_uwb(self, link_id: int, t_ns: int, cir: np.ndarray) -> None:
        self.sync_buffer.push_uwb(link_id, t_ns, cir)
