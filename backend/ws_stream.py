"""WebSocket akış kanalları (PDF Modül 5 Adım 3).

İki yönlü trafik:

    EDGE → SERVER  (/ws/ingest):
        ESP32/UWB cihazlarından gelen ham paket.
        Schema: {node_id, sensor, link_id, t_capture_ns, payload}
        → InferenceEngine.push_wifi() veya push_uwb()

    SERVER → CLIENT  (/ws/predict):
        Dashboard'a inference output (50 ms tick).
        Schema: {t_master_ns, detection_mask, materials, barcode_sparse,
                 uncertainty, tracking, telemetry}

Endpoint'ler `backend/main.py` içinde register edilir.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect


router = APIRouter()


# ── /ws/ingest — edge cihazlardan ham veri ─────────────────


@router.websocket("/ws/ingest")
async def ingest(ws: WebSocket) -> None:
    """Edge cihazlardan (ESP32/UWB master) gelen sensor packet'leri.

    Beklenen JSON şema:
        {
            "node_id": "M2",
            "sensor": "wifi" | "uwb",
            "link_id": 0..27 (wifi) | 0..5 (uwb bistatic),
            "t_capture_ns": 1715800000123456789,
            "payload": [...] (flat float list)
        }

    payload shape (sensor'a göre):
        wifi: 108 * 2 = 216 float (subc × real/imag flatten)
        uwb:  32 * 2  = 64 float  (tap × real/imag flatten)
    """
    await ws.accept()
    engine = ws.app.state.engine
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"error": "invalid JSON"})
                continue

            sensor = msg.get("sensor")
            link_id = int(msg.get("link_id", 0))
            t_ns = int(msg.get("t_capture_ns", 0))
            payload = np.asarray(msg.get("payload", []), dtype=np.float32)

            try:
                if sensor == "wifi":
                    # (108, 2) flatten edilmiş 216-d
                    csi = payload.reshape(108, 2)
                    engine.push_wifi(link_id, t_ns, csi)
                elif sensor == "uwb":
                    cir = payload.reshape(32, 2)
                    engine.push_uwb(link_id, t_ns, cir)
                else:
                    await ws.send_json({"error": f"unknown sensor: {sensor}"})
            except Exception as e:
                await ws.send_json({"error": str(e)})
    except WebSocketDisconnect:
        pass


# ── /ws/predict — dashboard'a inference yayını ─────────────


@router.websocket("/ws/predict")
async def predict(ws: WebSocket) -> None:
    """Dashboard subscriber'ı. InferenceEngine her 50 ms paket yayınlar.

    İlk bağlanan client'a son paket gönderilir (warm-up için).
    """
    await ws.accept()
    engine = ws.app.state.engine
    engine.add_subscriber(ws)
    # İlk paket: son cached
    if engine.last_packet is not None:
        try:
            await ws.send_json(engine.last_packet)
        except Exception:
            pass
    try:
        while True:
            # Heartbeat — client'tan ping bekle (engine asıl broadcast)
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        engine.remove_subscriber(ws)


# NOT: /telemetry HTTP endpoint backend/main.py içinde tanımlı,
# app.state.engine üzerinden gerçek veri döner. ws_stream.py sadece
# WebSocket router'ı.
