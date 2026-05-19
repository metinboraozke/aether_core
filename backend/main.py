"""FastAPI ana uygulama — Aether Core backend entry point.

PDF Modül 5 Adım 3 (README.md). Lifecycle:
    startup: InferenceEngine oluştur + checkpoint yükle + master tick başlat
    shutdown: engine durdur + baseline state'i diske kaydet

Endpoint'ler:
    GET  /                   — health check
    GET  /telemetry          — engine + sync_buffer telemetry
    POST /reset_baseline     — adaptive baseline manuel reset
    WS   /ws/ingest          — edge cihaz paket girişi
    WS   /ws/predict         — dashboard tahmin yayını

Çalıştırma:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Çevre değişkenleri:
    AETHER_CHECKPOINT  — model .pt path (yoksa MOCK mode)
    AETHER_DEVICE      — cpu / cuda (default cpu)
    AETHER_USE_STUDENT — 1 ise StudentFusedNet, 0 ise FusedCSIUWBNet
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.inference_server import InferenceEngine
from backend.ws_stream import router as ws_router


# ── Lifespan (startup + shutdown) ─────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: engine init + start. Shutdown: engine stop."""
    checkpoint = os.environ.get("AETHER_CHECKPOINT", "")
    device = os.environ.get("AETHER_DEVICE", "cpu")
    use_student = os.environ.get("AETHER_USE_STUDENT", "0") == "1"
    scene_yaml = os.environ.get("AETHER_SCENE_YAML", "configs/scene.yaml")

    model = None
    tracker_model = None

    if checkpoint and Path(checkpoint).exists():
        try:
            import torch
            if use_student:
                from adaptation import StudentFusedNet
                model = StudentFusedNet().to(device)
            else:
                from models.fused_model import FusedCSIUWBNet
                model = FusedCSIUWBNet().to(device)
            state = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state, strict=False)
            model.eval()
            print(f"[main] Model yüklendi: {checkpoint} (student={use_student})")

            from models.heads import SlidingTracker
            tracker_model = SlidingTracker().to(device)
            tracker_model.eval()
        except Exception as e:
            print(f"[main] Model yükleme hatası: {e} → MOCK mode'a düşülüyor")
            model = None
            tracker_model = None
    else:
        print(f"[main] MOCK mode (checkpoint yok)")

    engine = InferenceEngine(
        model=model,
        tracker_model=tracker_model,
        scene_yaml_path=scene_yaml,
        master_tick_ms=50,
        sync_tolerance_ms=5,
        ring_capacity=200,
        device=device,
    )
    app.state.engine = engine
    engine.start()
    print(f"[main] InferenceEngine başlatıldı (tick={engine.master_tick_ms} ms)")

    yield

    engine.stop()
    print(f"[main] InferenceEngine durduruldu (toplam tick: {engine.tick_count})")
    try:
        engine.baseline.save_reference("data/baseline_ref.npy")
    except Exception as e:
        print(f"[main] Baseline save hatası: {e}")


# ── FastAPI app ────────────────────────────────────────────


app = FastAPI(
    title="Aether Core",
    description="WiFi CSI + UWB CIR ile 3D RF-vizyon sistemi",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)

# ── Dashboard static mount (Three.js, CDN bağımlı) ────────
_dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
if _dashboard_dir.exists():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(_dashboard_dir), html=True),
        name="dashboard",
    )


# ── HTTP endpoints ────────────────────────────────────────


@app.get("/")
async def root() -> dict:
    engine = app.state.engine
    return {
        "service": "Aether Core",
        "status": "ok",
        "tick_count": engine.tick_count,
        "model_loaded": engine.model is not None,
        "subscribers": len(engine.subscribers),
        "dashboard": "/dashboard/",
    }


@app.get("/ui")
async def ui_redirect() -> RedirectResponse:
    """Kısayol: / yerine /ui de dashboard'a yönlendirir."""
    return RedirectResponse(url="/dashboard/")


@app.get("/telemetry")
async def telemetry() -> dict:
    engine = app.state.engine
    avg_ms = (engine.inference_total_ms / engine.tick_count) \
              if engine.tick_count > 0 else 0.0
    return {
        "tick_count": engine.tick_count,
        "last_inference_ms": engine.last_inference_ms,
        "avg_inference_ms": round(avg_ms, 3),
        "sync_buffer": engine.sync_buffer.telemetry(),
        "subscribers": len(engine.subscribers),
        "baseline_quiet_count": engine.baseline.quiet_count,
        "last_detection_mask": engine.last_detection_mask.tolist(),
    }


@app.post("/reset_baseline")
async def reset_baseline() -> dict:
    engine = app.state.engine
    engine.baseline.reset()
    return {"status": "reset_ok"}


# ── Uvicorn entry point ───────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )
