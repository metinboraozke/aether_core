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


# ── C Paketi (2026-06-03): Ürün katalog API ────────────────


@app.get("/api/products")
async def list_products() -> dict:
    """Tüm 16 ürün catalog'unu döndürür (dashboard products view için).

    Recognition head bypass — material classification DB üzerinden yapılır.
    Identification head'in 7-bit codeword çıktısı → Hamming(7,4) decode →
    4-bit ID → bu catalog'daki ürün.
    """
    engine = app.state.engine
    if engine.product_db is None:
        return {"products": [], "count": 0, "error": "ProductDB yüklenmedi"}
    items = engine.product_db.all()
    return {
        "products": items,
        "count": len(items),
        "material_palette": engine.product_db.material_palette,
    }


@app.get("/api/products/{product_id}")
async def get_product(product_id: int) -> dict:
    """Tek bir ürünü ID ile getir."""
    engine = app.state.engine
    if engine.product_db is None:
        return {"error": "ProductDB yüklenmedi"}
    return engine.product_db.lookup(product_id)


# ── Demo data injection (2026-06-03): jüri test + sistem doğrulama ──


@app.post("/api/demo/tick")
async def demo_tick(sample_idx: int | None = None) -> dict:
    """Lokal dataset'ten bir sample alıp model forward yap + WS broadcast.

    Gerçek sensör yokken dashboard'da canlı sahne göstermek için.
    sample_idx None ise random sample seçilir.

    Veri yolu: data/classroom_default/ (lokal symlink veya direkt klasör).
    İlk çağrıda dataset lazy load edilir, sonraki çağrılarda cache kullanılır.
    """
    engine = app.state.engine

    # Lazy load dataset
    if not hasattr(engine, '_demo_dataset') or engine._demo_dataset is None:
        try:
            from data_synthesis.multi_domain_dataset import MultiDomainDataset
            import os as _os
            # data/classroom_default varsa onu kullan, yoksa hata
            if not _os.path.isdir('data/classroom_default'):
                return {
                    "error": "data/classroom_default klasörü yok",
                    "hint": "Drive senkron veya lokal mini veri lazım."
                }
            engine._demo_dataset = MultiDomainDataset(
                data_root='data', config_root='configs',
                presets=['classroom_default'], return_aux=False,
            )
            print(f"[demo] dataset yüklendi: {len(engine._demo_dataset)} sample")
        except Exception as e:
            return {"error": f"dataset load: {e}"}

    ds = engine._demo_dataset
    import numpy as _np
    if sample_idx is None or sample_idx < 0 or sample_idx >= len(ds):
        sample_idx = int(_np.random.randint(0, len(ds)))

    try:
        sample = ds[sample_idx]
        # MultiDomainDataset.__getitem__ → dict {csi, uwb, link_geo, labels, ...}
        packet = engine.inject_dataset_sample(
            csi_raw=_np.asarray(sample['csi']),
            uwb_raw=_np.asarray(sample['uwb']),
            link_geo=_np.asarray(sample['link_geo']),
        )
        # WS broadcast (dashboard subscribers'a yayın)
        await engine._broadcast(packet)
        return {
            "status": "ok",
            "sample_idx": sample_idx,
            "tick_count": engine.tick_count,
            "inference_ms": packet["telemetry"]["inference_ms"],
            "n_filled_slots": int(sum(packet["detection_mask"])),
            "products_detected": [
                {"slot": p["slot"], "id": p["id"], "name": p["name"]}
                for p in packet["products"] if not p.get("empty")
            ],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"tick fail (idx={sample_idx}): {e}"}


@app.get("/api/demo/info")
async def demo_info() -> dict:
    """Demo dataset bilgisi (boyut, hangi preset)."""
    engine = app.state.engine
    if not hasattr(engine, '_demo_dataset') or engine._demo_dataset is None:
        return {"loaded": False, "hint": "İlk /api/demo/tick çağrısında yüklenir"}
    return {
        "loaded": True,
        "n_samples": len(engine._demo_dataset),
        "presets": list(engine._demo_dataset.presets),
    }


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
