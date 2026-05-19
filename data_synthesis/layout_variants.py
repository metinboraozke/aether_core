"""FAZ 0.5 — Sensor Placement Randomization.

Aynı odada anchor + ESP32 düğümlerinin pozisyonlarını sample seti boyunca
farklı kılarak modelin spesifik koordinatları ezberlemesini engelle.
Hedef: deployment-invariant generalization (gerçek dünyada anchor 10-30 cm
kaydığında sistem çökmesin).

Tetikleyici: Kullanıcı 2026-05-15 "model anchor pozisyonunu ezberlemesin".
Bkz: upcoming/07_SENSOR_PLACEMENT_RANDOMIZATION.txt

Kullanım:
    from data_synthesis.layout_variants import generate_layout_variants

    variants = generate_layout_variants(
        base_nodes=cfg["nodes"],
        n_variants=5,
        jitter_xy_m=0.3,
        jitter_z_m=0.1,
        room_dims_m=(cfg["room"]["width_m"], cfg["room"]["depth_m"], cfg["room"]["height_m"]),
        clip_margin_m=0.3,
        seed=cfg["domain"]["master_seed"],
    )
    # variants[v_idx] = jittered nodes listesi (same shape as base_nodes)

Önemli:
    - id / type / sensors alanları korunur, sadece position_m jitter'lanır
    - UWB ve WiFi script'leri AYNI seed + AYNI n_variants ile çağrılırsa
      variant rotasyonu birebir senkron, paired dataset garantisi bozulmaz
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def jitter_node_position(
    pos: list[float] | tuple[float, ...] | np.ndarray,
    rng: np.random.Generator,
    jitter_xy_m: float,
    jitter_z_m: float,
    room_dims_m: tuple[float, float, float],
    clip_margin_m: float,
) -> list[float]:
    """Tek bir düğüm pozisyonuna gaussian jitter + oda sınırı clip uygula."""
    p = np.asarray(pos, dtype=np.float64).copy()
    p[0] += rng.normal(0.0, jitter_xy_m)
    p[1] += rng.normal(0.0, jitter_xy_m)
    p[2] += rng.normal(0.0, jitter_z_m)
    for k in range(3):
        lo = clip_margin_m
        hi = float(room_dims_m[k]) - clip_margin_m
        p[k] = float(min(max(p[k], lo), hi))
    return p.tolist()


def generate_layout_variants(
    base_nodes: list[dict[str, Any]],
    n_variants: int = 5,
    jitter_xy_m: float = 0.3,
    jitter_z_m: float = 0.1,
    room_dims_m: tuple[float, float, float] = (10.0, 5.0, 3.0),
    clip_margin_m: float = 0.3,
    seed: int = 42,
) -> list[list[dict[str, Any]]]:
    """Base node layout'tan N adet jittered varyant üret.

    Args:
        base_nodes: scene.yaml'daki nodes listesi (örn. 8 düğüm: 4M + 4S)
        n_variants: kaç farklı yerleşim
        jitter_xy_m: XY plane gaussian std (metre)
        jitter_z_m:  Z gaussian std (metre)
        room_dims_m: (width, depth, height) — clip sınırları için
        clip_margin_m: oda kenarından minimum mesafe
        seed: deterministic; UWB ve WiFi script'lerinde aynı kullanılırsa
              variant pozisyonları birebir aynı olur

    Returns:
        variants: uzunluk n_variants; her eleman base_nodes ile aynı yapıda
                  ama position_m alanları jittered. id/type/sensors korunur.

    Notlar:
        - v=0 her zaman base layout (jitter=0) — backward compatibility
        - v>=1 deterministic gaussian jitter (seed'e göre)
        - Slot konumu DEĞİŞMEZ (bu fonksiyon yalnız node listesini etkiler)
    """
    if n_variants < 1:
        raise ValueError(f"n_variants >= 1 olmalı, geldi: {n_variants}")

    variants: list[list[dict[str, Any]]] = []

    # v=0 → base (identity)
    variants.append(copy.deepcopy(base_nodes))

    if n_variants == 1:
        return variants

    # v>=1 → jittered, deterministic per-variant seed
    for v in range(1, n_variants):
        variant_rng = np.random.default_rng(seed + 1000 * v)
        new_nodes = []
        for node in base_nodes:
            new_pos = jitter_node_position(
                node["position_m"], variant_rng,
                jitter_xy_m=jitter_xy_m,
                jitter_z_m=jitter_z_m,
                room_dims_m=room_dims_m,
                clip_margin_m=clip_margin_m,
            )
            new_nodes.append({**node, "position_m": new_pos})
        variants.append(new_nodes)

    return variants


def assign_layout_ids(n: int, n_variants: int) -> np.ndarray:
    """N sample'ı n_variants varyanta contiguous block'lar halinde dağıt.

    Sample sırasıyla: [v0, v0, ..., v0, v1, v1, ..., v_{n_variants-1}]
    Kalan sample'lar son variant'a düşer (n % n_variants != 0 durumunda).

    Returns:
        layout_ids: shape (n,) uint8, değerler 0..n_variants-1
    """
    samples_per_variant = n // n_variants
    ids = np.empty(n, dtype=np.uint8)
    for v in range(n_variants):
        start = v * samples_per_variant
        end = (v + 1) * samples_per_variant if v < n_variants - 1 else n
        ids[start:end] = v
    return ids


__all__ = [
    "jitter_node_position",
    "generate_layout_variants",
    "assign_layout_ids",
]
