"""Diferansiyellenebilir Ray Tracing (D-RT) ve kanal tepkisi (H) çözümü.

Modül 1 — Adım 2 (README.md):
    - compute_paths(max_depth=5): LoS + NLoS yansımaları.
    - RadioMapSolver: hızlı toplu CSI üretimi.
    - Diferansiyel: boş oda + nesneli oda → CSI_Δ.

TODO:
    * scene_builder'dan gelen Scene'i al.
    * 8 ESP32 → 28 link için Tx/Rx çiftlerini kur.
    * compute_paths çağır, ChannelResponse(H) döndür [link, 108 subcarrier].
    * iki tarama yap (boş + nesneli) ve farkı hesapla.
"""
