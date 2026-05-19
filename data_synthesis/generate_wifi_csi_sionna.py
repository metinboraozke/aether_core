"""Modül 1B ENTRY POINT — WiFi 2.4 GHz CSI Diferansiyel Üretici (Sionna 2.0 RT).

OPTİMİZE 2026-05-15: Önceki sürümde her link için ayrı PathSolver çağrısı
yapılıyordu (28 link × 2 pass = 56 solver/örnek → 1.5 s/it). Yeni sürümde
8 düğüm hep birden sahnede aktif, **tek PathSolver çağrısı** ile 8×8 link
matrisi paralel hesaplanır. ~10-15x hızlanma → ~5-10 it/s.

Çıktılar:
    csivec_delta.npy        [N, 28, 108, 2]   diferansiyel CSI (Real/Imag)
    path_params_csi.npy     [N, 28, 20, 4]    Sionna ground-truth path
                                              (DML-AP teacher hedefi, sim2real bridge)

PAIRED DATASET KURALI:
    UWB ile aynı --seed kullanılırsa (default 42), aynı stochastic slot
    konfigürasyonları üretilir. Yani örnek i'nin csivec_delta'sı, aynı
    örneğin uwb_cir_oracle/anchor'ı ile birebir eşleşir.

DİFERANSİYEL CSI:
    Her örnek için iki path-solve:
        1. Boş oda (slot küpleri vacuum) → csi_ref
        2. Nesneli oda (slot materyalleri set) → csi_full
    Çıktı: csi_delta = csi_full - csi_ref

Kullanım (Colab):
    !python -m data_synthesis.generate_wifi_csi_sionna \\
        --n 10000 \\
        --out /content/drive/MyDrive/aether_core/data \\
        --scene scenes/aether_classroom.xml \\
        --config configs/scene.yaml \\
        --checkpoint_every 1000 \\
        --seed 42

Süre tahmini (L4 GPU, optimize sonrası):
    Beklenen ~5-10 it/s. 10k örnek için ~20-30 dakika.
"""

from __future__ import annotations

import argparse
import os
from itertools import combinations
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from data_synthesis.generate_hybrid_sionna import (
    assign_slot_materials,
    build_sionna_scene,
    load_scene_config,
    get_esp32_nodes,
    _save_checkpoint,
    _merge_checkpoints,
)
from data_synthesis.noise_models import (
    apply_agc,
    apply_cfo,
    apply_wifi_interference,
    apply_bluetooth_bursts,
    apply_awgn,
)
from data_synthesis.resonator_inject import (
    complex_to_realimag,
    int_to_data_bits,
    hamming_encode_7_4,
)
from data_synthesis.layout_variants import (
    generate_layout_variants,
    assign_layout_ids,
)


# =====================================================================
# WIFI BATCHED YARDIMCILAR (8 TX + 8 RX hep birlikte)
# =====================================================================


def add_all_wifi_nodes(scene, esp32_positions: list[list[float]]):
    """8 ESP32 düğümünü hepsini birden sahneye ekler (her biri hem TX hem RX).

    Sionna 2.0 RT API'sinde TX ve RX ayrı objelerdir; gerçek ESP32 yarı-dupleks
    olsa da burada her düğümü iki kez (TX + RX) eklemek mathematical reciprocity'i
    bozmaz çünkü C(8,2) link çiftlerini sonradan matristen seçeriz.
    """
    from sionna.rt import Transmitter, Receiver  # type: ignore

    txs, rxs = [], []
    for i, pos in enumerate(esp32_positions):
        tx = Transmitter(name=f"wifi_tx_{i}", position=list(pos))
        rx = Receiver(name=f"wifi_rx_{i}", position=list(pos))
        scene.add(tx)
        scene.add(rx)
        txs.append(tx)
        rxs.append(rx)
    return txs, rxs


def remove_all_wifi_nodes(scene, txs: list, rxs: list) -> None:
    """Sahneden tüm WiFi düğümlerini çıkarır (sonraki sample için temiz başla)."""
    for node in (*txs, *rxs):
        try:
            scene.remove(node.name)
        except Exception:
            pass


# =====================================================================
# BATCHED CFR HESAPLAMA — TEK PATH-SOLVE, 8x8 MATRİS
# =====================================================================


def _squeeze_antenna_axes(arr: np.ndarray) -> np.ndarray:
    """[num_rx, num_rx_ant=1, num_tx, num_tx_ant=1, num_paths] →
       [num_rx, num_tx, num_paths]. Tek antenli düğümler için."""
    while arr.ndim > 3:
        squeezed = False
        for ax in range(1, arr.ndim - 1):
            if arr.shape[ax] == 1:
                arr = np.squeeze(arr, axis=ax)
                squeezed = True
                break
        if not squeezed:
            break
    return arr


def compute_all_links_cfr_batched(
    scene,
    link_pairs: list[tuple[int, int]],
    n_subcarriers: int = 108,
    bandwidth_hz: float = 40e6,
    max_depth: int = 5,
    max_paths: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """8 TX + 8 RX hepsi sahnede aktif iken TEK PathSolver çağrısı,
    tüm 28 link CFR'ı + path parametre tensörü extract et.

    REVİZE 2026-05-15: path_params eklendi.

    Gerekçe (sim2real bridge için kritik):
        Gerçek ESP32'den ham CSI gelir, path-by-path bilgi YOK. Real-time
        inference'ta DML-AP koşturmak imkansız (~100 ms/sample, 1 saat for
        500k). Çözüm: sentetik tarafta Sionna'nın bedava verdiği ground-truth
        path parametrelerini kayıt et → CSI Transformer Encoder'a auxiliary
        loss olarak supervise et. Model "raw CFR'dan path parametre tahmini"
        öğrenir → gerçek dünyada DML-AP koşturmaya GEREK KALMAZ.

    Sionna 2.0 RT döndürdüğü:
        a   shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
        tau shape: aynı

    Squeeze sonrası [num_rx, num_tx, num_paths]. C(8,2) link çiftlerini
    matristen alırız.

    Döner:
        cfr         : shape (28, n_subcarriers), complex64
        path_params : shape (28, max_paths, 4), float32
                      kanal: [real(a), imag(a), tau_seconds, validity_mask]
                      En güçlü max_paths path amplitude'a göre seçilir.
    """
    from sionna.rt import PathSolver  # type: ignore

    solver = PathSolver()
    paths = solver(
        scene=scene,
        max_depth=max_depth,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        synthetic_array=False,
    )
    a, tau = paths.cir(out_type="numpy", normalize_delays=True)

    a_np = _squeeze_antenna_axes(np.asarray(a))         # [num_rx, num_tx, num_paths]
    tau_np = _squeeze_antenna_axes(np.asarray(tau))     # [num_rx, num_tx, num_paths]

    # Bazı sürümlerde son axis time-step (1) olabilir
    if a_np.ndim == 4 and a_np.shape[-1] == 1:
        a_np = a_np[..., 0]
    if tau_np.ndim == 4 and tau_np.shape[-1] == 1:
        tau_np = tau_np[..., 0]

    # Subcarrier frekansları (baseband, ±BW/2)
    sub_freqs = np.linspace(
        -bandwidth_hz / 2.0, bandwidth_hz / 2.0, n_subcarriers, dtype=np.float64
    )

    n_links = len(link_pairs)
    out = np.zeros((n_links, n_subcarriers), dtype=np.complex64)
    path_params = np.zeros((n_links, max_paths, 4), dtype=np.float32)

    for link_idx, (tx_i, rx_j) in enumerate(link_pairs):
        # link_pairs (i, j) → tx=i, rx=j ; reciprocity ile (j, i) ile aynı
        a_link = a_np[rx_j, tx_i, :]
        tau_link = tau_np[rx_j, tx_i, :]
        valid = np.isfinite(tau_link) & (tau_link >= 0)
        if not valid.any():
            continue
        valid_a = a_link[valid]
        valid_t = tau_link[valid]

        # Path params: amplitude'a göre en güçlü max_paths
        n_valid = int(valid_a.size)
        if n_valid > 0:
            mag = np.abs(valid_a)
            order = np.argsort(-mag)[:max_paths]
            sel_a = valid_a[order]
            sel_t = valid_t[order]
            n_sel = int(sel_a.size)
            path_params[link_idx, :n_sel, 0] = np.real(sel_a).astype(np.float32)
            path_params[link_idx, :n_sel, 1] = np.imag(sel_a).astype(np.float32)
            path_params[link_idx, :n_sel, 2] = sel_t.astype(np.float32)
            path_params[link_idx, :n_sel, 3] = 1.0  # validity mask

        # CFR (analitik): H(f_k) = sum_p a_p * exp(-j 2π f_k τ_p)
        a_v = valid_a.astype(np.complex128)
        tau_v = valid_t.astype(np.float64)
        phase = np.exp(-1j * 2.0 * np.pi * sub_freqs[:, None] * tau_v[None, :])
        out[link_idx] = (a_v[None, :] * phase).sum(axis=-1).astype(np.complex64)

    return out, path_params


# =====================================================================
# WIFI-SPESİFİK GÜRÜLTÜ
# =====================================================================


def apply_wifi_csi_noise(
    cfr: np.ndarray,
    rng: np.random.Generator,
    snr_db: float = 25.0,
) -> np.ndarray:
    """CSI'a WiFi-spesifik gürültüleri uygula.

    cfr: shape (28, 108), complex
    Sıra: AGC → CFO → narrow-band interference → BT bursts → AWGN.
    """
    cfr = apply_agc(cfr, gain_db_range=(-6.0, 6.0), rng=rng)
    cfr = apply_cfo(cfr, cfo_phase_max_rad=np.pi, rng=rng)
    cfr = apply_wifi_interference(cfr, n_interferers_max=3,
                                   interferer_power_range=(0.005, 0.02),
                                   rng=rng)
    cfr = apply_bluetooth_bursts(cfr, burst_prob=0.05, rng=rng)
    cfr = apply_awgn(cfr, snr_db=snr_db, rng=rng)
    return cfr


# =====================================================================
# TEK ÖRNEK ÜRETİMİ (UWB ile birebir paired)
# =====================================================================


def build_one_csi_sample(
    scene,
    cfg: dict[str, Any],
    link_pairs: list[tuple[int, int]],
    rng: np.random.Generator,
    max_depth: int = 5,
) -> dict[str, np.ndarray]:
    """Tek paired sample (sadece CSI tarafı).

    UWB script'inin build_one_paired_sample'ı ile aynı stochastic seçimleri
    yapar (aynı seed → aynı slot konfigürasyonu).
    """
    n_slots = cfg["shelf"]["n_slots"]
    fill_prob = cfg["randomization"]["slot_fill_probability"]
    # REVİZE 2026-05-15: ESP32 düğümleri hibrit nodes listesinden (4M+4S=8)
    esp32_nodes = get_esp32_nodes(cfg)
    esp32_positions = [n["position_m"] for n in esp32_nodes]
    n_subcarriers = int(cfg["wifi"]["n_subcarriers"])
    bandwidth_hz = float(cfg["wifi"]["bandwidth_hz"])

    # 1) Stochastic seçim (UWB ile birebir AYNI sıralama → aynı sonuç)
    slot_filled = (rng.random(n_slots) < fill_prob).astype(np.uint8)
    material_ids = np.where(
        slot_filled == 1,
        rng.integers(1, 5, size=n_slots),
        0,
    ).astype(np.uint8)
    data_ids = rng.integers(0, 16, size=n_slots).astype(np.uint8)
    data_bits = int_to_data_bits(data_ids)
    codewords = hamming_encode_7_4(data_bits)
    # NOT: data_ids/codewords burada CSI için kullanılmaz; sadece RNG state'i
    # UWB ile senkron tutmak için aynı sırayla çağırıyoruz.

    # 2) BOŞ ODA pass — slotlar vacuum, 8 düğüm sahnede, TEK PathSolver
    #    NOT: bos pass path_params'i KAYIT EDILMIYOR (sadece duvar yansimasi,
    #    modelin ogrenmesine deger katmaz).
    empty_filled = np.zeros(n_slots, dtype=np.uint8)
    empty_mats = np.zeros(n_slots, dtype=np.uint8)
    assign_slot_materials(scene, empty_filled, empty_mats, cfg, rng)

    txs, rxs = add_all_wifi_nodes(scene, esp32_positions)
    csi_ref, _ = compute_all_links_cfr_batched(
        scene=scene,
        link_pairs=link_pairs,
        n_subcarriers=n_subcarriers,
        bandwidth_hz=bandwidth_hz,
        max_depth=max_depth,
    )
    remove_all_wifi_nodes(scene, txs, rxs)

    # 3) NESNELİ ODA pass — slot materyalleri set, TEK PathSolver
    #    Bu pass'in path_params'i AUXILIARY HEDEF olur (nesne yansimalari dahil).
    assign_slot_materials(scene, slot_filled, material_ids, cfg, rng)

    txs, rxs = add_all_wifi_nodes(scene, esp32_positions)
    csi_full, path_params_csi = compute_all_links_cfr_batched(
        scene=scene,
        link_pairs=link_pairs,
        n_subcarriers=n_subcarriers,
        bandwidth_hz=bandwidth_hz,
        max_depth=max_depth,
    )
    remove_all_wifi_nodes(scene, txs, rxs)

    # 4) Diferansiyel CSI
    csi_delta = csi_full - csi_ref

    # 5) WiFi gürültüsü (sadece CSI'ya, path_params clean kalir — supervision target)
    csi_delta = apply_wifi_csi_noise(csi_delta, rng=rng, snr_db=25.0)

    return {
        "csivec_delta": complex_to_realimag(csi_delta),  # (28, 108, 2)
        "path_params_csi": path_params_csi,              # (28, 20, 4)
        "slot_labels": slot_filled,
        "material_labels": material_ids,
    }


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aether Core — WiFi 2.4 GHz CSI diferansiyel üretici (BATCHED)"
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument(
        "--out", type=str, default="./data",
        help="çıktı klasörü (UWB ile aynı olmalı, paired dataset)"
    )
    parser.add_argument(
        "--scene", type=str, default="scenes/aether_classroom.xml"
    )
    parser.add_argument("--config", type=str, default="configs/scene.yaml")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="UWB script'iyle AYNI seed kullan → paired dataset garantisi"
    )
    parser.add_argument("--checkpoint_every", type=int, default=500)
    # FAZ 0.5 — Sensor Placement Randomization (UWB script ile AYNI değerler)
    parser.add_argument("--n_variants", type=int, default=1,
                        help="UWB script ile AYNI değer + AYNI --seed olmalı "
                             "(paired dataset variant rotasyonu)")
    parser.add_argument("--layout_jitter_xy", type=float, default=0.3)
    parser.add_argument("--layout_jitter_z", type=float, default=0.1)
    parser.add_argument("--layout_clip_margin", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[wifi-init] config: {args.config}")
    cfg = load_scene_config(args.config)

    print(f"[wifi-init] sahne yükleniyor (2.412 GHz): {args.scene}")
    wifi_freq = float(cfg["wifi"]["freq_hz"])
    scene = build_sionna_scene(args.scene, cfg, uwb_freq_hz=wifi_freq)

    # REVİZE 2026-05-15: ESP32 düğümleri hibrit nodes listesinden okunur (8 düğüm)
    esp32_nodes = get_esp32_nodes(cfg)
    n_nodes = len(esp32_nodes)
    link_pairs = list(combinations(range(n_nodes), 2))
    print(f"[wifi-init] {n_nodes} ESP32 düğümü ({sum(1 for n in esp32_nodes if n.get('type')=='master')} master + "
          f"{sum(1 for n in esp32_nodes if n.get('type')=='satellite')} satellite) → "
          f"{len(link_pairs)} link C({n_nodes},2) — BATCHED tek PathSolver/pass")

    # FAZ 0.5 — layout varyantları (UWB script ile AYNI master_seed + AYNI parametreler)
    room_dims = (
        float(cfg["room"]["width_m"]),
        float(cfg["room"]["depth_m"]),
        float(cfg["room"]["height_m"]),
    )
    layout_seed = int(cfg.get("domain", {}).get("master_seed", args.seed))
    variants = generate_layout_variants(
        base_nodes=cfg["nodes"],
        n_variants=args.n_variants,
        jitter_xy_m=args.layout_jitter_xy,
        jitter_z_m=args.layout_jitter_z,
        room_dims_m=room_dims,
        clip_margin_m=args.layout_clip_margin,
        seed=layout_seed,
    )
    layout_ids = assign_layout_ids(args.n, args.n_variants)
    print(f"[wifi-init] LAYOUT: {args.n_variants} variant × ~{args.n // args.n_variants} sample/variant "
          f"(jitter XY={args.layout_jitter_xy}m Z={args.layout_jitter_z}m, layout_seed={layout_seed})")

    # Per-sample bağımsız seed — UWB script ile AYNI master seed kullanılırsa
    # her sample i için aynı sample_seeds[i] üretilir → slot/material/data_id
    # garantili eşleşir (RNG drift etkisi yok).
    master_rng = np.random.default_rng(args.seed)
    sample_seeds = master_rng.integers(0, 2**31 - 1, size=args.n)

    buffers: dict[str, list] = {
        "csivec_delta": [],
        "path_params_csi": [],     # REVİZE 2026-05-15: DML-AP teacher hedefi
        "slot_labels_wifi": [],
        "material_labels_wifi": [],
    }

    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        tf = None

    current_v = -1
    cfg_variant = cfg

    for i in tqdm(range(args.n), desc="generating-wifi"):
        v_idx = int(layout_ids[i])
        if v_idx != current_v:
            cfg_variant = {**cfg, "nodes": variants[v_idx]}
            current_v = v_idx
            tqdm.write(f"[wifi-layout] variant {v_idx} aktif (sample {i})")

        sample_rng = np.random.default_rng(int(sample_seeds[i]))
        sample = build_one_csi_sample(
            scene=scene,
            cfg=cfg_variant,
            link_pairs=link_pairs,
            rng=sample_rng,
            max_depth=args.max_depth,
        )
        buffers["csivec_delta"].append(sample["csivec_delta"])
        buffers["path_params_csi"].append(sample["path_params_csi"])
        buffers["slot_labels_wifi"].append(sample["slot_labels"])
        buffers["material_labels_wifi"].append(sample["material_labels"])

        if (i + 1) % args.checkpoint_every == 0:
            _save_checkpoint(buffers, args.out, i + 1)
            for key in buffers:
                buffers[key] = []
            if tf is not None:
                tf.keras.backend.clear_session()

    if any(len(v) > 0 for v in buffers.values()):
        _save_checkpoint(buffers, args.out, args.n)

    print("[merge] WiFi parçaları birleştiriliyor...")
    _merge_checkpoints(args.out, "csivec_delta",         "csivec_delta.npy")
    _merge_checkpoints(args.out, "path_params_csi",      "path_params_csi.npy")
    _merge_checkpoints(args.out, "slot_labels_wifi",     "slot_labels_wifi.npy")
    _merge_checkpoints(args.out, "material_labels_wifi", "material_labels_wifi.npy")

    # FAZ 0.5 — layout_ids_wifi (UWB tarafıyla birebir eşleşmeli; ayrı dosya
    # paired check için faydalı)
    np.save(os.path.join(args.out, "layout_ids_wifi.npy"), layout_ids)

    print("[done] WiFi CSI dosyası kaydedildi:", args.out)
    print("       UWB ile paired index garantisi: aynı seed →",
          "slot_labels_wifi.npy ile slot_labels.npy birebir aynı olmalı")
    print("       Layout paired garantisi: layout_ids_wifi.npy == layout_ids.npy")


if __name__ == "__main__":
    main()
