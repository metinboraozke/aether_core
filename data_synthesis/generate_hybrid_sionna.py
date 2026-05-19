"""Modül 1 ENTRY POINT — Teacher-Student paired UWB CIR üretici (Sionna 2.0 RT).

REVİZE 2026-05-15: Hibrit topoloji + Round-Robin TDMA
    Donanım: 4 UWB master kit (M1..M4), her biri hem TX hem RX.
    TDMA cycle: 40 ms (25 Hz), her master 10 ms TX → 4 frame × 3 RX = 12 ölçüm.
    Reciprocity (kanal simetri) → C(4,2) = 6 unique bistatic link.
    Eski "1 ayrı TX + 4 RX anchor" varsayımı kaldırıldı (5 UWB gerekiyordu).

Mimari (Knowledge Distillation):
    Teacher (Oracle):  6 sanal RX, slot merkezlerinde (offset -0.20 m Y),
                       GÜRÜLTÜSÜZ ideal CIR → [N, 6, 32, 2]

    Student (Bistatic): 4 master UWB cihazı bistatic links,
                        path-loss + AGC + CFO + WiFi/BT + AWGN ile gürültülü
                        → [N, 6, 32, 2]   ← 6 bistatic link (4→6 revize)

Çıktılar (data/<preset>/ klasörüne):
    uwb_cir_oracle.npy        [N, 6, 32, 2]   Teacher ideal CIR
    uwb_cir_anchor.npy        [N, 6, 32, 2]   Student bistatic CIR (REVİZE: 4→6)
    path_params_oracle.npy    [N, 6, 20, 4]   Sionna ground-truth (DML-AP teacher)
    path_params_anchor.npy    [N, 6, 20, 4]   Sionna ground-truth (REVİZE: 4→6)
    slot_labels.npy           [N, 6]
    material_labels.npy       [N, 6]
    codeword_labels.npy       [N, 6, 7]
    data_id_labels.npy        [N, 6]
"""

from __future__ import annotations

import argparse
import os
from itertools import combinations
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from data_synthesis.resonator_inject import (
    complex_to_realimag,
    hamming_encode_7_4,
    inject_resonator_into_cir,
    int_to_data_bits,
)
from data_synthesis.noise_models import (
    apply_all_anchor_noise,
)
from data_synthesis.layout_variants import (
    generate_layout_variants,
    assign_layout_ids,
)


# =====================================================================
# SAHNE KURULUMU
# =====================================================================


def load_scene_config(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_master_nodes(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Hibrit topoloji'den UWB sensörü olan düğümleri (4 master) çıkar."""
    return [n for n in cfg["nodes"] if "uwb" in n.get("sensors", [])]


def get_esp32_nodes(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Hibrit topoloji'den ESP32 sensörü olan düğümleri (8 = 4M + 4S) çıkar."""
    return [n for n in cfg["nodes"] if "esp32" in n.get("sensors", [])]


def build_sionna_scene(
    scene_xml_path: str,
    cfg: dict[str, Any],
    uwb_freq_hz: float = 6.5e9,
):
    """Mitsuba XML sahnesini Sionna RT'ye yükler ve antenna array'lerini ayarlar."""
    from sionna.rt import load_scene, PlanarArray  # type: ignore

    scene = load_scene(scene_xml_path)
    scene.frequency = uwb_freq_hz

    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    return scene


def assign_slot_materials(
    scene,
    slot_filled: np.ndarray,
    material_ids: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.Generator,
):
    """Slot küplerinin radio_material'ını runtime'da değiştirir."""
    from sionna.rt import ITURadioMaterial  # type: ignore

    iturm_map = {
        0: "vacuum",
        1: "metal",
        2: "plasterboard",
        3: "wood",
        4: "plywood",
    }

    for i in range(cfg["shelf"]["n_slots"]):
        obj = scene.get(f"slot_obj_{i + 1}")
        if obj is None:
            continue

        if not slot_filled[i]:
            iturm_name = iturm_map[0]
        else:
            iturm_name = iturm_map[int(material_ids[i])]

        try:
            obj.radio_material = ITURadioMaterial(iturm_name)
        except Exception as e:
            print(f"[uyarı] slot {i + 1} materyal atama: {e}; varsayılan kullanılıyor")


# ── UWB master düğüm yönetimi (Round-Robin TDMA) ─────────────


def add_all_uwb_masters(scene, master_nodes: list[dict[str, Any]]):
    """4 master UWB cihazı hep birden sahnede aktif (hem TX hem RX).

    Sionna 2.0 RT API'sinde TX ve RX ayrı obje; gerçek donanımda DW1000
    yarı-dupleks ama TDMA cycle (40 ms) içinde her cihaz hem TX hem RX
    rolünü oynar. Sionna tarafında tek path-solve ile 4×4 link matrisi
    elde edilir, C(4,2)=6 unique bistatic link extract edilir.
    """
    from sionna.rt import Transmitter, Receiver  # type: ignore

    txs, rxs = [], []
    for n in master_nodes:
        pos = list(n["position_m"])
        tx = Transmitter(name=f"uwb_tx_{n['id']}", position=pos)
        rx = Receiver(name=f"uwb_rx_{n['id']}", position=pos)
        scene.add(tx)
        scene.add(rx)
        txs.append(tx)
        rxs.append(rx)
    return txs, rxs


def remove_all_uwb_masters(scene, txs: list, rxs: list) -> None:
    for n in (*txs, *rxs):
        try:
            scene.remove(n.name)
        except Exception:
            pass


# ── Oracle (Teacher) RX'leri — slot merkezleri ───────────────


def add_oracle_receivers(scene, slot_centers_m: list[list[float]]):
    """Teacher data için 6 sanal receiver (slot küplerinin 20 cm önünde).

    Slot küpleri scene'de slot merkezinde duruyor; RX'i tam slot merkezine
    koymak Sionna'da "RX küp içinde" durumuna yol açar → path = 0. Çözüm:
    Y ekseninde -0.20 m kaydır (rafın açık tarafına).
    """
    from sionna.rt import Receiver  # type: ignore

    ORACLE_OFFSET_M = (0.0, -0.20, 0.0)

    rxs = []
    for i, pos in enumerate(slot_centers_m):
        offset_pos = [float(p + o) for p, o in zip(pos, ORACLE_OFFSET_M)]
        rx = Receiver(name=f"rx_oracle_{i + 1}", position=offset_pos)
        scene.add(rx)
        rxs.append(rx)
    return rxs


def remove_receivers(scene, rxs: list):
    for rx in rxs:
        try:
            scene.remove(rx.name)
        except Exception:
            pass


# =====================================================================
# CIR HESAPLAMA — TEK PATH-SOLVE, MATRİS ÇIKARMA
# =====================================================================


def _squeeze_antenna_axes(arr: np.ndarray) -> np.ndarray:
    """[num_rx, 1, num_tx, 1, num_paths] → [num_rx, num_tx, num_paths]."""
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


def compute_cir_for_current_rxs(
    scene,
    n_taps: int = 32,
    sample_rate_hz: float = 500e6,
    max_depth: int = 5,
    max_paths: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Tek TX, çoklu RX (oracle pass için) — CIR + path_params.

    Sadece bir TX sahnede aktifken çağrılır (oracle pass'inde geçici bir TX
    eklenir veya sahnede zaten 1 TX vardır). Tüm RX'ler için CIR hesaplanır.

    Döner:
        cir         : (num_rx, n_taps), complex64
        path_params : (num_rx, max_paths, 4), float32
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

    a_np = np.asarray(a)
    tau_np = np.asarray(tau)

    while a_np.ndim > 3:
        a_np = np.squeeze(a_np, axis=1) if a_np.shape[1] == 1 else a_np
        if a_np.ndim > 3 and a_np.shape[1] == 1:
            a_np = np.squeeze(a_np, axis=1)
        else:
            break
    while tau_np.ndim > 2:
        tau_np = np.squeeze(tau_np, axis=1) if tau_np.shape[1] == 1 else tau_np
        if tau_np.ndim > 2 and tau_np.shape[1] == 1:
            tau_np = np.squeeze(tau_np, axis=1)
        else:
            break
    if a_np.ndim == 3 and a_np.shape[-1] == 1:
        a_np = a_np[..., 0]

    num_rx = a_np.shape[0]
    cir_out = np.zeros((num_rx, n_taps), dtype=np.complex64)
    path_params = np.zeros((num_rx, max_paths, 4), dtype=np.float32)

    tap_dt = 1.0 / sample_rate_hz
    for r in range(num_rx):
        amps = a_np[r]
        delays = tau_np[r]
        valid_mask = np.isfinite(delays) & (delays >= 0)
        valid_amps = amps[valid_mask]
        valid_delays = delays[valid_mask]

        n_valid = int(valid_amps.size)
        if n_valid > 0:
            mag = np.abs(valid_amps)
            order = np.argsort(-mag)[:max_paths]
            sel_a = valid_amps[order]
            sel_t = valid_delays[order]
            n_sel = int(sel_a.size)
            path_params[r, :n_sel, 0] = np.real(sel_a).astype(np.float32)
            path_params[r, :n_sel, 1] = np.imag(sel_a).astype(np.float32)
            path_params[r, :n_sel, 2] = sel_t.astype(np.float32)
            path_params[r, :n_sel, 3] = 1.0

        for amp, d in zip(valid_amps, valid_delays):
            tap_idx = int(round(float(d) / tap_dt))
            if 0 <= tap_idx < n_taps:
                cir_out[r, tap_idx] += np.complex64(amp)

    return cir_out, path_params


def compute_bistatic_cir_round_robin(
    scene,
    master_nodes: list[dict[str, Any]],
    n_taps: int = 32,
    sample_rate_hz: float = 500e6,
    max_depth: int = 5,
    max_paths: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """4 master UWB cihazı bistatic round-robin TDMA → 6 unique link.

    Sionna 2.0'da 4 master hem TX hem RX olarak sahnede aktif. Tek
    PathSolver çağrısı ile 4×4 matris (her TX'ten her RX'e) elde edilir.
    Self-loop (i==i) atılır, C(4,2) = 6 unique bistatic link extract edilir.

    Reciprocity: TX i → RX j ile TX j → RX i kanalı kanal teorisinde
    aynıdır. Pratikte iki yön ortalanır (gürültü temizleme).

    Master sırası: nodes config'inden geldiği sırada (M1, M2, M3, M4).
    Link sırası: combinations(range(4), 2) = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
                 → 6 link.

    Döner:
        cir         : (6, n_taps), complex64
        path_params : (6, max_paths, 4), float32
    """
    from sionna.rt import PathSolver  # type: ignore

    n_masters = len(master_nodes)
    if n_masters != 4:
        raise ValueError(
            f"Hibrit topoloji 4 UWB master beklenir, gelen: {n_masters}"
        )

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

    a_np = _squeeze_antenna_axes(np.asarray(a))
    tau_np = _squeeze_antenna_axes(np.asarray(tau))

    if a_np.ndim == 4 and a_np.shape[-1] == 1:
        a_np = a_np[..., 0]
    if tau_np.ndim == 4 and tau_np.shape[-1] == 1:
        tau_np = tau_np[..., 0]

    # a_np: [num_rx, num_tx, num_paths] = [4, 4, P]
    link_pairs = list(combinations(range(n_masters), 2))  # 6 unique
    n_links = len(link_pairs)

    tap_dt = 1.0 / sample_rate_hz
    cir_out = np.zeros((n_links, n_taps), dtype=np.complex64)
    path_params = np.zeros((n_links, max_paths, 4), dtype=np.float32)

    for link_idx, (i, j) in enumerate(link_pairs):
        # Reciprocity: TX i → RX j ile TX j → RX i ortalaması
        a_ij = a_np[j, i, :]  # RX j, TX i
        a_ji = a_np[i, j, :]  # RX i, TX j
        tau_ij = tau_np[j, i, :]
        tau_ji = tau_np[i, j, :]

        # İki yönü birleştir (concat) — daha fazla path örneği
        a_combined = np.concatenate([a_ij, a_ji])
        tau_combined = np.concatenate([tau_ij, tau_ji])

        valid_mask = np.isfinite(tau_combined) & (tau_combined >= 0)
        valid_a = a_combined[valid_mask]
        valid_t = tau_combined[valid_mask]

        # Path params: amplitude'a göre en güçlü max_paths
        if valid_a.size > 0:
            mag = np.abs(valid_a)
            order = np.argsort(-mag)[:max_paths]
            sel_a = valid_a[order]
            sel_t = valid_t[order]
            n_sel = int(sel_a.size)
            path_params[link_idx, :n_sel, 0] = np.real(sel_a).astype(np.float32)
            path_params[link_idx, :n_sel, 1] = np.imag(sel_a).astype(np.float32)
            path_params[link_idx, :n_sel, 2] = sel_t.astype(np.float32)
            path_params[link_idx, :n_sel, 3] = 1.0

        # CIR binning (tüm valid path'ler) — reciprocity ortalaması için /2
        for amp, d in zip(valid_a, valid_t):
            tap_idx = int(round(float(d) / tap_dt))
            if 0 <= tap_idx < n_taps:
                cir_out[link_idx, tap_idx] += np.complex64(amp) * 0.5

    return cir_out, path_params


# =====================================================================
# TEK ÖRNEK ÜRETİMİ
# =====================================================================


def build_one_paired_sample(
    scene,
    cfg: dict[str, Any],
    sample_rate_hz: float,
    n_taps: int,
    rng: np.random.Generator,
    max_depth: int = 5,
) -> dict[str, np.ndarray]:
    """Tek paired sample (oracle + bistatic anchor).

    REVİZE 2026-05-15: Anchor artık 4 anchor → 6 bistatic link
    (4 master arası, round-robin TDMA simülasyonu).
    """
    from sionna.rt import Transmitter  # type: ignore

    n_slots = cfg["shelf"]["n_slots"]
    fill_prob = cfg["randomization"]["slot_fill_probability"]
    master_nodes = get_master_nodes(cfg)

    # 1) Stochastic seçim (UWB ile aynı sıralama — paired RNG)
    slot_filled = (rng.random(n_slots) < fill_prob).astype(np.uint8)
    material_ids = np.where(
        slot_filled == 1,
        rng.integers(1, 5, size=n_slots),
        0,
    ).astype(np.uint8)
    data_ids = rng.integers(0, 16, size=n_slots).astype(np.uint8)
    data_bits = int_to_data_bits(data_ids)
    codewords = hamming_encode_7_4(data_bits)

    # 2) Materyal atama
    assign_slot_materials(scene, slot_filled, material_ids, cfg, rng)

    # 3) ORACLE pass — 6 RX slot merkezleri, 1 master TX (M1) ile path-solve
    #    Master M1'in TX'ini geçici olarak ekle (M1 zaten anchor pass'inde de TX)
    #    Oracle pass için tek TX yeter (slot merkezlerinde RX'ler)
    m1 = master_nodes[0]
    oracle_tx = Transmitter(name="oracle_tx", position=list(m1["position_m"]))
    scene.add(oracle_tx)
    oracle_rxs = add_oracle_receivers(scene, cfg["slot_centers_m"])
    cir_oracle, path_params_oracle = compute_cir_for_current_rxs(
        scene, n_taps=n_taps, sample_rate_hz=sample_rate_hz, max_depth=max_depth,
    )  # (6, n_taps), (6, 20, 4)
    remove_receivers(scene, oracle_rxs)
    try:
        scene.remove("oracle_tx")
    except Exception:
        pass

    # 4) BISTATIC ANCHOR pass — 4 master TX+RX, round-robin TDMA, 6 unique link
    txs, rxs = add_all_uwb_masters(scene, master_nodes)
    cir_anchor, path_params_anchor = compute_bistatic_cir_round_robin(
        scene, master_nodes=master_nodes,
        n_taps=n_taps, sample_rate_hz=sample_rate_hz, max_depth=max_depth,
    )  # (6, n_taps), (6, 20, 4)
    remove_all_uwb_masters(scene, txs, rxs)

    # 5) Rezonatör imzası — dolu slotlar için (oracle ve anchor)
    # float() cast: YAML 1.2'de "500e6" string olarak parse olabilir
    band_center = float(cfg["uwb"]["freq_hz"])
    band_width = float(cfg["uwb"]["bandwidth_hz"])

    for slot_idx in range(n_slots):
        if slot_filled[slot_idx] == 1:
            cir_oracle[slot_idx] = inject_resonator_into_cir(
                cir_oracle[slot_idx], codewords[slot_idx],
                sample_rate_hz=sample_rate_hz,
                band_center_hz=band_center, band_width_hz=band_width,
                rng=rng,
            )
            for a_idx in range(cir_anchor.shape[0]):
                cir_anchor[a_idx] = inject_resonator_into_cir(
                    cir_anchor[a_idx], codewords[slot_idx],
                    sample_rate_hz=sample_rate_hz,
                    band_center_hz=band_center, band_width_hz=band_width,
                    rng=rng,
                )

    # 6) Anchor'a gürültü uygula (oracle clean kalır — teacher hedefi)
    cir_anchor = apply_all_anchor_noise(
        cir_anchor,
        rng=rng,
        snr_db=30.0,
        gain_db_range=tuple(cfg["randomization"]["noise"]["agc_gain_db_range"]),
        cfo_phase_max_rad=float(cfg["randomization"]["noise"]["cfo_phase_max_rad"]),
        wifi_interferer_max=int(cfg["randomization"]["noise"]["wifi_interferer_count_max"]),
        bt_burst_prob=float(cfg["randomization"]["noise"]["bt_burst_dropout_prob"]),
    )

    return {
        "cir_oracle":  complex_to_realimag(cir_oracle),    # (6, 32, 2)
        "cir_anchor":  complex_to_realimag(cir_anchor),    # (6, 32, 2)  REVİZE: 4→6
        "path_params_oracle": path_params_oracle,           # (6, 20, 4)
        "path_params_anchor": path_params_anchor,           # (6, 20, 4)  REVİZE: 4→6
        "slot_labels": slot_filled,
        "material_labels": material_ids,
        "codeword_labels": codewords,
        "data_id_labels": data_ids,
    }


# =====================================================================
# CHECKPOINT YÖNETİMİ
# =====================================================================


def _save_checkpoint(buffers: dict[str, list], out_dir: str, idx: int) -> None:
    for key, buf in buffers.items():
        if not buf:
            continue
        np.save(os.path.join(out_dir, f"_part_{key}_{idx}.npy"), np.stack(buf))


def _merge_checkpoints(out_dir: str, key: str, final_name: str) -> None:
    import glob
    parts = sorted(
        glob.glob(os.path.join(out_dir, f"_part_{key}_*.npy")),
        key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
    )
    if not parts:
        print(f"[uyarı] {key} için parça bulunamadı, atlandı")
        return
    arrs = [np.load(p) for p in parts]
    merged = np.concatenate(arrs, axis=0)
    np.save(os.path.join(out_dir, final_name), merged)
    for p in parts:
        os.remove(p)
    print(f"  {final_name}: {merged.shape} {merged.dtype}")


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Aether Core — Teacher-Student UWB CIR generator (HİBRİT TDMA)")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=str, default="./data")
    parser.add_argument("--scene", type=str, default="scenes/aether_classroom.xml")
    parser.add_argument("--config", type=str, default="configs/scene.yaml")
    parser.add_argument("--n_taps", type=int, default=32)
    parser.add_argument("--sample_rate_hz", type=float, default=500e6)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    # FAZ 0.5 — Sensor Placement Randomization (varsayılan kapalı = 1 variant)
    parser.add_argument("--n_variants", type=int, default=1,
                        help="Aynı odada kaç farklı sensör yerleşimi üretilsin "
                             "(>=1; 1 = klasik sabit layout). UWB ve WiFi script'lerinde "
                             "AYNI değer + AYNI --seed → variant rotasyonu senkron")
    parser.add_argument("--layout_jitter_xy", type=float, default=0.3,
                        help="XY plane gaussian jitter std (metre)")
    parser.add_argument("--layout_jitter_z", type=float, default=0.1,
                        help="Z gaussian jitter std (metre)")
    parser.add_argument("--layout_clip_margin", type=float, default=0.3,
                        help="Oda kenarından minimum mesafe (metre)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[init] config: {args.config}")
    cfg = load_scene_config(args.config)

    n_master = len(get_master_nodes(cfg))
    n_esp32 = len(get_esp32_nodes(cfg))
    print(f"[init] HİBRİT topoloji: {n_master} master + {n_esp32 - n_master} satellite "
          f"(WiFi {n_esp32} düğüm, UWB {n_master} kit → C({n_master},2)={n_master*(n_master-1)//2} bistatic)")

    # FAZ 0.5 — layout varyantları
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
    print(f"[init] LAYOUT: {args.n_variants} variant × ~{args.n // args.n_variants} sample/variant "
          f"(jitter XY={args.layout_jitter_xy}m Z={args.layout_jitter_z}m, layout_seed={layout_seed})")

    print(f"[init] sahne yükleniyor: {args.scene}")
    scene = build_sionna_scene(args.scene, cfg, uwb_freq_hz=float(cfg["uwb"]["freq_hz"]))

    # Per-sample bağımsız seed — paired dataset garantisi
    master_rng = np.random.default_rng(args.seed)
    sample_seeds = master_rng.integers(0, 2**31 - 1, size=args.n)

    buffers: dict[str, list] = {
        "cir_oracle": [],
        "cir_anchor": [],
        "path_params_oracle": [],
        "path_params_anchor": [],
        "slot_labels": [],
        "material_labels": [],
        "codeword_labels": [],
        "data_id_labels": [],
    }

    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        tf = None

    current_v = -1
    cfg_variant = cfg  # ilk iterasyon override eder

    for i in tqdm(range(args.n), desc="generating"):
        v_idx = int(layout_ids[i])
        if v_idx != current_v:
            cfg_variant = {**cfg, "nodes": variants[v_idx]}
            current_v = v_idx
            tqdm.write(f"[layout] variant {v_idx} aktif (sample {i})")

        sample_rng = np.random.default_rng(int(sample_seeds[i]))
        sample = build_one_paired_sample(
            scene=scene, cfg=cfg_variant,
            sample_rate_hz=args.sample_rate_hz,
            n_taps=args.n_taps,
            rng=sample_rng,
            max_depth=args.max_depth,
        )
        for key in buffers:
            buffers[key].append(sample[key])

        if (i + 1) % args.checkpoint_every == 0:
            _save_checkpoint(buffers, args.out, i + 1)
            for key in buffers:
                buffers[key] = []
            if tf is not None:
                tf.keras.backend.clear_session()

    if any(len(v) > 0 for v in buffers.values()):
        _save_checkpoint(buffers, args.out, args.n)

    print("[merge] parçalar birleştiriliyor...")
    _merge_checkpoints(args.out, "cir_oracle",          "uwb_cir_oracle.npy")
    _merge_checkpoints(args.out, "cir_anchor",          "uwb_cir_anchor.npy")
    _merge_checkpoints(args.out, "path_params_oracle",  "path_params_oracle.npy")
    _merge_checkpoints(args.out, "path_params_anchor",  "path_params_anchor.npy")
    _merge_checkpoints(args.out, "slot_labels",         "slot_labels.npy")
    _merge_checkpoints(args.out, "material_labels",     "material_labels.npy")
    _merge_checkpoints(args.out, "codeword_labels",     "codeword_labels.npy")
    _merge_checkpoints(args.out, "data_id_labels",      "data_id_labels.npy")

    # FAZ 0.5 — layout_ids kaydı
    np.save(os.path.join(args.out, "layout_ids.npy"), layout_ids)

    # FAZ 0.5 — node pozisyonları (her variant) referans için diske
    layout_meta = {
        "n_variants": args.n_variants,
        "jitter_xy_m": args.layout_jitter_xy,
        "jitter_z_m": args.layout_jitter_z,
        "clip_margin_m": args.layout_clip_margin,
        "layout_seed": layout_seed,
        "variants": [
            [{"id": n["id"], "type": n.get("type"), "sensors": n.get("sensors", []),
              "position_m": list(n["position_m"])} for n in var]
            for var in variants
        ],
    }
    with open(os.path.join(args.out, "layout_meta.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(layout_meta, f, sort_keys=False)

    print("[done] tüm dosyalar kaydedildi:", args.out)
    print(f"[done] layout_ids.npy + layout_meta.yaml ({args.n_variants} variant)")


if __name__ == "__main__":
    main()
