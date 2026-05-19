"""MultiDomainDataset — 5 preset paired veri loader.

PDF Modül 6 — Training (training/train.py'da kullanılır).

Klasör yapısı:
    data/<preset>/<12 dosya>
        uwb_cir_oracle.npy     [N, 6, 32, 2]
        uwb_cir_anchor.npy     [N, 6, 32, 2]
        path_params_oracle.npy [N, 6, 20, 4]
        path_params_anchor.npy [N, 6, 20, 4]
        csivec_delta.npy       [N, 28, 108, 2]
        path_params_csi.npy    [N, 28, 20, 4]
        slot_labels.npy        [N, 6]
        material_labels.npy    [N, 6]
        codeword_labels.npy    [N, 6, 7]
        data_id_labels.npy     [N, 6]
        slot_labels_wifi.npy   [N, 6]  (paired check)
        material_labels_wifi.npy [N, 6]

Çıktı (her __getitem__):
    {
        'csi':       (28, 108, 2),    raw CSI (preprocess'siz)
        'uwb':       (6, 32, 2),      raw UWB anchor CIR
        'link_geo':  (28, 3),         link midpoint coordinates
        'labels': {
            'slot':     (6,),
            'material': (6,),
            'codeword': (6, 7),
        },
        'domain_id': int,             0..4 (preset index)
        'aux': {                       (return_aux=True ise)
            'oracle_cir':   (6, 32, 2),
            'path_oracle':  (6, 20, 4),
            'path_anchor':  (6, 20, 4),
            'path_csi':     (28, 20, 4),
        }
    }
"""

from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  # type: ignore


DEFAULT_PRESETS = [
    "classroom_default",
    "warehouse_large",
    "office_small",
    "lab_medium",
    "room_low_ceiling",
]


# ── Helper: link geometry per preset ───────────────────────


def compute_link_geo_per_preset(scene_yaml_path: str | Path) -> np.ndarray:
    """scene.yaml → ESP32 link midpoint koordinatları.

    Returns: (28, 3) float32
    """
    with open(scene_yaml_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return _link_geo_from_nodes(cfg['nodes'])


def _link_geo_from_nodes(nodes: list[dict]) -> np.ndarray:
    """Node listesinden ESP32 link midpoint koordinatları (28, 3) float32."""
    esp32 = np.array([
        n['position_m'] for n in nodes
        if 'esp32' in n.get('sensors', [])
    ], dtype=np.float32)
    pairs = list(combinations(range(len(esp32)), 2))
    return np.array([
        (esp32[i] + esp32[j]) / 2.0 for i, j in pairs
    ], dtype=np.float32)


def load_link_geo_variants(
    preset_dir: str | Path,
    fallback_scene_yaml: str | Path | None = None,
) -> list[np.ndarray]:
    """FAZ 0.5 — Preset klasöründen layout_meta.yaml oku, her variant için
    link_geo (28, 3) hesapla.

    Eğer layout_meta.yaml yoksa (eski FAZ 0 veri) tek-eleman liste döner;
    içerik fallback_scene_yaml'dan üretilir.

    Returns:
        list of np.ndarray, len = n_variants, her eleman (28, 3) float32
    """
    preset_dir = Path(preset_dir)
    meta_path = preset_dir / "layout_meta.yaml"
    if meta_path.exists():
        with open(meta_path, encoding='utf-8') as f:
            meta = yaml.safe_load(f)
        return [_link_geo_from_nodes(var) for var in meta['variants']]
    # Geriye dönük: tek layout, scene.yaml'dan
    if fallback_scene_yaml is None:
        raise FileNotFoundError(
            f"{meta_path} yok ve fallback_scene_yaml verilmedi"
        )
    return [compute_link_geo_per_preset(fallback_scene_yaml)]


# ── Dataset class ─────────────────────────────────────────


class MultiDomainDataset(Dataset):
    """5 preset paired multi-domain veri."""

    def __init__(
        self,
        data_root: str | Path = "data",
        config_root: str | Path = "configs",
        presets: list[str] | None = None,
        return_aux: bool = True,
        cache_link_geo: bool = True,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch gerekli")

        self.data_root = Path(data_root)
        self.config_root = Path(config_root)
        self.presets = presets or DEFAULT_PRESETS
        self.return_aux = return_aux

        # Per-preset veri yükle (memmap)
        self.preset_data: dict[str, dict[str, np.ndarray]] = {}
        self.preset_lens: dict[str, int] = {}
        self.cum_lens: list[int] = [0]
        self.domain_ids: dict[str, int] = {}
        # FAZ 0.5 — link_geo artık variant başına liste (preset → list[ndarray])
        self.link_geo_variants: dict[str, list[np.ndarray]] = {}

        for i, preset in enumerate(self.presets):
            pdir = self.data_root / preset
            if not pdir.exists():
                raise FileNotFoundError(f"Preset klasoru yok: {pdir}")
            self.preset_data[preset] = self._load_preset(pdir)
            n = self.preset_data[preset]["slot_labels"].shape[0]
            self.preset_lens[preset] = n
            self.cum_lens.append(self.cum_lens[-1] + n)
            self.domain_ids[preset] = i
            # Link geo: variant rotation aktifse layout_meta.yaml'dan, değilse scene.yaml
            scene_yaml = self.config_root / (
                "scene.yaml" if preset == "classroom_default"
                else f"scene_{preset}.yaml"
            )
            self.link_geo_variants[preset] = load_link_geo_variants(
                pdir, fallback_scene_yaml=scene_yaml,
            )

    def _load_preset(self, pdir: Path) -> dict[str, np.ndarray]:
        """Tek preset'in tüm .npy dosyalarını memmap olarak yükle."""
        files = {
            "csivec_delta": "csivec_delta.npy",
            "uwb_cir_anchor": "uwb_cir_anchor.npy",
            "uwb_cir_oracle": "uwb_cir_oracle.npy",
            "path_params_anchor": "path_params_anchor.npy",
            "path_params_oracle": "path_params_oracle.npy",
            "path_params_csi": "path_params_csi.npy",
            "slot_labels": "slot_labels.npy",
            "material_labels": "material_labels.npy",
            "codeword_labels": "codeword_labels.npy",
            "data_id_labels": "data_id_labels.npy",
        }
        out = {}
        for key, fname in files.items():
            p = pdir / fname
            if not p.exists():
                raise FileNotFoundError(f"{p}")
            out[key] = np.load(p, mmap_mode='r')

        # FAZ 0.5 — layout_ids opsiyonel (eski veri uyumlu: yoksa hepsi 0)
        lid_path = pdir / "layout_ids.npy"
        n = out["slot_labels"].shape[0]
        if lid_path.exists():
            out["layout_ids"] = np.load(lid_path, mmap_mode='r')
        else:
            out["layout_ids"] = np.zeros(n, dtype=np.uint8)
        return out

    def __len__(self) -> int:
        return self.cum_lens[-1]

    def _find_preset_local(self, idx: int) -> tuple[str, int]:
        """Global idx → (preset_name, local_idx)."""
        for i, preset in enumerate(self.presets):
            if idx < self.cum_lens[i + 1]:
                return preset, idx - self.cum_lens[i]
        raise IndexError(f"idx {idx} >= len {len(self)}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        preset, local = self._find_preset_local(idx)
        d = self.preset_data[preset]

        # FAZ 0.5 — sample'a göre doğru variant link_geo
        layout_id = int(d["layout_ids"][local])
        variants = self.link_geo_variants[preset]
        # Güvenlik: layout_id liste sınırını aşarsa 0'a düş
        if layout_id >= len(variants):
            layout_id = 0
        link_geo = variants[layout_id]

        sample = {
            "csi": np.asarray(d["csivec_delta"][local], dtype=np.float32),
            "uwb": np.asarray(d["uwb_cir_anchor"][local], dtype=np.float32),
            "link_geo": link_geo,
            "labels": {
                "slot": np.asarray(d["slot_labels"][local], dtype=np.int64),
                "material": np.asarray(d["material_labels"][local], dtype=np.int64),
                "codeword": np.asarray(d["codeword_labels"][local], dtype=np.float32),
            },
            "domain_id": self.domain_ids[preset],
            "layout_id": layout_id,
            "preset_name": preset,
        }

        if self.return_aux:
            sample["aux"] = {
                "oracle_cir": np.asarray(d["uwb_cir_oracle"][local], dtype=np.float32),
                "path_oracle": np.asarray(d["path_params_oracle"][local], dtype=np.float32),
                "path_anchor": np.asarray(d["path_params_anchor"][local], dtype=np.float32),
                "path_csi": np.asarray(d["path_params_csi"][local], dtype=np.float32),
            }

        return sample


# ── Collate function (DataLoader için) ────────────────────


def collate_with_preprocess(batch: list[dict[str, Any]],
                              apply_preprocess: bool = True
                              ) -> dict[str, Any]:
    """Batch toplama + opsiyonel preprocessing.

    Preprocessing batch-level uygulanır (vectorized).
    Output tensorler torch.float32 / torch.int64.
    """
    if HAS_TORCH:
        import torch as _torch
    else:
        raise ImportError("torch gerekli")

    # Stack raw arrays
    csi_raw = np.stack([b["csi"] for b in batch])          # (B, 28, 108, 2)
    uwb_raw = np.stack([b["uwb"] for b in batch])          # (B, 6, 32, 2)
    link_geo = np.stack([b["link_geo"] for b in batch])    # (B, 28, 3)

    if apply_preprocess:
        from preprocessing import preprocess_csi, preprocess_uwb
        csi_t = _torch.from_numpy(preprocess_csi(csi_raw, apply_dwt=False)).float()
        uwb_t = _torch.from_numpy(preprocess_uwb(uwb_raw)).float()
    else:
        csi_t = _torch.from_numpy(csi_raw).float()
        uwb_t = _torch.from_numpy(uwb_raw).float()

    out = {
        "csi": csi_t,
        "uwb": uwb_t,
        "link_geo": _torch.from_numpy(link_geo).float(),
        "labels": {
            "slot": _torch.from_numpy(
                np.stack([b["labels"]["slot"] for b in batch])
            ).long(),
            "material": _torch.from_numpy(
                np.stack([b["labels"]["material"] for b in batch])
            ).long(),
            "codeword": _torch.from_numpy(
                np.stack([b["labels"]["codeword"] for b in batch])
            ).float(),
        },
        "domain_id": _torch.tensor(
            [b["domain_id"] for b in batch], dtype=_torch.long
        ),
        "layout_id": _torch.tensor(
            [b.get("layout_id", 0) for b in batch], dtype=_torch.long
        ),
        "preset_names": [b["preset_name"] for b in batch],
    }

    # Aux varsa
    if "aux" in batch[0]:
        from preprocessing import preprocess_path_params
        path_csi = np.stack([b["aux"]["path_csi"] for b in batch])
        path_anc = np.stack([b["aux"]["path_anchor"] for b in batch])
        out["aux"] = {
            "oracle_cir": _torch.from_numpy(
                np.stack([b["aux"]["oracle_cir"] for b in batch])
            ).float(),
            "path_csi": _torch.from_numpy(
                preprocess_path_params(path_csi)
            ).float(),
            "path_anchor": _torch.from_numpy(
                preprocess_path_params(path_anc)
            ).float(),
        }

    return out


__all__ = [
    "DEFAULT_PRESETS", "MultiDomainDataset",
    "compute_link_geo_per_preset", "collate_with_preprocess",
    "load_link_geo_variants",
]
