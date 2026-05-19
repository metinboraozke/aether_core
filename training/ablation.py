"""Ablation çalışmaları — 5 varyantı sırayla koştur.

Modül 6 — Ablation (upcoming/06_TRAINING_ABLATION.txt).

Varyantlar:
    full:           path_aux=True,  dann=True,  oracle_kd=True   (referans)
    no_path_aux:    path_aux=False, dann=True,  oracle_kd=True
    no_dann:        path_aux=True,  dann=False, oracle_kd=True
    no_kd:          path_aux=True,  dann=True,  oracle_kd=False
    minimal:        path_aux=False, dann=False, oracle_kd=False  (worst case)

Her varyant için:
    1. Eğitim (eşit epoch'la)
    2. Validation set üzerinde eval
    3. Sonuç tablosuna yaz

Çıktı: training/ablation_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from data_synthesis.multi_domain_dataset import (
    MultiDomainDataset, collate_with_preprocess,
)
from models.fused_model import FusedCSIUWBNet, DEFAULT_LAMBDAS
from training.train import train_baseline, create_optimizer
from training.eval import evaluate


ABLATION_VARIANTS = [
    {"name": "full",        "path_aux": True,  "dann": True,  "oracle_kd": True},
    {"name": "no_path_aux", "path_aux": False, "dann": True,  "oracle_kd": True},
    {"name": "no_dann",     "path_aux": True,  "dann": False, "oracle_kd": True},
    {"name": "no_kd",       "path_aux": True,  "dann": True,  "oracle_kd": False},
    {"name": "minimal",     "path_aux": False, "dann": False, "oracle_kd": False},
]


def build_lambdas(variant: dict) -> dict:
    """Varyant flag'lerine göre lambda sözlüğü."""
    lam = dict(DEFAULT_LAMBDAS)
    if not variant["path_aux"]:
        lam["csi_path"] = 0.0
        lam["uwb_path"] = 0.0
    if not variant["oracle_kd"]:
        lam["kd_oracle"] = 0.0
    # dann flag'i train loop'unda kullanılır, lambda'da değil
    return lam


def run_single_ablation(
    variant: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: str,
    lr: float = 1e-3,
) -> dict:
    """Tek bir varyantı eğit + eval."""
    print(f"\n{'='*60}")
    print(f"ABLATION: {variant['name']}")
    print(f"{'='*60}")
    print(f"  path_aux={variant['path_aux']}  dann={variant['dann']}  oracle_kd={variant['oracle_kd']}")

    lambdas = build_lambdas(variant)
    use_aux = variant["path_aux"] or variant["oracle_kd"]

    model = FusedCSIUWBNet().to(device)
    optimizer = create_optimizer(model, lr=lr)

    t_start = time.perf_counter()
    train_history = train_baseline(
        model, train_loader, epochs=epochs, optimizer=optimizer,
        device=device, lambdas=lambdas, use_aux=use_aux,
        verbose=False, log_every=999,
    )
    train_time = round(time.perf_counter() - t_start, 2)

    print(f"  Eğitim süresi: {train_time}s")
    print(f"  Train L final: {train_history[-1]['total']:.4f}")

    # Eval
    val_metrics = evaluate(model, val_loader, device=device, return_aux=False)
    print(f"  Val metrics:")
    for k, v in val_metrics.items():
        print(f"    {k:30s} = {v:.4f}")

    return {
        "name": variant["name"],
        "config": variant,
        "lambdas": lambdas,
        "train_time_s": train_time,
        "train_history": train_history,
        "val_metrics": val_metrics,
    }


def run_ablation(
    data_root: str = "data",
    config_root: str = "configs",
    presets: list[str] | None = None,
    epochs: int = 2,
    batch_size: int = 16,
    val_split: float = 0.2,
    device: str = "cpu",
    lr: float = 1e-3,
    out_path: str = "training/ablation_results.json",
    variants: list[dict] | None = None,
    seed: int = 42,
) -> list[dict]:
    """5 varyantı sırayla koştur, sonuçları JSON'a kaydet."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = MultiDomainDataset(
        data_root=data_root, config_root=config_root,
        presets=presets, return_aux=True,
    )
    n_total = len(ds)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Dataset: {n_total} sample → train={n_train}, val={n_val}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_with_preprocess,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_with_preprocess,
    )

    variants = variants or ABLATION_VARIANTS
    results = []
    for v in variants:
        res = run_single_ablation(
            v, train_loader, val_loader,
            epochs=epochs, device=device, lr=lr,
        )
        results.append(res)

    # JSON kaydet
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[ablation] sonuçlar kaydedildi: {out_path}")

    # Özet tablo
    print(f"\n{'='*70}")
    print(f"{'VARYANT':15s} {'F1':>8s} {'REC_ACC':>10s} {'ID_BIT':>10s} {'TIME':>8s}")
    print(f"{'-'*70}")
    for r in results:
        m = r["val_metrics"]
        print(f"{r['name']:15s} "
              f"{m.get('det_f1', 0):>8.3f} "
              f"{m.get('rec_accuracy', 0):>10.3f} "
              f"{m.get('id_bit_accuracy', 0):>10.3f} "
              f"{r['train_time_s']:>7.1f}s")

    return results


__all__ = ["ABLATION_VARIANTS", "run_single_ablation", "run_ablation",
            "build_lambdas"]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="training/ablation_results.json")
    args = p.parse_args()
    run_ablation(
        epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, out_path=args.out,
    )
