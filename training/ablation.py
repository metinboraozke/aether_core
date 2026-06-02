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
from training.train import (
    train_baseline, create_optimizer, create_scheduler,
    save_checkpoint,
)
from training.eval import evaluate


ABLATION_VARIANTS = [
    {"name": "full",        "path_aux": True,  "dann": True,  "oracle_kd": True},
    {"name": "no_path_aux", "path_aux": False, "dann": True,  "oracle_kd": True},
    {"name": "no_dann",     "path_aux": True,  "dann": False, "oracle_kd": True},
    {"name": "no_kd",       "path_aux": True,  "dann": True,  "oracle_kd": False},
    {"name": "minimal",     "path_aux": False, "dann": False, "oracle_kd": False},
]


def build_lambdas(variant: dict) -> dict:
    """Varyant flag'lerine göre lambda sözlüğü.

    NOT (P3 — 2026-06-02): DEFAULT_LAMBDAS değiştirilmez, manuel override
    KAPALI. Varyantlar sadece flag (path_aux, dann, oracle_kd) ile fark
    edebilir; lambdaları manuel set etme ablation runner üzerinden imkansız.

    Sebep: Mayıs sonu production run'da `rec=5.0, id=3.0, aux=0.05, kd=0.05`
    manuel override edildi → recognition + identification head'leri dead,
    encoder physics-aware feature öğrenemedi (teacher signal kapalı). Bu
    hata tekrarlanmasın.
    """
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
    warmup_epochs: int = 2,
    eta_min: float = 1e-5,
    val_interval: int = 5,
    patience: int = 5,
    checkpoint_dir: str = "checkpoints",
) -> dict:
    """Tek bir varyantı eğit + eval.

    LR scheduler (warmup + cosine) tüm varyantlara aynı uygulanır → adil
    karşılaştırma. warmup_epochs=0 verilirse scheduler kapanır (sabit lr).

    P4: Her val_interval epoch'ta validation çağrısı (mini test 2 epoch'ta
        otomatik devre dışı — val sadece eğitim sonunda).
    P5: Best checkpoint (val det_f1 max) diske kaydedilir; sonunda best
        state geri yüklenir.
    P7: Early stopping — patience val_interval boyunca iyileşme yoksa dur.
    """
    print(f"\n{'='*60}")
    print(f"ABLATION: {variant['name']}")
    print(f"{'='*60}")
    print(f"  path_aux={variant['path_aux']}  dann={variant['dann']}  oracle_kd={variant['oracle_kd']}")

    lambdas = build_lambdas(variant)
    use_aux = variant["path_aux"] or variant["oracle_kd"]

    model = FusedCSIUWBNet().to(device)
    optimizer = create_optimizer(model, lr=lr)
    if warmup_epochs > 0 and epochs > warmup_epochs:
        scheduler = create_scheduler(
            optimizer, total_epochs=epochs,
            warmup_epochs=warmup_epochs, eta_min=eta_min,
        )
    else:
        scheduler = None  # Çok az epoch (mini test) → scheduler devre dışı

    # P4 + P5 + P7: per-epoch val + best ckpt + early stop
    best_f1 = -1.0
    best_state: dict | None = None
    best_epoch = 0
    no_improve = 0
    train_history: list[dict] = []
    val_history: list[dict] = []
    early_stopped = False
    do_per_epoch_val = epochs > val_interval  # mini test'te per-epoch val devre dışı

    t_start = time.perf_counter()

    if not do_per_epoch_val:
        # Mini test path: tek pass train + final eval (eski davranış)
        train_history = train_baseline(
            model, train_loader, epochs=epochs, optimizer=optimizer,
            device=device, lambdas=lambdas, use_aux=use_aux,
            verbose=False, log_every=999, scheduler=scheduler,
        )
    else:
        # Production path: val_interval bloklarına böl
        ep_done = 0
        while ep_done < epochs:
            chunk = min(val_interval, epochs - ep_done)
            hist = train_baseline(
                model, train_loader, epochs=chunk, optimizer=optimizer,
                device=device, lambdas=lambdas, use_aux=use_aux,
                verbose=True, log_every=999, scheduler=scheduler,
            )
            # Train history'deki epoch numaralarını absolute'a kaydır
            for h in hist:
                h["epoch"] = ep_done + h["epoch"]
            train_history.extend(hist)
            ep_done += chunk

            # P4: validation
            model.eval()
            vm = evaluate(model, val_loader, device=device, return_aux=False)
            model.train()
            vm["epoch"] = ep_done
            val_history.append(vm)
            print(f"  [val ep {ep_done}/{epochs}] "
                  f"det_f1={vm['det_f1']:.3f} "
                  f"rec={vm.get('rec_accuracy', 0):.3f} "
                  f"id={vm.get('id_bit_accuracy', 0):.3f}")

            # P5: best ckpt
            if vm["det_f1"] > best_f1:
                best_f1 = vm["det_f1"]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                best_epoch = ep_done
                no_improve = 0
                print(f"  [best] yeni best det_f1={best_f1:.3f} @ ep {best_epoch}")
            else:
                no_improve += 1
                # P7: early stopping
                if no_improve >= patience:
                    print(f"  [early-stop] {patience} val_interval iyileşme yok, "
                          f"ep {ep_done}'da dur")
                    early_stopped = True
                    break

    train_time = round(time.perf_counter() - t_start, 2)

    print(f"  Eğitim süresi: {train_time}s ({len(train_history)} epoch)")
    if train_history:
        print(f"  Train L final: {train_history[-1].get('total', 0):.4f}")
    if scheduler is not None and train_history:
        lrs = [h.get('lr', 0) for h in train_history]
        print(f"  LR profili: {lrs[0]:.2e} → {lrs[-1]:.2e} "
              f"(peak {max(lrs):.2e})")

    # P5: best state'i geri yükle + checkpoint kaydet
    ckpt_path = None
    if best_state is not None:
        model.load_state_dict(best_state)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        ckpt_path = f"{checkpoint_dir}/{variant['name']}_best.pt"
        save_checkpoint(model, ckpt_path, extras={
            "best_epoch": best_epoch,
            "best_f1": best_f1,
            "variant": variant,
            "lambdas": lambdas,
            "early_stopped": early_stopped,
        })
        print(f"  [save] best checkpoint: {ckpt_path} (ep {best_epoch}, "
              f"det_f1={best_f1:.3f})")

    # Final eval (best state ile, mini test'te sadece final state)
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
        "val_history": val_history,
        "val_metrics": val_metrics,
        "best_epoch": best_epoch,
        "best_f1": best_f1 if best_f1 >= 0 else None,
        "early_stopped": early_stopped,
        "checkpoint_path": ckpt_path,
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
    warmup_epochs: int = 2,
    eta_min: float = 1e-5,
    val_interval: int = 5,
    patience: int = 5,
    num_workers: int = 2,
    checkpoint_dir: str = "checkpoints",
    out_path: str = "training/ablation_results.json",
    variants: list[dict] | None = None,
    seed: int = 42,
) -> list[dict]:
    """5 varyantı sırayla koştur, sonuçları JSON'a kaydet.

    Tüm varyantlar AYNI LR schedule (warmup + cosine) ile eğitilir →
    flag'ler dışındaki tek değişken kalmaz.

    P4/P5/P7: val_interval, patience kontrolü her varyanta uygulanır.
    P8: DataLoader num_workers + pin_memory (cuda'da) + persistent_workers.
    """
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

    # P8: DataLoader optimization (num_workers + pin_memory + persistent)
    pin = (device == "cuda")
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=persistent,
        collate_fn=collate_with_preprocess,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=persistent,
        collate_fn=collate_with_preprocess,
    )
    print(f"DataLoader: num_workers={num_workers}, pin_memory={pin}, "
          f"persistent={persistent}")

    variants = variants or ABLATION_VARIANTS
    results = []
    for v in variants:
        res = run_single_ablation(
            v, train_loader, val_loader,
            epochs=epochs, device=device, lr=lr,
            warmup_epochs=warmup_epochs, eta_min=eta_min,
            val_interval=val_interval, patience=patience,
            checkpoint_dir=checkpoint_dir,
        )
        results.append(res)

    # JSON kaydet
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[ablation] sonuçlar kaydedildi: {out_path}")

    # Özet tablo
    print(f"\n{'='*88}")
    print(f"{'VARYANT':15s} {'F1':>8s} {'REC_ACC':>10s} {'ID_BIT':>10s} "
          f"{'BEST_EP':>8s} {'EARLY':>7s} {'TIME':>8s}")
    print(f"{'-'*88}")
    for r in results:
        m = r["val_metrics"]
        best_ep_str = str(r.get("best_epoch") or "-")
        early_str = "Y" if r.get("early_stopped") else "N"
        print(f"{r['name']:15s} "
              f"{m.get('det_f1', 0):>8.3f} "
              f"{m.get('rec_accuracy', 0):>10.3f} "
              f"{m.get('id_bit_accuracy', 0):>10.3f} "
              f"{best_ep_str:>8s} {early_str:>7s} "
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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_epochs", type=int, default=2,
                   help="Lineer warmup epoch'ları (0 = scheduler kapalı)")
    p.add_argument("--eta_min", type=float, default=1e-5,
                   help="Cosine annealing min lr")
    p.add_argument("--val_interval", type=int, default=5,
                   help="Her N epoch'ta validation (P4)")
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping val_interval cinsinden (P7)")
    p.add_argument("--num_workers", type=int, default=2,
                   help="DataLoader worker sayısı (P8, GPU önerisi 2)")
    p.add_argument("--checkpoint_dir", default="checkpoints",
                   help="Best checkpoint kayıt dizini (P5)")
    args = p.parse_args()
    run_ablation(
        epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, out_path=args.out,
        lr=args.lr, warmup_epochs=args.warmup_epochs, eta_min=args.eta_min,
        val_interval=args.val_interval, patience=args.patience,
        num_workers=args.num_workers, checkpoint_dir=args.checkpoint_dir,
    )
