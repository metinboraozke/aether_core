"""Eğitim script'i — baseline + DANN + Phase 1/2 + distillation.

Modül 6 implementasyonu (upcoming/06_TRAINING_ABLATION.txt).

Komut satırı:
    python -m training.train --phase=baseline --epochs=5 \
        --checkpoint_out=checkpoints/baseline.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_synthesis.multi_domain_dataset import (
    MultiDomainDataset, collate_with_preprocess, DEFAULT_PRESETS,
)
from models.fused_model import FusedCSIUWBNet, compute_loss, DEFAULT_LAMBDAS
from adaptation import (
    DomainDiscriminator, DANNWrapper, grl_lambda_schedule,
    freeze_encoder, unfreeze_all,
    StudentFusedNet, distillation_loss,
)


# ── Helper'lar ─────────────────────────────────────────────


def create_optimizer(model: torch.nn.Module, lr: float = 1e-3,
                      weight_decay: float = 1e-4
                      ) -> torch.optim.Optimizer:
    """AdamW optimizer, sadece requires_grad=True olan param'ler."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 2,
    eta_min: float = 1e-5,
    warmup_start_factor: float = 0.01,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Lineer warmup + cosine annealing (sequential).

    Multi-domain eğitim için TEK scheduler — tüm preset/variant'lar boyunca
    devam eder. Domain shift "anı" yok (interleaved batching), o yüzden
    restart gerekmez.

    Profil (total_epochs=40, warmup_epochs=2):
        Epoch 0-2: lr 1e-5 → 1e-3   (lineer warmup, büyük model + DANN için)
        Epoch 2-40: lr 1e-3 → 1e-5  (cosine annealing)

    Args:
        optimizer: AdamW (lr = peak lr, warmup start_factor ile çarpılır)
        total_epochs: tüm eğitim süresinin epoch sayısı
        warmup_epochs: lineer warmup süresi
        eta_min: cosine'in sonunda lr ulaşacağı taban
        warmup_start_factor: warmup başlangıcında lr * factor (0.01 = peak/100)

    Returns:
        SequentialLR — her epoch sonunda .step() çağrılmalı
    """
    if total_epochs <= warmup_epochs:
        raise ValueError(
            f"total_epochs ({total_epochs}) > warmup_epochs ({warmup_epochs}) olmalı"
        )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    total_epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    eta_min: float = 1e-5,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Convenience: AdamW + lineer warmup + cosine annealing birlikte."""
    opt = create_optimizer(model, lr=lr, weight_decay=weight_decay)
    sched = create_scheduler(
        opt, total_epochs=total_epochs,
        warmup_epochs=warmup_epochs, eta_min=eta_min,
    )
    return opt, sched


def save_checkpoint(model: torch.nn.Module, path: str | Path,
                     extras: dict | None = None) -> None:
    state = {"model": model.state_dict()}
    if extras:
        state["extras"] = extras
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model: torch.nn.Module, path: str | Path,
                     strict: bool = False) -> dict | None:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"], strict=strict)
    return state.get("extras")


# ── train_baseline ─────────────────────────────────────────


def train_baseline(
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    lambdas: dict | None = None,
    use_aux: bool = True,
    log_every: int = 10,
    verbose: bool = True,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip_norm: float = 1.0,
) -> list[dict]:
    """Sentetik veride supervised eğitim.

    scheduler verilirse her epoch sonunda .step() çağrılır ve history'ye
    `lr` alanı eklenir. Verilmezse LR sabit (geriye dönük uyumluluk).

    grad_clip_norm > 0 ise her step'te clip_grad_norm_ uygulanır (P1, default 1.0).
    grad_clip_norm <= 0 → clip kapalı.

    P2: NaN guard — epoch loss NaN ise en son sağlam state'e geri yüklenir,
    optimizer reset, epoch atlanır.
    """
    if lambdas is None:
        lambdas = DEFAULT_LAMBDAS

    history = []
    # P2: NaN guard için son sağlam state cache
    last_safe_state: dict | None = None
    nan_recovery_count = 0

    model.train()
    for epoch in range(epochs):
        epoch_losses: dict[str, list[float]] = {}
        t_start = time.perf_counter()
        for step, batch in enumerate(loader):
            csi = batch["csi"].to(device)
            uwb = batch["uwb"].to(device)
            link_geo = batch["link_geo"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            aux = None
            if use_aux and "aux" in batch:
                aux = {k: v.to(device) for k, v in batch["aux"].items()}
                # Sadece compute_loss'un beklediği anahtarlar
                aux = {
                    "oracle_cir": aux["oracle_cir"],
                    "path_csi": aux["path_csi"],
                    "path_uwb": aux["path_anchor"],
                }

            optimizer.zero_grad()
            out = model(csi, uwb, link_geo, return_aux=use_aux)
            losses = compute_loss(out, labels, aux_targets=aux, lambdas=lambdas)
            losses["total"].backward()
            # P1: gradient clipping (NaN patlamasını önler)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )
            optimizer.step()

            for k, v in losses.items():
                epoch_losses.setdefault(k, []).append(v.item())

            if verbose and step % log_every == 0:
                print(f"  ep {epoch+1}/{epochs} step {step:>3}: "
                      f"L={losses['total'].item():.4f}")

        epoch_avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        epoch_avg["epoch"] = epoch + 1
        epoch_avg["time_s"] = round(time.perf_counter() - t_start, 2)
        # LR'i (epoch sonu, scheduler.step öncesi) kayda al
        epoch_avg["lr"] = float(optimizer.param_groups[0]["lr"])

        # P2: NaN guard — epoch ortalaması finite mi?
        if not np.isfinite(epoch_avg.get("total", 0.0)):
            nan_recovery_count += 1
            print(f"[NAN-GUARD] ep {epoch+1} loss={epoch_avg['total']} → "
                  f"recovery (#{nan_recovery_count})")
            if last_safe_state is not None:
                # 1) Model'i son sağlam state'e geri yükle
                # Cache CPU'da → orijinal device'a tekrar gönder
                target_device = next(model.parameters()).device
                model.load_state_dict({
                    k: v.to(target_device) for k, v in last_safe_state.items()
                })
                # 2) Optimizer state'i IN-PLACE temizle (PyTorch sürüm uyumlu)
                #    optimizer.state container type'ını koru, sadece içini boşalt
                #    type(optimizer.state)() PyTorch 2.x'te yanlış container üretiyor
                #    → AdamW.step() KeyError fırlatıyor. clear() güvenli.
                optimizer.state.clear()
                # 3) Lr'i geçici olarak yarıya düşür (bir sonraki adım daha temkinli)
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] * 0.5
                print(f"[NAN-GUARD] son sağlam state yüklendi, "
                      f"optimizer.state.clear() + lr×0.5 = {optimizer.param_groups[0]['lr']:.2e}")
            else:
                print(f"[NAN-GUARD] sağlam state yok, epoch atlanıyor (cache henüz dolmadı)")
            # NaN epoch'u history'ye yine de kaydet (debug için)
            epoch_avg["nan_recovery"] = True
            history.append(epoch_avg)
            if scheduler is not None:
                scheduler.step()
            # 4 ardışık NaN'da eğitimi tamamen durdur (felaket recovery)
            if nan_recovery_count >= 4:
                print(f"[NAN-GUARD] {nan_recovery_count} ardışık NaN → eğitim durduruluyor")
                break
            continue
        else:
            # Loss sağlamsa NaN sayacını sıfırla (tek seferlik NaN affedilebilir)
            nan_recovery_count = 0

        # Loss sağlamsa state'i cache'le (CPU clone — bellek dostu)
        last_safe_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

        history.append(epoch_avg)

        # Multi-domain için tek scheduler (warmup + cosine) — restart yok
        if scheduler is not None:
            scheduler.step()

        # P6: per-component loss log
        if verbose:
            aux_sum = epoch_avg.get('csi_path', 0) + epoch_avg.get('uwb_path', 0)
            print(f"[ep {epoch+1}] L={epoch_avg['total']:.3f} "
                  f"(det={epoch_avg.get('detection', 0):.3f} "
                  f"rec={epoch_avg.get('recognition', 0):.3f} "
                  f"id={epoch_avg.get('identification', 0):.3f} "
                  f"tsdf={epoch_avg.get('tsdf', 0):.3f} "
                  f"aux={aux_sum:.3f} "
                  f"kd={epoch_avg.get('kd_oracle', 0):.3f}) "
                  f"lr={epoch_avg['lr']:.2e} "
                  f"({epoch_avg['time_s']}s)")
    return history


# ── train_dann_phase (DANN + supervised) ──────────────────


def train_dann_phase(
    model: torch.nn.Module,
    discriminator: torch.nn.Module,
    dann_wrapper: DANNWrapper,
    loader: DataLoader,
    epochs: int,
    optimizer_main: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    total_epochs: int,
    device: str = "cpu",
    lambdas: dict | None = None,
    use_aux: bool = True,
    verbose: bool = True,
    scheduler_main: torch.optim.lr_scheduler.LRScheduler | None = None,
    scheduler_disc: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip_norm: float = 1.0,
) -> list[dict]:
    """DANN ile (sentetik + simulated_real_domain_id) eğitim.

    Domain labels MultiDomainDataset'ten geliyor (5 preset = 5 domain).
    Burada binary için: classroom = 0, diğerleri = 1 (simulated 'real').

    scheduler_main / scheduler_disc verilirse her epoch sonunda .step().
    grad_clip_norm > 0 → main ve discriminator ayrı clip (P1).
    """
    history = []
    model.train()
    discriminator.train()

    for epoch in range(epochs):
        progress = (epoch + 1) / total_epochs
        epoch_losses: dict[str, list[float]] = {}
        for batch in loader:
            csi = batch["csi"].to(device)
            uwb = batch["uwb"].to(device)
            link_geo = batch["link_geo"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            domain_id = batch["domain_id"].to(device)
            # Binary domain: 0 = sim (classroom_default), 1 = "diğer" (simulated real)
            domain_binary = (domain_id > 0).long()
            aux = None
            if use_aux and "aux" in batch:
                aux = {
                    "oracle_cir": batch["aux"]["oracle_cir"].to(device),
                    "path_csi": batch["aux"]["path_csi"].to(device),
                    "path_uwb": batch["aux"]["path_anchor"].to(device),
                }

            optimizer_main.zero_grad()
            optimizer_disc.zero_grad()

            out = model(csi, uwb, link_geo, return_aux=True)
            L_task = compute_loss(out, labels, aux_targets=aux, lambdas=lambdas)
            L_dom, lam = dann_wrapper.compute_domain_loss(
                out["uwb_global_latent"], domain_binary, progress=progress
            )

            total = L_task["total"] + L_dom
            total.backward()
            # P1: gradient clipping (main + discriminator ayrı)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(), max_norm=grad_clip_norm
                )
            optimizer_main.step()
            optimizer_disc.step()

            for k, v in L_task.items():
                epoch_losses.setdefault(k, []).append(v.item())
            epoch_losses.setdefault("domain", []).append(L_dom.item())
            epoch_losses.setdefault("grl_lambda", []).append(lam)

        epoch_avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        epoch_avg["epoch"] = epoch + 1
        epoch_avg["lr_main"] = float(optimizer_main.param_groups[0]["lr"])
        epoch_avg["lr_disc"] = float(optimizer_disc.param_groups[0]["lr"])
        history.append(epoch_avg)

        if scheduler_main is not None:
            scheduler_main.step()
        if scheduler_disc is not None:
            scheduler_disc.step()

        if verbose:
            print(f"[DANN ep {epoch+1}] L_total={epoch_avg['total']:.3f}  "
                  f"L_dom={epoch_avg['domain']:.3f}  λ={epoch_avg['grl_lambda']:.3f}  "
                  f"lr_main={epoch_avg['lr_main']:.2e}")
    return history


# ── CLI entry ──────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["baseline", "dann", "distill"],
                   default="baseline")
    p.add_argument("--data_root", default="data")
    p.add_argument("--config_root", default="configs")
    p.add_argument("--presets", nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--checkpoint_in", default=None)
    p.add_argument("--checkpoint_out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no_aux", action="store_true")
    # LR scheduler (multi-domain için tek warmup + cosine)
    p.add_argument("--warmup_epochs", type=int, default=2,
                   help="Lineer warmup epoch sayısı (0 = scheduler kapalı)")
    p.add_argument("--eta_min", type=float, default=1e-5,
                   help="Cosine annealing'in ulaşacağı min lr")
    args = p.parse_args()

    print(f"[train] phase={args.phase} epochs={args.epochs} "
          f"batch={args.batch_size} device={args.device}")

    ds = MultiDomainDataset(
        data_root=args.data_root,
        config_root=args.config_root,
        presets=args.presets,
        return_aux=not args.no_aux,
    )
    print(f"[train] dataset: {len(ds)} sample × {len(ds.presets)} preset")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_with_preprocess,
    )

    model = FusedCSIUWBNet().to(args.device)
    if args.checkpoint_in:
        load_checkpoint(model, args.checkpoint_in)
        print(f"[train] checkpoint yüklendi: {args.checkpoint_in}")

    use_sched = args.warmup_epochs > 0 and args.epochs > args.warmup_epochs

    if args.phase == "baseline":
        if use_sched:
            opt, sched = create_optimizer_and_scheduler(
                model, total_epochs=args.epochs, lr=args.lr,
                warmup_epochs=args.warmup_epochs, eta_min=args.eta_min,
            )
            print(f"[train] LR scheduler: warmup {args.warmup_epochs} ep + cosine "
                  f"({args.lr:.0e} → {args.eta_min:.0e}) tek geçişli, multi-domain")
        else:
            opt, sched = create_optimizer(model, lr=args.lr), None
        train_baseline(model, loader, args.epochs, opt,
                        device=args.device, use_aux=not args.no_aux,
                        scheduler=sched)

    elif args.phase == "dann":
        disc = DomainDiscriminator(in_dim=128).to(args.device)
        dann = DANNWrapper(discriminator=disc, max_lambda=0.7, alpha=5.0)
        if use_sched:
            opt_main, sched_main = create_optimizer_and_scheduler(
                model, total_epochs=args.epochs, lr=args.lr,
                warmup_epochs=args.warmup_epochs, eta_min=args.eta_min,
            )
            opt_disc, sched_disc = create_optimizer_and_scheduler(
                disc, total_epochs=args.epochs, lr=args.lr,
                warmup_epochs=args.warmup_epochs, eta_min=args.eta_min,
            )
            print(f"[train] DANN LR scheduler: warmup {args.warmup_epochs} ep + cosine, "
                  f"main+disc ayrı (aynı profil)")
        else:
            opt_main = create_optimizer(model, lr=args.lr)
            opt_disc = create_optimizer(disc, lr=args.lr)
            sched_main = sched_disc = None
        train_dann_phase(
            model, disc, dann, loader,
            epochs=args.epochs,
            optimizer_main=opt_main, optimizer_disc=opt_disc,
            total_epochs=args.epochs,
            device=args.device, use_aux=not args.no_aux,
            scheduler_main=sched_main, scheduler_disc=sched_disc,
        )

    elif args.phase == "distill":
        student = StudentFusedNet().to(args.device)
        # Teacher = model (yüklü), Student = yeni light
        # Basit KD loop (eval/train alternative — Modül 6'da extend)
        teacher = model
        teacher.eval()
        opt_s = create_optimizer(student, lr=args.lr)
        for ep in range(args.epochs):
            for batch in loader:
                csi = batch["csi"].to(args.device)
                uwb = batch["uwb"].to(args.device)
                geo = batch["link_geo"].to(args.device)
                labels = {k: v.to(args.device) for k, v in batch["labels"].items()}
                with torch.no_grad():
                    t_out = teacher(csi, uwb, geo, return_aux=True)
                s_out = student(csi, uwb, geo)
                kd = distillation_loss(s_out, t_out, labels=labels)
                opt_s.zero_grad()
                kd["total"].backward()
                opt_s.step()
            print(f"[distill ep {ep+1}] KD total={kd['total'].item():.3f}")
        # Student'ı kaydet (model dahil değil)
        save_checkpoint(student, args.checkpoint_out)
        print(f"[train] student kaydedildi: {args.checkpoint_out}")
        return

    save_checkpoint(model, args.checkpoint_out)
    print(f"[train] model kaydedildi: {args.checkpoint_out}")


if __name__ == "__main__":
    main()
