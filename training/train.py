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
) -> list[dict]:
    """Sentetik veride supervised eğitim."""
    if lambdas is None:
        lambdas = DEFAULT_LAMBDAS

    history = []
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
            optimizer.step()

            for k, v in losses.items():
                epoch_losses.setdefault(k, []).append(v.item())

            if verbose and step % log_every == 0:
                print(f"  ep {epoch+1}/{epochs} step {step:>3}: "
                      f"L={losses['total'].item():.4f}")

        epoch_avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        epoch_avg["epoch"] = epoch + 1
        epoch_avg["time_s"] = round(time.perf_counter() - t_start, 2)
        history.append(epoch_avg)
        if verbose:
            print(f"[ep {epoch+1}] L_total={epoch_avg['total']:.4f}  "
                  f"det={epoch_avg.get('detection', 0):.3f}  "
                  f"rec={epoch_avg.get('recognition', 0):.3f}  "
                  f"id={epoch_avg.get('identification', 0):.3f}  "
                  f"tsdf={epoch_avg.get('tsdf', 0):.3f}  "
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
) -> list[dict]:
    """DANN ile (sentetik + simulated_real_domain_id) eğitim.

    Domain labels MultiDomainDataset'ten geliyor (5 preset = 5 domain).
    Burada binary için: classroom = 0, diğerleri = 1 (simulated 'real').
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
            optimizer_main.step()
            optimizer_disc.step()

            for k, v in L_task.items():
                epoch_losses.setdefault(k, []).append(v.item())
            epoch_losses.setdefault("domain", []).append(L_dom.item())
            epoch_losses.setdefault("grl_lambda", []).append(lam)

        epoch_avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        epoch_avg["epoch"] = epoch + 1
        history.append(epoch_avg)
        if verbose:
            print(f"[DANN ep {epoch+1}] L_total={epoch_avg['total']:.3f}  "
                  f"L_dom={epoch_avg['domain']:.3f}  λ={epoch_avg['grl_lambda']:.3f}")
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

    if args.phase == "baseline":
        opt = create_optimizer(model, lr=args.lr)
        train_baseline(model, loader, args.epochs, opt,
                        device=args.device, use_aux=not args.no_aux)

    elif args.phase == "dann":
        disc = DomainDiscriminator(in_dim=128).to(args.device)
        dann = DANNWrapper(discriminator=disc, max_lambda=0.7, alpha=5.0)
        opt_main = create_optimizer(model, lr=args.lr)
        opt_disc = create_optimizer(disc, lr=args.lr)
        train_dann_phase(
            model, disc, dann, loader,
            epochs=args.epochs,
            optimizer_main=opt_main, optimizer_disc=opt_disc,
            total_epochs=args.epochs,
            device=args.device, use_aux=not args.no_aux,
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
