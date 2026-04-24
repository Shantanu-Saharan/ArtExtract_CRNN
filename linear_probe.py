"""
Train a frozen-backbone linear probe for the multitask classifiers.

The script is used for the EVA02 and CLIP-H probe checkpoints that feed the
final stacked ensemble.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from models.multitask_crnn import MultiTaskCRNN
from datasets.dataset import WikiArtMultiTaskDataset
from utils.transforms import get_train_transforms, get_val_transforms
from utils.seed import set_seed


def compute_f1(preds, labels):
    mask = np.array(labels) >= 0
    if mask.sum() == 0:
        return 0.0
    return f1_score(np.array(labels)[mask], np.array(preds)[mask],
                    average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",          default="clip_vitl14",
                        choices=["clip_vitl14", "clip_vith14", "dinov2_vitl14",
                                 "eva02_large", "convnext_large", "convnextv2_large"])
    parser.add_argument("--pretrained_path",   default="")
    parser.add_argument("--train_csv",         required=True)
    parser.add_argument("--val_csv",           required=True)
    parser.add_argument("--root_dir",          default="wikiart")
    parser.add_argument("--num_artist_classes",type=int, default=23)
    parser.add_argument("--num_style_classes", type=int, default=27)
    parser.add_argument("--num_genre_classes", type=int, default=8)
    parser.add_argument("--use_hybrid",        action="store_true", default=True)
    parser.add_argument("--image_size",        type=int, default=448)
    parser.add_argument("--batch_size",        type=int, default=64)
    parser.add_argument("--epochs",            type=int, default=20)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--weight_decay",      type=float, default=0.01)
    parser.add_argument("--num_workers",       type=int, default=4)
    parser.add_argument("--save_path",         default="checkpoints/linear_probe.pt")
    parser.add_argument("--seed",              type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}  |  Linear probe (backbone FROZEN)")

    # ── Build model ────────────────────────────────────────────────────────
    model = MultiTaskCRNN(
        num_artist_classes=args.num_artist_classes,
        num_style_classes=args.num_style_classes,
        num_genre_classes=args.num_genre_classes,
        backbone=args.backbone,
        pretrained_path=args.pretrained_path,
        use_cross_attn=True,
    ).to(device)

    # Freeze backbone entirely — only heads train
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    model.backbone.eval()

    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone.") and p.requires_grad]
    n_head = sum(p.numel() for p in head_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_head:,} / {n_total:,} ({100*n_head/n_total:.1f}%)")

    # ── Data ───────────────────────────────────────────────────────────────
    train_ds = WikiArtMultiTaskDataset(
        args.train_csv, root_dir=args.root_dir,
        transform=get_train_transforms(args.image_size),
    )
    val_ds = WikiArtMultiTaskDataset(
        args.val_csv, root_dir=args.root_dir,
        transform=get_val_transforms(args.image_size),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Optimizer — only head params ───────────────────────────────────────
    optimizer = AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    active_tasks = ["artist", "style", "genre"]
    best_f1, best_ep = 0.0, -1

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        model.backbone.eval()   # keep BN/dropout in eval mode for frozen backbone
        losses = []
        for images, labels in tqdm(train_loader, desc=f"Ep{epoch} Train", leave=False):
            images = images.to(device, non_blocking=True)
            batch  = {t: labels[t].to(device) for t in active_tasks}
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(images)
                loss = sum(
                    criterion(outputs[t], batch[t])
                    for t in active_tasks
                    if batch[t].max() >= 0
                )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head_params, 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        preds_all  = {t: [] for t in active_tasks}
        labels_all = {t: [] for t in active_tasks}
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Ep{epoch} Val", leave=False):
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(images)
                for t in active_tasks:
                    preds_all[t].extend(outputs[t].argmax(1).cpu().tolist())
                    labels_all[t].extend(labels[t].tolist())

        f1s = {t: compute_f1(preds_all[t], labels_all[t]) for t in active_tasks}
        mean_f1 = sum(f1s.values()) / len(f1s)
        lr_now  = scheduler.get_last_lr()[0]

        print(f"Ep{epoch:3d} | Loss:{np.mean(losses):.4f} | "
              f"Artist:{f1s['artist']:.4f} Style:{f1s['style']:.4f} "
              f"Genre:{f1s['genre']:.4f} | MeanF1:{mean_f1:.4f} | LR:{lr_now:.2e}")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_ep = epoch
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "val_macro_f1":       best_f1,
                "num_artist_classes": args.num_artist_classes,
                "num_style_classes":  args.num_style_classes,
                "num_genre_classes":  args.num_genre_classes,
                "backbone":           args.backbone,
                "config": {
                    "backbone": args.backbone,
                    "use_cross_attn": True,
                },
            }, args.save_path)
            print(f"  ✓ Saved (MeanF1={best_f1:.4f})")

    print(f"\nBest MeanF1: {best_f1:.4f} at epoch {best_ep}")
    print(f"Checkpoint:  {args.save_path}")
    print()
    print("Next — warm-start full fine-tuning from this checkpoint:")
    print(f"  torchrun --nproc_per_node=2 train_multitask_ddp.py \\")
    print(f"    --backbone {args.backbone} \\")
    print(f"    --resume_checkpoint {args.save_path} \\")
    print(f"    --freeze_epochs 0 \\")
    print(f"    --lr 5e-5 --use_bfloat16 ...")


if __name__ == "__main__":
    main()
