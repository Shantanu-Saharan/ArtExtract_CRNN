"""
siglip2_linear_probe.py
=======================
Frozen-backbone linear probe for SigLIP 2 vision encoders.

This keeps the large SigLIP 2 vision tower frozen and trains only
lightweight task heads for artist / style / genre classification.
"""

import argparse
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, SiglipVisionModel

from datasets.dataset import WikiArtMultiTaskDataset
from utils.seed import set_seed


def compute_f1(preds, labels):
    mask = np.array(labels) >= 0
    if mask.sum() == 0:
        return 0.0
    return f1_score(
        np.array(labels)[mask],
        np.array(preds)[mask],
        average="macro",
        zero_division=0,
    )


class PILIdentity:
    def __call__(self, image):
        return image


class TaskHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        hidden_dim = max(512, in_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskProbe(nn.Module):
    def __init__(self, in_dim: int, num_artist_classes: int,
                 num_style_classes: int, num_genre_classes: int):
        super().__init__()
        self.artist_head = TaskHead(in_dim, num_artist_classes)
        self.style_head = TaskHead(in_dim, num_style_classes)
        self.genre_head = TaskHead(in_dim, num_genre_classes)

    def forward(self, feats):
        return {
            "artist": self.artist_head(feats),
            "style": self.style_head(feats),
            "genre": self.genre_head(feats),
        }


def build_collate_fn(processor):
    def collate_fn(batch):
        images = [image for image, _ in batch]
        labels = {
            "artist": torch.tensor([item[1]["artist"] for item in batch], dtype=torch.long),
            "style": torch.tensor([item[1]["style"] for item in batch], dtype=torch.long),
            "genre": torch.tensor([item[1]["genre"] for item in batch], dtype=torch.long),
        }
        enc = processor(images=images, return_tensors="pt")
        return enc, labels

    return collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="google/siglip2-so400m-patch16-384",
        help="HF model id or local directory containing a SigLIP 2 checkpoint.",
    )
    parser.add_argument("--cache_dir", default="weights/hf_cache")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--root_dir", default="wikiart")
    parser.add_argument("--num_artist_classes", type=int, default=25)
    parser.add_argument("--num_style_classes", type=int, default=27)
    parser.add_argument("--num_genre_classes", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", default="checkpoints/linear_probe_top25_siglip2.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    def amp_context():
        return torch.cuda.amp.autocast(dtype=torch.bfloat16) if amp_enabled else nullcontext()

    print(f"Device: {device}")
    print(f"SigLIP 2 source: {args.model_name_or_path}")

    processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=False,
    )
    backbone = SiglipVisionModel.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16 if amp_enabled else torch.float32,
    ).to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    feat_dim = backbone.config.hidden_size
    probe = MultiTaskProbe(
        in_dim=feat_dim,
        num_artist_classes=args.num_artist_classes,
        num_style_classes=args.num_style_classes,
        num_genre_classes=args.num_genre_classes,
    ).to(device)

    head_params = [p for p in probe.parameters() if p.requires_grad]
    n_head = sum(p.numel() for p in head_params)
    n_backbone = sum(p.numel() for p in backbone.parameters())
    print(f"Trainable params: {n_head:,} | Frozen backbone params: {n_backbone:,}")

    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    eval_aug = PILIdentity()

    train_ds = WikiArtMultiTaskDataset(
        args.train_csv,
        root_dir=args.root_dir,
        transform=train_aug,
    )
    val_ds = WikiArtMultiTaskDataset(
        args.val_csv,
        root_dir=args.root_dir,
        transform=eval_aug,
    )
    collate_fn = build_collate_fn(processor)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    optimizer = AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    active_tasks = ["artist", "style", "genre"]

    best_f1 = 0.0
    best_ep = -1

    for epoch in range(1, args.epochs + 1):
        probe.train()
        losses = []
        for batch_inputs, labels in tqdm(train_loader, desc=f"Ep{epoch} Train", leave=False):
            batch_inputs = {
                k: v.to(device, non_blocking=True) for k, v in batch_inputs.items()
            }
            batch = {
                t: labels[t].to(device, non_blocking=True) for t in active_tasks
            }

            with torch.no_grad():
                with amp_context():
                    feats = backbone(**batch_inputs).pooler_output
            feats = feats.float()
            outputs = probe(feats)

            loss = sum(
                criterion(outputs[t], batch[t])
                for t in active_tasks
                if batch[t].max() >= 0
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head_params, 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        probe.eval()
        preds_all = {t: [] for t in active_tasks}
        labels_all = {t: [] for t in active_tasks}
        with torch.no_grad():
            for batch_inputs, labels in tqdm(val_loader, desc=f"Ep{epoch} Val", leave=False):
                batch_inputs = {
                    k: v.to(device, non_blocking=True) for k, v in batch_inputs.items()
                }
                with amp_context():
                    feats = backbone(**batch_inputs).pooler_output
                feats = feats.float()
                outputs = probe(feats)
                for t in active_tasks:
                    preds_all[t].extend(outputs[t].argmax(1).cpu().tolist())
                    labels_all[t].extend(labels[t].tolist())

        f1s = {t: compute_f1(preds_all[t], labels_all[t]) for t in active_tasks}
        mean_f1 = sum(f1s.values()) / len(f1s)
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"Ep{epoch:3d} | Loss:{np.mean(losses):.4f} | "
            f"Artist:{f1s['artist']:.4f} Style:{f1s['style']:.4f} "
            f"Genre:{f1s['genre']:.4f} | MeanF1:{mean_f1:.4f} | LR:{lr_now:.2e}"
        )

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_ep = epoch
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "probe_state_dict": probe.state_dict(),
                    "val_macro_f1": best_f1,
                    "model_name_or_path": args.model_name_or_path,
                    "num_artist_classes": args.num_artist_classes,
                    "num_style_classes": args.num_style_classes,
                    "num_genre_classes": args.num_genre_classes,
                    "feature_dim": feat_dim,
                },
                args.save_path,
            )
            print(f"  ✓ Saved (MeanF1={best_f1:.4f})")

    print(f"\nBest MeanF1: {best_f1:.4f} at epoch {best_ep}")
    print(f"Checkpoint: {args.save_path}")


if __name__ == "__main__":
    main()
