#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from contextlib import nullcontext

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import AutoImageProcessor

from datasets.dataset import WikiArtMultiTaskDataset
from models.siglip2_multitask import Siglip2MultiTaskModel
from utils.losses import FocalLoss, make_class_weights_from_counts
from utils.seed import set_seed


TASKS = ["artist", "style", "genre"]


class PILIdentity:
    def __call__(self, image):
        return image


def build_train_transform(image_size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.70, 1.0),
            ratio=(0.85, 1.18),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.12, hue=0.03),
        transforms.RandomGrayscale(p=0.03),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))
        ], p=0.08),
    ])


def build_val_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.08), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
    ])


def build_collate_fn(processor):
    def collate_fn(batch):
        images = [image for image, _ in batch]
        labels = {
            "artist": torch.tensor([item[1]["artist"] for item in batch], dtype=torch.long),
            "style": torch.tensor([item[1]["style"] for item in batch], dtype=torch.long),
            "genre": torch.tensor([item[1]["genre"] for item in batch], dtype=torch.long),
        }
        enc = processor(images=images, return_tensors="pt")
        return enc["pixel_values"], labels

    return collate_fn


def compute_macro_f1(preds, labels):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    mask = labels >= 0
    if mask.sum() == 0:
        return 0.0
    return f1_score(labels[mask], preds[mask], average="macro", zero_division=0)


def rank0_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def gather_lists(local_store):
    world_size = dist.get_world_size()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_store)
    out = []
    for chunk in gathered:
        out.extend(chunk)
    return out


def gather_task_metrics(pred_store, label_store, top5_store):
    rank = dist.get_rank()
    metrics = {}
    for task in TASKS:
        preds = gather_lists(pred_store[task])
        labels = gather_lists(label_store[task])
        top5 = gather_lists(top5_store[task])
        if rank == 0:
            labels_np = np.asarray(labels)
            preds_np = np.asarray(preds)
            valid = labels_np >= 0
            if valid.sum() == 0:
                metrics[task] = {"macro_f1": 0.0, "accuracy": 0.0, "top5_accuracy": 0.0}
            else:
                metrics[task] = {
                    "macro_f1": compute_macro_f1(preds_np[valid], labels_np[valid]),
                    "accuracy": float((preds_np[valid] == labels_np[valid]).mean()),
                    "top5_accuracy": float(np.asarray(top5, dtype=np.float32)[valid].mean()),
                }
    if rank == 0:
        metrics["macro_f1"] = float(sum(metrics[t]["macro_f1"] for t in TASKS) / len(TASKS))
    else:
        metrics["macro_f1"] = 0.0
    return metrics


def build_class_weights(csv_path, dims):
    df = pd.read_csv(csv_path)
    out = {}
    mapping = {
        "artist": ("artist_label", dims["artist"]),
        "style": ("style_label", dims["style"]),
        "genre": ("genre_label", dims["genre"]),
    }
    for task, (col, num_classes) in mapping.items():
        counts = (
            df[col][df[col] >= 0]
            .value_counts()
            .reindex(range(num_classes), fill_value=0)
            .sort_index()
            .values
        )
        out[task] = make_class_weights_from_counts(counts, power=0.35)
    return out


def build_optimizer(raw_model, args):
    head_params = [p for n, p in raw_model.named_parameters() if not n.startswith("backbone.")]
    groups = [{"params": head_params, "lr": args.head_lr}]

    if args.use_llrd:
        vision = raw_model.backbone.vision_model
        stage_modules = [vision.embeddings] + list(vision.encoder.layers) + [
            vision.post_layernorm,
            vision.head,
        ]
        for depth, module in enumerate(reversed(stage_modules)):
            params = list(module.parameters())
            if not params:
                continue
            lr = args.backbone_lr * (args.llrd_decay ** depth)
            groups.append({"params": params, "lr": lr})
    else:
        groups.append({"params": raw_model.backbone.parameters(), "lr": args.backbone_lr})

    return AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))


def build_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio):
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def set_backbone_trainable(model, trainable):
    raw_model = model.module if hasattr(model, "module") else model
    for param in raw_model.backbone.parameters():
        param.requires_grad_(trainable)
    if trainable:
        raw_model.backbone.train()
    else:
        raw_model.backbone.eval()


def set_gradient_checkpointing(model, enabled):
    raw_model = model.module if hasattr(model, "module") else model
    if enabled:
        raw_model.gradient_checkpointing_enable()
    else:
        raw_model.gradient_checkpointing_disable()


def maybe_load_probe_heads(model, ckpt_path, device, rank):
    if not ckpt_path or not os.path.exists(ckpt_path):
        rank0_print(rank, f"[INFO] Probe checkpoint not found, skipping: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    probe_state = ckpt.get("probe_state_dict")
    if probe_state is None:
        rank0_print(rank, f"[INFO] No probe_state_dict found in: {ckpt_path}")
        return

    model_state = model.state_dict()
    filtered = {
        k: v for k, v in probe_state.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    rank0_print(
        rank,
        f"[INFO] Loaded probe heads from {ckpt_path} | "
        f"matched={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}",
    )


def maybe_load_init_weights(model, ckpt_path, device, rank):
    if not ckpt_path or not os.path.exists(ckpt_path):
        if ckpt_path:
            rank0_print(rank, f"[INFO] Init checkpoint not found, skipping: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    if "probe_state_dict" in ckpt:
        raw_state = ckpt["probe_state_dict"]
    elif "ema_state_dict" in ckpt:
        raw_state = ckpt["ema_state_dict"]
    else:
        raw_state = ckpt.get("model_state_dict", ckpt)

    model_state = model.state_dict()
    filtered = {
        k: v for k, v in raw_state.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    rank0_print(
        rank,
        f"[INFO] Warm-started model from {ckpt_path} | "
        f"matched={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}",
    )


def build_ema_model(model):
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema(ema_model, model, decay):
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for key, ema_val in ema_state.items():
        model_val = model_state[key].detach()
        if torch.is_floating_point(ema_val):
            ema_val.mul_(decay).add_(model_val, alpha=1.0 - decay)
        else:
            ema_val.copy_(model_val)


def maybe_load_resume(model, optimizer, scheduler, ckpt_path, device, rank, ema_model=None):
    if not ckpt_path or not os.path.exists(ckpt_path):
        if ckpt_path:
            rank0_print(rank, f"[INFO] Resume checkpoint not found, starting fresh: {ckpt_path}")
        return 0, -1.0

    ckpt = torch.load(ckpt_path, map_location=device)
    raw_state = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in raw_state.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    model.load_state_dict(filtered, strict=False)

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ema_model is not None:
        ema_state = ckpt.get("ema_state_dict")
        if ema_state is not None:
            filtered_ema = {
                k: v for k, v in ema_state.items()
                if k in ema_model.state_dict() and tuple(v.shape) == tuple(ema_model.state_dict()[k].shape)
            }
            ema_model.load_state_dict(filtered_ema, strict=False)
        else:
            ema_model.load_state_dict(model.state_dict(), strict=False)

    start_epoch = int(ckpt.get("epoch", 0))
    best_f1 = float(ckpt.get("best_val_macro_f1", ckpt.get("val_macro_f1", -1.0)))
    rank0_print(rank, f"[INFO] Resumed full fine-tune from {ckpt_path} at epoch {start_epoch}")
    return start_epoch, best_f1


def compute_task_loss(criterion, logits, targets):
    valid = targets >= 0
    if valid.sum() == 0:
        return None
    return criterion(logits[valid], targets[valid])


@torch.no_grad()
def tta_forward(model, pixel_values, n_views, amp_dtype):
    _, _, h, w = pixel_values.shape
    views = [pixel_values]
    if n_views >= 2:
        views.append(torch.flip(pixel_values, dims=[3]))
    for scale in [0.95, 1.05, 0.90, 1.10]:
        if len(views) >= n_views:
            break
        if scale < 1.0:
            nh, nw = int(h * scale), int(w * scale)
            y0, x0 = (h - nh) // 2, (w - nw) // 2
            crop = pixel_values[:, :, y0:y0 + nh, x0:x0 + nw]
            view = torch.nn.functional.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)
        else:
            nh, nw = int(h * scale), int(w * scale)
            up = torch.nn.functional.interpolate(pixel_values, size=(nh, nw), mode="bilinear", align_corners=False)
            y0, x0 = (nh - h) // 2, (nw - w) // 2
            view = up[:, :, y0:y0 + h, x0:x0 + w]
        views.append(view)

    acc = {task: [] for task in TASKS}
    for view in views[:n_views]:
        with autocast(dtype=amp_dtype):
            outputs = model(view)
        for task in TASKS:
            acc[task].append(outputs[task].float())
    return {task: torch.stack(acc[task]).mean(0) for task in TASKS}


def train_one_epoch(model, loader, criterions, optimizer, scheduler, scaler, device, args, amp_dtype):
    rank = dist.get_rank()
    model.train()
    losses = []
    pred_store = {task: [] for task in TASKS}
    label_store = {task: [] for task in TASKS}
    top5_store = {task: [] for task in TASKS}
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="Train", leave=False, disable=(rank != 0))
    for step, (pixel_values, labels) in enumerate(progress, start=1):
        pixel_values = pixel_values.to(device, non_blocking=True)
        batch = {task: labels[task].to(device, non_blocking=True) for task in TASKS}

        with autocast(dtype=amp_dtype):
            outputs = model(pixel_values)
            task_losses = []
            for task in TASKS:
                loss_t = compute_task_loss(criterions[task], outputs[task], batch[task])
                if loss_t is not None:
                    task_losses.append(getattr(args, f"{task}_weight") * loss_t)
            if not task_losses:
                continue
            total_loss = sum(task_losses) / args.accum_steps

        scaler.scale(total_loss).backward()

        if step % args.accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if args.use_ema:
                update_ema(args.ema_model, model.module, args.ema_decay)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        losses.append(float(total_loss.item()) * args.accum_steps)

        with torch.no_grad():
            for task in TASKS:
                logits = outputs[task]
                targets = batch[task]
                valid = targets >= 0
                if valid.sum() == 0:
                    continue
                logits_v = logits[valid]
                targets_v = targets[valid]
                pred_store[task].extend(logits_v.argmax(1).cpu().tolist())
                label_store[task].extend(targets_v.cpu().tolist())
                k = min(5, logits_v.shape[1])
                top5 = logits_v.topk(k, dim=1).indices
                top5_store[task].extend((targets_v.unsqueeze(1) == top5).any(dim=1).cpu().tolist())

        if rank == 0:
            progress.set_postfix(loss=f"{np.mean(losses):.4f}", lr=f"{max(pg['lr'] for pg in optimizer.param_groups):.2e}")

    metrics = {"loss": float(np.mean(losses)) if losses else 0.0}
    gathered = gather_task_metrics(pred_store, label_store, top5_store)
    if rank == 0:
        for task in TASKS:
            metrics[task] = gathered[task]
    metrics["macro_f1"] = gathered["macro_f1"]
    return metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterions, device, args, amp_dtype):
    rank = dist.get_rank()
    model.eval()
    losses = []
    pred_store = {task: [] for task in TASKS}
    label_store = {task: [] for task in TASKS}
    top5_store = {task: [] for task in TASKS}

    for pixel_values, labels in tqdm(loader, desc="Val", leave=False, disable=(rank != 0)):
        pixel_values = pixel_values.to(device, non_blocking=True)
        batch = {task: labels[task].to(device, non_blocking=True) for task in TASKS}

        if args.use_tta:
            outputs = tta_forward(model, pixel_values, n_views=args.tta_views, amp_dtype=amp_dtype)
        else:
            with autocast(dtype=amp_dtype):
                outputs = model(pixel_values)
            outputs = {task: outputs[task].float() for task in TASKS}

        batch_losses = []
        for task in TASKS:
            loss_t = compute_task_loss(criterions[task], outputs[task], batch[task])
            if loss_t is not None:
                batch_losses.append(getattr(args, f"{task}_weight") * loss_t)
        if batch_losses:
            losses.append(float(sum(batch_losses).item()))

        for task in TASKS:
            logits = outputs[task]
            targets = batch[task]
            valid = targets >= 0
            if valid.sum() == 0:
                continue
            logits_v = logits[valid]
            targets_v = targets[valid]
            pred_store[task].extend(logits_v.argmax(1).cpu().tolist())
            label_store[task].extend(targets_v.cpu().tolist())
            k = min(5, logits_v.shape[1])
            top5 = logits_v.topk(k, dim=1).indices
            top5_store[task].extend((targets_v.unsqueeze(1) == top5).any(dim=1).cpu().tolist())

    metrics = {"loss": float(np.mean(losses)) if losses else 0.0}
    gathered = gather_task_metrics(pred_store, label_store, top5_store)
    if rank == 0:
        for task in TASKS:
            metrics[task] = gathered[task]
    metrics["macro_f1"] = gathered["macro_f1"]
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Full SigLIP 2 fine-tuning for top-25 multitask art classification")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--cache_dir", default="weights/hf_cache")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--root_dir", default="wikiart")
    parser.add_argument("--num_artist_classes", type=int, default=25)
    parser.add_argument("--num_style_classes", type=int, default=27)
    parser.add_argument("--num_genre_classes", type=int, default=7)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=float, default=1.0)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=1)
    parser.add_argument("--head_lr", type=float, default=8e-5)
    parser.add_argument("--backbone_lr", type=float, default=1.5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--min_lr_ratio", type=float, default=0.10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artist_weight", type=float, default=0.35)
    parser.add_argument("--style_weight", type=float, default=0.41)
    parser.add_argument("--genre_weight", type=float, default=0.24)
    parser.add_argument("--artist_focal_gamma", type=float, default=1.0)
    parser.add_argument("--style_focal_gamma", type=float, default=1.25)
    parser.add_argument("--genre_focal_gamma", type=float, default=1.0)
    parser.add_argument("--artist_label_smoothing", type=float, default=0.03)
    parser.add_argument("--style_label_smoothing", type=float, default=0.05)
    parser.add_argument("--genre_label_smoothing", type=float, default=0.02)
    parser.add_argument("--use_llrd", action="store_true", default=False)
    parser.add_argument("--llrd_decay", type=float, default=0.92)
    parser.add_argument("--use_patch_style", action="store_true", default=False)
    parser.add_argument("--use_style_fusion", action="store_true", default=False)
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=False)
    parser.add_argument("--ema_decay", type=float, default=0.9996)
    parser.add_argument("--use_tta", action="store_true", default=False)
    parser.add_argument("--tta_views", type=int, default=2)
    parser.add_argument("--use_bfloat16", action="store_true", default=False)
    parser.add_argument("--probe_checkpoint", default="")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--save_path", default="checkpoints/best_top25_siglip2_ft.pt")
    parser.add_argument("--history_json", default="")
    parser.add_argument("--early_stop_patience", type=int, default=6)
    args = parser.parse_args()

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("train_siglip2_multitask_ddp.py must be launched with torchrun")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed + rank)

    dims = {
        "artist": args.num_artist_classes,
        "style": args.num_style_classes,
        "genre": args.num_genre_classes,
    }
    amp_dtype = torch.bfloat16 if args.use_bfloat16 else torch.float16
    scaler = GradScaler(enabled=not args.use_bfloat16)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=False,
    )
    collate_fn = build_collate_fn(processor)

    train_dataset = WikiArtMultiTaskDataset(
        args.train_csv,
        root_dir=args.root_dir,
        transform=build_train_transform(args.image_size),
    )
    val_dataset = WikiArtMultiTaskDataset(
        args.val_csv,
        root_dir=args.root_dir,
        transform=build_val_transform(args.image_size),
    )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )

    rank0_print(rank, f"Device: {device} | world_size={world_size}")
    rank0_print(rank, f"Model: {args.model_name_or_path}")
    rank0_print(rank, f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")
    rank0_print(
        rank,
        f"Per-GPU batch={args.batch_size} | accum={args.accum_steps} | "
        f"effective batch={args.batch_size * world_size * args.accum_steps}",
    )
    rank0_print(
        rank,
        f"Epochs={args.epochs} | freeze_backbone_epochs={args.freeze_backbone_epochs} | "
        f"TTA={args.use_tta} ({args.tta_views} views)",
    )
    rank0_print(
        rank,
        f"Style path: patch_only={args.use_patch_style} fusion={args.use_style_fusion} | "
        f"EMA={args.use_ema} decay={args.ema_decay}",
    )

    class_weights = build_class_weights(args.train_csv, dims)

    model = Siglip2MultiTaskModel(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        num_artist_classes=args.num_artist_classes,
        num_style_classes=args.num_style_classes,
        num_genre_classes=args.num_genre_classes,
        use_patch_style=args.use_patch_style,
        use_style_fusion=args.use_style_fusion,
    ).to(device)
    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        pass
    elif args.init_checkpoint:
        maybe_load_init_weights(model, args.init_checkpoint, device, rank)
    else:
        maybe_load_probe_heads(model, args.probe_checkpoint, device, rank)

    ema_model = build_ema_model(model).to(device) if args.use_ema else None
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )

    optimizer = build_optimizer(model.module, args)
    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.accum_steps))
    total_steps = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(round(updates_per_epoch * args.warmup_epochs))
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    start_epoch, best_f1 = maybe_load_resume(
        model.module, optimizer, scheduler, args.resume_checkpoint, device, rank, ema_model=ema_model
    )
    if args.use_ema and not args.resume_checkpoint:
        ema_model.load_state_dict(model.module.state_dict(), strict=False)

    criterions = {
        "artist": FocalLoss(
            weight=class_weights["artist"].to(device),
            gamma=args.artist_focal_gamma,
            label_smoothing=args.artist_label_smoothing,
        ),
        "style": FocalLoss(
            weight=class_weights["style"].to(device),
            gamma=args.style_focal_gamma,
            label_smoothing=args.style_label_smoothing,
        ),
        "genre": FocalLoss(
            weight=class_weights["genre"].to(device),
            gamma=args.genre_focal_gamma,
            label_smoothing=args.genre_label_smoothing,
        ),
    }

    history = []
    no_improve = 0
    args.ema_model = ema_model

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        backbone_on = epoch > args.freeze_backbone_epochs
        set_backbone_trainable(model, backbone_on)
        if args.use_gradient_checkpointing:
            # Checkpointing the fully frozen backbone triggers warnings and gives no benefit.
            set_gradient_checkpointing(model, backbone_on)

        rank0_print(rank, f"\nEpoch {epoch}/{args.epochs} | backbone_trainable={backbone_on}")

        train_metrics = train_one_epoch(
            model, train_loader, criterions, optimizer, scheduler, scaler, device, args, amp_dtype
        )
        eval_model = ema_model if args.use_ema else model
        val_metrics = validate_one_epoch(eval_model, val_loader, criterions, device, args, amp_dtype)
        current_lr = max(pg["lr"] for pg in optimizer.param_groups)

        if rank == 0:
            print(
                f"Train | Loss:{train_metrics['loss']:.4f} MeanF1:{train_metrics['macro_f1']:.4f} | "
                f"Artist:F1:{train_metrics['artist']['macro_f1']:.4f} Acc:{train_metrics['artist']['accuracy']:.4f} Top5:{train_metrics['artist']['top5_accuracy']:.4f} | "
                f"Style:F1:{train_metrics['style']['macro_f1']:.4f} Acc:{train_metrics['style']['accuracy']:.4f} Top5:{train_metrics['style']['top5_accuracy']:.4f} | "
                f"Genre:F1:{train_metrics['genre']['macro_f1']:.4f} Acc:{train_metrics['genre']['accuracy']:.4f} Top5:{train_metrics['genre']['top5_accuracy']:.4f}"
            )
            print(
                f"Val   | Loss:{val_metrics['loss']:.4f} MeanF1:{val_metrics['macro_f1']:.4f} | "
                f"Artist:F1:{val_metrics['artist']['macro_f1']:.4f} Acc:{val_metrics['artist']['accuracy']:.4f} Top5:{val_metrics['artist']['top5_accuracy']:.4f} | "
                f"Style:F1:{val_metrics['style']['macro_f1']:.4f} Acc:{val_metrics['style']['accuracy']:.4f} Top5:{val_metrics['style']['top5_accuracy']:.4f} | "
                f"Genre:F1:{val_metrics['genre']['macro_f1']:.4f} Acc:{val_metrics['genre']['accuracy']:.4f} Top5:{val_metrics['genre']['top5_accuracy']:.4f} | "
                f"LR:{current_lr:.2e}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                no_improve = 0
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "ema_state_dict": ema_model.state_dict() if args.use_ema else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_macro_f1": best_f1,
                    "val_macro_f1": val_metrics["macro_f1"],
                    "model_name_or_path": args.model_name_or_path,
                    "num_artist_classes": args.num_artist_classes,
                    "num_style_classes": args.num_style_classes,
                    "num_genre_classes": args.num_genre_classes,
                    "backbone": "siglip2_so400m_patch16_384",
                    "config": {
                        "backbone": "siglip2_so400m_patch16_384",
                        "use_patch_style": args.use_patch_style,
                        "use_style_fusion": args.use_style_fusion,
                        "use_ema": args.use_ema,
                        "ema_decay": args.ema_decay,
                        "head_lr": args.head_lr,
                        "backbone_lr": args.backbone_lr,
                        "freeze_backbone_epochs": args.freeze_backbone_epochs,
                        "probe_checkpoint": args.probe_checkpoint,
                        "init_checkpoint": args.init_checkpoint,
                    },
                }
                torch.save(save_dict, args.save_path)
                print(
                    f"✓ Saved best → {args.save_path} "
                    f"(MeanF1={best_f1:.4f} Artist:{val_metrics['artist']['macro_f1']:.4f} "
                    f"Style:{val_metrics['style']['macro_f1']:.4f} Genre:{val_metrics['genre']['macro_f1']:.4f})"
                )
            else:
                no_improve += 1
                print(f"No improvement {no_improve} epoch(s) | best MeanF1={best_f1:.4f}")

        stop_flag = torch.tensor(0, device=device)
        if rank == 0 and no_improve >= args.early_stop_patience:
            stop_flag.fill_(1)
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item():
            rank0_print(rank, f"Early stopping at epoch {epoch}")
            break

    if rank == 0:
        print(f"\n=== SigLIP 2 fine-tuning complete ===")
        print(f"Best MeanF1: {best_f1:.4f}")
        if args.history_json:
            os.makedirs(os.path.dirname(args.history_json) or ".", exist_ok=True)
            with open(args.history_json, "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=2)
            print(f"History: {args.history_json}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
