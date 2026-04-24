"""
Distributed multitask training — artist, style, genre.
"""

import argparse
import copy
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets.dataset import WikiArtMultiTaskDataset
from models.multitask_crnn import MultiTaskCRNN
from utils.losses import make_class_weights_from_counts, FocalLoss, ArcFaceLoss
from utils.metrics import compute_classification_metrics, numpy_mean
from utils.seed import set_seed
from utils.transforms import get_train_transforms, get_train_transforms_strong, get_val_transforms
from utils.visualization import save_training_curves


# --- scheduler ---

def build_scheduler(optimizer, args):
    warmup = args.freeze_epochs + 2

    if getattr(args, "no_sgdr_restart", False):
        remaining = max(args.epochs - warmup, 1)
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=remaining,
            eta_min=args.lr * 1e-3,
        )
    else:
        base_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.sgdr_t0,
            T_mult=args.sgdr_tmult,
            eta_min=args.lr * 1e-3,
        )

    class WarmupWrapper:
        def __init__(self, sched, warmup_ep):
            self.sched = sched
            self.warmup_ep = warmup_ep
            self.epoch = 0

        def step(self):
            self.epoch += 1
            if self.epoch <= self.warmup_ep:
                scale = self.epoch / self.warmup_ep
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * scale
            else:
                self.sched.step()

    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    return WarmupWrapper(base_scheduler, warmup)


# --- SAM ---

class SAM(torch.optim.Optimizer):
    def __init__(self, base_optimizer, rho=0.05, adaptive=False):
        assert rho >= 0.0, f"rho must be >= 0, got {rho}"
        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(base_optimizer.param_groups, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# --- LLRD ---

def build_optimizer_llrd(raw_model, args):
    backbone_lr = args.lr * args.backbone_lr_scale
    decay = getattr(args, "llrd_decay", 0.85)
    bb = raw_model.backbone

    if hasattr(bb, "features"):
        stage_groups = [
            list(bb.features[0:2].parameters()),
            list(bb.features[2:4].parameters()),
            list(bb.features[4:6].parameters()),
            list(bb.features[6:].parameters()),
        ]

    elif hasattr(bb, "blocks"):
        n_blocks = len(bb.blocks)
        q = n_blocks // 4
        stage_groups = [
            list(bb.patch_embed.parameters()),
            list(bb.blocks[:q].parameters()),
            list(bb.blocks[q:2*q].parameters()),
            list(bb.blocks[2*q:3*q].parameters()),
            list(bb.blocks[3*q:].parameters()),
        ]
        if hasattr(bb, "norm"):
            stage_groups[-1] = stage_groups[-1] + list(bb.norm.parameters())

    else:
        backbone_param_ids = {id(p) for p in bb.parameters()}
        head_params = [p for p in raw_model.parameters() if id(p) not in backbone_param_ids]
        return AdamW([
            {"params": head_params,       "lr": args.lr},
            {"params": list(bb.parameters()), "lr": backbone_lr},
        ], weight_decay=args.weight_decay)

    backbone_param_ids = {id(p) for p in bb.parameters()}
    head_params = [p for p in raw_model.parameters() if id(p) not in backbone_param_ids]

    param_groups = [{"params": head_params, "lr": args.lr}]
    for i, stage_params in enumerate(reversed(stage_groups)):   # deepest first → lowest decay
        if not stage_params:
            continue
        lr_i = backbone_lr * (decay ** i)
        param_groups.append({"params": stage_params, "lr": lr_i})

    return AdamW(param_groups, weight_decay=args.weight_decay)


# --- augmentation ---

def mixup_batch(images, labels_dict, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    B = images.size(0)
    idx = torch.randperm(B, device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, (labels_dict, idx, lam)


def cutmix_batch(images, labels_dict, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha))
    B, C, H, W = images.shape
    idx = torch.randperm(B, device=images.device)
    cut_ratio = math.sqrt(1 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam_actual = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, (labels_dict, idx, lam_actual)


def apply_augmentation(images, labels_dict, p=0.25):
    if random.random() > p:
        return images, None
    if random.random() < 0.5:
        return mixup_batch(images, labels_dict, alpha=0.4)
    return cutmix_batch(images, labels_dict, alpha=1.0)


def build_task_batches(images, labels_dict, active_tasks, arcface_losses=None, augmentation_prob=0.25):
    arcface_tasks = set((arcface_losses or {}).keys())
    mixed_images, mixup_info = apply_augmentation(images, labels_dict, p=augmentation_prob)

    task_images = {}
    task_mixup_info = {}
    for task in active_tasks:
        if task in arcface_tasks:
            task_images[task] = images
            task_mixup_info[task] = None
        else:
            task_images[task] = mixed_images
            task_mixup_info[task] = mixup_info
    return task_images, task_mixup_info


# --- TTA ---

@torch.no_grad()
def tta_forward(model, images, active_tasks, n_views=6, return_embeddings=False):
    B, C, H, W = images.shape

    views = [images, torch.flip(images, dims=[3])]

    for scale in [0.95, 1.05, 0.90, 1.10, 0.85, 1.15, 0.80, 1.20]:
        if len(views) >= n_views:
            break
        if scale < 1.0:
            new_h, new_w = int(H * scale), int(W * scale)
            y0, x0 = (H - new_h) // 2, (W - new_w) // 2
            cropped = images[:, :, y0:y0 + new_h, x0:x0 + new_w]
            view = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
        else:
            new_h, new_w = int(H * scale), int(W * scale)
            upscaled = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
            y0, x0 = (new_h - H) // 2, (new_w - W) // 2
            view = upscaled[:, :, y0:y0 + H, x0:x0 + W]
        views.append(view)

    all_logits = {t: [] for t in active_tasks}
    for view in views[:n_views]:
        with autocast(dtype=_AMP_DTYPE):
            out = model(view, return_embeddings=return_embeddings)
        for t in active_tasks:
            all_logits[t].append(out[t].float())

    return {t: torch.stack(all_logits[t]).mean(0) for t in active_tasks}


# --- logit adjustment ---

def get_class_counts_from_csv(csv_path, num_artist, num_style, num_genre, hybrid=False):
    df = pd.read_csv(csv_path)
    counts = {}
    for col, n, key in [
        ("artist_label", num_artist, "artist"),
        ("style_label",  num_style,  "style"),
        ("genre_label",  num_genre,  "genre"),
    ]:
        if hybrid and key == "artist":
            valid_df = df[df[col] >= 0]
            cnt = valid_df[col].value_counts().reindex(range(n), fill_value=0).sort_index().values
        else:
            cnt = df[col].value_counts().reindex(range(n), fill_value=0).sort_index().values
        counts[key] = cnt.tolist()
    return counts


def build_logit_adjustments(class_counts_dict, active_tasks, device, tau_dict=None):
    if tau_dict is None:
        tau_dict = {t: 0.0 for t in active_tasks}
    adjustments = {}
    for t in active_tasks:
        tau = tau_dict.get(t, 0.0)
        if tau <= 0.0:
            continue   # no adjustment for this task
        counts = torch.tensor(class_counts_dict[t], dtype=torch.float32).clamp(min=1.0)
        log_prior = torch.log(counts / counts.sum())
        adjustments[t] = (tau * log_prior).to(device)
    return adjustments


def apply_logit_adjustment(outputs_dict, adjustments):
    return {t: outputs_dict[t] - adjustments[t] for t in outputs_dict if t in adjustments}


# --- class weights ---

def build_multitask_class_weights(csv_path, num_artist, num_style, num_genre, power, hybrid=False):
    df = pd.read_csv(csv_path)
    weights = {}
    for col, n, key in [
        ("artist_label", num_artist, "artist"),
        ("style_label",  num_style,  "style"),
        ("genre_label",  num_genre,  "genre"),
    ]:
        if hybrid and key == "artist":
            valid_df = df[df[col] >= 0]
            counts = valid_df[col].value_counts().reindex(range(n), fill_value=0).sort_index().values
        else:
            counts = df[col].value_counts().reindex(range(n), fill_value=0).sort_index().values
        weights[key] = make_class_weights_from_counts(counts, power=power)
    return weights


# --- loss helpers ---

def compute_mixed_loss(criterion, output, batch, mixup_info):
    if mixup_info is None:
        return criterion(output, batch)
    labels_dict, idx, lam = mixup_info
    return lam * criterion(output, batch) + (1 - lam) * criterion(output, batch[idx])


def set_backbone_trainable(model, trainable):
    base = model.module if hasattr(model, "module") else model
    for param in base.backbone.parameters():
        param.requires_grad = trainable


class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device
        self.num_updates = 0
        src = model.module if hasattr(model, "module") else model
        self.shadow = copy.deepcopy(src)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.shadow.to(device)

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        actual_decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        src = model.module if hasattr(model, "module") else model
        for ema_p, src_p in zip(self.shadow.parameters(), src.parameters()):
            ema_p.mul_(actual_decay).add_(src_p.data, alpha=1.0 - actual_decay)
        for ema_b, src_b in zip(self.shadow.buffers(), src.buffers()):
            ema_b.copy_(src_b)

    def get_model(self):
        return self.shadow


# --- metrics ---

def gather_and_compute_metrics(pred_store, label_store, active_tasks,
                               return_per_class=False, top5_store=None):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    all_preds  = {}
    all_labels = {}
    all_top5   = {}

    for t in active_tasks:
        gathered_p = [None] * world_size
        gathered_l = [None] * world_size
        dist.all_gather_object(gathered_p, pred_store[t])
        dist.all_gather_object(gathered_l, label_store[t])
        all_preds[t]  = [x for sub in gathered_p for x in sub]
        all_labels[t] = [x for sub in gathered_l for x in sub]

        if top5_store is not None:
            gathered_t5 = [None] * world_size
            dist.all_gather_object(gathered_t5, top5_store[t])
            all_top5[t] = [x for sub in gathered_t5 for x in sub]

    metrics = {}
    if rank == 0:
        for t in active_tasks:
            if not all_labels[t]:
                metrics[t] = {"macro_f1": 0.0, "accuracy": 0.0, "top5_accuracy": 0.0}
            else:
                m = compute_classification_metrics(
                    all_labels[t], all_preds[t],
                    logits=None,
                    return_per_class=return_per_class,
                )
                if t in all_top5 and all_top5[t]:
                    m["top5_accuracy"] = float(sum(all_top5[t]) / len(all_top5[t]))
                else:
                    m["top5_accuracy"] = m.get("top5_accuracy", 0.0)
                metrics[t] = m
    return metrics


# --- training loop ---

def train_one_epoch_ddp(model, loader, criterions, optimizer, device,
                        task_weights, active_tasks, grad_clip,
                        accum_steps=1, use_hybrid=False, ema=None,
                        use_sam=False, scaler=None, augmentation_prob=0.25,
                        log_grad_norm=False, log_per_class_f1=False,
                        arcface_losses=None):
    model.train()
    losses    = []
    grad_norms = []
    pred_store  = {t: [] for t in active_tasks}
    label_store = {t: [] for t in active_tasks}
    top5_store  = {t: [] for t in active_tasks}

    use_arcface = bool(arcface_losses)
    optimizer.zero_grad(set_to_none=True)
    # import sys  # debug

    for step, (images, labels) in enumerate(tqdm(loader, desc="Train", leave=False,
                                                   disable=dist.get_rank() != 0)):
        images = images.to(device, non_blocking=True)
        batch  = {t: labels[t].to(device, non_blocking=True) for t in labels}

        task_images, task_mixup_info = build_task_batches(
            images, batch, active_tasks,
            arcface_losses=arcface_losses,
            augmentation_prob=augmentation_prob,
        )

        with autocast(dtype=_AMP_DTYPE):
            clean_emb_outputs = None
            aug_logit_outputs = None

            need_clean_forward = any(t in (arcface_losses or {}) for t in active_tasks)
            need_aug_forward = any(t not in (arcface_losses or {}) for t in active_tasks)

            if need_clean_forward:
                clean_emb_outputs = model(images, return_embeddings=True)

            if need_aug_forward:
                aug_images = next(task_images[t] for t in active_tasks if t not in (arcface_losses or {}))
                aug_logit_outputs = model(aug_images)

            task_losses = {}
            for t in active_tasks:
                current_images = task_images[t]
                current_mixup = task_mixup_info[t]

                if use_hybrid and t == "artist":
                    valid_mask = batch["artist"] >= 0
                    if valid_mask.sum() == 0:
                        task_losses[t] = torch.tensor(0.0, device=device, requires_grad=False)
                        continue
                    if arcface_losses and t in arcface_losses:
                        task_losses[t] = arcface_losses[t](
                            clean_emb_outputs[t][valid_mask], batch[t][valid_mask]
                        )
                    else:
                        task_losses[t] = compute_mixed_loss(
                            criterions[t], aug_logit_outputs[t][valid_mask], batch[t][valid_mask], current_mixup
                        )
                elif arcface_losses and t in arcface_losses:
                    task_losses[t] = arcface_losses[t](clean_emb_outputs[t], batch[t])
                else:
                    task_losses[t] = compute_mixed_loss(
                        criterions[t], aug_logit_outputs[t], batch[t], current_mixup
                    )

            total_loss = sum(task_weights[t] * task_losses[t] for t in active_tasks) / accum_steps

            total_loss = sum(task_weights[t] * task_losses[t] for t in active_tasks) / accum_steps

        scaler.scale(total_loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if use_sam:
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if log_grad_norm:
                    grad_norms.append(float(gn))
                optimizer.first_step(zero_grad=True)
                scaler.update()

                # SAM second step
                with autocast(dtype=_AMP_DTYPE):
                    _clean_emb2 = None
                    _aug_log2 = None

                    need_clean_forward = any(t in (arcface_losses or {}) for t in active_tasks)
                    need_aug_forward = any(t not in (arcface_losses or {}) for t in active_tasks)

                    if need_clean_forward:
                        _clean_emb2 = model(images, return_embeddings=True)

                    if need_aug_forward:
                        aug_images = next(task_images[t] for t in active_tasks if t not in (arcface_losses or {}))
                        _aug_log2 = model(aug_images)

                    task_losses = {}
                    for t in active_tasks:
                        current_mixup = task_mixup_info[t]
                        if use_hybrid and t == "artist":
                            valid_mask = batch["artist"] >= 0
                            if valid_mask.sum() > 0:
                                if arcface_losses and t in arcface_losses:
                                    task_losses[t] = arcface_losses[t](_clean_emb2[t][valid_mask], batch[t][valid_mask])
                                else:
                                    task_losses[t] = compute_mixed_loss(
                                        criterions[t], _aug_log2[t][valid_mask], batch[t][valid_mask], current_mixup
                                    )
                            else:
                                task_losses[t] = torch.tensor(0.0, device=device, requires_grad=False)
                        elif arcface_losses and t in arcface_losses:
                            task_losses[t] = arcface_losses[t](_clean_emb2[t], batch[t])
                        else:
                            task_losses[t] = compute_mixed_loss(criterions[t], _aug_log2[t], batch[t], current_mixup)
                    total_loss = sum(task_weights[t] * task_losses[t] for t in active_tasks) / accum_steps
                    aug_logit_outputs = _aug_log2

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)   # safe now — scaler was reset above
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if log_grad_norm:
                    grad_norms.append(float(gn))
                optimizer.second_step(zero_grad=True)
                scaler.update()
            else:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if log_grad_norm:
                        grad_norms.append(float(gn))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

        losses.append(float(total_loss.item()) * accum_steps)

        with torch.no_grad():
            for t in active_tasks:
                if arcface_losses and t in arcface_losses:
                    logits_t = arcface_losses[t].get_logits(clean_emb_outputs[t])
                else:
                    logits_t = aug_logit_outputs[t]
                k = min(5, logits_t.shape[1])
                if use_hybrid and t == "artist":
                    valid_mask = batch["artist"] >= 0
                    if valid_mask.sum() > 0:
                        logits_v = logits_t[valid_mask]
                        lbls_v   = batch[t][valid_mask]
                        preds = logits_v.argmax(1).cpu().tolist()
                        lbls  = lbls_v.cpu().tolist()
                        top5  = logits_v.topk(k, dim=1).indices
                        top5c = (lbls_v.unsqueeze(1) == top5).any(dim=1).cpu().tolist()
                        pred_store[t].extend(preds)
                        label_store[t].extend(lbls)
                        top5_store[t].extend(top5c)
                else:
                    top5 = logits_t.topk(k, dim=1).indices
                    top5c = (batch[t].unsqueeze(1) == top5).any(dim=1).cpu().tolist()
                    pred_store[t].extend(logits_t.argmax(1).cpu().tolist())
                    label_store[t].extend(batch[t].cpu().tolist())
                    top5_store[t].extend(top5c)

    metrics = {"loss": numpy_mean(losses)}
    if log_grad_norm and grad_norms:
        metrics["grad_norm"] = numpy_mean(grad_norms)

    gathered = gather_and_compute_metrics(pred_store, label_store, active_tasks,
                                          return_per_class=log_per_class_f1,
                                          top5_store=top5_store)
    if dist.get_rank() == 0:
        for t in active_tasks:
            metrics[t] = gathered[t]
        metrics["macro_f1"] = float(sum(metrics[t]["macro_f1"] for t in active_tasks) / len(active_tasks))
    else:
        metrics["macro_f1"] = 0.0

    return metrics


# --- validation ---

@torch.no_grad()
def validate_one_epoch_ddp(model, loader, criterions, device,
                           task_weights, active_tasks, use_hybrid=False,
                           log_per_class_f1=False, use_tta=False,
                           tta_views=6, logit_adjustments=None,
                           arcface_losses=None):
    model.eval()
    losses      = []
    pred_store  = {t: [] for t in active_tasks}
    label_store = {t: [] for t in active_tasks}
    top5_store  = {t: [] for t in active_tasks}

    for images, labels in tqdm(loader, desc="Val", leave=False,
                               disable=dist.get_rank() != 0):
        images = images.to(device, non_blocking=True)
        batch  = {t: labels[t].to(device, non_blocking=True) for t in labels}

        if use_tta:
            if arcface_losses:
                emb_out = tta_forward(model, images, active_tasks, n_views=tta_views,
                                      return_embeddings=True)
                raw_out = tta_forward(model, images, active_tasks, n_views=tta_views)
            else:
                raw_out = tta_forward(model, images, active_tasks, n_views=tta_views)
                emb_out = None
        else:
            with autocast(dtype=_AMP_DTYPE):
                if arcface_losses:
                    emb_out = model(images, return_embeddings=True)
                    raw_out = model(images)
                else:
                    raw_out = model(images)
                    emb_out = None

        outputs = {}
        for t in active_tasks:
            if arcface_losses and t in arcface_losses and emb_out is not None:
                outputs[t] = arcface_losses[t].get_logits(emb_out[t])
            else:
                outputs[t] = raw_out[t]

        if logit_adjustments is not None:
            outputs = {t: (outputs[t] - logit_adjustments[t]
                           if t in logit_adjustments else outputs[t])
                       for t in active_tasks if t in outputs}

        batch_losses = []
        for t in active_tasks:
            if use_hybrid and t == "artist":
                valid_mask = batch["artist"] >= 0
                if valid_mask.sum() > 0:
                    loss_t = criterions[t](outputs[t][valid_mask], batch[t][valid_mask])
                    batch_losses.append(task_weights[t] * loss_t)
            else:
                batch_losses.append(task_weights[t] * criterions[t](outputs[t], batch[t]))
        if batch_losses:
            losses.append(float(sum(batch_losses).item()))

        for t in active_tasks:
            logits_t = outputs[t]
            k = min(5, logits_t.shape[1])
            if use_hybrid and t == "artist":
                valid_mask = batch["artist"] >= 0
                if valid_mask.sum() > 0:
                    logits_v = logits_t[valid_mask]
                    lbls_v   = batch[t][valid_mask]
                    top5  = logits_v.topk(k, dim=1).indices
                    top5c = (lbls_v.unsqueeze(1) == top5).any(dim=1).cpu().tolist()
                    pred_store[t].extend(logits_v.argmax(1).cpu().tolist())
                    label_store[t].extend(lbls_v.cpu().tolist())
                    top5_store[t].extend(top5c)
            else:
                top5  = logits_t.topk(k, dim=1).indices
                top5c = (batch[t].unsqueeze(1) == top5).any(dim=1).cpu().tolist()
                pred_store[t].extend(logits_t.argmax(1).cpu().tolist())
                label_store[t].extend(batch[t].cpu().tolist())
                top5_store[t].extend(top5c)

    metrics = {"loss": numpy_mean(losses)}

    gathered = gather_and_compute_metrics(pred_store, label_store, active_tasks,
                                          return_per_class=log_per_class_f1,
                                          top5_store=top5_store)
    if dist.get_rank() == 0:
        for t in active_tasks:
            metrics[t] = gathered[t]
        metrics["macro_f1"] = float(sum(metrics[t]["macro_f1"] for t in active_tasks) / len(active_tasks))
    else:
        metrics["macro_f1"] = 0.0

    return metrics


# --- main ---

_AMP_DTYPE: torch.dtype = torch.float16


def main():
    parser = argparse.ArgumentParser(
        description="Multitask art classification — DDP + AMP + EMA + TTA + LogitAdj"
    )
    parser.add_argument("--train_csv",            required=True)
    parser.add_argument("--val_csv",              required=True)
    parser.add_argument("--root_dir",             default="")
    parser.add_argument("--num_artist_classes",   type=int, required=True)
    parser.add_argument("--num_style_classes",    type=int, required=True)
    parser.add_argument("--num_genre_classes",    type=int, required=True)
    parser.add_argument("--backbone",             default="convnext_small",
                        choices=["convnext_tiny", "convnext_small",
                                 "convnext_base", "convnext_large",
                                 "dinov2_vitl14", "convnextv2_large",
                                 "clip_vitl14",   "clip_vith14",
                                 "eva02_large"])
    parser.add_argument("--use_cross_attn",       action="store_true", default=True)
    parser.add_argument("--no_pretrained",        action="store_true", default=False)
    parser.add_argument("--pretrained_path",      type=str, default="")
    parser.add_argument("--strong_augment",       action="store_true", default=False)
    parser.add_argument("--epochs",               type=int,   default=100)
    parser.add_argument("--batch_size",           type=int,   default=48)
    parser.add_argument("--accum_steps",          type=int,   default=2)
    parser.add_argument("--lr",                   type=float, default=3e-4)
    parser.add_argument("--backbone_lr_scale",    type=float, default=0.15)
    parser.add_argument("--weight_decay",         type=float, default=0.05)
    parser.add_argument("--image_size",           type=int,   default=384)
    parser.add_argument("--num_workers",          type=int,   default=4)
    parser.add_argument("--freeze_epochs",        type=int,   default=5)
    parser.add_argument("--grad_clip",            type=float, default=1.0)
    parser.add_argument("--early_stop_patience",  type=int,   default=25)
    parser.add_argument("--class_weight_power",   type=float, default=0.4)
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--sgdr_t0",              type=int,   default=10)
    parser.add_argument("--sgdr_tmult",           type=int,   default=2)
    parser.add_argument("--no_sgdr_restart",      action="store_true", default=False)
    parser.add_argument("--artist_weight",        type=float, default=0.35)
    parser.add_argument("--style_weight",         type=float, default=0.40)
    parser.add_argument("--genre_weight",         type=float, default=0.25)
    parser.add_argument("--artist_focal_gamma",   type=float, default=1.0)
    parser.add_argument("--style_focal_gamma",    type=float, default=1.0)
    parser.add_argument("--genre_focal_gamma",    type=float, default=1.0)
    parser.add_argument("--artist_label_smoothing", type=float, default=0.05)
    parser.add_argument("--style_label_smoothing",  type=float, default=0.05)
    parser.add_argument("--genre_label_smoothing",  type=float, default=0.02)
    parser.add_argument("--use_llrd",             action="store_true", default=False)
    parser.add_argument("--llrd_decay",           type=float, default=0.85)
    parser.add_argument("--save_path",            default="checkpoints/best_model.pt")
    parser.add_argument("--curves_path",          default="checkpoints/curves.png")
    parser.add_argument("--history_json",         default="")
    parser.add_argument("--resume_checkpoint",    type=str, default=None)
    parser.add_argument("--use_hybrid",           action="store_true", default=False)
    parser.add_argument("--use_ema",              action="store_true", default=True)
    parser.add_argument("--ema_decay",            type=float, default=0.9999)
    parser.add_argument("--use_sam",              action="store_true", default=False)
    parser.add_argument("--use_arcface",          action="store_true", default=False)
    parser.add_argument("--arcface_s",            type=float, default=32.0)
    parser.add_argument("--arcface_m_artist",     type=float, default=0.25)
    parser.add_argument("--arcface_m_style",      type=float, default=0.30)
    parser.add_argument("--arcface_tasks",        type=str, default="artist,style")
    parser.add_argument("--augmentation_prob",    type=float, default=0.25)
    parser.add_argument("--aug_end_prob",         type=float, default=None)
    parser.add_argument("--use_tta",              action="store_true", default=False)
    parser.add_argument("--tta_views",            type=int, default=6)
    parser.add_argument("--artist_logit_tau",     type=float, default=0.3)
    parser.add_argument("--style_logit_tau",      type=float, default=0.0)
    parser.add_argument("--genre_logit_tau",      type=float, default=0.5)
    parser.add_argument("--log_grad_norm",        action="store_true", default=False)
    parser.add_argument("--log_per_class_f1",     action="store_true", default=False)
    parser.add_argument("--use_bfloat16",         action="store_true", default=False)
    args = parser.parse_args()

    global _AMP_DTYPE
    _AMP_DTYPE = torch.bfloat16 if args.use_bfloat16 else torch.float16

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main    = (local_rank == 0)

    set_seed(args.seed + local_rank)

    if is_main:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        print(f"Device: {device}  |  world_size={world_size}")
        print(f"Per-GPU batch={args.batch_size}  →  "
              f"Effective batch={args.batch_size * world_size * args.accum_steps}")
        print(f"Image size: {args.image_size}  |  Epochs: {args.epochs}")
        amp_str = "bfloat16" if args.use_bfloat16 else "float16"
        print(f"EMA={args.use_ema} (decay={args.ema_decay})  |  SAM={args.use_sam}  |  AMP={amp_str}")
        print(f"LLRD={args.use_llrd} (decay={args.llrd_decay})  |  "
              f"No-SGDR-restart={args.no_sgdr_restart}")
        print(f"TTA={args.use_tta} ({args.tta_views} views)  |  "
              f"LogitAdjust τ: artist={args.artist_logit_tau} "
              f"style={args.style_logit_tau} genre={args.genre_logit_tau}")
        print(f"Focal γ: artist={args.artist_focal_gamma}  "
              f"style={args.style_focal_gamma}  genre={args.genre_focal_gamma}")
        print(f"Smoothing: artist={args.artist_label_smoothing}  "
              f"style={args.style_label_smoothing}  genre={args.genre_label_smoothing}")
        print(f"Aug prob: {args.augmentation_prob} → "
              f"{args.aug_end_prob if args.aug_end_prob is not None else args.augmentation_prob}")
        if args.pretrained_path:
            exists = os.path.isfile(args.pretrained_path)
            print(f"Pretrained path: {args.pretrained_path} "
                  f"({'EXISTS ✓' if exists else 'NOT FOUND ✗ — will use internet/random'})")

    active_tasks = ["artist", "style"]
    if args.num_genre_classes > 1:
        active_tasks.append("genre")

    task_weights = {
        "artist": args.artist_weight,
        "style":  args.style_weight,
        "genre":  args.genre_weight,
    }

    if is_main:
        for name, path in [("train", args.train_csv), ("val", args.val_csv)]:
            df = pd.read_csv(path)
            print(f"\n{name} label sanity check:")
            for col, n in [("artist_label", args.num_artist_classes),
                            ("style_label",  args.num_style_classes),
                            ("genre_label",  args.num_genre_classes)]:
                lbls = df[col]
                min_valid = -1 if (args.use_hybrid and col == "artist_label") else 0
                valid_lbls = lbls[lbls >= 0]
                print(f"  {col}: min={lbls.min()}, max={lbls.max()}, "
                      f"unique={lbls.nunique()}, num_classes={n}")
                if lbls.min() < min_valid:
                    raise ValueError(f"[{name}] {col} has min={lbls.min()}, "
                                     f"expected >= {min_valid}")
                if len(valid_lbls) > 0 and valid_lbls.max() >= n:
                    raise ValueError(f"[{name}] {col} max label={valid_lbls.max()} "
                                     f"but num_classes={n}.")

        if is_main:
            train_df = pd.read_csv(args.train_csv)
            present  = set(train_df["genre_label"].unique())
            expected = set(range(args.num_genre_classes))
            missing_genre = sorted(expected - present)
            if missing_genre:
                print(f"\n[WARNING] Genre classes missing from training data: {missing_genre}")
                print(f"  These classes have zero training samples and CANNOT be learned.")
                print(f"  logit-adjustment with genre_logit_tau>0 will BOOST these ghost")
                print(f"  classes at inference, causing false predictions and lower F1.")
                print(f"  → Set --genre_logit_tau 0.0 (already recommended in optimized script)")
                print(f"  → Or use num_genre_classes={args.num_genre_classes - len(missing_genre)}")
                print(f"     after remapping labels with: python diagnose_data.py --fix_genre")
            else:
                print(f"\n[OK] All {args.num_genre_classes} genre classes present in training data.")

    dist.barrier()

    class_weights = build_multitask_class_weights(
        args.train_csv, args.num_artist_classes,
        args.num_style_classes, args.num_genre_classes,
        power=args.class_weight_power,
        hybrid=args.use_hybrid,
    )


    logit_adjustments = None
    tau_dict = {
        "artist": args.artist_logit_tau,
        "style":  args.style_logit_tau,
        "genre":  args.genre_logit_tau,
    }
    if any(v > 0 for v in tau_dict.values()):
        class_counts = get_class_counts_from_csv(
            args.train_csv, args.num_artist_classes,
            args.num_style_classes, args.num_genre_classes,
            hybrid=args.use_hybrid,
        )
        logit_adjustments = build_logit_adjustments(
            class_counts, active_tasks, device, tau_dict=tau_dict
        )
        if is_main:
            active_tau = {t: v for t, v in tau_dict.items() if v > 0}
            print(f"Logit adjustment enabled: {active_tau}")

    train_transform = (get_train_transforms_strong(args.image_size)
                       if args.strong_augment
                       else get_train_transforms(args.image_size))
    if is_main:
        aug_name = "TrivialAugmentWide (strong)" if args.strong_augment else "moderate"
        print(f"Training augmentation: {aug_name}")

    train_dataset = WikiArtMultiTaskDataset(
        args.train_csv, root_dir=args.root_dir,
        transform=train_transform,
    )
    val_dataset = WikiArtMultiTaskDataset(
        args.val_csv, root_dir=args.root_dir,
        transform=get_val_transforms(args.image_size),
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=local_rank, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_dataset, num_replicas=world_size,
                                       rank=local_rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size,
                              sampler=val_sampler, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True)

    model = MultiTaskCRNN(
        num_artist_classes=args.num_artist_classes,
        num_style_classes=args.num_style_classes,
        num_genre_classes=args.num_genre_classes,
        dropout=0.35,
        pretrained=not args.no_pretrained,
        backbone=args.backbone,
        use_cross_attn=args.use_cross_attn,
        pretrained_path=args.pretrained_path,
    ).to(device)

    start_epoch = 0
    resume_ckpt = None
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        if is_main:
            print(f"\n>>> Resuming from: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        resume_ckpt = ckpt
        state_key = "ema_state_dict" if "ema_state_dict" in ckpt else "model_state_dict"
        raw_sd = ckpt[state_key]
        model_sd = model.state_dict()
        filtered_sd = {k: v for k, v in raw_sd.items()
                       if k not in model_sd or v.shape == model_sd[k].shape}
        skipped = [k for k in raw_sd if k in model_sd and raw_sd[k].shape != model_sd[k].shape]
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        start_epoch = ckpt.get("epoch", 0)
        if is_main:
            print(f"    Loaded '{state_key}' from epoch {start_epoch}")
            if skipped:
                print(f"    Shape-mismatch skipped ({len(skipped)}): {skipped}"
                      f" — re-initialised randomly")
            if missing:
                print(f"    Missing keys  ({len(missing)}): {missing[:5]}"
                      f"{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"    Unexpected keys ({len(unexpected)}): {unexpected[:5]}"
                      f"{'...' if len(unexpected) > 5 else ''}")

        # re-init cross-attn projections if missing from checkpoint
        new_proj_keys = {"artist_proj.weight", "style_proj.weight"}
        if new_proj_keys.intersection(set(missing)):
            for proj_name in ["artist_proj", "style_proj"]:
                proj = getattr(model, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    nn.init.eye_(proj.weight)      # identity: output = input
                    nn.init.zeros_(proj.bias)
            if is_main:
                print("    Initialized artist_proj / style_proj as identity "
                      "(not found in checkpoint — prevents cross-attn cold-start)")
    elif args.resume_checkpoint:
        if is_main:
            print(f"[WARNING] resume_checkpoint not found: {args.resume_checkpoint}")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    ema = None
    if args.use_ema:
        ema = ModelEMA(model, decay=args.ema_decay, device=device)
        if is_main:
            print(f"EMA enabled (decay={args.ema_decay})")

    criterions = {
        "artist": FocalLoss(weight=class_weights["artist"].to(device),
                            gamma=args.artist_focal_gamma,
                            label_smoothing=args.artist_label_smoothing),
        "style":  FocalLoss(weight=class_weights["style"].to(device),
                            gamma=args.style_focal_gamma,
                            label_smoothing=args.style_label_smoothing),
        "genre":  FocalLoss(weight=class_weights["genre"].to(device),
                            gamma=args.genre_focal_gamma,
                            label_smoothing=args.genre_label_smoothing),
    }

    arcface_losses = {}
    arcface_task_set = set()
    if getattr(args, "use_arcface", False):
        arcface_task_set = {t.strip() for t in args.arcface_tasks.split(",")}
        _tmp_model = model.module if hasattr(model, "module") else model
        _head_hidden = _tmp_model.artist_head.fc1.out_features

        _arcface_margins = {
            "artist": args.arcface_m_artist,
            "style":  args.arcface_m_style,
        }
        _arcface_num_classes = {
            "artist": args.num_artist_classes,
            "style":  args.num_style_classes,
            "genre":  args.num_genre_classes,
        }
        for t in arcface_task_set:
            if t not in active_tasks:
                continue
            af = ArcFaceLoss(
                in_features=_head_hidden,
                num_classes=_arcface_num_classes[t],
                s=args.arcface_s,
                m=_arcface_margins.get(t, 0.30),
                label_smoothing=getattr(args, f"{t}_label_smoothing", 0.05),
                class_weights=class_weights[t].to(device),
            ).to(device)
            arcface_losses[t] = af
            if is_main:
                print(f"[ArcFace] task={t}  in_features={_head_hidden}  "
                      f"num_classes={_arcface_num_classes[t]}  "
                      f"s={args.arcface_s}  m={_arcface_margins.get(t, 0.30):.2f}")
        if is_main and arcface_losses:
            print(f"ArcFace active for: {list(arcface_losses.keys())}  "
                  f"(others use FocalLoss)")

        if resume_ckpt is not None and "arcface_state_dicts" in resume_ckpt:
            saved_af = resume_ckpt["arcface_state_dicts"]
            for t, af in arcface_losses.items():
                if t not in saved_af:
                    continue
                missing, unexpected = af.load_state_dict(saved_af[t], strict=False)
                if is_main:
                    msg = f"[ArcFace resume] loaded task={t}"
                    if missing:
                        msg += f" | missing={len(missing)}"
                    if unexpected:
                        msg += f" | unexpected={len(unexpected)}"
                    print(msg)

    raw = model.module
    if args.use_llrd:
        base_optimizer = build_optimizer_llrd(raw, args)
        if is_main:
            n_groups = len(base_optimizer.param_groups)
            print(f"LLRD optimizer: {n_groups} param groups, decay={args.llrd_decay}")
    else:
        head_params = [p for n, p in raw.named_parameters()
                       if not n.startswith("backbone.")]
        base_optimizer = AdamW([
            {"params": raw.backbone.parameters(),
             "lr": args.lr * args.backbone_lr_scale},
            {"params": head_params, "lr": args.lr},
        ], weight_decay=args.weight_decay)

    if arcface_losses:
        af_params = [p for af in arcface_losses.values() for p in af.parameters()]
        base_optimizer.add_param_group({"params": af_params, "lr": args.lr})
        if is_main:
            total_af = sum(p.numel() for p in af_params)
            print(f"ArcFace params added to optimizer: {total_af:,} params at lr={args.lr:.2e}")

    optimizer = SAM(base_optimizer, rho=0.05) if args.use_sam else base_optimizer

    scheduler = build_scheduler(optimizer, args)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(start_epoch):
            scheduler.step()

    scaler = GradScaler(enabled=(not args.use_bfloat16))

    best_f1, best_epoch, no_improve = -1.0, -1, 0
    curve_history = {k: [] for k in [
        "train_loss", "val_loss", "train_macro_f1", "val_macro_f1", "lr",
        "train_artist_macro_f1", "train_style_macro_f1", "train_genre_macro_f1",
        "val_artist_macro_f1",   "val_style_macro_f1",   "val_genre_macro_f1",
        "train_artist_acc", "train_style_acc", "train_genre_acc",
        "val_artist_acc",   "val_style_acc",   "val_genre_acc",
        "val_artist_top5",  "val_style_top5",  "val_genre_top5",
    ]}

    aug_end_prob = (args.aug_end_prob if args.aug_end_prob is not None
                    else args.augmentation_prob)
    total_train_epochs = max(args.epochs - start_epoch, 1)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        if is_main:
            print(f"\nEpoch {epoch}/{args.epochs}")

        backbone_on = epoch > args.freeze_epochs
        set_backbone_trainable(model, backbone_on)

        frac = (epoch - start_epoch - 1) / max(total_train_epochs - 1, 1)
        current_aug_prob = args.augmentation_prob + frac * (aug_end_prob - args.augmentation_prob)
        current_aug_prob = float(max(0.0, min(1.0, current_aug_prob)))

        train_m = train_one_epoch_ddp(
            model, train_loader, criterions, optimizer, device,
            task_weights, active_tasks, args.grad_clip,
            accum_steps=args.accum_steps,
            use_hybrid=args.use_hybrid,
            ema=ema,
            use_sam=args.use_sam,
            scaler=scaler,
            augmentation_prob=current_aug_prob,
            log_grad_norm=args.log_grad_norm,
            log_per_class_f1=args.log_per_class_f1,
            arcface_losses=arcface_losses,
        )

        val_model = ema.get_model() if ema is not None else model
        val_m = validate_one_epoch_ddp(
            val_model, val_loader, criterions, device,
            task_weights, active_tasks,
            use_hybrid=args.use_hybrid,
            log_per_class_f1=args.log_per_class_f1,
            use_tta=args.use_tta,
            tta_views=args.tta_views,
            logit_adjustments=logit_adjustments,
            arcface_losses=arcface_losses,
        )

        scheduler.step()
        current_lr = max(pg["lr"] for pg in optimizer.param_groups)

        if is_main:
            def fmt(m, t):
                d = m[t]
                top5 = d.get("top5_accuracy", 0.0)
                top5_str = f" Top5:{top5:.4f}" if top5 > 0 else ""
                return (f"F1:{d['macro_f1']:.4f} "
                        f"Acc:{d.get('top1_accuracy', d.get('accuracy', 0)):.4f}"
                        f"{top5_str}")

            gn_str = f" GradNorm:{train_m.get('grad_norm', 0):.3f}" if args.log_grad_norm else ""
            print(
                f"Train | Loss:{train_m['loss']:.4f} MeanF1:{train_m['macro_f1']:.4f}"
                f"{gn_str} | "
                + " | ".join(f"{t.title()}:{fmt(train_m, t)}" for t in active_tasks)
            )
            print(
                f"Val   | Loss:{val_m['loss']:.4f} MeanF1:{val_m['macro_f1']:.4f} | "
                + " | ".join(f"{t.title()}:{fmt(val_m, t)}" for t in active_tasks)
                + f" | LR:{current_lr:.2e}"
            )

            if args.log_per_class_f1:
                for t in active_tasks:
                    if "per_class_f1" in val_m.get(t, {}):
                        pcf = val_m[t]["per_class_f1"]
                        worst = np.argsort(pcf)[:3]
                        print(f"  {t.title()} per-class: "
                              f"min={min(pcf):.3f} max={max(pcf):.3f} "
                              f"std={np.std(pcf):.3f}")
                        print(f"    Worst: "
                              + ", ".join(f"c{i}={pcf[i]:.3f}" for i in worst))

            curve_history["train_loss"].append(train_m["loss"])
            curve_history["val_loss"].append(val_m["loss"])
            curve_history["train_macro_f1"].append(train_m["macro_f1"])
            curve_history["val_macro_f1"].append(val_m["macro_f1"])
            curve_history["lr"].append(current_lr)
            for t in ["artist", "style", "genre"]:
                if t in active_tasks:
                    for split, m in [("train", train_m), ("val", val_m)]:
                        curve_history[f"{split}_{t}_macro_f1"].append(m[t]["macro_f1"])
                        curve_history[f"{split}_{t}_acc"].append(
                            m[t].get("top1_accuracy", m[t].get("accuracy", 0)))
                    curve_history[f"val_{t}_top5"].append(
                        val_m[t].get("top5_accuracy", 0))

            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                best_epoch = epoch
                no_improve = 0
                save_dict = {
                    "epoch":              epoch,
                    "model_state_dict":   raw.state_dict(),
                    "val_macro_f1":       best_f1,
                    "num_artist_classes": args.num_artist_classes,
                    "num_style_classes":  args.num_style_classes,
                    "num_genre_classes":  args.num_genre_classes,
                    "backbone":           args.backbone,
                }
                if ema is not None:
                    save_dict["ema_state_dict"] = ema.get_model().state_dict()
                if arcface_losses:
                    save_dict["arcface_state_dicts"] = {
                        t: af.state_dict() for t, af in arcface_losses.items()
                    }
                torch.save(save_dict, args.save_path)
                print(
                    f"✓ Saved best → {args.save_path} "
                    f"(MeanF1={best_f1:.4f}  "
                    f"Artist:{val_m['artist']['macro_f1']:.4f}  "
                    f"Style:{val_m['style']['macro_f1']:.4f}  "
                    f"Genre:{val_m.get('genre', {}).get('macro_f1', 0):.4f})"
                )
            else:
                no_improve += 1
                print(f"No improvement {no_improve} epoch(s) "
                      f"(best ep {best_epoch}, F1={best_f1:.4f})")

        stop_flag = torch.tensor([no_improve >= args.early_stop_patience], device=device)
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item():
            if is_main:
                print(f"Early stopping at epoch {epoch}.")
            break

    if is_main:
        save_training_curves(curve_history, args.curves_path)
        if args.history_json:
            os.makedirs(os.path.dirname(args.history_json) or ".", exist_ok=True)
            with open(args.history_json, "w") as f:
                json.dump(curve_history, f, indent=2)
        print(f"\n=== Training Complete ===")
        print(f"Best MeanF1: {best_f1:.4f} at epoch {best_epoch}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
