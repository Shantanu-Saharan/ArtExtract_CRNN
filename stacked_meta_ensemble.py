#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

from datasets.dataset import WikiArtMultiTaskDataset
from models.multitask_crnn import MultiTaskCRNN
from models.siglip2_multitask import Siglip2MultiTaskModel
from utils.losses import ArcFaceLoss
from utils.transforms import get_val_transforms


TASKS = ["artist", "style", "genre"]
TASK_DIMS = {"artist": 25, "style": 27, "genre": 7}


class PILIdentity:
    def __call__(self, image):
        return image


@dataclass
class ModelSpec:
    label: str
    ckpt_path: str
    image_size: int
    use_tta: bool = False


def compute_macro_f1(labels, preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    mask = labels >= 0
    if mask.sum() == 0:
        return 0.0
    return f1_score(labels[mask], preds[mask], average="macro", zero_division=0)


def top5_accuracy(logits, labels):
    labels = np.asarray(labels)
    mask = labels >= 0
    if mask.sum() == 0:
        return 0.0
    logits = logits[mask]
    labels = labels[mask]
    top5 = np.argsort(-logits, axis=1)[:, : min(5, logits.shape[1])]
    return float((top5 == labels[:, None]).any(axis=1).mean())


def eval_task_logits(logits, labels):
    preds = logits.argmax(axis=1)
    mask = labels >= 0
    return {
        "macro_f1": compute_macro_f1(labels, preds),
        "accuracy": float(accuracy_score(labels[mask], preds[mask])) if mask.sum() else 0.0,
        "top5_accuracy": top5_accuracy(logits, labels),
    }


def build_collate_fn(processor):
    def collate_fn(batch):
        images = [image for image, _ in batch]
        labels = {
            task: torch.tensor([item[1][task] for item in batch], dtype=torch.long)
            for task in TASKS
        }
        enc = processor(images=images, return_tensors="pt")
        return enc["pixel_values"], labels

    return collate_fn


def tta_tensor_multitask(model, images, n_views=2):
    _, _, h, w = images.shape
    views = [images]
    if n_views >= 2:
        views.append(torch.flip(images, dims=[3]))
    acc = {task: [] for task in TASKS}
    for view in views[:n_views]:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(view)
        for task in TASKS:
            acc[task].append(outputs[task].float())
    return {task: torch.stack(acc[task]).mean(0) for task in TASKS}


def tta_tensor_siglip(model, pixel_values, n_views=2):
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
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(view)
        for task in TASKS:
            acc[task].append(outputs[task].float())
    return {task: torch.stack(acc[task]).mean(0) for task in TASKS}


def load_multitask_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    backbone = ckpt.get("backbone", cfg.get("backbone", "dinov2_vitl14"))
    model = MultiTaskCRNN(
        num_artist_classes=ckpt.get("num_artist_classes", 25),
        num_style_classes=ckpt.get("num_style_classes", 27),
        num_genre_classes=ckpt.get("num_genre_classes", 7),
        dropout=0.0,
        pretrained=False,
        backbone=backbone,
        use_cross_attn=cfg.get("use_cross_attn", True),
    ).to(device)
    state_key = "ema_state_dict" if "ema_state_dict" in ckpt else "model_state_dict"
    raw_state = ckpt[state_key]
    if any(k.startswith("module.") for k in raw_state):
        raw_state = {k.replace("module.", ""): v for k, v in raw_state.items()}
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in raw_state.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    model.load_state_dict(filtered, strict=False)
    model.eval()

    arcface_losses = {}
    saved_arcface = ckpt.get("arcface_state_dicts") or {}
    if saved_arcface:
        hidden_dim = model.artist_head.fc1.out_features
        task_dims = {
            "artist": ckpt.get("num_artist_classes", 25),
            "style": ckpt.get("num_style_classes", 27),
            "genre": ckpt.get("num_genre_classes", 7),
        }
        for task, state_dict in saved_arcface.items():
            if task not in TASKS:
                continue
            af = ArcFaceLoss(
                in_features=hidden_dim,
                num_classes=task_dims[task],
                s=32.0,
                m=0.30,
            ).to(device)
            af.load_state_dict(state_dict, strict=False)
            af.eval()
            arcface_losses[task] = af

    return model, {
        "family": "multitask",
        "backbone": backbone,
        "arcface_losses": arcface_losses,
        "arcface_tasks": sorted(list(arcface_losses.keys())),
    }


def load_siglip_probe(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Siglip2MultiTaskModel(
        model_name_or_path=ckpt["model_name_or_path"],
        num_artist_classes=ckpt.get("num_artist_classes", 25),
        num_style_classes=ckpt.get("num_style_classes", 27),
        num_genre_classes=ckpt.get("num_genre_classes", 7),
        dropout=0.2,
    ).to(device)
    probe_state = ckpt["probe_state_dict"]
    filtered = {
        k: v for k, v in probe_state.items()
        if k in model.state_dict() and tuple(v.shape) == tuple(model.state_dict()[k].shape)
    }
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model, {"family": "siglip_probe", "model_name_or_path": ckpt["model_name_or_path"]}


def load_siglip_finetune(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = Siglip2MultiTaskModel(
        model_name_or_path=ckpt["model_name_or_path"],
        num_artist_classes=ckpt.get("num_artist_classes", 25),
        num_style_classes=ckpt.get("num_style_classes", 27),
        num_genre_classes=ckpt.get("num_genre_classes", 7),
        dropout=0.2,
        use_patch_style=cfg.get("use_patch_style", False),
        use_style_fusion=cfg.get("use_style_fusion", False),
    ).to(device)
    raw_state = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    filtered = {
        k: v for k, v in raw_state.items()
        if k in model.state_dict() and tuple(v.shape) == tuple(model.state_dict()[k].shape)
    }
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model, {"family": "siglip_ft", "model_name_or_path": ckpt["model_name_or_path"]}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "probe_state_dict" in ckpt and ckpt.get("model_name_or_path"):
        return load_siglip_probe(ckpt_path, device)
    if ckpt.get("backbone") == "siglip2_so400m_patch16_384":
        return load_siglip_finetune(ckpt_path, device)
    return load_multitask_model(ckpt_path, device)


def maybe_wrap_data_parallel(model, device):
    if device.type != "cuda":
        return model
    n_gpu = torch.cuda.device_count()
    if n_gpu <= 1:
        return model
    return torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))


def build_loader(csv_path, root_dir, family, image_size, batch_size, num_workers, model_meta):
    if family.startswith("siglip"):
        processor = AutoImageProcessor.from_pretrained(
            model_meta["model_name_or_path"],
            use_fast=False,
        )
        dataset = WikiArtMultiTaskDataset(csv_path, root_dir=root_dir, transform=PILIdentity())
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=build_collate_fn(processor),
        )

    dataset = WikiArtMultiTaskDataset(
        csv_path,
        root_dir=root_dir,
        transform=get_val_transforms(image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def infer_model_on_split(model, family, loader, device, use_tta=False, tta_views=2, model_meta=None):
    logits_store = {task: [] for task in TASKS}
    labels_store = {task: [] for task in TASKS}
    model_meta = model_meta or {}
    arcface_losses = model_meta.get("arcface_losses") or {}
    for batch in tqdm(loader, leave=False):
        if family.startswith("siglip"):
            pixel_values, labels = batch
            pixel_values = pixel_values.to(device, non_blocking=True)
            with torch.no_grad():
                if use_tta:
                    outputs = tta_tensor_siglip(model, pixel_values, n_views=tta_views)
                else:
                    outputs = model(pixel_values)
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                if arcface_losses:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        emb_outputs = model(images, return_embeddings=True)
                        raw_outputs = model(images)
                    outputs = {}
                    for task in TASKS:
                        if task in arcface_losses:
                            outputs[task] = arcface_losses[task].get_logits(emb_outputs[task]).float()
                        else:
                            outputs[task] = raw_outputs[task].float()
                elif use_tta:
                    outputs = tta_tensor_multitask(model, images, n_views=tta_views)
                else:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = model(images)
                    outputs = {task: outputs[task].float() for task in TASKS}

        for task in TASKS:
            logits_store[task].append(outputs[task].detach().cpu())
            labels_store[task].append(labels[task].cpu())

    return (
        {task: torch.cat(logits_store[task], dim=0).numpy() for task in TASKS},
        {task: torch.cat(labels_store[task], dim=0).numpy() for task in TASKS},
    )


def build_features(logits_per_model, task, feature_mode):
    feats = []
    for model_idx, logits in enumerate(logits_per_model):
        x = sanitize_logits(logits[task], task=task, model_idx=model_idx)
        if feature_mode in {"logits_probs", "probs"}:
            probs = stable_softmax(x)
        if feature_mode == "logits":
            feats.append(x)
        elif feature_mode == "probs":
            feats.append(probs)
        elif feature_mode == "logits_probs":
            feats.extend([x, probs])
        else:
            raise ValueError(f"Unknown feature_mode={feature_mode}")
    return np.concatenate(feats, axis=1).astype(np.float32)


def sanitize_logits(x, task, model_idx):
    x = np.asarray(x, dtype=np.float32)
    bad_mask = ~np.isfinite(x)
    if not bad_mask.any():
        return x

    repaired = x.copy()
    bad_count = int(bad_mask.sum())
    print(
        f"[warn] repairing {bad_count} non-finite logits "
        f"for task={task} model_index={model_idx}"
    )

    finite_mask = np.isfinite(repaired)
    col_fill = np.zeros(repaired.shape[1], dtype=np.float32)
    for col in range(repaired.shape[1]):
        col_vals = repaired[finite_mask[:, col], col]
        if col_vals.size:
            col_fill[col] = np.float32(col_vals.mean())

    row_idx, col_idx = np.where(bad_mask)
    repaired[row_idx, col_idx] = col_fill[col_idx]
    repaired = np.nan_to_num(repaired, nan=0.0, posinf=50.0, neginf=-50.0)
    return repaired


def stable_softmax(x):
    x = np.asarray(x, dtype=np.float32)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    denom = np.sum(exp_x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return exp_x / denom


def fit_meta_classifier(X_train, y_train, task, c_grid, seed, standardize=True):
    mask = y_train >= 0
    X = X_train[mask]
    y = y_train[mask]
    X_tr, X_dev, y_tr, y_dev = train_test_split(
        X, y, test_size=0.10, random_state=seed, stratify=y
    )

    best = None
    for c in c_grid:
        base = LogisticRegression(
            C=c,
            max_iter=1500,
            class_weight="balanced",
            solver="lbfgs",
        )
        clf = OneVsRestClassifier(base, n_jobs=1)
        if standardize:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ])
        clf.fit(X_tr, y_tr)
        if hasattr(clf, "decision_function"):
            dev_scores = clf.decision_function(X_dev)
        else:
            dev_scores = clf.predict_proba(X_dev)
        dev_preds = dev_scores.argmax(axis=1)
        dev_f1 = compute_macro_f1(y_dev, dev_preds)
        if best is None or dev_f1 > best["dev_f1"]:
            best = {"C": c, "dev_f1": dev_f1}

    base = LogisticRegression(
        C=best["C"],
        max_iter=1500,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf = OneVsRestClassifier(base, n_jobs=1)
    if standardize:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
    clf.fit(X, y)
    return clf, best


def decision_scores(clf, X):
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
    else:
        scores = clf.predict_proba(X)
    return np.asarray(scores)


def main():
    parser = argparse.ArgumentParser(description="Stacked class-wise meta-ensemble for top-25 checkpoints")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--root_dir", default="wikiart")
    parser.add_argument("--model_ckpt", action="append", required=True)
    parser.add_argument("--model_label", action="append", required=True)
    parser.add_argument("--model_image_size", action="append", type=int, required=True)
    parser.add_argument("--model_use_tta", action="append", type=int, default=[])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--feature_mode", choices=["logits", "probs", "logits_probs"], default="logits_probs")
    parser.add_argument("--c_grid", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--tta_views", type=int, default=2)
    parser.add_argument("--no_standardize", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_out", default="reports/top25_meta_ensemble_report.json")
    parser.add_argument("--cache_dir", default="reports/meta_cache")
    parser.add_argument("--models_out", default="reports/top25_meta_models.pkl")
    args = parser.parse_args()

    if len(args.model_ckpt) != len(args.model_label) or len(args.model_ckpt) != len(args.model_image_size):
        raise ValueError("Repeat --model_ckpt, --model_label, and --model_image_size once per model.")

    if args.model_use_tta and len(args.model_use_tta) not in {0, len(args.model_ckpt)}:
        raise ValueError("--model_use_tta must be omitted or repeated once per model.")

    tta_flags = args.model_use_tta or [0] * len(args.model_ckpt)
    specs = [
        ModelSpec(label=label, ckpt_path=ckpt, image_size=image_size, use_tta=bool(tta))
        for label, ckpt, image_size, tta in zip(args.model_label, args.model_ckpt, args.model_image_size, tta_flags)
    ]

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Visible CUDA devices: {torch.cuda.device_count()}")
    print(f"Models: {[spec.label for spec in specs]}")

    train_logits_by_model = []
    val_logits_by_model = []
    labels_train = None
    labels_val = None
    individual_val = {}
    meta_info = []

    for spec in specs:
        print("\n" + "=" * 80)
        print(f"Loading {spec.label}")
        print("=" * 80)
        train_cache = os.path.join(args.cache_dir, f"{spec.label}_train.pkl")
        val_cache = os.path.join(args.cache_dir, f"{spec.label}_val.pkl")

        if os.path.exists(train_cache) and os.path.exists(val_cache):
            print(f"[cache] using cached logits for {spec.label}")
            with open(train_cache, "rb") as fh:
                train_logits, labels_train_cur = pickle.load(fh)
            with open(val_cache, "rb") as fh:
                val_logits, labels_val_cur = pickle.load(fh)
            model_meta = {
                "label": spec.label,
                "ckpt_path": spec.ckpt_path,
                "family": "cached",
            }
        else:
            model, model_meta = load_model(spec.ckpt_path, device)
            model = maybe_wrap_data_parallel(model, device)
            runtime_meta = dict(model_meta)
            model_meta = {
                k: v for k, v in model_meta.items()
                if k != "arcface_losses"
            }
            model_meta["label"] = spec.label
            model_meta["ckpt_path"] = spec.ckpt_path
            train_loader = build_loader(
                args.train_csv, args.root_dir, runtime_meta["family"], spec.image_size,
                args.batch_size, args.num_workers, runtime_meta
            )
            val_loader = build_loader(
                args.val_csv, args.root_dir, runtime_meta["family"], spec.image_size,
                args.batch_size, args.num_workers, runtime_meta
            )
            print(f"  infer train ({len(train_loader.dataset)} samples)")
            train_logits, labels_train_cur = infer_model_on_split(
                model, runtime_meta["family"], train_loader, device,
                use_tta=spec.use_tta, tta_views=args.tta_views, model_meta=runtime_meta
            )
            print(f"  infer val   ({len(val_loader.dataset)} samples)")
            val_logits, labels_val_cur = infer_model_on_split(
                model, runtime_meta["family"], val_loader, device,
                use_tta=spec.use_tta, tta_views=args.tta_views, model_meta=runtime_meta
            )
            with open(train_cache, "wb") as fh:
                pickle.dump((train_logits, labels_train_cur), fh)
            with open(val_cache, "wb") as fh:
                pickle.dump((val_logits, labels_val_cur), fh)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        meta_info.append(model_meta)

        train_logits_by_model.append(train_logits)
        val_logits_by_model.append(val_logits)
        labels_train = labels_train_cur if labels_train is None else labels_train
        labels_val = labels_val_cur if labels_val is None else labels_val

        individual_val[spec.label] = {}
        for task in TASKS:
            individual_val[spec.label][task] = eval_task_logits(val_logits[task], labels_val_cur[task])
        mean_f1 = np.mean([individual_val[spec.label][task]["macro_f1"] for task in TASKS])
        print(
            f"{spec.label:<18} Artist={individual_val[spec.label]['artist']['macro_f1']:.4f} "
            f"Style={individual_val[spec.label]['style']['macro_f1']:.4f} "
            f"Genre={individual_val[spec.label]['genre']['macro_f1']:.4f} "
            f"MeanF1={mean_f1:.4f}"
        )

    meta_models = {}
    meta_results = {}
    final_metrics = {}

    print("\n" + "=" * 80)
    print("STACKED META-ENSEMBLE")
    print("=" * 80)
    for task in TASKS:
        X_train = build_features(train_logits_by_model, task, args.feature_mode)
        X_val = build_features(val_logits_by_model, task, args.feature_mode)
        y_train = labels_train[task]
        y_val = labels_val[task]
        clf, meta_results[task] = fit_meta_classifier(
            X_train, y_train, task, args.c_grid, args.seed, standardize=not args.no_standardize
        )
        val_scores = decision_scores(clf, X_val)
        final_metrics[task] = eval_task_logits(val_scores, y_val)
        meta_models[task] = clf
        print(
            f"{task:<8} C={meta_results[task]['C']} "
            f"devF1={meta_results[task]['dev_f1']:.4f} "
            f"valF1={final_metrics[task]['macro_f1']:.4f} "
            f"valAcc={final_metrics[task]['accuracy']:.4f}"
        )

    ensemble_mean = float(np.mean([final_metrics[task]["macro_f1"] for task in TASKS]))
    print("-" * 80)
    print(f"Stacked MeanF1={ensemble_mean:.4f}")
    print("-" * 80)

    best_single_mean = max(
        float(np.mean([individual_val[label][task]["macro_f1"] for task in TASKS]))
        for label in individual_val
    )
    print(f"Best single-model MeanF1={best_single_mean:.4f}")
    print(f"Gain={ensemble_mean - best_single_mean:+.4f}")

    report = {
        "models": [
            {
                "label": spec.label,
                "checkpoint": spec.ckpt_path,
                "image_size": spec.image_size,
                "use_tta": spec.use_tta,
                "meta": meta,
            }
            for spec, meta in zip(specs, meta_info)
        ],
        "feature_mode": args.feature_mode,
        "tta_views": args.tta_views,
        "standardized_features": not args.no_standardize,
        "individual_val": individual_val,
        "meta_search": meta_results,
        "ensemble_val": final_metrics,
        "ensemble_mean_f1": ensemble_mean,
        "best_single_mean_f1": best_single_mean,
    }
    with open(args.report_out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(args.models_out, "wb") as fh:
        pickle.dump(meta_models, fh)
    print(f"Report: {args.report_out}")
    print(f"Meta models: {args.models_out}")


if __name__ == "__main__":
    main()
