"""
download_pretrained.py
======================
Downloads pretrained backbone weights to a local file so they can be
used on an air-gapped cluster (no internet access).

Supported backbones
-------------------
  ConvNeXt (torchvision):
    convnext_tiny, convnext_small, convnext_base, convnext_large  [DEFAULT]

  Strong timm backbones (requires: pip install timm):
    dinov2_vitl14    — DINOv2 ViT-Large.
    clip_vith14      — CLIP ViT-H/14 (LAION-2B / fine-tuned).
    eva02_large      — EVA-02 Large, native 448px backbone.
    convnextv2_large — ConvNeXtV2-Large (ImageNet-22k → 1k finetune).

RUN THIS ON YOUR LOCAL MACHINE (with internet access):
    python download_pretrained.py                     # ConvNeXt-Large (default)
    python download_pretrained.py --backbone dinov2_vitl14
    python download_pretrained.py --backbone clip_vith14
    python download_pretrained.py --backbone eva02_large

Then upload to cluster:
    scp weights/convnext_large_1k.pth  <user>@<cluster>:~/CRNN/weights/
    scp weights/dinov2_vitl14.pth      <user>@<cluster>:~/CRNN/weights/
    scp weights/clip_vith14.pth        <user>@<cluster>:~/CRNN/weights/
    scp weights/eva02_large.pth        <user>@<cluster>:~/CRNN/weights/

Then add to your training command:
    --pretrained_path weights/convnext_large_1k.pth
    --backbone convnext_large

    --pretrained_path weights/dinov2_vitl14.pth
    --backbone dinov2_vitl14
"""

import os
import torch
from torchvision import models


# ── torchvision backbones ─────────────────────────────────────────────────────

TV_BACKBONE_MAP = {
    "convnext_tiny":  (models.convnext_tiny,  models.ConvNeXt_Tiny_Weights,  "convnext_tiny_1k.pth"),
    "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights, "convnext_small_1k.pth"),
    "convnext_base":  (models.convnext_base,  models.ConvNeXt_Base_Weights,  "convnext_base_1k.pth"),
    "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights, "convnext_large_1k.pth"),
}

# ── timm backbones ─────────────────────────────────────────────────────────────
# model_name → (timm_id, filename)
TIMM_BACKBONE_MAP = {
    "dinov2_vitl14":    ("vit_large_patch14_dinov2.lvd142m",                         "dinov2_vitl14.pth"),
    "clip_vith14":      ("vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k",         "clip_vith14.pth"),
    "eva02_large":      ("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",          "eva02_large.pth"),
    "convnextv2_large": ("convnextv2_large.fcmae_ft_in22k_in1k_384",                 "convnextv2_large.pth"),
}

ALL_BACKBONES = list(TV_BACKBONE_MAP) + list(TIMM_BACKBONE_MAP)


def download_tv_backbone(name: str, out_dir: str) -> str:
    fn, weights_cls, filename = TV_BACKBONE_MAP[name]
    out_path = os.path.join(out_dir, filename)

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"[SKIP] Already exists: {out_path} ({size_mb:.0f} MB)")
        return out_path

    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading {name} pretrained weights (ImageNet-1k)...")
    print("This may take a few minutes depending on your connection.")

    weights = weights_cls.IMAGENET1K_V1
    model   = fn(weights=weights)
    torch.save(model.state_dict(), out_path)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Saved: {out_path}  ({size_mb:.0f} MB)")
    return out_path


def download_timm_backbone(name: str, out_dir: str) -> str:
    timm_id, filename = TIMM_BACKBONE_MAP[name]
    out_path = os.path.join(out_dir, filename)

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"[SKIP] Already exists: {out_path} ({size_mb:.0f} MB)")
        return out_path

    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for DINOv2/ConvNeXtV2 backbones.\n"
            "Install with:  pip install timm"
        )

    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading {name} ({timm_id}) from timm hub...")
    print("This may take several minutes (ViT-Large ~1.2 GB).")

    # num_classes=0, global_pool="" → feature extractor mode (no classification head).
    # pretrained=True downloads from the timm model hub.
    model = timm.create_model(timm_id, pretrained=True, num_classes=0, global_pool="")
    model.eval()

    # Save the full state dict — the model loading code in multitask_crnn.py
    # uses timm.create_model(..., pretrained=False) and then loads via load_state_dict,
    # so the saved dict must match timm's own state dict keys exactly.
    torch.save(model.state_dict(), out_path)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Saved: {out_path}  ({size_mb:.0f} MB)")
    return out_path


def download_backbone(name: str, out_dir: str = "weights") -> str:
    if name in TV_BACKBONE_MAP:
        return download_tv_backbone(name, out_dir)
    if name in TIMM_BACKBONE_MAP:
        return download_timm_backbone(name, out_dir)
    raise ValueError(f"Unknown backbone '{name}'. Choose from: {ALL_BACKBONES}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Download pretrained backbone weights for offline cluster use"
    )
    parser.add_argument(
        "--backbone", default="convnext_large",
        choices=ALL_BACKBONES,
        help=(
            "Which backbone to download (default: convnext_large).\n"
            "timm backbones (DINOv2 / CLIP-H / EVA02 / ConvNeXtV2) require: pip install timm"
        ),
    )
    parser.add_argument("--out_dir", default="weights",
                        help="Directory to save the weights file (default: weights/)")
    args = parser.parse_args()

    out_path = download_backbone(args.backbone, args.out_dir)

    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"1. Upload to cluster:")
    print(f"     scp {out_path} <user>@<cluster>:~/CRNN/{out_path}")
    print(f"2. On cluster, create the weights dir if it doesn't exist:")
    print(f"     mkdir -p ~/CRNN/weights")
    print(f"3. Add to your training command:")
    print(f"     --pretrained_path {out_path}")
    print(f"     --backbone {args.backbone}")
    print()
    if args.backbone in TV_BACKBONE_MAP:
        print("Or just run the provided optimized script which already includes this flag:")
        print("     bash run_optimized_2gpu.sh")
    else:
        print("Note: timm backbones (DINOv2, ConvNeXtV2) are loaded differently at runtime.")
        print("The model loading in multitask_crnn.py handles this automatically via timm.create_model.")
        print("Just pass --pretrained_path and --backbone as shown above.")


if __name__ == "__main__":
    main()
