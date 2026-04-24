# multitask model — artist / style / genre
# supports DINO and ConvNeXt backbones via timm and torchvision

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# --- pooling ---

class GeM(nn.Module):

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0, max=10.0)
        return (
            F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(p), (1, 1))
            .pow(1.0 / p)
            .flatten(1)
        )


# --- heads ---

class TaskHead(nn.Module):

    def __init__(self, in_dim: int, num_classes: int,
                 dropout: float = 0.3, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(512, in_dim // 2)
        self.norm  = nn.LayerNorm(in_dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(in_dim, hidden_dim)
        self.act   = nn.GELU()
        self.drop2 = nn.Dropout(dropout * 0.5)
        self.fc2   = nn.Linear(hidden_dim, num_classes)

        # small init so logits start near-uniform
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x = self.norm(x)
        x = self.drop1(x)
        emb = self.act(self.fc1(x))
        if return_embedding:
            # pre-logit embedding for arcface
            return emb
        emb = self.drop2(emb)
        return self.fc2(emb)


# --- cross-task attention ---

class CrossTaskAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True,
                                          dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q  = query.unsqueeze(1)
        kv = context.unsqueeze(1)
        out, _ = self.attn(q, kv, kv)
        return self.norm(query + out.squeeze(1))


# --- main model ---

class MultiTaskCRNN(nn.Module):

    _TV_BACKBONES: dict = {
        "convnext_tiny":  (models.convnext_tiny,  "ConvNeXt_Tiny_Weights",  768),
        "convnext_small": (models.convnext_small, "ConvNeXt_Small_Weights", 768),
        "convnext_base":  (models.convnext_base,  "ConvNeXt_Base_Weights",  1024),
        "convnext_large": (models.convnext_large, "ConvNeXt_Large_Weights", 1536),
    }

    _TIMM_BACKBONES: dict = {
        "dinov2_vitl14":    ("vit_large_patch14_dinov2.lvd142m",                    1024),
        "convnextv2_large": ("convnextv2_large.fcmae_ft_in22k_in1k_384",            1536),
        "clip_vitl14":      ("vit_large_patch14_clip_224.openai",                   1024),
        "clip_vith14":      ("vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k",    1280),
        "eva02_large":      ("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",     1024),
    }

    def __init__(
        self,
        num_artist_classes: int = 23,
        num_style_classes:  int = 27,
        num_genre_classes:  int = 8,
        dropout:            float = 0.3,
        pretrained:         bool  = True,
        backbone:           str   = "convnext_large",
        use_cross_attn:     bool  = True,
        pretrained_path:    str   = "",
        proj_dim:           int   | None = None,
        hidden_size:        int   | None = None,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self._backbone_name = backbone

        self.backbone, self.feature_dim = self._build_backbone(
            backbone, pretrained, pretrained_path
        )

        self.pool = GeM(p=3.0)

        if use_cross_attn:
            nh = 8 if self.feature_dim >= 1024 else 4
            self.artist_proj = nn.Linear(self.feature_dim, self.feature_dim)
            self.style_proj  = nn.Linear(self.feature_dim, self.feature_dim)
            self.cross_as    = CrossTaskAttention(self.feature_dim, nh)
            self.cross_sa    = CrossTaskAttention(self.feature_dim, nh)

        self.artist_head = TaskHead(self.feature_dim, num_artist_classes, dropout)
        self.style_head  = TaskHead(self.feature_dim, num_style_classes,  dropout)
        self.genre_head  = TaskHead(self.feature_dim, num_genre_classes,  dropout)

    def _build_backbone(self, name: str, pretrained: bool, pretrained_path: str = ""):

        if name in self._TV_BACKBONES:
            fn, weights_attr, dim = self._TV_BACKBONES[name]

            if pretrained_path and os.path.isfile(pretrained_path):
                print(f"[INFO] Loading backbone weights from local file: {pretrained_path}")
                bb = fn(weights=None)
                state = torch.load(pretrained_path, map_location="cpu")
                for wrap_key in ("model", "state_dict", "backbone", "net"):
                    if wrap_key in state and isinstance(state[wrap_key], dict):
                        state = state[wrap_key]
                        break
                missing, unexpected = bb.load_state_dict(state, strict=False)
                missing_bb = [k for k in missing
                              if not k.startswith(("classifier", "avgpool", "head"))]
                print(f"[INFO] Backbone loaded — "
                      f"missing_backbone={len(missing_bb)}, "
                      f"unexpected={len(unexpected)} "
                      f"(classifier keys ignored)")
                bb.classifier = nn.Identity()
                bb.avgpool    = nn.Identity()
                return bb, dim

            if pretrained:
                try:
                    w  = getattr(models, weights_attr).DEFAULT
                    bb = fn(weights=w)
                    print(f"[INFO] Loaded pretrained {name} from internet.")
                except Exception as exc:
                    print(f"[WARN] Could not load pretrained {name} "
                          f"({exc}).\n"
                          f"       Falling back to random init.\n"
                          f"       To fix: run `python download_pretrained.py` locally,\n"
                          f"       upload weights/ folder to cluster, then pass\n"
                          f"       --pretrained_path weights/convnext_large_1k.pth")
                    bb = fn(weights=None)
            else:
                bb = fn(weights=None)

            bb.classifier = nn.Identity()
            bb.avgpool    = nn.Identity()
            return bb, dim

        if name in self._TIMM_BACKBONES:
            timm_name, dim = self._TIMM_BACKBONES[name]
            try:
                import timm
            except ImportError:
                raise ImportError(
                    f"Backbone '{name}' requires `timm`. Install with: pip install timm"
                )
            try:
                timm_kwargs = dict(num_classes=0, global_pool="", dynamic_img_size=True)

                if pretrained_path and os.path.isfile(pretrained_path):
                    print(f"[INFO] Loading timm backbone '{name}' from local file: {pretrained_path}")
                    bb = timm.create_model(timm_name, pretrained=False, **timm_kwargs)
                    state = torch.load(pretrained_path, map_location="cpu")
                    for wrap_key in ("model", "state_dict", "backbone", "net"):
                        if wrap_key in state and isinstance(state[wrap_key], dict):
                            state = state[wrap_key]
                            break
                    missing, unexpected = bb.load_state_dict(state, strict=False)
                    missing_bb = [k for k in missing
                                  if not k.startswith(("head", "fc", "classifier"))]
                    print(f"[INFO] Timm backbone loaded — "
                          f"missing={len(missing_bb)}, unexpected={len(unexpected)}")
                else:
                    bb = timm.create_model(timm_name, pretrained=pretrained, **timm_kwargs)
                    if pretrained:
                        print(f"[INFO] Loaded pretrained timm backbone '{name}' from internet.")
                return bb, dim
            except Exception as exc:
                raise RuntimeError(f"Failed to load timm backbone '{name}': {exc}") from exc

        all_keys = list(self._TV_BACKBONES) + list(self._TIMM_BACKBONES)
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {all_keys}")

    def _extract_features(self, x: torch.Tensor):
        bb = self.backbone
        if hasattr(bb, "features"):
            feat = bb.features(x)
            return self.pool(feat)  # (B, feature_dim)

        # timm models
        feat = bb.forward_features(x)
        if feat.ndim == 3:
            cls_token  = feat[:, 0, :]
            patch_mean = feat[:, 1:, :].mean(1)
            return {"cls": cls_token, "patch": patch_mean}
        elif feat.ndim == 4 and feat.shape[1] != self.feature_dim:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        return self.pool(feat)  # (B, feature_dim)

    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> dict:
        raw = self._extract_features(x)

        if isinstance(raw, dict):
            artist_base = raw["cls"]
            style_base  = raw["patch"]
            genre_base  = raw["cls"]
            pooled      = raw["cls"]
        else:
            artist_base = style_base = genre_base = pooled = raw

        if self.use_cross_attn:
            artist_init = F.gelu(self.artist_proj(artist_base))
            style_init  = F.gelu(self.style_proj(style_base))
            artist_feat = self.cross_as(artist_init, style_init)
            style_feat  = self.cross_sa(style_init,  artist_init)
        else:
            artist_feat = artist_base
            style_feat  = style_base

        if return_embeddings:
            return {
                "artist":    self.artist_head(artist_feat, return_embedding=True),
                "style":     self.style_head(style_feat,  return_embedding=True),
                "genre":     self.genre_head(genre_base,  return_embedding=True),
                "embedding": pooled,
            }

        return {
            "artist":    self.artist_head(artist_feat),
            "style":     self.style_head(style_feat),
            "genre":     self.genre_head(genre_base),
            "embedding": pooled,
        }
