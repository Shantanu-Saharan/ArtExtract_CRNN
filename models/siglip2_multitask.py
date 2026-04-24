from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SiglipVisionModel


class TaskHead(nn.Module):
    """Probe-compatible task head used for SigLIP experiments."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Siglip2MultiTaskModel(nn.Module):
    """Full fine-tuning model for SigLIP vision backbones."""

    def __init__(
        self,
        model_name_or_path: str,
        num_artist_classes: int,
        num_style_classes: int,
        num_genre_classes: int,
        cache_dir: str | None = None,
        dropout: float = 0.2,
        use_patch_style: bool = False,
        use_style_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.use_patch_style = use_patch_style
        self.use_style_fusion = use_style_fusion
        self.backbone = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        self.feature_dim = int(self.backbone.config.hidden_size)
        if self.use_style_fusion:
            self.style_pool_proj = nn.Linear(self.feature_dim, self.feature_dim)
            self.style_patch_proj = nn.Linear(self.feature_dim, self.feature_dim)
            self.style_gate = nn.Sequential(
                nn.LayerNorm(self.feature_dim * 2),
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.GELU(),
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.Sigmoid(),
            )
            self._init_style_fusion()
        self.artist_head = TaskHead(self.feature_dim, num_artist_classes, dropout=dropout)
        self.style_head = TaskHead(self.feature_dim, num_style_classes, dropout=dropout)
        self.genre_head = TaskHead(self.feature_dim, num_genre_classes, dropout=dropout)

    def _init_style_fusion(self) -> None:
        # Keep warm-starts stable: old checkpoints used pooled features for style,
        # so the new fusion path starts as an almost-pooled identity mapping.
        nn.init.eye_(self.style_pool_proj.weight)
        nn.init.zeros_(self.style_pool_proj.bias)
        nn.init.eye_(self.style_patch_proj.weight)
        nn.init.zeros_(self.style_patch_proj.bias)

        gate_in = self.style_gate[1]
        gate_out = self.style_gate[3]
        nn.init.zeros_(gate_in.weight)
        nn.init.zeros_(gate_in.bias)
        nn.init.zeros_(gate_out.weight)
        nn.init.constant_(gate_out.bias, -6.0)

    def gradient_checkpointing_enable(self) -> None:
        # Non-reentrant checkpointing plays much more nicely with DDP than the
        # Transformers default reentrant mode.
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def gradient_checkpointing_disable(self) -> None:
        self.backbone.gradient_checkpointing_disable()

    def extract_features(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        patch_mean = outputs.last_hidden_state.mean(dim=1)
        if self.use_style_fusion:
            pooled_style = self.style_pool_proj(pooled)
            patch_style = self.style_patch_proj(patch_mean)
            fused = torch.cat([pooled_style, patch_style], dim=-1)
            gate = self.style_gate(fused)
            style_feat = gate * patch_style + (1.0 - gate) * pooled_style
        elif self.use_patch_style:
            style_feat = patch_mean
        else:
            style_feat = pooled
        return {
            "artist": pooled,
            "style": style_feat,
            "genre": pooled,
            "pooled": pooled,
            "patch_mean": patch_mean,
        }

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.extract_features(pixel_values)
        return {
            "artist": self.artist_head(feats["artist"]),
            "style": self.style_head(feats["style"]),
            "genre": self.genre_head(feats["genre"]),
            "embedding": feats["pooled"],
        }
