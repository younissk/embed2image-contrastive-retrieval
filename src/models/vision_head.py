"""Vision transformer head for pseudo-image embeddings."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embed2image import Embed2Image

__all__ = ["SharedViT", "VisionProjectionHead"]


class SharedViT(nn.Module):
    """Wrap a timm ViT backbone with an MLP projection head."""

    FALLBACK_BACKBONES = {
        "vit_base_patch32_224_clip_laion2b": "vit_base_patch32_224",
        "vit_base_patch16_224_clip_laion2b": "vit_base_patch16_224",
        "vit_large_patch14_clip_224.openai": "vit_large_patch16_224",
        "eva02_base_patch14_clip_224": "vit_base_patch16_224",
        "eva02_large_patch14_clip_336": "vit_large_patch16_224",
    }

    def __init__(
        self,
        *,
        backbone_name: str = "vit_small_patch16_224",
        out_dim: int = 512,
        pretrained: bool = True,
        feature_pooling: Optional[str] = None,
        dropout: float = 0.0,
        force_download: bool = False,
        **timm_kwargs: Any,
    ) -> None:
        super().__init__()

        candidates: List[str] = [backbone_name]
        if backbone_name in self.FALLBACK_BACKBONES:
            candidates.append(self.FALLBACK_BACKBONES[backbone_name])
        if "clip" in backbone_name.lower() or "eva" in backbone_name.lower():
            candidates.extend(["vit_base_patch16_224", "vit_small_patch16_224"])

        self.backbone = None
        last_error: Optional[Exception] = None
        for name in candidates:
            try:
                self.backbone = timm.create_model(
                    name,
                    pretrained=pretrained,
                    num_classes=0,
                    **timm_kwargs,
                )
                self.backbone_name = name
                break
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_error = exc
                continue
        if self.backbone is None:  # pragma: no cover - defensive fallback
            raise RuntimeError(f"Failed to create timm model. Last error: {last_error}")

        embed_dim = getattr(self.backbone, "num_features", None) or getattr(self.backbone, "embed_dim", None)
        if embed_dim is None:
            raise ValueError(f"Unable to infer embedding dimension for backbone '{self.backbone_name}'")

        self.feature_pooling = feature_pooling if feature_pooling not in {None, "none"} else None

        layers: List[nn.Module] = [nn.Linear(embed_dim, embed_dim * 2), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(embed_dim * 2, out_dim))
        self.projection_head = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if isinstance(features, (list, tuple)):
            features = features[0]

        if features.ndim == 3:
            if self.feature_pooling == "avg":
                features = features.mean(dim=1)
            elif self.feature_pooling == "max":
                features = features.max(dim=1).values
            else:  # 'cls' token
                features = features[:, 0]

        projected = self.projection_head(features)
        return F.normalize(projected, dim=-1)


class VisionProjectionHead(nn.Module):
    """Projection head that maps embeddings to pseudo-images and through a ViT."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        *,
        image_size: int = 128,
        channel_mode: str = "split",
        interp_mode: str = "nearest",
        backbone_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        feature_pooling: Optional[str] = None,
        vision_dropout: float = 0.0,
        force_download: bool = False,
        **timm_kwargs: Any,
    ) -> None:
        super().__init__()
        self.embedder = Embed2Image(target_hw=image_size, mode=interp_mode, channel_mode=channel_mode)
        self.vision_head = SharedViT(
            backbone_name=backbone_name,
            out_dim=out_dim,
            pretrained=pretrained,
            feature_pooling=feature_pooling,
            dropout=vision_dropout,
            force_download=force_download,
            **timm_kwargs,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        pseudo_images = self.embedder(embeddings)
        return self.vision_head(pseudo_images)
