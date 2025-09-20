"""Utility to reshape embeddings into pseudo-images."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Embed2Image"]


class Embed2Image(nn.Module):
    """Reshape a (B, D) embedding to a (B, 3, H, W) pseudo-image."""

    def __init__(self, target_hw: int = 128, mode: str = "nearest", *, channel_mode: str = "split") -> None:
        super().__init__()
        if channel_mode not in {"replicate", "split"}:
            raise ValueError("channel_mode must be either 'replicate' or 'split'")

        self.target_hw = target_hw
        self.mode = mode
        self.channel_mode = channel_mode

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor, *, clip: bool = False) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("Expected tensor of shape (B, D)")

        batch, dim = x.shape
        if self.channel_mode == "split":
            side = int(math.sqrt(dim / 3))
            if dim > side * side * 3:
                side += 1
            pad = side * side * 3 - dim
        else:
            side = int(math.sqrt(dim))
            if dim > side * side:
                side += 1
            pad = side * side - dim

        if pad:
            x = F.pad(x, (0, pad))

        if self.channel_mode == "split":
            img = x.view(batch, 3, side, side).contiguous()
        else:
            img = x.view(batch, 1, side, side).contiguous()

        if clip:
            img = img.clamp_(-1.0, 1.0)

        if self.channel_mode == "replicate":
            img = img.repeat(1, 3, 1, 1)

        img = self.stem(img)
        img = F.interpolate(img, size=(self.target_hw, self.target_hw), mode=self.mode)
        img_min = img.amin(dim=(-2, -1), keepdim=True)
        img_max = img.amax(dim=(-2, -1), keepdim=True)
        img = (img - img_min) / (img_max - img_min + 1e-8)
        return img
