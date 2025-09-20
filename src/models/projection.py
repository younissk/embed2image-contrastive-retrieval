"""Projection head modules."""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn

from .vision_head import VisionProjectionHead

__all__ = ["ProjectionHead", "VisionProjectionHead", "HEAD_REGISTRY", "resolve_projection_head"]


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head used in the baseline."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if hidden_dim > 0:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.net(inputs)


HEAD_REGISTRY: dict[str, Type[nn.Module]] = {
    "mlp": ProjectionHead,
    "vision": VisionProjectionHead,
}


def resolve_projection_head(name: str) -> Type[nn.Module]:
    try:
        return HEAD_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown projection head '{name}'. Available: {list(HEAD_REGISTRY)}") from exc
