"""Shared helpers for exporting embeddings as pseudo-images."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch

from .embed2image import Embed2Image

try:
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("Pillow is required for pseudo-image export") from exc


@dataclass(slots=True)
class EmbeddingImageExporter:
    """Export embedding tensors as RGB pseudo-image PNGs."""

    root: Path
    image_size: int
    mode: str
    channel_mode: str
    clip_values: bool = False

    _converter: Embed2Image = field(init=False)
    _exported_audio: set[str] = field(default_factory=set, init=False)
    _exported_text: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "audio").mkdir(parents=True, exist_ok=True)
        (self.root / "text").mkdir(parents=True, exist_ok=True)
        self._converter = Embed2Image(
            target_hw=self.image_size,
            mode=self.mode,
            channel_mode=self.channel_mode,
        )

    def export_batch(
        self,
        kind: str,
        embeddings: torch.Tensor,
        ids: Sequence[str],
    ) -> list[Path]:
        if kind not in {"audio", "text"}:
            raise ValueError("kind must be 'audio' or 'text'")

        exported = self._exported_audio if kind == "audio" else self._exported_text
        pending_embeddings: list[torch.Tensor] = []
        pending_ids: list[str] = []

        for emb, sample_id in zip(embeddings, ids, strict=True):
            if sample_id in exported:
                continue
            exported.add(sample_id)
            pending_embeddings.append(emb.detach().cpu())
            pending_ids.append(sample_id)

        if not pending_embeddings:
            return []

        batch = torch.stack(pending_embeddings, dim=0).float()
        images = self._converter(batch, clip=self.clip_values).cpu()

        output_dir = self.root / kind
        paths: list[Path] = []
        for tensor, sample_id in zip(images, pending_ids, strict=True):
            filename = _sanitize_identifier(f"{kind}_{sample_id}") + ".png"
            path = output_dir / filename
            _save_tensor_image(tensor, path)
            paths.append(path)

        return paths

    def export_pair(
        self,
        audio_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        sample_id: str,
    ) -> list[Path]:
        paths: list[Path] = []
        paths.extend(self.export_batch("audio", audio_embedding.unsqueeze(0), [sample_id]))
        paths.extend(self.export_batch("text", text_embedding.unsqueeze(0), [sample_id]))
        return paths


_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_identifier(value: str) -> str:
    sanitized = _SANITIZE_PATTERN.sub("_", value)
    return sanitized.strip("._") or "sample"


def _save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    array = (tensor.clamp(0.0, 1.0) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)

