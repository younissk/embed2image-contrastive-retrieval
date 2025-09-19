"""Evaluate trained projection heads on cached embeddings with recall metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Image = None

from .train_projection import (
    DEFAULT_CACHE_DIR,
    ProjectionHead,
    VisionProjectionHead,
    _append_log,
    _sanitize_identifier,
    _setup_log_file,
)
from .utils.embed2image import Embed2Image


def _load_embeddings(cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    audio_path = cache_dir / "audio_embeddings.pt"
    text_path = cache_dir / "text_embeddings.pt"

    if not audio_path.exists() or not text_path.exists():
        raise FileNotFoundError(
            "Missing embedding tensors. Re-run 'make download-embeddings' or 'make cache-embeddings'."
        )

    audio = torch.load(str(audio_path), map_location="cpu")
    text = torch.load(str(text_path), map_location="cpu")

    if audio.shape != text.shape:
        raise ValueError(f"Audio and text tensor shapes differ: {audio.shape} vs {text.shape}")

    return audio, text


def _load_projection_heads(
    checkpoint_state: dict,
    *,
    device: torch.device,
    embedding_dim: int,
    text_dim: int,
    fallback: argparse.Namespace,
) -> tuple[nn.Module, ProjectionHead, dict[str, object]]:
    config = checkpoint_state.get("config", {})
    anchor_conf = config.get("anchor_head")
    text_conf = config.get("text_head")
    input_type = config.get("input_type", fallback.input_type)

    # Backward compatibility with older checkpoints
    if anchor_conf is None:
        anchor_conf = {
            "type": "projection",
            "in_dim": embedding_dim,
            "hidden_dim": fallback.hidden_dim,
            "out_dim": fallback.out_dim,
            "activation": fallback.activation,
            "dropout": fallback.dropout,
        }
    if text_conf is None:
        text_conf = {
            "type": "projection",
            "in_dim": text_dim,
            "hidden_dim": fallback.hidden_dim,
            "out_dim": fallback.out_dim,
            "activation": fallback.activation,
            "dropout": fallback.dropout,
        }

    anchor_type = anchor_conf.get("type", "projection")
    if anchor_type == "vision":
        anchor_head: nn.Module = VisionProjectionHead(
            out_dim=anchor_conf.get("out_dim", fallback.out_dim),
            base_channels=anchor_conf.get("base_channels", fallback.vision_base_channels),
            hidden_dim=anchor_conf.get("hidden_dim", fallback.vision_hidden_dim),
            dropout=anchor_conf.get("dropout", fallback.dropout),
        ).to(device)
    else:
        anchor_head = ProjectionHead(
            in_dim=anchor_conf.get("in_dim", embedding_dim),
            hidden_dim=anchor_conf.get("hidden_dim", fallback.hidden_dim),
            out_dim=anchor_conf.get("out_dim", fallback.out_dim),
            activation=anchor_conf.get("activation", fallback.activation),
            dropout=anchor_conf.get("dropout", fallback.dropout),
        ).to(device)

    text_head = ProjectionHead(
        in_dim=text_conf.get("in_dim", text_dim),
        hidden_dim=text_conf.get("hidden_dim", fallback.hidden_dim),
        out_dim=text_conf.get("out_dim", fallback.out_dim),
        activation=text_conf.get("activation", fallback.activation),
        dropout=text_conf.get("dropout", fallback.dropout),
    ).to(device)

    anchor_state = checkpoint_state.get("anchor_head") or checkpoint_state.get("audio_head")
    if anchor_state is None:
        raise KeyError("Checkpoint is missing anchor head weights")
    anchor_head.load_state_dict(anchor_state)
    text_head.load_state_dict(checkpoint_state["text_head"])

    return anchor_head.eval(), text_head.eval(), {"input_type": input_type, **config}


def _load_sample_ids(cache_dir: Path, expected: int) -> list[str]:
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        ids = payload.get("ids")
        if ids and len(ids) == expected:
            return list(ids)
    return [f"sample_{idx}" for idx in range(expected)]


def _prepare_vision_inputs(
    audio_embeddings: torch.Tensor,
    sample_ids: Sequence[str],
    *,
    cache_dir: Path,
    vision_conf: dict[str, object],
    args: argparse.Namespace,
) -> torch.Tensor:
    image_dir_str = vision_conf.get("image_dir", args.vision_image_dir)
    image_dir: Optional[Path] = None
    if isinstance(image_dir_str, str) and image_dir_str.lower() != "none":
        tentative = Path(image_dir_str)
        if not tentative.is_absolute():
            tentative = cache_dir / tentative
        if tentative.exists():
            image_dir = tentative

    image_size = int(vision_conf.get("image_size", args.vision_image_size))
    image_mode = vision_conf.get("image_mode", args.vision_image_mode)
    channel_mode = vision_conf.get("channel_mode", args.vision_image_channel_mode)
    clip = bool(vision_conf.get("clip", args.vision_clip_images))

    total = audio_embeddings.shape[0]
    images: list[Optional[torch.Tensor]] = [None] * total

    if image_dir is not None and Image is not None:
        for idx, sample_id in enumerate(sample_ids):
            filename = _sanitize_identifier(f"audio_{sample_id}") + ".png"
            path = image_dir / filename
            if not path.exists():
                continue
            with Image.open(path) as img:
                array = np.array(img, dtype="float32") / 255.0
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=-1)
            tensor = torch.as_tensor(array).permute(2, 0, 1).contiguous()
            images[idx] = tensor

    missing = [idx for idx, value in enumerate(images) if value is None]
    if missing:
        converter = Embed2Image(
            target_hw=image_size,
            mode=image_mode,
            channel_mode=channel_mode,
        )
        converter = converter.eval()
        for param in converter.parameters():
            param.requires_grad = False

        batch_size = max(1, min(args.batch_size, len(missing)))
        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            batch = audio_embeddings[chunk]
            with torch.no_grad():
                imgs = converter(batch, clip=clip).cpu()
            for offset, idx in enumerate(chunk):
                images[idx] = imgs[offset]

    if any(img is None for img in images):
        raise RuntimeError("Failed to materialise pseudo-images for all samples")

    stacked = torch.stack([img for img in images], dim=0)
    return stacked


@torch.no_grad()
def _project_embeddings(
    embeddings: torch.Tensor,
    model: nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    loader = torch.utils.data.DataLoader(embeddings, batch_size=batch_size, shuffle=False)
    for batch in tqdm(loader, desc="Projecting", leave=False):
        batch = batch.to(device)
        proj = model(batch)
        outputs.append(proj.cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def _recall_at_k(
    query: torch.Tensor,
    candidates: torch.Tensor,
    *,
    ks: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> dict[int, float]:
    ks = sorted(set(int(k) for k in ks))
    num_items = query.shape[0]
    recalls = {k: 0 for k in ks}

    query = F.normalize(query.to(device), dim=-1)
    candidates = F.normalize(candidates.to(device), dim=-1)

    total = 0
    loader = torch.utils.data.DataLoader(torch.arange(num_items), batch_size=batch_size, shuffle=False)
    for indices in tqdm(loader, desc="Computing recalls", leave=False):
        q = query[indices]
        scores = q @ candidates.t()
        targets = indices.to(device)

        for k in ks:
            topk = scores.topk(k, dim=1).indices
            recalls[k] += (topk == targets.unsqueeze(1)).any(dim=1).float().sum().item()
        total += indices.numel()

    return {k: recalls[k] / total for k in ks}


def evaluate_projection_heads(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()

    audio_emb, text_emb = _load_embeddings(cache_dir)
    sample_ids = _load_sample_ids(cache_dir, len(audio_emb))

    state = torch.load(str(checkpoint), map_location=device)

    anchor_head, text_head, config = _load_projection_heads(
        state,
        device=device,
        embedding_dim=audio_emb.shape[1],
        text_dim=text_emb.shape[1],
        fallback=args,
    )

    input_type = config.get("input_type", "embedding")
    vision_conf = config.get("vision_images", {})

    if input_type == "vision":
        anchor_inputs = _prepare_vision_inputs(
            audio_emb,
            sample_ids,
            cache_dir=cache_dir,
            vision_conf=vision_conf,
            args=args,
        )
    else:
        anchor_inputs = audio_emb

    log_path = _setup_log_file(
        Path(args.log_dir).expanduser(),
        "eval",
        args.run_name,
        {
            "checkpoint": str(checkpoint),
            "input_type": input_type,
            "args": vars(args),
        },
    )

    anchor_proj = _project_embeddings(anchor_inputs, anchor_head, device=device, batch_size=args.batch_size)
    text_proj = _project_embeddings(text_emb, text_head, device=device, batch_size=args.batch_size)

    ks = (1, 5, 10)
    recalls_a = _recall_at_k(anchor_proj, text_proj, ks=ks, device=device, batch_size=args.sim_batch_size)
    recalls_t = _recall_at_k(text_proj, anchor_proj, ks=ks, device=device, batch_size=args.sim_batch_size)

    metrics = {}
    for k in ks:
        metrics[f"R@{k}_audio_to_text"] = recalls_a[k]
        metrics[f"R@{k}_text_to_audio"] = recalls_t[k]

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    _append_log(
        log_path,
        {
            "event": "metrics",
            **{key: float(value) for key, value in metrics.items()},
        },
    )

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Directory with audio_embeddings.pt / text_embeddings.pt",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/projection_heads.pt",
        help="Path to the trained projection head checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size used when projecting embeddings",
    )
    parser.add_argument(
        "--sim-batch-size",
        type=int,
        default=512,
        help="Batch size used when computing similarities for recall metrics",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension fallback if not stored in checkpoint",
    )
    parser.add_argument(
        "--out-dim",
        type=int,
        default=256,
        help="Output dimension fallback if not stored in checkpoint",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "gelu"],
        default="relu",
        help="Activation fallback if not stored in checkpoint",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout fallback if not stored in checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for projection and similarity scoring",
    )
    parser.add_argument(
        "--input-type",
        choices=["embedding", "vision"],
        default="embedding",
        help="Fallback input type when checkpoint metadata is missing",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where evaluation logs will be stored",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run identifier for the log filename",
    )
    parser.add_argument(
        "--vision-base-channels",
        type=int,
        default=32,
        help="Fallback base channels for vision head reconstruction",
    )
    parser.add_argument(
        "--vision-hidden-dim",
        type=int,
        default=512,
        help="Fallback hidden dimension for the vision head",
    )
    parser.add_argument(
        "--vision-image-size",
        type=int,
        default=128,
        help="Fallback pseudo-image size when regenerating",
    )
    parser.add_argument(
        "--vision-image-mode",
        default="nearest",
        help="Fallback interpolation mode when regenerating pseudo-images",
    )
    parser.add_argument(
        "--vision-image-channel-mode",
        choices=["split", "replicate"],
        default="split",
        help="Fallback channel handling when regenerating pseudo-images",
    )
    parser.add_argument(
        "--vision-clip-images",
        action="store_true",
        help="Clip regenerated pseudo-images to [-1, 1] before normalisation",
    )
    parser.add_argument(
        "--vision-image-dir",
        default="pseudo_images/audio",
        help="Fallback directory containing cached pseudo-images",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, float]:  # pragma: no cover
    parser = build_parser()
    args = parser.parse_args(argv)
    return evaluate_projection_heads(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
