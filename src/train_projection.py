"""Train a projection head on cached audio/text embeddings using contrastive loss."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

try:  # optional logging backend
    import wandb
except ModuleNotFoundError:  # pragma: no cover - dependency optional
    wandb = None

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Image = None

from .utils.embed2image import Embed2Image

DEFAULT_CACHE_DIR = Path.home() / "data" / "CLOTHO_v2.1" / "embeddings" / "development"


def _setup_log_file(base_dir: Path, category: str, run_name: Optional[str], meta: dict) -> Path:
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_name or f"{category}_{timestamp}"
    log_dir = base_dir / category
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.jsonl"
    header = {"event": "metadata", "timestamp": timestamp, "data": meta}
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
    return log_path


class _WandbRun:
    """Lightweight context manager for optional wandb usage."""

    def __init__(
        self,
        enabled: bool,
        *,
        project: Optional[str],
        run_name: Optional[str],
        config: dict[str, object],
    ) -> None:
        self.enabled = enabled and wandb is not None
        self.project = project
        self.run_name = run_name
        self.config = config
        self._active = False

    def __enter__(self) -> "_WandbRun":
        if self.enabled:
            wandb.init(project=self.project, name=self.run_name, config=self.config, reinit=True)
            self._active = True
        return self

    def log(self, payload: dict[str, object]) -> None:
        if self._active:
            wandb.log(payload)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._active:
            wandb.finish()
            self._active = False


def _append_log(log_path: Path, payload: dict) -> None:
    payload["timestamp"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    sanitized = sanitized.strip("._")
    return sanitized or "sample"


class PairedDataset(Dataset):
    """Dataset returning either raw embeddings or pseudo-images paired with text embeddings."""

    def __init__(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        sample_ids: Sequence[str],
        *,
        input_type: Literal["embedding", "vision"],
        image_converter: Optional[Embed2Image] = None,
        image_dir: Optional[Path] = None,
        image_clip: bool = False,
    ) -> None:
        if audio_embeddings.shape != text_embeddings.shape:
            raise ValueError(
                "Audio and text embeddings must share the same shape. "
                f"Got {audio_embeddings.shape} vs {text_embeddings.shape}."
            )

        if input_type == "vision" and image_converter is None and image_dir is None:
            raise ValueError("Vision mode requires either an image converter or an image directory")

        self._audio = audio_embeddings
        self._text = text_embeddings
        self._ids = list(sample_ids)
        self._input_type = input_type
        self._image_dir = image_dir
        self._image_clip = image_clip
        self._converter = image_converter

        if self._converter is not None:
            self._converter = self._converter.eval()
            for param in self._converter.parameters():
                param.requires_grad = False

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._audio.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - trivial
        anchor = self._get_anchor(index)
        text = self._text[index]
        return anchor, text

    def _get_anchor(self, index: int) -> torch.Tensor:
        if self._input_type == "embedding":
            return self._audio[index]

        # vision mode
        image: Optional[torch.Tensor] = None
        if self._image_dir is not None and Image is not None:
            sample_id = self._ids[index]
            filename = _sanitize_identifier(f"audio_{sample_id}") + ".png"
            path = self._image_dir / filename
            if path.exists():
                with Image.open(path) as img:
                    tensor = torch.as_tensor(np.array(img, dtype="float32")) / 255.0
                if tensor.ndim == 2:  # grayscale fallback
                    tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
                image = tensor.permute(2, 0, 1).contiguous()

        if image is None:
            if self._converter is None:
                raise RuntimeError(
                    "Unable to load pseudo-image: neither disk image found nor converter available"
                )
            embedding = self._audio[index].unsqueeze(0)
            with torch.no_grad():
                image = self._converter(embedding, clip=self._image_clip).squeeze(0).cpu()

        return image


class ProjectionHead(nn.Module):
    """Small MLP that projects embeddings before contrastive training."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = 512,
        out_dim: int = 256,
        activation: Literal["relu", "gelu"] = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim or None
        self.out_dim = out_dim
        self.activation = activation
        self.dropout = dropout

        layers: list[nn.Module] = []

        if self.hidden_dim:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU() if activation == "relu" else nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.net(inputs)


class VisionProjectionHead(nn.Module):
    """CNN-based head that maps pseudo-images into the shared embedding space."""

    def __init__(
        self,
        out_dim: int = 256,
        base_channels: int = 32,
        hidden_dim: Optional[int] = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.out_dim = out_dim
        self.base_channels = base_channels
        self.hidden_dim = hidden_dim or None
        self.dropout = dropout

        layers: list[nn.Module] = []
        in_channels = 3
        channels = base_channels

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )

        for stage in range(4):
            layers.append(block(in_channels, channels))
            layers.append(nn.MaxPool2d(2))
            in_channels = channels
            channels = min(channels * 2, base_channels * 8)

        layers.append(block(in_channels, channels))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*layers)

        projector_input_dim = channels
        projector: list[nn.Module] = []
        if self.hidden_dim:
            projector.append(nn.Linear(projector_input_dim, self.hidden_dim))
            projector.append(nn.GELU())
            if dropout > 0:
                projector.append(nn.Dropout(p=dropout))
            projector.append(nn.Linear(self.hidden_dim, out_dim))
        else:
            projector.append(nn.Linear(projector_input_dim, out_dim))

        self.projector = nn.Sequential(*projector)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        feats = self.features(inputs)
        feats = feats.flatten(1)
        return self.projector(feats)


@dataclass(slots=True)
class TrainBatch:
    anchor: torch.Tensor
    text: torch.Tensor


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_dataset(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    if val_fraction <= 0:
        return dataset, Subset(dataset, [])

    num_items = len(dataset)
    num_val = max(1, int(math.floor(num_items * val_fraction)))
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _contrastive_loss(anchor_proj: torch.Tensor, text_proj: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_norm = F.normalize(anchor_proj, dim=-1)
    text_norm = F.normalize(text_proj, dim=-1)

    logits = anchor_norm @ text_norm.t() / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)

    loss_a = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    loss = (loss_a + loss_t) * 0.5
    return loss, logits


@torch.no_grad()
def _compute_metrics(logits: torch.Tensor) -> dict[str, float]:
    targets = torch.arange(logits.shape[0], device=logits.device)
    top1_audio = (logits.argmax(dim=1) == targets).float().mean().item()
    top1_text = (logits.argmax(dim=0) == targets).float().mean().item()
    return {
        "acc_audio_to_text": top1_audio,
        "acc_text_to_audio": top1_text,
    }


def train_projection_head(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device)
    _seed_everything(args.seed)

    cache_dir = Path(args.cache_dir).expanduser()
    audio_path = cache_dir / "audio_embeddings.pt"
    text_path = cache_dir / "text_embeddings.pt"

    if not audio_path.exists() or not text_path.exists():
        raise FileNotFoundError(
            "Expected embedding tensors not found. Run 'make download-embeddings' first."
        )

    audio_embeddings = torch.load(str(audio_path), map_location="cpu")
    text_embeddings = torch.load(str(text_path), map_location="cpu")

    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata_payload = json.load(handle)
        sample_ids = metadata_payload.get("ids") or [f"sample_{idx}" for idx in range(len(audio_embeddings))]
    else:
        sample_ids = [f"sample_{idx}" for idx in range(len(audio_embeddings))]

    image_dir: Optional[Path] = None
    image_converter: Optional[Embed2Image] = None
    if args.input_type == "vision":
        raw_image_dir = Path(args.vision_image_dir)
        image_dir = raw_image_dir if raw_image_dir.is_absolute() else cache_dir / raw_image_dir
        if not image_dir.exists():
            image_dir = None

        need_converter = args.vision_image_source in {"generated", "auto"}
        if args.vision_image_source == "auto" and image_dir is not None:
            need_converter = False

        if need_converter:
            image_converter = Embed2Image(
                target_hw=args.vision_image_size,
                mode=args.vision_image_mode,
                channel_mode=args.vision_image_channel_mode,
            )

    dataset = PairedDataset(
        audio_embeddings,
        text_embeddings,
        sample_ids,
        input_type=args.input_type,
        image_converter=image_converter,
        image_dir=image_dir,
        image_clip=args.vision_clip_images,
    )
    train_dataset, val_dataset = _split_dataset(dataset, args.val_fraction, args.seed)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        if len(val_dataset) > 0
        else None
    )

    input_dim = audio_embeddings.shape[1]
    anchor_head: nn.Module
    if args.input_type == "vision":
        anchor_head = VisionProjectionHead(
            out_dim=args.out_dim,
            base_channels=args.vision_base_channels,
            hidden_dim=args.vision_hidden_dim,
            dropout=args.dropout,
        ).to(device)
    else:
        anchor_head = ProjectionHead(
            in_dim=input_dim,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            activation=args.activation,
            dropout=args.dropout,
        ).to(device)

    text_head = ProjectionHead(
        in_dim=text_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(anchor_head.parameters()) + list(text_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_path = _setup_log_file(
        Path(args.log_dir).expanduser(),
        "train",
        args.run_name,
        {
            "input_type": args.input_type,
            "dataset_size": len(dataset),
            "args": vars(args),
        },
    )

    wandb_project = args.wandb_project or os.getenv("WANDB_PROJECT")
    wandb_run = _WandbRun(
        args.use_wandb,
        project=wandb_project,
        run_name=args.run_name,
        config={
            "input_type": args.input_type,
            "dataset_size": len(dataset),
            **{f"arg/{key}": value for key, value in vars(args).items()},
        },
    )

    best_val_loss: float | None = None
    best_epoch: Optional[int] = None
    history: dict[str, float] = {}

    with wandb_run:
        for epoch in range(1, args.epochs + 1):
            anchor_head.train()
            text_head.train()
            running_loss = 0.0
            num_batches = 0
            for batch_anchor, batch_text in tqdm(
                train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False
            ):
                batch_anchor = batch_anchor.to(device)
                batch_text = batch_text.to(device)

                optimizer.zero_grad(set_to_none=True)

                anchor_proj = anchor_head(batch_anchor)
                text_proj = text_head(batch_text)

                loss, logits = _contrastive_loss(anchor_proj, text_proj, args.temperature)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

            scheduler.step()

            avg_train_loss = running_loss / max(1, num_batches)
            log_items: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "lr": scheduler.get_last_lr()[0],
            }

        if val_loader is not None:
            anchor_head.eval()
            text_head.eval()
            val_loss = 0.0
            val_batches = 0
            metrics_total = {"acc_audio_to_text": 0.0, "acc_text_to_audio": 0.0}

            for batch_anchor, batch_text in val_loader:
                batch_anchor = batch_anchor.to(device)
                batch_text = batch_text.to(device)

                anchor_proj = anchor_head(batch_anchor)
                text_proj = text_head(batch_text)

                loss, logits = _contrastive_loss(anchor_proj, text_proj, args.temperature)
                metrics = _compute_metrics(logits)

                val_loss += loss.item()
                val_batches += 1
                for key in metrics_total:
                    metrics_total[key] += metrics[key]

            avg_val_loss = val_loss / max(1, val_batches)
            log_items["val_loss"] = avg_val_loss
            for key in metrics_total:
                log_items[key] = metrics_total[key] / max(1, val_batches)

            if best_val_loss is None or avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                _persist_heads(
                    args.output_dir,
                    anchor_head,
                    text_head,
                    epoch,
                    avg_val_loss,
                    _build_checkpoint_config(
                        input_type=args.input_type,
                        anchor_head=anchor_head,
                        text_head=text_head,
                        image_settings={
                            "image_size": args.vision_image_size,
                            "image_mode": args.vision_image_mode,
                            "channel_mode": args.vision_image_channel_mode,
                            "clip": args.vision_clip_images,
                            "image_dir": str(image_dir) if image_dir else None,
                        },
                    ),
                )
        else:
            # Without validation we keep last epoch weights.
            best_epoch = epoch
            if best_val_loss is None or avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
            _persist_heads(
                args.output_dir,
                anchor_head,
                text_head,
                epoch,
                avg_train_loss,
                _build_checkpoint_config(
                    input_type=args.input_type,
                    anchor_head=anchor_head,
                    text_head=text_head,
                    image_settings={
                        "image_size": args.vision_image_size,
                        "image_mode": args.vision_image_mode,
                        "channel_mode": args.vision_image_channel_mode,
                        "clip": args.vision_clip_images,
                        "image_dir": str(image_dir) if image_dir else None,
                    },
                ),
            )

        history = log_items
        tqdm.write(
            " | ".join(
                f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
                for key, value in log_items.items()
            )
        )

        payload = {
            "event": "epoch",
            **{key: float(value) if isinstance(value, (int, float)) else value for key, value in log_items.items()},
        }
        _append_log(log_path, payload)
        wandb_run.log({k: v for k, v in payload.items() if isinstance(v, (int, float))})

        if best_val_loss is not None:
            summary = {
                "event": "best",
                "epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }
            _append_log(log_path, summary)
            wandb_run.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})

    return history


def _build_checkpoint_config(
    *,
    input_type: str,
    anchor_head: nn.Module,
    text_head: nn.Module,
    image_settings: dict[str, Optional[str | bool | int]],
) -> dict[str, object]:
    def projection_config(head: ProjectionHead) -> dict[str, object]:
        return {
            "type": "projection",
            "in_dim": head.in_dim,
            "hidden_dim": head.hidden_dim,
            "out_dim": head.out_dim,
            "activation": head.activation,
            "dropout": head.dropout,
        }

    def vision_config(head: VisionProjectionHead) -> dict[str, object]:
        return {
            "type": "vision",
            "base_channels": head.base_channels,
            "hidden_dim": head.hidden_dim,
            "out_dim": head.out_dim,
            "dropout": head.dropout,
        }

    if isinstance(anchor_head, VisionProjectionHead):
        anchor_conf = vision_config(anchor_head)
    elif isinstance(anchor_head, ProjectionHead):
        anchor_conf = projection_config(anchor_head)
    else:  # pragma: no cover - defensive
        anchor_conf = {"type": anchor_head.__class__.__name__}

    text_conf = projection_config(text_head)

    return {
        "input_type": input_type,
        "anchor_head": anchor_conf,
        "text_head": text_conf,
        "vision_images": image_settings,
    }


def _persist_heads(
    output_dir: Path | str,
    anchor_head: nn.Module,
    text_head: nn.Module,
    epoch: int,
    loss: float,
    config: dict[str, object],
) -> None:
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "loss": loss,
        "config": config,
        "anchor_head": anchor_head.state_dict(),
        "text_head": text_head.state_dict(),
    }
    torch.save(state, output_path / "projection_heads.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Directory containing audio_embeddings.pt and text_embeddings.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Where to store the trained projection heads",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Optimizer learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay applied to AdamW",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden layer size in the projection head (set to 0 for linear)",
    )
    parser.add_argument(
        "--out-dim",
        type=int,
        default=256,
        help="Output dimensionality of the projection head",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "gelu"],
        default="relu",
        help="Activation used between projection layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability applied after the hidden layer",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature parameter for the contrastive loss",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of pairs used for validation",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (e.g. 'cuda', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--input-type",
        choices=["embedding", "vision"],
        default="embedding",
        help="Primary modality used for the anchor branch",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where JSONL training logs will be written",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional identifier to include in log filenames",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Stream metrics to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Project name for Weights & Biases (defaults to WANDB_PROJECT env)",
    )
    parser.add_argument(
        "--vision-base-channels",
        type=int,
        default=32,
        help="Base number of channels for the vision projection head",
    )
    parser.add_argument(
        "--vision-hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension inside the vision projection head (set to 0 for linear)",
    )
    parser.add_argument(
        "--vision-image-source",
        choices=["auto", "generated", "disk"],
        default="auto",
        help="Where vision inputs are loaded from: existing PNGs, generated on the fly, or auto",
    )
    parser.add_argument(
        "--vision-image-dir",
        default="pseudo_images/audio",
        help="Relative or absolute path to cached pseudo-image PNGs",
    )
    parser.add_argument(
        "--vision-image-size",
        type=int,
        default=128,
        help="Spatial resolution used when generating pseudo-images",
    )
    parser.add_argument(
        "--vision-image-mode",
        default="nearest",
        help="Interpolation mode for generated pseudo-images",
    )
    parser.add_argument(
        "--vision-image-channel-mode",
        choices=["split", "replicate"],
        default="split",
        help="Channel strategy when folding embeddings into pseudo-images",
    )
    parser.add_argument(
        "--vision-clip-images",
        action="store_true",
        help="Clip generated pseudo-images to [-1, 1] before normalisation",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, float]:  # pragma: no cover - CLI entry point
    parser = build_parser()
    args = parser.parse_args(argv)
    return train_projection_head(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
