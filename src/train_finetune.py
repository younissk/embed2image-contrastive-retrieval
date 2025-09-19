"""Fine-tune audio/text encoders with contrastive loss, baseline and vision variants."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer

from hear21passt.base import get_basic_model

from .train_projection import (
    VisionProjectionHead,
    _append_log,
    _sanitize_identifier,
    _setup_log_file,
)
from .utils.embed2image import Embed2Image
from .utils.embeddings import _load_samples

DEFAULT_METADATA = (
    Path.home()
    / "data"
    / "CLOTHO_v2.1"
    / "clotho_csv_files"
    / "clotho_captions_development.csv"
)


@dataclass(slots=True)
class Sample:
    audio: Path
    text: str
    sample_id: str


class ContrastiveDataset(Dataset):
    """Dataset that loads raw audio waveforms and paired captions."""

    def __init__(
        self,
        samples: Iterable[dict[str, str]],
        *,
        sample_rate: int = 32000,
        max_seconds: Optional[float] = None,
    ) -> None:
        self._entries: list[Sample] = []
        self.sample_rate = sample_rate
        self.max_length = int(max_seconds * sample_rate) if max_seconds else None

        for item in samples:
            path = Path(item["audio"]).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")
            text = str(item["text"])
            sample_id = str(item.get("id") or _sanitize_identifier(path.stem))
            self._entries.append(Sample(path, text, sample_id))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, object]:  # pragma: no cover - trivial
        entry = self._entries[index]
        waveform, sr = torchaudio.load(entry.audio)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sr, self.sample_rate
            ).squeeze(0)

        if self.max_length and waveform.numel() > self.max_length:
            waveform = waveform[: self.max_length]

        return {
            "audio": waveform,
            "text": entry.text,
            "id": entry.sample_id,
            "length": waveform.numel(),
        }


class ContrastiveCollate:
    """Pad waveforms and tokenize captions."""

    def __init__(self, tokenizer: AutoTokenizer, max_text_length: int = 32) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        if not batch:
            raise ValueError("Batch is empty")

        lengths = [int(item["length"]) for item in batch]
        max_len = max(lengths)
        audio_batch = torch.zeros(len(batch), max_len)
        for idx, item in enumerate(batch):
            waveform = item["audio"]
            audio_batch[idx, : waveform.numel()] = waveform

        texts = [str(item["text"]) for item in batch]
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        )

        sample_ids = [str(item["id"]) for item in batch]

        return {
            "audio": audio_batch,
            "audio_lengths": torch.tensor(lengths, dtype=torch.long),
            "text_inputs": tokenized,
            "ids": sample_ids,
        }


class WarmupCosineScheduler:
    """Cosine scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, 1)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_num = 0
        self.current_lr = min_lr
        self._set_lr(min_lr)

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value
        self.current_lr = value

    def step(self) -> float:
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            progress = self.step_num / max(1, self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        else:
            decay_progress = (self.step_num - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            decay_progress = min(decay_progress, 1.0)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * decay_progress)
            )
        self._set_lr(lr)
        return lr

    def get_last_lr(self) -> float:  # pragma: no cover - trivial
        return self.current_lr


class RetrievalModel(nn.Module):
    """Audio-text retrieval model with optional vision projection head."""

    def __init__(
        self,
        *,
        audio_arch: str,
        text_model_name: str,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
        model_type: str,
        vision_image_size: int,
        vision_image_mode: str,
        vision_channel_mode: str,
    ) -> None:
        super().__init__()

        self.model_type = model_type
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.audio_model = get_basic_model(mode="embed_only", arch=audio_arch)
        self.audio_model.to(self.device)

        self.text_model = AutoModel.from_pretrained(
            text_model_name,
            add_pooling_layer=False,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.text_model.to(self.device)

        audio_dim = getattr(self.audio_model, "embed_dim", 768)
        text_dim = self.text_model.config.hidden_size

        if model_type == "vision":
            self.embed_to_image = Embed2Image(
                target_hw=vision_image_size,
                mode=vision_image_mode,
                channel_mode=vision_channel_mode,
            )
            self.vision_head = VisionProjectionHead(
                out_dim=projection_dim,
                base_channels=32,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            head_layers: list[nn.Module] = []
            if hidden_dim:
                head_layers.append(nn.Linear(audio_dim, hidden_dim))
                head_layers.append(nn.GELU())
                if dropout > 0:
                    head_layers.append(nn.Dropout(dropout))
                head_layers.append(nn.Linear(hidden_dim, projection_dim))
            else:
                head_layers.append(nn.Linear(audio_dim, projection_dim))
            self.audio_projection = nn.Sequential(*head_layers)

        text_layers: list[nn.Module] = []
        if hidden_dim:
            text_layers.append(nn.Linear(text_dim, hidden_dim))
            text_layers.append(nn.GELU())
            if dropout > 0:
                text_layers.append(nn.Dropout(dropout))
            text_layers.append(nn.Linear(hidden_dim, projection_dim))
        else:
            text_layers.append(nn.Linear(text_dim, projection_dim))
        self.text_projection = nn.Sequential(*text_layers)

        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode_audio(self, audio_waveforms: torch.Tensor) -> torch.Tensor:
        waveforms = audio_waveforms.to(self.device)
        audio_embeddings = self.audio_model.get_scene_embeddings(waveforms)
        if self.model_type == "vision":
            images = self.embed_to_image(audio_embeddings)
            audio_proj = self.vision_head(images)
        else:
            audio_proj = self.audio_projection(audio_embeddings)
        return audio_proj

    def encode_text(self, text_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
        outputs = self.text_model(**inputs)
        hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (hidden * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1.0)
        text_features = summed / counts
        return self.text_projection(text_features)

    def forward(
        self,
        audio_waveforms: torch.Tensor,
        text_inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = self.encode_audio(audio_waveforms)
        text = self.encode_text(text_inputs)
        anchor = F.normalize(anchor, dim=-1)
        text = F.normalize(text, dim=-1)
        return anchor, text

    def current_temperature(self) -> torch.Tensor:
        return torch.abs(self.temperature) + 1e-6


def contrastive_loss(
    anchor_proj: torch.Tensor, text_proj: torch.Tensor, temperature: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = anchor_proj @ text_proj.T / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_a + loss_t), logits


def compute_metrics(logits: torch.Tensor) -> dict[str, float]:
    targets = torch.arange(logits.size(0), device=logits.device)
    top1_anchor = (logits.argmax(dim=1) == targets).float().mean().item()
    top1_text = (logits.argmax(dim=0) == targets).float().mean().item()
    return {
        "acc_audio_to_text": top1_anchor,
        "acc_text_to_audio": top1_text,
    }


def split_dataset(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    if val_fraction <= 0:
        return dataset, Subset(dataset, [])
    generator = torch.Generator()
    generator.manual_seed(seed)
    num_val = max(1, int(len(dataset) * val_fraction))
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    metadata_path = Path(args.metadata).expanduser()
    samples = _load_samples(metadata_path)

    dataset = ContrastiveDataset(
        samples,
        sample_rate=args.sample_rate,
        max_seconds=args.max_audio_seconds,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    collate = ContrastiveCollate(tokenizer, max_text_length=args.max_text_length)

    train_dataset, val_dataset = split_dataset(dataset, args.val_fraction, args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            drop_last=False,
        )
        if len(val_dataset) > 0
        else None
    )

    model = RetrievalModel(
        audio_arch=args.audio_arch,
        text_model_name=args.text_model,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        model_type=args.model_type,
        vision_image_size=args.vision_image_size,
        vision_image_mode=args.vision_image_mode,
        vision_channel_mode=args.vision_channel_mode,
    )
    model.to(model.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=int(args.warmup_epochs * max(1, len(train_loader))),
        total_steps=total_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
    )

    log_path = _setup_log_file(
        Path(args.log_dir).expanduser(),
        "finetune",
        args.run_name,
        {
            "metadata": str(metadata_path),
            "model_type": args.model_type,
            "dataset_size": len(dataset),
            "args": vars(args),
        },
    )

    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"finetune_{args.model_type}.pt"

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            global_step += 1
            lr = scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            audio = batch["audio"].to(model.device)
            text_inputs = {
                key: value.to(model.device) for key, value in batch["text_inputs"].items()
            }

            anchor_proj, text_proj = model(audio, text_inputs)
            loss, logits = contrastive_loss(anchor_proj, text_proj, model.current_temperature())
            loss.backward()

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(1, num_batches)
        log_items: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "lr": scheduler.get_last_lr(),
        }

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            metrics_accumulator = {"acc_audio_to_text": 0.0, "acc_text_to_audio": 0.0}

            with torch.no_grad():
                for batch in val_loader:
                    audio = batch["audio"].to(model.device)
                    text_inputs = {
                        key: value.to(model.device)
                        for key, value in batch["text_inputs"].items()
                    }
                    anchor_proj, text_proj = model(audio, text_inputs)
                    loss, logits = contrastive_loss(
                        anchor_proj, text_proj, model.current_temperature()
                    )
                    metrics = compute_metrics(logits)
                    val_loss += loss.item()
                    val_batches += 1
                    for key in metrics_accumulator:
                        metrics_accumulator[key] += metrics[key]

            avg_val_loss = val_loss / max(1, val_batches)
            log_items["val_loss"] = avg_val_loss
            for key in metrics_accumulator:
                log_items[key] = metrics_accumulator[key] / max(1, val_batches)

            metric_to_track = log_items.get(args.selection_metric, avg_val_loss)
            if best_metric is None or (metric_to_track < best_metric if args.metric_mode == "min" else metric_to_track > best_metric):
                best_metric = metric_to_track
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "loss": avg_val_loss,
                        "metric": metric_to_track,
                        "args": vars(args),
                        "model_state": model.state_dict(),
                        "tokenizer": args.text_model,
                        "audio_arch": args.audio_arch,
                        "model_type": args.model_type,
                    },
                    checkpoint_path,
                )
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "loss": avg_train_loss,
                    "metric": avg_train_loss,
                    "args": vars(args),
                    "model_state": model.state_dict(),
                    "tokenizer": args.text_model,
                    "audio_arch": args.audio_arch,
                    "model_type": args.model_type,
                },
                checkpoint_path,
            )

        _append_log(
            log_path,
            {
                "event": "epoch",
                **{k: float(v) if isinstance(v, (int, float)) else v for k, v in log_items.items()},
            },
        )

    if best_metric is not None:
        _append_log(
            log_path,
            {
                "event": "best",
                "epoch": best_epoch,
                "metric": best_metric,
                "checkpoint": str(checkpoint_path),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Metadata CSV/JSON path")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--log-dir", default="logs", help="Directory for JSONL logs")
    parser.add_argument("--run-name", default=None, help="Optional run identifier for logging")

    parser.add_argument("--model-type", choices=["baseline", "vision"], default="baseline")
    parser.add_argument("--audio-arch", default="passt_s_swa_p16_128_ap476", help="PaSST architecture")
    parser.add_argument("--text-model", default="roberta-base", help="Hugging Face text model")
    parser.add_argument("--projection-dim", type=int, default=1024, help="Shared embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden size for projection MLPs")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for projection heads")
    parser.add_argument("--sample-rate", type=int, default=32000, help="Target audio sample rate")
    parser.add_argument("--max-audio-seconds", type=float, default=None, help="Optional max audio length in seconds")
    parser.add_argument("--max-text-length", type=int, default=32, help="Tokenization max length")

    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="Warmup duration in epochs")
    parser.add_argument("--max-lr", type=float, default=3e-6, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Final learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm (0 to disable)")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")

    parser.add_argument("--selection-metric", default="val_loss", help="Metric key used for best checkpoint")
    parser.add_argument("--metric-mode", choices=["min", "max"], default="min", help="Direction for selection metric")

    parser.add_argument("--vision-image-size", type=int, default=128, help="Pseudo-image size for vision model")
    parser.add_argument("--vision-image-mode", default="nearest", help="Interpolation mode for pseudo-images")
    parser.add_argument(
        "--vision-channel-mode", choices=["split", "replicate"], default="split", help="Channel mapping for pseudo-images"
    )

    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entry point
    args = parse_args()
    train(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
