"""Fine-tune PaSST + RoBERTa encoders on Clotho with a contrastive objective."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
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

try:  # optional logging backend
    import wandb
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wandb = None

DEFAULT_METADATA = (
    Path.home()
    / "data"
    / "CLOTHO_v2.1"
    / "clotho_csv_files"
    / "clotho_captions_development.csv"
)


@dataclass(slots=True)
class Sample:
    audio_path: Path
    caption: str
    sample_id: str


def _sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "sample"


def _normalized(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _load_clotho_samples(metadata_path: Path) -> list[Sample]:
    if metadata_path.suffix.lower() != ".csv":
        raise ValueError("Expected a Clotho caption CSV as metadata input")

    dataset_root = metadata_path.parent.parent
    split = _infer_split(metadata_path)
    audio_dir = dataset_root / "clotho_audio_files" / split
    if not audio_dir.exists():
        raise FileNotFoundError(
            f"Missing audio directory '{audio_dir}'. Run 'make download-dataset' first."
        )

    # Build a lookup table mapping both original and normalized filenames to actual file paths
    lookup = {}
    for wav_path in audio_dir.glob("*.wav"):
        lookup[wav_path.name] = wav_path
        lookup.setdefault(_normalized(wav_path.name), wav_path)

    samples: list[Sample] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        caption_columns = [col for col in reader.fieldnames or [] if col.startswith("caption_")]
        if "file_name" not in (reader.fieldnames or []):
            raise ValueError("CSV must provide a 'file_name' column")
        if not caption_columns:
            raise ValueError("CSV must contain caption_* columns")

        for row in reader:
            filename = row["file_name"].strip()
            audio_path = lookup.get(filename) or lookup.get(_normalized(filename))
            if audio_path is None:
                raise FileNotFoundError(f"Audio file not found: {filename} (tried both original and normalized names)")
            for idx, column in enumerate(caption_columns, start=1):
                caption = row.get(column, "").strip()
                if not caption:
                    continue
                sample_id = f"{filename}#{idx}"
                samples.append(
                    Sample(audio_path=audio_path, caption=caption, sample_id=_sanitize_identifier(sample_id))
                )

    if not samples:
        raise ValueError(f"No samples found in metadata file {metadata_path}")

    return samples


def _infer_split(metadata_path: Path) -> str:
    stem = metadata_path.stem.lower()
    if "development" in stem:
        return "development"
    if "validation" in stem:
        return "validation"
    if "evaluation" in stem:
        return "evaluation"
    raise ValueError(
        "Unable to infer split from metadata filename. Expected one of development/validation/evaluation."
    )


class ContrastiveDataset(Dataset):
    """Dataset yielding waveform/caption pairs for contrastive training."""

    def __init__(
        self,
        samples: Iterable[Sample],
        *,
        sample_rate: int,
        max_seconds: Optional[float] = None,
    ) -> None:
        self._items = list(samples)
        self.sample_rate = sample_rate
        self.max_length = int(max_seconds * sample_rate) if max_seconds else None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, object]:  # pragma: no cover - trivial
        sample = self._items[index]
        waveform, sr = torchaudio.load(sample.audio_path)
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
            "length": waveform.numel(),
            "text": sample.caption,
            "id": sample.sample_id,
        }


class ContrastiveCollate:
    """Pad audio and batch tokenized captions."""

    def __init__(self, tokenizer: AutoTokenizer, max_text_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:  # pragma: no cover - trivial
        if not batch:
            raise ValueError("Empty batch")

        lengths = [int(item["length"]) for item in batch]
        max_len = max(lengths)
        audio_batch = torch.zeros(len(batch), max_len)
        for idx, item in enumerate(batch):
            waveform = item["audio"]
            audio_batch[idx, : waveform.numel()] = waveform

        tokenized = self.tokenizer(
            [str(item["text"]) for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        )

        return {
            "audio": audio_batch,
            "text_inputs": tokenized,
        }


class WarmupCosineScheduler:
    """Cosine annealing with linear warmup wrapper."""

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
            decay_progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            decay_progress = min(decay_progress, 1.0)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * decay_progress))
        self._set_lr(lr)
        return lr

    def get_last_lr(self) -> float:  # pragma: no cover - trivial
        return self.current_lr


class ProjectionHead(nn.Module):
    """Two-layer MLP for projection to the shared embedding space."""

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


class RetrievalModel(nn.Module):
    """PaSST audio encoder + RoBERTa text encoder with projection heads."""

    def __init__(
        self,
        *,
        audio_arch: str,
        text_model: str,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.audio_model = get_basic_model(mode="embed_only", arch=audio_arch)
        self.audio_model.to(self.device)

        self.text_model = AutoModel.from_pretrained(
            text_model,
            add_pooling_layer=False,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.text_model.to(self.device)

        audio_dim = getattr(self.audio_model, "embed_dim", 768)
        text_dim = self.text_model.config.hidden_size

        self.audio_projection = ProjectionHead(audio_dim, hidden_dim, projection_dim, dropout)
        self.text_projection = ProjectionHead(text_dim, hidden_dim, projection_dim, dropout)

        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode_audio(self, waveforms: torch.Tensor) -> torch.Tensor:
        embeddings = self.audio_model.get_scene_embeddings(waveforms.to(self.device))
        return self.audio_projection(embeddings)

    def encode_text(self, text_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        outputs = self.text_model(**inputs)
        hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (hidden * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1.0)
        text_features = summed / counts
        return self.text_projection(text_features)

    def forward(self, waveforms: torch.Tensor, text_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = F.normalize(self.encode_audio(waveforms), dim=-1)
        text = F.normalize(self.encode_text(text_inputs), dim=-1)
        return anchor, text

    def current_temperature(self) -> torch.Tensor:
        return torch.abs(self.temperature) + 1e-6


def contrastive_loss(anchor: torch.Tensor, text: torch.Tensor, temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = anchor @ text.t() / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a + loss_t), logits


def compute_metrics(logits: torch.Tensor) -> dict[str, float]:
    targets = torch.arange(logits.size(0), device=logits.device)
    acc_anchor = (logits.argmax(dim=1) == targets).float().mean().item()
    acc_text = (logits.argmax(dim=0) == targets).float().mean().item()
    return {
        "acc_audio_to_text": acc_anchor,
        "acc_text_to_audio": acc_text,
    }


def split_dataset(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    if val_fraction <= 0:
        return dataset, Subset(dataset, [])
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    num_val = max(1, int(len(dataset) * val_fraction))
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _setup_log_file(log_dir: Path, prefix: str, run_name: Optional[str], metadata: dict) -> Path:
    """Create a log file with timestamp and metadata header."""
    from datetime import datetime
    
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_suffix = f"_{run_name}" if run_name else ""
    log_filename = f"{prefix}_{timestamp}{run_suffix}.jsonl"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_filename
    
    # Write metadata header
    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "metadata", **metadata}) + "\n")
    
    return log_path


def _append_log(log_path: Path, payload: dict) -> None:
    """Append a log entry to the JSONL log file."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    metadata_path = Path(args.metadata).expanduser()
    samples = _load_clotho_samples(metadata_path)

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
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
        if len(val_dataset) > 0
        else None
    )

    model = RetrievalModel(
        audio_arch=args.audio_arch,
        text_model=args.text_model,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
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
            "dataset_size": len(dataset),
            **{f"arg/{key}": value for key, value in vars(args).items()},
        },
    )

    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "finetune_baseline.pt"

    with wandb_run:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                audio = batch["audio"].to(model.device)
                text_inputs = {k: v.to(model.device) for k, v in batch["text_inputs"].items()}

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
                metrics_acc = {"acc_audio_to_text": 0.0, "acc_text_to_audio": 0.0}

                with torch.no_grad():
                    for batch in val_loader:
                        audio = batch["audio"].to(model.device)
                        text_inputs = {k: v.to(model.device) for k, v in batch["text_inputs"].items()}
                        anchor_proj, text_proj = model(audio, text_inputs)
                        loss, logits = contrastive_loss(anchor_proj, text_proj, model.current_temperature())
                        metrics = compute_metrics(logits)
                        val_loss += loss.item()
                        val_batches += 1
                        for key in metrics_acc:
                            metrics_acc[key] += metrics[key]

                avg_val_loss = val_loss / max(1, val_batches)
                log_items["val_loss"] = avg_val_loss
                for key in metrics_acc:
                    log_items[key] = metrics_acc[key] / max(1, val_batches)

                metric_to_track = log_items.get(args.selection_metric, avg_val_loss)
                improved = (
                    best_metric is None
                    or (
                        metric_to_track < best_metric if args.metric_mode == "min" else metric_to_track > best_metric
                    )
                )
                if improved:
                    best_metric = metric_to_track
                    best_epoch = epoch
                    torch.save(
                        {
                            "epoch": epoch,
                            "loss": avg_val_loss,
                            "metric": metric_to_track,
                            "args": vars(args),
                            "model_state": model.state_dict(),
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
                    },
                    checkpoint_path,
                )

            payload = {
                "event": "epoch",
                **{k: float(v) if isinstance(v, (int, float)) else v for k, v in log_items.items()},
            }
            _append_log(log_path, payload)
            wandb_run.log({k: v for k, v in payload.items() if isinstance(v, (int, float))})

        if best_metric is not None:
            summary = {
                "event": "best",
                "epoch": best_epoch,
                "metric": best_metric,
                "checkpoint": str(checkpoint_path),
            }
            _append_log(log_path, summary)
            wandb_run.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Path to Clotho caption CSV")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--log-dir", default="logs", help="Directory for JSONL logs")
    parser.add_argument("--run-name", default=None, help="Optional identifier for logging")
    parser.add_argument("--use-wandb", action="store_true", help="Stream metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default=None, help="Override WANDB_PROJECT for the run")

    parser.add_argument("--audio-arch", default="passt_s_swa_p16_128_ap476", help="PaSST architecture name")
    parser.add_argument("--text-model", default="roberta-base", help="Hugging Face text model")
    parser.add_argument("--projection-dim", type=int, default=1024, help="Shared embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden layer size for projection heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout applied in projection heads")
    parser.add_argument("--sample-rate", type=int, default=32000, help="Target sample rate for audio")
    parser.add_argument("--max-audio-seconds", type=float, default=None, help="Optional max waveform length in seconds")
    parser.add_argument("--max-text-length", type=int, default=32, help="Tokenization max length")

    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="Warmup duration (in epochs)")
    parser.add_argument("--max-lr", type=float, default=3e-6, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Final learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm (0 to disable)")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")

    parser.add_argument("--selection-metric", default="val_loss", help="Metric name for best checkpoint selection")
    parser.add_argument("--metric-mode", choices=["min", "max"], default="min", help="Direction of selection metric")

    return parser.parse_args()


class _WandbRun:
    """Context manager around wandb.init/finish (best-effort)."""

    def __init__(self, enabled: bool, *, project: Optional[str], run_name: Optional[str], config: dict[str, object]) -> None:
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


def main() -> None:  # pragma: no cover - CLI entry point
    args = parse_args()
    train(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
