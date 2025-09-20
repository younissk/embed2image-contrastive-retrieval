"""Fine-tune PaSST + RoBERTa encoders on Clotho using PyTorch Lightning."""

from __future__ import annotations

import argparse
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer

from hear21passt.base import get_basic_model
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

try:  # optional dependency
    from lightning.pytorch.loggers import WandbLogger
except ModuleNotFoundError:  # pragma: no cover - optional
    WandbLogger = None


@dataclass(slots=True)
class Sample:
    audio_path: Path
    caption: str
    sample_id: str


def _normalise_name(name: str) -> str:
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

    lookup: dict[str, Path] = {}
    for wav_path in audio_dir.glob("*.wav"):
        lookup[wav_path.name] = wav_path
        lookup.setdefault(_normalise_name(wav_path.name), wav_path)

    samples: list[Sample] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        import csv

        reader = csv.DictReader(handle)
        caption_columns = [col for col in reader.fieldnames or [] if col.startswith("caption_")]
        if "file_name" not in (reader.fieldnames or []):
            raise ValueError("CSV must provide a 'file_name' column")
        if not caption_columns:
            raise ValueError("CSV must contain caption_* columns")

        for row in reader:
            filename = row["file_name"].strip()
            audio_path = lookup.get(filename) or lookup.get(_normalise_name(filename))
            if audio_path is None:
                raise FileNotFoundError(
                    f"Audio file not found for caption entry '{filename}'. "
                    "Verify that the Clotho archives were extracted correctly."
                )
            for idx, column in enumerate(caption_columns, start=1):
                caption = row.get(column, "").strip()
                if not caption:
                    continue
                sample_id = f"{filename}#{idx}"
                samples.append(
                    Sample(audio_path=audio_path, caption=caption, sample_id=_normalise_name(sample_id))
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


class ClothoDataset(Dataset):
    """Dataset yielding waveform/caption examples for contrastive training."""

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
        }


class ContrastiveCollate:
    """Pad audio and tokenize captions."""

    def __init__(self, tokenizer: AutoTokenizer, max_text_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        if not batch:
            raise ValueError("Empty batch")

        lengths = [int(item["length"]) for item in batch]
        max_len = max(lengths)
        audio_batch = torch.zeros(len(batch), max_len, dtype=torch.float32)
        for idx, item in enumerate(batch):
            waveform = item["audio"].float()
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


class ClothoDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        metadata: Path,
        text_model: str,
        batch_size: int,
        val_fraction: float,
        sample_rate: int,
        max_audio_seconds: Optional[float],
        max_text_length: int,
        num_workers: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.text_model = text_model
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.max_text_length = max_text_length
        self.num_workers = num_workers
        self.seed = seed

        self.tokenizer: Optional[AutoTokenizer] = None
        self.collate: Optional[ContrastiveCollate] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        samples = _load_clotho_samples(self.metadata)
        rng = random.Random(self.seed)
        rng.shuffle(samples)

        num_val = max(1, int(len(samples) * self.val_fraction)) if self.val_fraction > 0 else 0
        val_samples = samples[:num_val]
        train_samples = samples[num_val:] if num_val else samples

        if not val_samples:
            val_samples = train_samples[: max(1, len(train_samples) // 10)]

        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model)
        self.collate = ContrastiveCollate(self.tokenizer, self.max_text_length)

        self.train_dataset = ClothoDataset(
            train_samples,
            sample_rate=self.sample_rate,
            max_seconds=self.max_audio_seconds,
        )
        self.val_dataset = ClothoDataset(
            val_samples,
            sample_rate=self.sample_rate,
            max_seconds=self.max_audio_seconds,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None and self.collate is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None and self.collate is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            persistent_workers=self.num_workers > 0,
        )


class ProjectionHead(nn.Module):
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


class RetrievalModule(L.LightningModule):
    def __init__(
        self,
        *,
        audio_arch: str,
        text_model: str,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
        max_lr: float,
        min_lr: float,
        weight_decay: float,
        warmup_epochs: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_model = get_basic_model(mode="embed_only", arch=audio_arch).to(device)
        self.text_model = AutoModel.from_pretrained(
            text_model,
            add_pooling_layer=False,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        ).to(device)

        audio_dim = getattr(self.audio_model, "embed_dim", 768)
        text_dim = self.text_model.config.hidden_size

        self.audio_projection = ProjectionHead(audio_dim, hidden_dim, projection_dim, dropout).to(device)
        self.text_projection = ProjectionHead(text_dim, hidden_dim, projection_dim, dropout).to(device)

        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode_audio(self, waveforms: torch.Tensor) -> torch.Tensor:
        embeddings = self.audio_model.get_scene_embeddings(waveforms)
        embeddings = embeddings.to(waveforms.device)
        return self.audio_projection(embeddings)

    def encode_text(self, text_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.text_model(**text_inputs)
        hidden = outputs.last_hidden_state
        attention_mask = text_inputs["attention_mask"].unsqueeze(-1)
        summed = (hidden * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1.0)
        text_features = summed / counts
        return self.text_projection(text_features)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        audio = batch["audio"]
        text_inputs = batch["text_inputs"]
        anchor = F.normalize(self.encode_audio(audio), dim=-1)
        text = F.normalize(self.encode_text(text_inputs), dim=-1)
        return anchor, text

    def current_temperature(self) -> torch.Tensor:
        return torch.abs(self.temperature) + 1e-6

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch = self._move_batch_to_device(batch)
        anchor, text = self(batch)
        loss, logits = contrastive_loss(anchor, text, self.current_temperature())
        metrics = compute_metrics(logits)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/tau", self.current_temperature(), prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/acc_audio_to_text", metrics["acc_audio_to_text"], prog_bar=False, on_epoch=True)
        self.log("train/acc_text_to_audio", metrics["acc_text_to_audio"], prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        batch = self._move_batch_to_device(batch)
        anchor, text = self(batch)
        loss, logits = contrastive_loss(anchor, text, self.current_temperature())
        metrics = compute_metrics(logits)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc_audio_to_text", metrics["acc_audio_to_text"], prog_bar=True, on_epoch=True)
        self.log("val/acc_text_to_audio", metrics["acc_text_to_audio"], prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.max_lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.warmup_epochs <= 0:
            return optimizer

        def lr_lambda(step: int) -> float:
            total_steps = max(1, self.trainer.estimated_stepping_batches)
            warmup_steps = int(self.hparams.warmup_epochs * total_steps / max(1, self.trainer.max_epochs))
            if warmup_steps <= 0:
                warmup_steps = 1
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            min_lr = self.hparams.min_lr / max(self.hparams.max_lr, 1e-12)
            return max(min_lr, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            }
        ]

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = self.device
        batch["audio"] = batch["audio"].to(device)
        batch["text_inputs"] = {k: v.to(device) for k, v in batch["text_inputs"].items()}
        return batch


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=str(Path.home() / "data" / "CLOTHO_v2.1" / "clotho_csv_files" / "clotho_captions_development.csv"))
    parser.add_argument("--output-dir", default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--log-dir", default="logs", help="Directory for logging")
    parser.add_argument("--run-name", default=None, help="Optional run name for loggers")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project name (defaults to WANDB_PROJECT env)",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity/team (defaults to WANDB_ENTITY env)",
    )

    parser.add_argument("--audio-arch", default="passt_s_swa_p16_128_ap476", help="PaSST architecture identifier")
    parser.add_argument("--text-model", default="roberta-base", help="Hugging Face text encoder")
    parser.add_argument("--projection-dim", type=int, default=1024, help="Shared embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension in projection heads (0 for linear)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout in projection heads")

    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="Number of warmup epochs")
    parser.add_argument("--max-lr", type=float, default=3e-6, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm (0 to disable)")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of samples used for validation")

    parser.add_argument("--max-audio-seconds", type=float, default=20.0, help="Truncate audio to this many seconds (None to disable)")
    parser.add_argument("--sample-rate", type=int, default=32000, help="Target audio sample rate")
    parser.add_argument("--max-text-length", type=int, default=32, help="Maximum number of BPE tokens per caption")

    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--precision", default="bf16-mixed", help="Trainer precision (e.g. 32, 16-mixed, bf16-mixed)")
    parser.add_argument("--devices", default=1, help="Number of devices (or 'auto')")
    parser.add_argument("--strategy", default="auto", help="Lightning strategy (e.g. ddp, auto)")
    parser.add_argument("--log-every-n-steps", type=int, default=25, help="Logging frequency")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic trainer mode")

    return parser


def main() -> None:  # pragma: no cover - CLI entry point
    parser = build_parser()
    args = parser.parse_args()

    metadata_path = Path(args.metadata).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(args.seed, workers=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

    data_module = ClothoDataModule(
        metadata=metadata_path,
        text_model=args.text_model,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        sample_rate=args.sample_rate,
        max_audio_seconds=args.max_audio_seconds,
        max_text_length=args.max_text_length,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = RetrievalModule(
        audio_arch=args.audio_arch,
        text_model=args.text_model,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
    )

    logger = None
    if args.use_wandb:
        if WandbLogger is None:
            raise RuntimeError("wandb is not installed but --use-wandb was set")
        project = args.wandb_project or os.getenv("WANDB_PROJECT") or "embed2image"
        entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
        logger = WandbLogger(project=project, name=args.run_name, entity=entity, log_model=False)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="finetune-{epoch:02d}-{val_loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        default_root_dir=str(output_dir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=args.strategy,
        max_epochs=args.epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        gradient_clip_val=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=args.deterministic,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":  # pragma: no cover
    main()
