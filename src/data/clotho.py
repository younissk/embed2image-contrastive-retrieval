"""Data utilities for the Clotho audio-caption dataset."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
from transformers import AutoTokenizer

__all__ = [
    "Sample",
    "load_clotho_samples",
    "ClothoDataset",
    "ContrastiveCollate",
    "ClothoDataModule",
]


@dataclass(slots=True)
class Sample:
    audio_path: Path
    caption: str
    sample_id: str


def _normalise_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def load_clotho_samples(metadata_path: Path) -> list[Sample]:
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
                    f"Audio file not found for caption entry '{filename}'."
                )
            for idx, column in enumerate(caption_columns, start=1):
                caption = row.get(column, "").strip()
                if not caption:
                    continue
                sample_id = f"{filename}#{idx}"
                samples.append(Sample(audio_path=audio_path, caption=caption, sample_id=sample_id))

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
            "id": sample.sample_id,
        }


class ContrastiveCollate:
    """Pad audio and tokenize captions."""

    def __init__(self, tokenizer: AutoTokenizer, max_text_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:  # pragma: no cover - trivial
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
            "ids": [str(item["id"]) for item in batch],
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
        full_val: bool = False,
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
        self.full_val = full_val

        self.tokenizer: Optional[AutoTokenizer] = None
        self.collate: Optional[ContrastiveCollate] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        samples = load_clotho_samples(self.metadata)
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
        if self.full_val:
            self.val_dataset = ClothoDataset(
                samples,
                sample_rate=self.sample_rate,
                max_seconds=self.max_audio_seconds,
            )
        else:
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
