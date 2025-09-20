"""Evaluate a fine-tuned PaSST + RoBERTa checkpoint on Clotho."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch

from .train_finetune import ClothoDataModule, RetrievalModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to Lightning checkpoint")
    parser.add_argument(
        "--metadata",
        default=str(Path.home() / "data" / "CLOTHO_v2.1" / "clotho_csv_files" / "clotho_captions_development.csv"),
        help="Path to Clotho caption CSV",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--sample-rate", type=int, default=32000, help="Target audio sample rate")
    parser.add_argument("--max-audio-seconds", type=float, default=10.0, help="Clip length passed to PaSST")
    parser.add_argument("--max-text-length", type=int, default=32, help="Tokenizer max length")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--precision", default="bf16-mixed", help="Lightning precision mode")
    parser.add_argument("--devices", default=1, help="Number of devices or 'auto'")
    parser.add_argument("--strategy", default="auto", help="Lightning strategy")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    return parser


def main() -> None:  # pragma: no cover - CLI entry point
    parser = build_parser()
    args = parser.parse_args()

    metadata_path = Path(args.metadata).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()

    model = RetrievalModule.load_from_checkpoint(str(checkpoint_path))

    L.seed_everything(args.seed, workers=True)

    data_module = ClothoDataModule(
        metadata=metadata_path,
        text_model=model.hparams.get("text_model", "roberta-base"),
        batch_size=args.batch_size,
        val_fraction=0.1,
        sample_rate=args.sample_rate,
        max_audio_seconds=args.max_audio_seconds,
        max_text_length=args.max_text_length,
        num_workers=args.num_workers,
        seed=args.seed,
        full_val=True,
    )

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        logger=False,
    )

    trainer.validate(model=model, datamodule=data_module, ckpt_path=None)


if __name__ == "__main__":  # pragma: no cover
    main()
