"""CLI for fine-tuning PaSST + RoBERTa on Clotho via Lightning."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Type

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

try:  # optional dependency
    from lightning.pytorch.loggers import WandbLogger
except ModuleNotFoundError:  # pragma: no cover - optional
    WandbLogger = None

from .data.clotho import ClothoDataModule
from .models.baseline import RetrievalModule
from .models.projection import ProjectionHead, resolve_projection_head


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        default=str(Path.home() / "data" / "CLOTHO_v2.1" / "clotho_csv_files" / "clotho_captions_development.csv"),
        help="Path to Clotho caption CSV",
    )
    parser.add_argument("--output-dir", default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--log-dir", default="logs", help="Directory for logging")
    parser.add_argument("--run-name", default=None, help="Optional run name for loggers")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=None, help="W&B project name (defaults to env)")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity/team (defaults to env)")

    parser.add_argument("--audio-arch", default="passt_s_swa_p16_128_ap476", help="PaSST architecture identifier")
    parser.add_argument("--text-model", default="roberta-base", help="Hugging Face text encoder")
    parser.add_argument("--projection-dim", type=int, default=1024, help="Shared embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension in projection heads (0 for linear)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout in projection heads")
    parser.add_argument("--projection-head", default="mlp", help="Projection head identifier (see registry)")

    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="Number of warmup epochs")
    parser.add_argument("--max-lr", type=float, default=3e-6, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm (0 to disable)")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of samples used for validation")

    parser.add_argument("--max-audio-seconds", type=float, default=20.0, help="Truncate audio to this many seconds")
    parser.add_argument("--sample-rate", type=int, default=32000, help="Target audio sample rate")
    parser.add_argument("--max-text-length", type=int, default=32, help="Tokenizer max length")

    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--precision", default="bf16-mixed", help="Trainer precision mode")
    parser.add_argument("--devices", default=1, help="Number of devices (or 'auto')")
    parser.add_argument("--strategy", default="auto", help="Lightning strategy (e.g. ddp, auto)")
    parser.add_argument("--log-every-n-steps", type=int, default=25, help="Logging frequency")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic trainer mode")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs). Set <=0 to disable",
    )
    parser.add_argument(
        "--early-stop-metric",
        default="val/mAP@10_text_to_audio",
        help="Metric monitored for early stopping",
    )
    parser.add_argument(
        "--early-stop-mode",
        choices=["min", "max"],
        default="max",
        help="Early stopping direction",
    )

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

    projection_head_cls: Type[ProjectionHead] = resolve_projection_head(args.projection_head)

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
        projection_head=projection_head_cls,
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
    callbacks = [checkpoint_cb, lr_monitor]

    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=args.early_stop_metric,
                patience=args.early_stop_patience,
                mode=args.early_stop_mode,
                verbose=True,
            )
        )

    trainer = L.Trainer(
        default_root_dir=str(output_dir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=args.strategy,
        max_epochs=args.epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=args.deterministic,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":  # pragma: no cover
    main()
