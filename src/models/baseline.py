"""Baseline PaSST + RoBERTa retrieval model."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from hear21passt.base import get_basic_model

from .projection import resolve_projection_head
from ..utils.metrics import contrastive_loss, compute_metrics, compute_global_metrics

__all__ = ["RetrievalModule"]


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
        projection_head_name: str = "mlp",
        projection_kwargs: Optional[Dict[str, Any]] = None,
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

        head_cls = resolve_projection_head(projection_head_name)
        kwargs = projection_kwargs or {}
        self.audio_projection = head_cls(audio_dim, hidden_dim, projection_dim, dropout, **kwargs).to(device)
        self.text_projection = head_cls(text_dim, hidden_dim, projection_dim, dropout, **kwargs).to(device)

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

        self._val_audio.append(anchor.detach().cpu())
        self._val_text.append(text.detach().cpu())
        self._val_ids.extend(batch.get("ids", []))

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
            warmup_steps = max(warmup_steps, 1)
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

    def on_validation_epoch_start(self) -> None:
        self._val_audio: list[torch.Tensor] = []
        self._val_text: list[torch.Tensor] = []
        self._val_ids: list[str] = []

    def on_validation_epoch_end(self) -> None:
        if not self._val_audio or not self._val_text:
            return

        audio = torch.cat(self._val_audio, dim=0)
        text = torch.cat(self._val_text, dim=0)
        ids = self._val_ids if self._val_ids else [f"sample_{i}" for i in range(len(audio))]

        metrics = compute_global_metrics(audio, text, ids)
        for key, value in metrics.items():
            self.log(f"val/{key}", value, prog_bar=True, on_epoch=True)
