"""Text encoder built on top of a RoBERTa backbone."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TextEncoder:
    """Encodes raw text prompts into fixed-size RoBERTa embeddings."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        device: Optional[Union[str, torch.device]] = None,
        normalize: bool = True,
        max_length: int = 256,
    ) -> None:
        self._device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._normalize = normalize
        self._max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self._device)
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return self._device

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        """Encode one or more texts into embeddings."""

        if isinstance(texts, (str, bytes)):
            batch_texts = [texts]
        else:
            batch_texts = list(texts)
            if not batch_texts:
                raise ValueError("texts cannot be empty")

        encoded = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        ).to(self.device)

        with torch.no_grad():
            model_out = self.model(**encoded)

        hidden = model_out.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        summed = (hidden * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1.0)
        embeddings = summed / counts

        should_normalize = self._normalize if normalize is None else normalize
        if should_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu()
