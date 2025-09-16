"""Audio encoder wrapper built on top of the PaSST model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Iterable, Optional, Union

from typing import TYPE_CHECKING

try:  # numpy is optional at runtime
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - numpy might be missing
    np = None

import torch
import torch.nn.functional as F
import torchaudio
from hear21passt.base import get_basic_model

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import numpy as _np
    NumpyArray = _np.ndarray
else:
    class NumpyArray:  # type: ignore[misc]
        """Fallback used when numpy is not installed."""

        pass


WaveformLike = Union[str, Path, torch.Tensor, "NumpyArray"]


class AudioEncoder:
    """Encodes raw audio waveforms into PaSST embeddings."""

    def __init__(
        self,
        arch: str = "passt_s_swa_p16_128_ap476",
        device: Optional[Union[str, torch.device]] = None,
        normalize: bool = True,
    ) -> None:
        self._device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # The wrapper already includes the mel front-end required by PaSST.
        self.model = get_basic_model(mode="embed_only", arch=arch)
        self.model.to(self._device)
        self.model.eval()
        self.sample_rate = getattr(self.model, "sample_rate", 32000)
        self._normalize = normalize

    @property
    def device(self) -> torch.device:
        return self._device

    def encode(
        self,
        audio: Union[WaveformLike, Sequence[WaveformLike]],
        sample_rate: Optional[Union[int, Sequence[int]]] = None,
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        """Encode a single audio clip or a batch.

        Args:
            audio: Path(s) or waveform(s) to encode.
            sample_rate: Optional sampling rate(s) corresponding to the provided
                waveform(s). Ignored for file paths.
            normalize: Whether to L2 normalise the resulting embeddings. Defaults
                to what was specified during initialisation.

        Returns:
            Tensor of shape `(batch, embedding_dim)`.
        """
        if isinstance(audio, Sequence) and not isinstance(audio, (str, bytes, Path)):
            sr_seq = self._expand_sample_rates(audio, sample_rate)
            waveforms = [
                self._load_waveform(item, sr)
                for item, sr in zip(audio, sr_seq, strict=True)
            ]
            waveform_batch = self._collate_waveforms(waveforms)
        else:
            waveform_batch = self._collate_waveforms(
                [self._load_waveform(audio, sample_rate)]  # pyright: ignore[reportArgumentType]
            )

        waveform_batch = waveform_batch.to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_scene_embeddings(waveform_batch)

        should_normalize = self._normalize if normalize is None else normalize
        if should_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu()

    def _load_waveform(
        self, audio: WaveformLike, sample_rate: Optional[int]
    ) -> torch.Tensor:
        if isinstance(audio, (str, Path)):
            waveform, sr = torchaudio.load(Path(audio))
        else:
            waveform = self._to_tensor(audio)
            if sample_rate is None:
                raise ValueError(
                    "sample_rate must be provided when passing raw waveforms"
                )
            sr = int(sample_rate)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.float()
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        return waveform.squeeze(0)

    def _collate_waveforms(self, waveforms: Iterable[torch.Tensor]) -> torch.Tensor:
        batches = []
        max_len = 0
        for waveform in waveforms:
            waveform = waveform.flatten().unsqueeze(0)
            batches.append(waveform)
            max_len = max(max_len, waveform.shape[-1])

        padded = []
        for waveform in batches:
            if waveform.shape[-1] == max_len:
                padded.append(waveform)
            else:
                pad_amount = max_len - waveform.shape[-1]
                padded.append(F.pad(waveform, (0, pad_amount)))

        return torch.cat(padded, dim=0)

    @staticmethod
    def _expand_sample_rates(
        audio: Sequence[WaveformLike], sample_rates: Optional[Union[int, Sequence[int]]]
    ) -> Sequence[Optional[int]]:
        if sample_rates is None or isinstance(sample_rates, int):
            return [sample_rates] * len(audio)
        if len(sample_rates) != len(audio):
            raise ValueError("sample_rate sequence length must match audio sequence")
        return sample_rates

    @staticmethod
    def _to_tensor(data: WaveformLike) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu()
        if np is not None and isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        raise TypeError("Unsupported audio input type")
