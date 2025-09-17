"""Helpers for generating and caching embeddings ahead of time."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

from collections.abc import Iterable, Mapping, MutableMapping, Sequence as SequenceABC

import torch
from tqdm import tqdm

from ..audio_encoder import AudioEncoder
from ..text_encoder import TextEncoder
from .hf_uploader import HFUploadError, HFDatasetUploader


Sample = Mapping[str, object]

DEFAULT_METADATA = (
    Path.home()
    / "data"
    / "CLOTHO_v2.1"
    / "clotho_csv_files"
    / "clotho_captions_development.csv"
)
DEFAULT_CACHE_DIR = (
    Path.home() / "data" / "CLOTHO_v2.1" / "embeddings" / "development"
)


def cache_embeddings(
    samples: Iterable[Sample],
    cache_dir: Path | str,
    audio_encoder: AudioEncoder,
    text_encoder: TextEncoder,
    *,
    id_key: str = "id",
    audio_key: str = "audio",
    text_key: str = "text",
    sample_rate_key: str = "sample_rate",
    batch_size: int = 8,
    force: bool = False,
    show_progress: bool = True,
    metadata_source: Optional[str] = None,
    upload_every: Optional[int] = None,
    hf_dataset: Optional[str] = None,
    hf_prefix: str = "",
    hf_token: Optional[str] = None,
) -> dict[str, Path]:
    """Generate embeddings for a dataset and persist them on disk.

    Args:
        samples: Iterable of sample dictionaries. Each sample must expose the
            audio data under `audio_key` and the text prompt under `text_key`.
            Audio items can be paths to audio files or waveforms compatible with
            :class:`AudioEncoder`.
        cache_dir: Directory where cache files should be written.
        audio_encoder: Instantiated :class:`AudioEncoder` used to produce audio
            embeddings.
        text_encoder: Instantiated :class:`TextEncoder` used to produce text
            embeddings.
        id_key: Optional key used to fetch a stable identifier per sample. If
            absent, the index is used.
        audio_key: Key used to retrieve audio inputs from each sample mapping.
        text_key: Key used to retrieve the text prompt from each sample mapping.
        sample_rate_key: Key used to retrieve the sampling rate when passing raw
            waveforms instead of file paths.
        batch_size: Number of samples to encode per batch.
        force: If ``True`` the cache will be regenerated even if the expected
            files already exist.
        show_progress: Whether to display a progress bar.
        metadata_source: Optional string describing the metadata origin. Stored
            in the output JSON for traceability.
        upload_every: If provided, upload intermediate checkpoints to the
            Hugging Face dataset every ``upload_every`` processed pairs. Applies
            only when ``hf_dataset`` is set.
        hf_dataset: Optional Hugging Face dataset repository ID (e.g.
            ``username/audio-text-embed-to-images``). When provided, the cache
            files are mirrored to that dataset repository.
        hf_prefix: Optional subdirectory inside the dataset repository for the
            uploaded files.
        hf_token: Optional Hugging Face access token. Falls back to the
            ``HF_TOKEN`` environment variable or the cached CLI token.

    Returns:
        A dictionary with paths to the generated cache files.
    """

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    audio_cache_path = cache_path / "audio_embeddings.pt"
    text_cache_path = cache_path / "text_embeddings.pt"
    meta_cache_path = cache_path / "metadata.json"

    if (
        not force
        and audio_cache_path.exists()
        and text_cache_path.exists()
        and meta_cache_path.exists()
    ):
        return {
            "audio": audio_cache_path,
            "text": text_cache_path,
            "metadata": meta_cache_path,
        }

    sample_list = list(samples)
    ids: list[str] = []
    audio_batches: list[torch.Tensor] = []
    text_tensors: list[torch.Tensor] = []
    audio_embedding_cache: dict[str, torch.Tensor] = {}

    uploader: HFDatasetUploader | None = None
    if hf_dataset:
        try:
            uploader = HFDatasetUploader(
                repo_id=hf_dataset,
                prefix=hf_prefix,
                token=hf_token,
            )
        except HFUploadError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HFUploadError(
                f"Failed to initialise Hugging Face uploader for repo '{hf_dataset}': {exc}"
            ) from exc

    last_upload_count = 0

    def _log(message: str) -> None:
        if show_progress:
            tqdm.write(message)
        else:
            print(message)

    iterator = range(0, len(sample_list), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Caching embeddings", unit="batch")

    for offset in iterator:
        batch = sample_list[offset : offset + batch_size]
        audio_inputs = [_require(sample, audio_key) for sample in batch]
        sample_rates = [sample.get(sample_rate_key) for sample in batch]
        text_inputs = [_require(sample, text_key) for sample in batch]

        batch_audio_embeddings = _encode_audio_batch(
            audio_encoder, audio_inputs, sample_rates, audio_embedding_cache  # pyright: ignore[reportArgumentType]
        )
        text_embeddings = text_encoder.encode(text_inputs)  # pyright: ignore[reportArgumentType]

        audio_batches.append(batch_audio_embeddings)
        text_tensors.append(text_embeddings)
        ids.extend(
            [
                _string_id(sample, id_key, idx + offset)
                for idx, sample in enumerate(batch)
            ]
        )

        processed = len(ids)
        should_upload = (
            uploader is not None
            and upload_every is not None
            and upload_every > 0
            and processed - last_upload_count >= upload_every
        )

        if should_upload:
            paths = _write_cache_files(
                cache_path,
                audio_batches,
                text_tensors,
                ids,
                audio_encoder,
                text_encoder,
                metadata_source,
            )
            upload_targets = _select_embedding_paths(paths)
            commit_message = (
                f"Upload partial embeddings ({processed} pairs processed)"
            )
            uris = uploader.upload_files(
                upload_targets,
                commit_message=commit_message,
            )
            last_upload_count = processed
            _log(
                f"Uploaded partial cache with {processed} pairs to"
                f" {', '.join(uris)}"
            )

    paths = _write_cache_files(
        cache_path,
        audio_batches,
        text_tensors,
        ids,
        audio_encoder,
        text_encoder,
        metadata_source,
    )

    if uploader is not None:
        upload_targets = _select_embedding_paths(paths)
        commit_message = f"Upload final embeddings ({len(ids)} pairs)"
        uris = uploader.upload_files(upload_targets, commit_message=commit_message)
        _log(
            f"Uploaded final cache with {len(ids)} pairs to {', '.join(uris)}"
        )

    return paths


def _string_id(sample: Sample, key: str, fallback: int) -> str:
    value = sample.get(key)
    if value is None:
        return str(fallback)
    return str(value)


def _write_cache_files(
    cache_path: Path,
    audio_batches: Sequence[torch.Tensor],
    text_batches: Sequence[torch.Tensor],
    ids: Sequence[str],
    audio_encoder: AudioEncoder,
    text_encoder: TextEncoder,
    metadata_source: Optional[str],
) -> dict[str, Path]:
    audio_cache_path = cache_path / "audio_embeddings.pt"
    text_cache_path = cache_path / "text_embeddings.pt"
    meta_cache_path = cache_path / "metadata.json"

    audio_mat = torch.cat(list(audio_batches), dim=0)
    text_mat = torch.cat(list(text_batches), dim=0)

    torch.save(audio_mat, audio_cache_path)
    torch.save(text_mat, text_cache_path)

    metadata: MutableMapping[str, object] = {
        "ids": list(ids),
        "audio_embedding_shape": list(audio_mat.shape),
        "text_embedding_shape": list(text_mat.shape),
        "audio_sample_rate": audio_encoder.sample_rate,
        "audio_encoder": type(audio_encoder).__name__,
        "text_encoder": type(text_encoder).__name__,
        "num_pairs": audio_mat.shape[0],
    }

    if metadata_source is not None:
        metadata["metadata_source"] = metadata_source

    meta_cache_path.write_text(json.dumps(metadata, indent=2))

    return {
        "audio": audio_cache_path,
        "text": text_cache_path,
        "metadata": meta_cache_path,
    }


def _select_embedding_paths(paths: Mapping[str, Path]) -> list[Path]:
    return [paths[key] for key in ("audio", "text") if key in paths]


def _audio_cache_key(audio: object) -> Optional[str]:
    if isinstance(audio, (str, Path)):
        return str(Path(audio).resolve())
    return None


def _encode_audio_batch(
    audio_encoder: AudioEncoder,
    audio_inputs: Sequence[object],
    sample_rates: Sequence[Optional[int]],
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    batch_embeddings: list[Optional[torch.Tensor]] = [None] * len(audio_inputs)

    cacheable_positions: dict[str, list[int]] = {}
    pending_inputs: list[object] = []
    pending_rates: list[Optional[int]] = []
    pending_keys: list[str] = []

    non_cacheable_inputs: list[object] = []
    non_cacheable_rates: list[Optional[int]] = []
    non_cacheable_positions: list[int] = []

    seen_pending_keys: set[str] = set()

    for idx, (audio, sr) in enumerate(zip(audio_inputs, sample_rates)):
        key = _audio_cache_key(audio)
        if key is None:
            non_cacheable_positions.append(idx)
            non_cacheable_inputs.append(audio)
            non_cacheable_rates.append(sr)
            continue

        if key in cache:
            batch_embeddings[idx] = cache[key]
            continue

        cacheable_positions.setdefault(key, []).append(idx)
        if key not in seen_pending_keys:
            seen_pending_keys.add(key)
            pending_inputs.append(audio)
            pending_rates.append(sr)
            pending_keys.append(key)

    if pending_inputs:
        new_embeddings = audio_encoder.encode(
            pending_inputs, sample_rate=pending_rates  # pyright: ignore[reportArgumentType]
        )
        for key, embedding in zip(pending_keys, new_embeddings, strict=True):
            cache[key] = embedding
            for position in cacheable_positions[key]:
                batch_embeddings[position] = embedding

    if non_cacheable_inputs:
        new_embeddings = audio_encoder.encode(
            non_cacheable_inputs, sample_rate=non_cacheable_rates  # pyright: ignore[reportArgumentType]
        )
        for position, embedding in zip(non_cacheable_positions, new_embeddings, strict=True):
            batch_embeddings[position] = embedding

    if any(embedding is None for embedding in batch_embeddings):
        raise RuntimeError("Failed to compute audio embeddings for one or more samples")

    return torch.stack([embedding for embedding in batch_embeddings if embedding is not None])


def _require(sample: Sample, key: str) -> object:
    if key not in sample:
        raise KeyError(f"Required key '{key}' missing from sample {sample}")
    return sample[key]


def _load_samples(metadata_path: Path) -> list[Sample]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    suffix = metadata_path.suffix.lower()
    if suffix == ".jsonl":
        records: list[Sample] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:  # pragma: no cover - parse guard
                    raise ValueError(
                        f"Failed to parse JSONL line {line_number}: {exc}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise TypeError(
                        "Each record in metadata must be a mapping of features"
                    )
                records.append(record)
        return records

    if suffix == ".json":
        with metadata_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, SequenceABC):
            raise TypeError("JSON metadata must contain a list of samples")
        samples: list[Sample] = []
        for item in data:
            if not isinstance(item, Mapping):
                raise TypeError("Each sample entry must be a mapping")
            samples.append(item)
        return samples

    if suffix == ".csv":
        return _load_clotho_csv(metadata_path)

    raise ValueError(
        "Unsupported metadata format. Use a JSON/JSONL list or the Clotho CSV."
    )


def _load_clotho_csv(metadata_path: Path) -> list[Sample]:
    dataset_root = metadata_path.parent.parent
    split = _infer_split(metadata_path)
    audio_dir = dataset_root / "clotho_audio_files" / split

    if not audio_dir.exists():
        raise FileNotFoundError(
            f"Expected audio directory not found: {audio_dir}. Did you run 'make prepare'?"
        )

    samples: list[Sample] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        caption_columns = [col for col in reader.fieldnames or [] if col.startswith("caption_")]
        if "file_name" not in (reader.fieldnames or []):
            raise ValueError("Clotho CSV must include a 'file_name' column")
        if not caption_columns:
            raise ValueError("Clotho CSV must include caption columns")

        for row in reader:
            file_name = row["file_name"]
            audio_path = audio_dir / file_name
            for idx, column in enumerate(caption_columns, start=1):
                caption = row.get(column, "").strip()
                if not caption:
                    continue
                sample_id = f"{file_name}#{idx}"
                samples.append(
                    {
                        "id": sample_id,
                        "audio": str(audio_path),
                        "text": caption,
                    }
                )

    if not samples:
        raise ValueError(
            f"No samples were generated from {metadata_path}. Please verify the file."
        )

    return samples


def _infer_split(metadata_path: Path) -> str:
    name = metadata_path.stem.lower()
    for split in ("development", "validation", "evaluation"):
        if split in name:
            return split
    raise ValueError(
        "Unable to infer dataset split from metadata filename. Expected one of"
        " 'development', 'validation', or 'evaluation'."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache dataset embeddings")
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA),
        help="Path to dataset metadata (JSON/JSONL/Clotho CSV).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Directory where the embedding cache should be stored",
    )
    parser.add_argument(
        "--audio-arch",
        default="passt_s_swa_p16_128_ap476",
        help="PaSST architecture identifier",
    )
    parser.add_argument(
        "--text-model",
        default="roberta-base",
        help="Hugging Face model identifier for the text encoder",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of samples to encode per batch",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for model execution (e.g. 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--id-key",
        default="id",
        help="Key used to read sample identifiers",
    )
    parser.add_argument(
        "--audio-key",
        default="audio",
        help="Key containing the audio path or waveform",
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="Key containing the text description",
    )
    parser.add_argument(
        "--sample-rate-key",
        default="sample_rate",
        help="Key containing the sampling rate when passing waveforms",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate cache files even if they already exist",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar",
    )
    parser.add_argument(
        "--upload-every",
        type=int,
        default=None,
        help=(
            "Upload intermediate caches to Hugging Face Hub after this many "
            "pairs. Requires --hf-dataset."
        ),
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Hugging Face dataset repository to mirror cache files to",
    )
    parser.add_argument(
        "--hf-prefix",
        default="",
        help="Optional folder path inside the dataset repository",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token (falls back to HF_TOKEN env)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Path]:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    metadata_path = Path(args.metadata)
    cache_dir = Path(args.cache_dir)

    samples = _load_samples(metadata_path)

    audio_encoder = AudioEncoder(arch=args.audio_arch, device=args.device)
    text_encoder = TextEncoder(model_name=args.text_model, device=args.device)

    return cache_embeddings(
        samples,
        cache_dir=cache_dir,
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        id_key=args.id_key,
        audio_key=args.audio_key,
        text_key=args.text_key,
        sample_rate_key=args.sample_rate_key,
        batch_size=args.batch_size,
        force=args.force,
        show_progress=not args.no_progress,
        metadata_source=str(metadata_path),
        upload_every=args.upload_every,
        hf_dataset=args.hf_dataset,
        hf_prefix=args.hf_prefix,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    paths = main()
    for key, path in paths.items():
        print(f"{key}: {path}")
