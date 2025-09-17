"""Generate pseudo-images from cached embeddings and upload to Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from tqdm import tqdm

from .hf_uploader import HFUploadError, HFDatasetUploader
from .pseudo_images import EmbeddingImageExporter


def generate_images(
    cache_dir: Path | str,
    *,
    hf_dataset: Optional[str] = None,
    hf_prefix: str = "",
    hf_token: Optional[str] = None,
    image_hw: int = 128,
    image_mode: str = "nearest",
    image_channel_mode: str = "split",
    clip_values: bool = False,
    output_dirname: str = "pseudo_images",
    upload_every: Optional[int] = None,
    include_embeddings: bool = False,
) -> dict[str, Path | None]:
    """Create pseudo-images from cached embeddings and optionally upload them.

    Args:
        cache_dir: Directory containing cached ``audio_embeddings.pt`` and
            ``text_embeddings.pt`` files.
        hf_dataset: Optional Hugging Face dataset repo to upload generated
            images (and optionally embeddings) to.
        hf_prefix: Optional subdirectory inside the dataset repo.
        hf_token: Optional Hugging Face token (falls back to ``HF_TOKEN`` env var).
        image_hw: Output resolution for pseudo-images.
        image_mode: Interpolation mode used during upsampling.
        image_channel_mode: Channel folding strategy for :class:`Embed2Image`.
        clip_values: Whether to clamp embeddings to ``[-1, 1]`` before
            normalisation.
        output_dirname: Folder name (under ``cache_dir``) where PNGs are stored.
        upload_every: When set, mirror artifacts to the Hub every ``N`` pairs.
        include_embeddings: If ``True`` also upload the cached `.pt` files and
            metadata during the next commit.

    Returns:
        Mapping describing key artifact paths that were processed.
    """

    cache_path = Path(cache_dir)
    audio_path = cache_path / "audio_embeddings.pt"
    text_path = cache_path / "text_embeddings.pt"
    metadata_path = cache_path / "metadata.json"

    if not audio_path.exists() or not text_path.exists():
        raise FileNotFoundError(
            f"Expected embeddings not found in {cache_path}. Run cache_embeddings first."
        )

    metadata_ids = _load_metadata_ids(metadata_path)

    audio_embeddings = torch.load(audio_path, map_location="cpu")
    text_embeddings = torch.load(text_path, map_location="cpu")

    if audio_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Expected 2D tensors for embeddings")
    if audio_embeddings.shape[0] != text_embeddings.shape[0]:
        raise ValueError("Audio and text embedding counts do not match")

    if metadata_ids is not None and len(metadata_ids) != audio_embeddings.shape[0]:
        raise ValueError("Metadata IDs count does not match audio embedding count")

    ids = metadata_ids or [str(idx) for idx in range(audio_embeddings.shape[0])]

    exporter = EmbeddingImageExporter(
        root=cache_path / output_dirname,
        image_size=image_hw,
        mode=image_mode,
        channel_mode=image_channel_mode,
        clip_values=clip_values,
    )

    uploader: HFDatasetUploader | None = None
    pending_paths: list[Path] = []
    last_upload_count = 0
    processed_pairs = 0

    if include_embeddings:
        pending_paths.extend([audio_path, text_path])
        if metadata_path.exists():
            pending_paths.append(metadata_path)

    if hf_dataset:
        try:
            uploader = HFDatasetUploader(
                repo_id=hf_dataset,
                prefix=hf_prefix,
                token=hf_token,
            )
        except HFUploadError:
            raise
        except Exception as exc:  # pragma: no cover
            raise HFUploadError(
                f"Failed to initialise Hugging Face uploader for repo '{hf_dataset}': {exc}"
            ) from exc

    def maybe_upload(processed: int, final: bool = False) -> None:
        nonlocal pending_paths, last_upload_count
        if uploader is None:
            return
        if upload_every is None and not final:
            return
        if upload_every is not None and not final and processed - last_upload_count < upload_every:
            return
        if not pending_paths:
            return
        commit_message = (
            f"Upload pseudo-images ({processed} items processed)"
            if not final
            else f"Upload final pseudo-images ({processed} items)"
        )
        uploader.upload_files(pending_paths, commit_message=commit_message)
        pending_paths.clear()
        last_upload_count = processed

    progress = tqdm(range(len(ids)), desc="Exporting pseudo-images", unit="pair")
    for index in progress:
        sample_id = ids[index]
        pending_paths.extend(
            exporter.export_pair(audio_embeddings[index], text_embeddings[index], sample_id)
        )
        processed_pairs += 1
        maybe_upload(processed_pairs, final=False)

    maybe_upload(processed_pairs, final=True)

    return {
        "audio_embeddings": audio_path,
        "text_embeddings": text_path,
        "images_dir": exporter.root,
        "metadata": metadata_path if metadata_path.exists() else None,
    }


def _load_metadata_ids(metadata_path: Path) -> Optional[list[str]]:
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse metadata JSON: {exc}") from exc
    except FileNotFoundError:
        return None

    if isinstance(data, Mapping):
        ids = data.get("ids")
    else:
        ids = None

    if ids is None:
        return None
    if not isinstance(ids, list):
        raise ValueError("Metadata 'ids' field must be a list")
    return [str(item) for item in ids]
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate pseudo-images from cached embeddings")
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory containing audio/text embeddings",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Hugging Face dataset repository to upload images to",
    )
    parser.add_argument(
        "--hf-prefix",
        default="",
        help="Prefix inside the dataset repository",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token (falls back to HF_TOKEN env)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Output pseudo-image resolution",
    )
    parser.add_argument(
        "--image-mode",
        default="nearest",
        help="Interpolation mode for upsampling",
    )
    parser.add_argument(
        "--image-channel-mode",
        default="split",
        choices=["split", "replicate"],
        help="Channel folding mode",
    )
    parser.add_argument(
        "--images-dir",
        default="pseudo_images",
        help="Subdirectory for generated images",
    )
    parser.add_argument(
        "--upload-every",
        type=int,
        default=None,
        help="Upload images to HF after this many processed pairs (requires --hf-dataset)",
    )
    parser.add_argument(
        "--clip-values",
        action="store_true",
        help="Clip embeddings to [-1, 1] before normalising",
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Upload cached embedding tensors/metadata alongside images",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Path]:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    return generate_images(
        cache_dir=args.cache_dir,
        hf_dataset=args.hf_dataset,
        hf_prefix=args.hf_prefix,
        hf_token=args.hf_token,
        image_hw=args.image_size,
        image_mode=args.image_mode,
        image_channel_mode=args.image_channel_mode,
        clip_values=args.clip_values,
        output_dirname=args.images_dir,
        upload_every=args.upload_every,
        include_embeddings=args.include_embeddings,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    result = main()
    for key, value in result.items():
        if value is None:
            continue
        print(f"{key}: {value}")
