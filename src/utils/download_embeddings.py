"""Download cached embeddings from the Hugging Face dataset snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "younissk/audio-text-embed-to-images"


def download_embeddings(
    target_dir: Path | str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    subset: Optional[str] = None,
    revision: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Materialise the cached embeddings for a dataset subset.

    Args:
        target_dir: Directory where the subset should end up (e.g. the value of
            ``CACHE_DIR`` in the Makefile).
        repo_id: Hugging Face dataset repository containing the cached files.
        subset: Optional dataset subset inside the repository. Defaults to the
            name of ``target_dir``.
        revision: Optional git revision (branch, tag or commit) of the dataset.
        force: Re-download the snapshot even when the expected files already
            exist locally.

    Returns:
        The path to the hydrated subset directory.
    """

    subset_dir = Path(target_dir).expanduser()
    root_dir = subset_dir.parent
    subset_name = subset or subset_dir.name

    metadata_path = subset_dir / "metadata.json"
    audio_path = subset_dir / "audio_embeddings.pt"
    text_path = subset_dir / "text_embeddings.pt"

    if not force and metadata_path.exists() and audio_path.exists() and text_path.exists():
        return subset_dir

    root_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(root_dir),
        allow_patterns=[f"{subset_name}/**"],
        ignore_patterns=None,
    )

    if not metadata_path.exists() or not audio_path.exists() or not text_path.exists():
        raise RuntimeError(
            "Expected embeddings files were not found after download. "
            f"Checked for: {metadata_path.name}, {audio_path.name}, {text_path.name}."
        )

    return subset_dir


def gather_stats(cache_dir: Path | str) -> dict[str, object]:
    """Produce quick statistics about the cached embeddings."""

    cache_path = Path(cache_dir).expanduser()
    stats: dict[str, object] = {}

    metadata_file = cache_path / "metadata.json"
    if metadata_file.exists():
        with metadata_file.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        ids = metadata.get("ids", [])
        stats["num_pairs"] = metadata.get("num_pairs", len(ids))
        stats["num_unique_items"] = len({item.split("#", 1)[0] for item in ids}) if ids else None
        stats["audio_embedding_shape"] = metadata.get("audio_embedding_shape")
        stats["text_embedding_shape"] = metadata.get("text_embedding_shape")
        stats["audio_encoder"] = metadata.get("audio_encoder")
        stats["text_encoder"] = metadata.get("text_encoder")
        stats["metadata_source"] = metadata.get("metadata_source")

    pseudo_dir = cache_path / "pseudo_images" / "audio"
    if pseudo_dir.exists():
        stats["num_pseudo_images"] = sum(1 for _ in pseudo_dir.glob("*.png"))

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Destination directory where embeddings should be stored",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repository ID",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional dataset subset (defaults to basename of cache dir)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision inside the dataset repository",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cache files already exist",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print dataset statistics after download",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    parser = build_parser()
    args = parser.parse_args(argv)

    cache_dir = download_embeddings(
        args.cache_dir,
        repo_id=args.repo_id,
        subset=args.subset,
        revision=args.revision,
        force=args.force,
    )

    if args.print_stats:
        stats = gather_stats(cache_dir)
        for key, value in stats.items():
            print(f"{key}: {value}")
        return stats

    return {}


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
