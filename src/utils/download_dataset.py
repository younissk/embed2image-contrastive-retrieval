"""Utility for ensuring the Clotho dataset is downloaded and extracted."""

from pathlib import Path

from typing import Iterable

from aac_datasets.datasets.functional.clotho import download_clotho_datasets

SUBSETS: list[str] = ["dev", "val", "eval"]
SPLIT_DIRS = {
    "dev": "development",
    "val": "validation",
    "eval": "evaluation",
}


def _audio_present(root: Path, subsets: Iterable[str]) -> bool:
    audio_root = root / "CLOTHO_v2.1" / "clotho_audio_files"
    for subset in subsets:
        split_dir = SPLIT_DIRS.get(subset, subset)
        subset_dir = audio_root / split_dir
        if not subset_dir.exists():
            return False
        try:
            next(subset_dir.glob("*.wav"))
        except StopIteration:
            return False
    return True


def download_clotho(data_path: str):
    root = Path(data_path).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    download_clotho_datasets(
        subsets=SUBSETS,
        root=str(root),
        clean_archives=True,
        force=not _audio_present(root, SUBSETS),
        verbose=5,
    )


if __name__ == "__main__":

    print("Downloading Clotho dataset in ~/data...")

    data_root = Path.home() / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    download_clotho(str(data_root))

    print("Done!")
