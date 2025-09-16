"""
Not my code. Courtesy of https://github.com/CPJKU/dcase2025_task6_baseline
"""

import os

from aac_datasets.datasets.functional.clotho import download_clotho_datasets


def download_clotho(data_path: str):

    download_clotho_datasets(
        subsets=["dev", "val", "eval"],
        root=data_path,
        clean_archives=False,
        verbose=5
    )


if __name__ == "__main__":

    print("Downloading Clotho dataset in ~/data...")

    if not os.path.exists(os.path.expanduser("~/data")):
        os.makedirs(os.path.expanduser("~/data"))

    download_clotho(os.path.expanduser("~/data"))

    print("Done!")
