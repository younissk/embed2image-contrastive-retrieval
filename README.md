# Embed2Image Contrastive Retrieval

## Setup

```bash
make prepare
```

This will do the following:

- Install the packages
- Create a virtual environment
- Download the dataset (Clotho)

Or, if you only want to download the dataset, you can run:

```bash
make download-dataset
```

## Cache Embeddings

After running `make prepare` the Clotho dataset lives in `~/data/CLOTHO_v2.1`.
You can pre-compute the audio and text embeddings for the development split with
a single command:

```bash
make cache-embeddings
```

This uses the development caption CSV from `clotho_csv_files` and writes the
resulting tensors to `~/data/CLOTHO_v2.1/embeddings/development`. You can
forward additional CLI flags through `CACHE_ARGS`, for example
`CACHE_ARGS="--batch-size 16"`.

To target a different split, override the make variables on the command line:

```bash
make cache-embeddings METADATA=~/data/CLOTHO_v2.1/clotho_csv_files/clotho_captions_validation.csv \
    CACHE_DIR=~/data/CLOTHO_v2.1/embeddings/validation
```

Under the hood this command reads the Clotho caption CSV, expands each of the
five captions per clip, and pairs them with the corresponding audio file from
`clotho_audio_files/<split>`. The same CLI still accepts JSON/JSONL metadata files
and exposes tuning options such as `--audio-arch`, `--text-model`, and
`--batch-size`; run `uv run python -m src.utils.embeddings --help` for details.
