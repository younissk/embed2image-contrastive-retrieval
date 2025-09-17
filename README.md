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

## Mirror Caches to Hugging Face Hub

The embedding CLI can automatically push the raw embedding tensors to a
Hugging Face dataset repository so you can version them alongside your runs.

1. Create the dataset repo (replace `username` with your handle or org):

   ```bash
   huggingface-cli repo create username/audio-text-embed-to-images --type dataset
   ```

2. Authenticate on the machine running the cache job via `huggingface-cli login`
   or by exporting `HF_TOKEN` (already supported by the CLI).
3. Kick off caching with Hub uploads enabled:

   ```bash
   uv run python -m src.utils.embeddings \
       --metadata ~/data/CLOTHO_v2.1/clotho_csv_files/clotho_captions_development.csv \
       --cache-dir ~/data/CLOTHO_v2.1/embeddings/development \
       --hf-dataset username/audio-text-embed-to-images \
       --hf-prefix runs/dev-cache \
       --upload-every 100
   ```

With `--upload-every` the command saves the partial tensors locally and pushes
`audio_embeddings.pt` and `text_embeddings.pt` every *N* pairs. The final
iteration always uploads the freshest files. Drop `--upload-every` if you only
need the terminal snapshot. Pass `--hf-token` explicitly when you prefer not to
rely on the cached CLI credentials or environment variable.
