# Embed2Image Contrastive Retrieval

## Setup

```bash
make prepare
```

This will do the following:

- Install the packages
- Create a virtual environment
- Download the cached Clotho development embeddings from Hugging Face

Or, if you only want the raw Clotho archives rather than the cached embeddings, run:

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
the `audio_embeddings.pt` / `text_embeddings.pt` files every *N* pairs. Drop
`--upload-every` if you only need the terminal snapshot. Pass `--hf-token`
explicitly when you prefer not to rely on the cached CLI credentials or
environment variable. Add `--with-images` if you want the same run to also
generate pseudo-images (see below).

### Pseudo-Image Export

Run pseudo-image generation separately once embeddings are cached:

```bash
uv run python -m src.utils.images --cache-dir ~/data/CLOTHO_v2.1/embeddings/development \
    --hf-dataset username/audio-text-embed-to-images --hf-prefix runs/dev-images \
    --upload-every 100
```

Or via the Makefile helper (propagate flags with `IMAGES_ARGS`):

```bash
make cache-images CACHE_DIR=~/data/CLOTHO_v2.1/embeddings/development \
    IMAGES_ARGS="--hf-dataset username/audio-text-embed-to-images --upload-every 100 --include-embeddings"
```

The image command reshapes each embedding into an RGB pseudo-image and stores it under
`pseudo_images/audio` and `pseudo_images/text` within the cache directory. These
PNGs are mirrored to the Hub alongside the tensors. Tune the behaviour with:

- `--image-size` – spatial resolution of the generated images (default 128)
- `--image-mode` – interpolation strategy used during resizing (default
  `nearest`)
- `--image-channel-mode` – fold embeddings by splitting into RGB channels or
  replicating a single channel (`split`/`replicate`)
- `--images-dir` – custom folder name inside the cache directory
- `--clip-values` / `--clip-images` – clamp embeddings to `[-1, 1]` before
  normalising (flag name differs between the standalone and combined CLIs)
- `--include-embeddings` – push the `.pt` tensors/metadata in the same commit

The same flags are honoured by `src.utils.embeddings` whenever `--with-images`
is supplied.

## Train Projection Heads

Once the cached embeddings are available you can learn lightweight projection
heads with a contrastive objective similar to the DCASE Task 6 baseline:

```bash
make train-projection TRAIN_ARGS="--epochs 5 --batch-size 512 --out-dim 128"
```

The helper wraps `uv run python -m src.train_projection` and saves the learned
audio/text projection weights under `checkpoints/projection_heads.pt` by
default. Use `TRAIN_ARGS` to pass any CLI flag defined in
`src/train_projection.py` (e.g. `--cache-dir`, `--output-dir`, `--temperature`,
`--device`). Validation metrics (symmetric top-1 accuracies) are logged at the
end of each epoch; the checkpoint reflects the best validation loss observed.
Every training run also writes a JSONL log to `logs/train/` capturing the epoch
history and the chosen hyper-parameters for later inspection.

Vision-style pseudo-image heads can be trained by switching the input type:

```bash
make train-projection TRAIN_ARGS="--input-type vision --vision-image-source generated --epochs 10"
```

This enables a small CNN (`VisionProjectionHead`) that learns from the
embedding-derived pseudo-images created on the fly (or loaded from
`pseudo_images/audio` when available). The text branch still uses the MLP
projection head so you can contrast image features against text embeddings.

Evaluate the resulting heads with retrieval recalls (R@1/5/10) via:

```bash
make eval-projection EVAL_ARGS="--checkpoint checkpoints/projection_heads.pt"
```

Under the hood this runs `src.eval_projection`, which projects all cached
embeddings, computes symmetric recalls, and prints the scores. Override
`--cache-dir`, `--device`, or the batch sizes (`--batch-size`,
`--sim-batch-size`) as needed. Evaluation runs also append their results to
`logs/eval/` so you have an audit trail of recall numbers.

## Fine-Tune Encoders (Baseline-style)

When you need to update the PaSST/Roberta backbones themselves, use the
fine-tuning entry points. They expect the raw Clotho audio to be available
under `$(DATA_ROOT)/clotho_audio_files/<split>` (which happens automatically
after `make download-dataset`).

Baseline-style audio/text projection (MLP heads on top of PaSST + RoBERTa):

```bash
make train-baseline BASELINE_ARGS="--epochs 20 --batch-size 32 --max-lr 3e-6"
```

Vision variant that inserts the pseudo-image CNN head between the audio encoder
and the contrastive projection:

```bash
make train-vision VISION_ARGS="--epochs 20 --batch-size 32 --vision-image-size 128"
```

Both commands invoke `src.train_finetune`, log their progress to
`logs/finetune/`, and save checkpoints as `checkpoints/finetune_<mode>.pt`. Tune
learning-rate schedule, warm-up, or projection dimensions through the
`BASELINE_ARGS`/`VISION_ARGS` override strings.
