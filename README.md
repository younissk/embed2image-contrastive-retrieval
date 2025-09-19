# Embed2Image Contrastive Retrieval

## Setup

```bash
make prepare
```

This installs the project dependencies via `uv`, creates the virtual
environment, and downloads the Clotho v2.1 audio/caption files into
`~/data/CLOTHO_v2.1`.

If you ever need to refresh the raw dataset without re-installing, run:

```bash
make download-dataset
```

## Fine-tuning (PaSST + RoBERTa)

All training now happens end-to-end on the original audio and captions, closely
mirroring the DCASE Task 6 baseline. Launch a run with:

```bash
make train TRAIN_ARGS="--epochs 20 --batch-size 32 --max-lr 3e-6"
```

Useful flags (`uv run python -m src.train_finetune --help` for the full list):

- `--audio-arch`: PaSST backbone (default `passt_s_swa_p16_128_ap476`)
- `--text-model`: Hugging Face text encoder (`roberta-base` by default)
- `--projection-dim` / `--hidden-dim`: projection head sizes
- `--val-fraction`: held-out portion of Clotho captions (default `0.1`)
- `--use-wandb` and `--wandb-project`: stream metrics to Weights & Biases

Checkpoints are written to `checkpoints/finetune_baseline.pt` and every training
run also drops a JSONL log in `logs/finetune/` for reproducibility.

### Weights & Biases

Authenticate once per machine:

```bash
uv run wandb login
```

Then pass the flag (and optionally a project name) to any training run:

```bash
WANDB_PROJECT=embed2image make train \
  TRAIN_ARGS="--use-wandb --run-name a10-baseline --epochs 20"
```

The W&B run mirrors the JSONL log, so you can inspect metrics either locally or
in the dashboard.
