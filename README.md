# Embed2Image Contrastive Retrieval

## Setup

```bash
make prepare
```

This installs dependencies with `uv`, sets up the virtual environment, and
ensures the Clotho v2.1 audio/caption files are downloaded into
`~/data/CLOTHO_v2.1`. You can re-run `make download-dataset` at any time to
refresh the archives.

## Lightning Training (PaSST + RoBERTa)

Training is driven by a PyTorch Lightning module that mirrors the DCASE Task 6
baseline (PaSST audio encoder + RoBERTa text encoder with learned projection
heads). Launch a run and forward any CLI options through `TRAIN_ARGS`:

```bash
make train-baseline
```

By default `make train-baseline` forwards the following Lightning arguments:

```
--batch-size 24
--accumulate-grad-batches 2
--max-audio-seconds 16
--precision bf16-mixed
--epochs 20
--warmup-epochs 1.0
--max-lr 3e-6
--grad-clip-norm 1.0
--num-workers 16
```

Each invocation also sets a default run name `baseline-<timestamp>`; override it
with `RUN_NAME=my-run make train-baseline ...` if you prefer. W&B logging is
enabled automatically and defaults to the `embed2image` project; change it via
`WANDB_PROJECT` (or `--wandb-project` in `TRAIN_ARGS`). Supply `WANDB_ENTITY` /`--wandb-entity`
if you want the run to land in a specific team/org.

Add or override any option via `TRAIN_ARGS`, e.g.

```bash
make train-baseline TRAIN_ARGS="--batch-size 8 --accumulate-grad-batches 4"
```

### Useful arguments (`uv run python -m src.train_finetune --help`)

- `--batch-size` / `--accumulate-grad-batches`: use micro-batches that fit
  comfortably in memory and recover the desired effective batch with gradient
  accumulation. The defaults (`24 × 2`) give an effective batch of 48; adjust as
  needed for your GPU.
- `--max-audio-seconds`: truncate clips before PaSST to control memory/latency.
  Values around 10–15 s keep utilisation high without exhausting VRAM.
- `--audio-arch`, `--text-model`, `--projection-dim`, `--hidden-dim`: model
  customisation if you want to deviate from the baseline defaults.
- `--precision`: set mixed precision (`bf16-mixed`, `16-mixed`, etc.) to leverage
  tensor cores.
- `--use-wandb` / `--wandb-project` / `--wandb-entity`: stream metrics to
  Weights & Biases and control where runs are stored.

Lightning handles checkpointing (best validation loss) and logs learning-rate
curves; checkpoints land under `--output-dir` (default `checkpoints/`).

## Weights & Biases

Authenticate once per machine:

```bash
uv run wandb login
```

Then enable logging on any run:

```bash
WANDB_PROJECT=embed2image make train-baseline \
  TRAIN_ARGS="--use-wandb --wandb-entity your-team --run-name h100-baseline \
              --batch-size 4 --accumulate-grad-batches 8"
```

If W&B is disabled the run still logs locally via PyTorch Lightning.
