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

Vision-head configuration (pseudo-image + ViT) has its own shortcut:

```bash
make train-vision
```

By default `make train-baseline` forwards the following Lightning arguments:

```
--batch-size 8
--accumulate-grad-batches 4
--max-audio-seconds 10
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
if you want the run to land in a specific team/org. After training finishes the
newest checkpoint is automatically evaluated (see below).

Add or override any option via `TRAIN_ARGS`, e.g.

```bash
make train-baseline TRAIN_ARGS="--batch-size 8 --accumulate-grad-batches 4"
```

`make train-vision` behaves similarly but defaults to:

```
--batch-size 6
--accumulate-grad-batches 4
--max-audio-seconds 10
--precision bf16-mixed
--epochs 20
--warmup-epochs 1.0
--max-lr 3e-6
--grad-clip-norm 1.0
--num-workers 16
--projection-head vision
--vision-image-size 224
--vision-backbone vit_small_patch16_224
--vision-feature-pooling cls
```

Override with `TRAIN_ARGS` as needed, for example:

```bash
make train-vision TRAIN_ARGS="--vision-backbone vit_base_patch16_224 --vision-image-size 192"
```

### Useful arguments (`uv run python -m src.train_finetune --help`)

- `--batch-size` / `--accumulate-grad-batches`: use micro-batches that fit
  comfortably in memory and recover the desired effective batch with gradient
  accumulation. The defaults (`8 × 4`) give an effective batch of 32; adjust as
  needed for your GPU.
- `--max-audio-seconds`: truncate clips before PaSST to control memory/latency.
  Values around 10–15 s keep utilisation high without exhausting VRAM.
- `--audio-arch`, `--text-model`, `--projection-dim`, `--hidden-dim`: model
  customisation if you want to deviate from the baseline defaults.
- `--projection-head`: choose an alternative projection head (defaults to the
  baseline MLP and can be extended via the head registry).
- `--vision-*`: when using `--projection-head vision`, control the pseudo-image
  pipeline (e.g. `--vision-image-size`, `--vision-backbone`,
  `--vision-feature-pooling`, `--vision-pretrained`, `--vision-channel-mode`).
- `--precision`: set mixed precision (`bf16-mixed`, `16-mixed`, etc.) to leverage
  tensor cores.
- `--use-wandb` / `--wandb-project` / `--wandb-entity`: stream metrics to
  Weights & Biases and control where runs are stored.
- `--early-stop-patience` / `--early-stop-metric` / `--early-stop-mode`: control
  Lightning’s early-stopping callback (defaults: patience 10 epochs, monitor
  `val/mAP@10_text_to_audio`, mode `max`). Set patience ≤ 0 to disable.

Lightning handles checkpointing (best validation loss) and logs learning-rate
curves. Validation additionally reports retrieval metrics (`R@{1,5,10}` in both
directions) and `mAP@10`, mirroring the DCASE baseline. Checkpoints land under
`--output-dir` (default `checkpoints/`).

## Standalone Evaluation

After training, evaluate a checkpoint across the full validation set:

```bash
make evaluate-baseline CHECKPOINT=checkpoints/finetune-epoch12.ckpt \
  EVAL_ARGS="--batch-size 16"
```

If `CHECKPOINT` is omitted, the most recent `finetune-*.ckpt` in `checkpoints/`
is used. The command prints the same recall/mAP metrics logged during training.
Pass additional options through `EVAL_ARGS` (e.g. `--precision bf16-mixed`).

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
