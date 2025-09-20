
DATA_ROOT ?= $(HOME)/data/CLOTHO_v2.1
METADATA ?= $(DATA_ROOT)/clotho_csv_files/clotho_captions_development.csv
WANDB_PROJECT ?= embed2image

TRAIN_DEFAULT_ARGS ?= \
	--batch-size 12 \
	--accumulate-grad-batches 4 \
	--max-audio-seconds 12 \
	--precision bf16-mixed \
	--epochs 20 \
	--warmup-epochs 1.0 \
	--max-lr 3e-6 \
	--grad-clip-norm 1.0 \
	--num-workers 16 \
	--use-wandb \
	--wandb-project $(WANDB_PROJECT)

export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: test prepare download-dataset check-dataset train-baseline wandb ensure-uv

ensure-uv:
	@command -v uv >/dev/null 2>&1 || (echo "uv not found. Installing..." && curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --install-dir $$HOME/.local/bin --force)
	@command -v uv >/dev/null 2>&1 || (echo "uv still not found after installation" && exit 1)

test: ensure-uv
	uv run python -m src.main

prepare: ensure-uv
	uv sync
	make check-dataset

download-dataset: ensure-uv
	uv run python -m src.utils.download_dataset

check-dataset: ensure-uv
	@echo "Checking if Clotho dataset exists..."
	@uv run python -c "from src.utils.download_dataset import _audio_present; from pathlib import Path; import sys; exists = _audio_present(Path.home() / 'data', ['dev', 'val', 'eval']); print('Dataset found!' if exists else 'Dataset not found.'); sys.exit(0 if exists else 1)" || (echo "Dataset not found. Downloading..." && make download-dataset)

wandb: ensure-uv
	@uv run python -c "import wandb; import sys; sys.exit(0) if wandb.Api().api_key else sys.exit(1)" || (echo 'wandb not logged in. Please login:' && wandb login)

train-baseline: check-dataset ensure-uv
	@run_name=$${RUN_NAME:-baseline-$$(date +%Y%m%d-%H%M%S)}; \
	uv run python -m src.train_finetune --metadata "$(METADATA)" --run-name "$$run_name" $(TRAIN_DEFAULT_ARGS) $(TRAIN_ARGS)
