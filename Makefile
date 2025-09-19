
DATA_ROOT ?= $(HOME)/data/CLOTHO_v2.1
METADATA ?= $(DATA_ROOT)/clotho_csv_files/clotho_captions_development.csv

.PHONY: test prepare download-dataset check-dataset train wandb ensure-uv

ensure-uv:
	@command -v uv >/dev/null 2>&1 || (echo "uv not found. Installing..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
	@command -v uv >/dev/null 2>&1 || (echo "uv still not found after installation" && exit 1)
	@if [ ! -f /usr/local/bin/uv ] && [ -f $$HOME/.local/bin/uv ]; then \
		echo "Creating symlink to /usr/local/bin for system-wide access..."; \
		sudo ln -sf $$HOME/.local/bin/uv /usr/local/bin/uv; \
		sudo ln -sf $$HOME/.local/bin/uvx /usr/local/bin/uvx; \
	fi

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

train: check-dataset ensure-uv
	uv run python -m src.train_finetune --metadata "$(METADATA)" $(TRAIN_ARGS)
