
DATA_ROOT ?= $(HOME)/data/CLOTHO_v2.1
METADATA ?= $(DATA_ROOT)/clotho_csv_files/clotho_captions_development.csv

.PHONY: test prepare download-dataset train

test:
	uv run python -m src.main

prepare:
	uv sync
	make download-dataset

download-dataset:
	uv run python -m src.utils.download_dataset

train:
	uv run python -m src.train_finetune --metadata "$(METADATA)" $(TRAIN_ARGS)
