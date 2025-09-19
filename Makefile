
DATA_ROOT ?= $(HOME)/data/CLOTHO_v2.1
METADATA ?= $(DATA_ROOT)/clotho_csv_files/clotho_captions_development.csv
CACHE_DIR ?= $(DATA_ROOT)/embeddings/development

.PHONY: test, prepare download-dataset download-embeddings cache-embeddings cache-images \
	train-projection eval-projection train-baseline train-vision

test:
	uv run python -m src.main

prepare:
	uv sync
	make download-embeddings

download-dataset:
	uv run python -m src.utils.download_dataset

download-embeddings:
	uv run python -m src.utils.download_embeddings --cache-dir "$(CACHE_DIR)" --print-stats


cache-embeddings:
	uv run python -m src.utils.embeddings --metadata "$(METADATA)" --cache-dir "$(CACHE_DIR)" $(CACHE_ARGS)

cache-images:
	uv run python -m src.utils.images --cache-dir "$(CACHE_DIR)" $(IMAGES_ARGS)

train-projection:
	uv run python -m src.train_projection --cache-dir "$(CACHE_DIR)" $(TRAIN_ARGS)

eval-projection:
	uv run python -m src.eval_projection --cache-dir "$(CACHE_DIR)" $(EVAL_ARGS)

train-baseline:
	uv run python -m src.train_finetune --metadata "$(METADATA)" --model-type baseline $(BASELINE_ARGS)

train-vision:
	uv run python -m src.train_finetune --metadata "$(METADATA)" --model-type vision $(VISION_ARGS)
