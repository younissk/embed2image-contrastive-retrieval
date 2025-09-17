
DATA_ROOT ?= $(HOME)/data/CLOTHO_v2.1
METADATA ?= $(DATA_ROOT)/clotho_csv_files/clotho_captions_development.csv
CACHE_DIR ?= $(DATA_ROOT)/embeddings/development

.PHONY: test, prepare download-dataset cache-embeddings cache-images

test:
	uv run python -m src.main

prepare:
	uv sync
	make download-dataset

download-dataset:
	uv run python -m src.utils.download_dataset

cache-embeddings:
	uv run python -m src.utils.embeddings --metadata "$(METADATA)" --cache-dir "$(CACHE_DIR)" $(CACHE_ARGS)

cache-images:
	uv run python -m src.utils.images --cache-dir "$(CACHE_DIR)" $(IMAGES_ARGS)
