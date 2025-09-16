

.PHONY: test, prepare download-dataset

test:
	uv run python -m src.main

prepare:
	uv sync

download-dataset:
	uv run python -m src.utils.download_dataset