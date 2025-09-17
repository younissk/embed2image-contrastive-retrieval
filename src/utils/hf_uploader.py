"""Utilities for uploading cache artifacts to Hugging Face datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


class HFUploadError(RuntimeError):
    """Raised when uploading to Hugging Face Hub fails."""


@dataclass(slots=True)
class HFDatasetUploader:
    """Uploads files to a Hugging Face dataset repository."""

    repo_id: str
    prefix: str = ""
    token: str | None = None
    create_if_missing: bool = True

    def __post_init__(self) -> None:
        hub = _load_hub()
        self._token = self.token or os.getenv("HF_TOKEN") or hub.HfFolder.get_token()
        if not self._token:
            raise HFUploadError(
                "HF token not provided. Set HF_TOKEN env var or pass --hf-token."
            )

        self._api = hub.HfApi()
        if self.create_if_missing:
            try:
                self._api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self._token,
                    exist_ok=True,
                )
            except Exception as exc:  # pragma: no cover - network interaction
                raise HFUploadError(
                    f"Failed to ensure dataset repo '{self.repo_id}' exists: {exc}"
                ) from exc

        self._prefix = self.prefix.strip("/")

    def upload_files(
        self,
        files: Mapping[str, Path] | Iterable[Path],
        *,
        commit_message: str,
    ) -> list[str]:
        """Upload files to the dataset and return their hub URIs."""

        hub = _load_hub()
        operations: list[hub.CommitOperationAdd] = []
        uris: list[str] = []

        if isinstance(files, Mapping):
            iterable = files.values()
        else:
            iterable = files

        for path in iterable:
            repo_path = path.name if not self._prefix else f"{self._prefix}/{path.name}"
            operations.append(
                hub.CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=path)
            )
            uris.append(f"hf://datasets/{self.repo_id}/{repo_path}")

        if not operations:
            return uris

        try:
            self._api.create_commit(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self._token,
                operations=operations,
                commit_message=commit_message,
            )
        except Exception as exc:  # pragma: no cover - network interaction
            raise HFUploadError(
                f"Failed to upload files to dataset '{self.repo_id}': {exc}"
            ) from exc

        return uris


def _load_hub():
    try:
        import huggingface_hub
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise HFUploadError(
            "huggingface-hub is required for hub uploads. Install it via 'uv sync'."
        ) from exc
    return huggingface_hub

