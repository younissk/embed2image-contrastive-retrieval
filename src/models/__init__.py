"""Model components for embed2image baseline."""

from .baseline import RetrievalModule
from .projection import ProjectionHead, HEAD_REGISTRY, resolve_projection_head

__all__ = [
    "RetrievalModule",
    "ProjectionHead",
    "HEAD_REGISTRY",
    "resolve_projection_head",
]
