"""Loss and retrieval metrics used by the baseline."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

__all__ = [
    "contrastive_loss",
    "compute_metrics",
    "compute_global_metrics",
]


def contrastive_loss(anchor: torch.Tensor, text: torch.Tensor, temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = anchor @ text.t() / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a + loss_t), logits


def compute_metrics(logits: torch.Tensor) -> dict[str, float]:
    targets = torch.arange(logits.size(0), device=logits.device)
    acc_anchor = (logits.argmax(dim=1) == targets).float().mean().item()
    acc_text = (logits.argmax(dim=0) == targets).float().mean().item()
    return {
        "acc_audio_to_text": acc_anchor,
        "acc_text_to_audio": acc_text,
    }


def compute_global_metrics(
    audio_proj: torch.Tensor,
    text_proj: torch.Tensor,
    ids: List[str],
) -> dict[str, float]:
    audio_norm = F.normalize(audio_proj, dim=-1)
    text_norm = F.normalize(text_proj, dim=-1)

    canonical_ids = _canonical(ids)

    metrics = {}
    metrics.update(
        _retrieval_metrics(
            query_embeddings=text_norm,
            candidate_embeddings=audio_norm,
            query_ids=canonical_ids,
            candidate_ids=canonical_ids,
            prefix="text_to_audio",
        )
    )

    unique_audio_idx, unique_audio_ids = _unique_audio_indices(canonical_ids)
    metrics.update(
        _retrieval_metrics(
            query_embeddings=audio_norm[unique_audio_idx],
            candidate_embeddings=text_norm,
            query_ids=unique_audio_ids,
            candidate_ids=canonical_ids,
            prefix="audio_to_text",
        )
    )
    return metrics


def _retrieval_metrics(
    *,
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    query_ids: List[str],
    candidate_ids: List[str],
    prefix: str,
) -> dict[str, float]:
    sims = query_embeddings @ candidate_embeddings.t()
    metrics = {}
    metrics[f"R@1_{prefix}"] = _recall_at_k(sims, query_ids, candidate_ids, k=1)
    metrics[f"R@5_{prefix}"] = _recall_at_k(sims, query_ids, candidate_ids, k=5)
    metrics[f"R@10_{prefix}"] = _recall_at_k(sims, query_ids, candidate_ids, k=10)
    metrics[f"mAP@10_{prefix}"] = _mean_average_precision(sims, query_ids, candidate_ids, k=10)
    return metrics


def _canonical(values: List[str]) -> List[str]:
    return [value.split("#", 1)[0] for value in values]


def _unique_audio_indices(ids: List[str]) -> tuple[List[int], List[str]]:
    seen = set()
    indices: List[int] = []
    uniq_ids: List[str] = []
    for idx, cid in enumerate(ids):
        if cid not in seen:
            seen.add(cid)
            indices.append(idx)
            uniq_ids.append(cid)
    return indices, uniq_ids


def _recall_at_k(
    similarities: torch.Tensor,
    query_ids: List[str],
    candidate_ids: List[str],
    *,
    k: int,
) -> float:
    candidate_canon = _canonical(candidate_ids)
    topk = similarities.topk(min(k, similarities.size(1)), dim=1).indices
    hits = 0
    for q_idx, row in enumerate(topk.tolist()):
        query_id = query_ids[q_idx]
        if any(candidate_canon[idx] == query_id for idx in row):
            hits += 1
    return hits / len(query_ids) if query_ids else 0.0


def _mean_average_precision(
    similarities: torch.Tensor,
    query_ids: List[str],
    candidate_ids: List[str],
    *,
    k: int,
) -> float:
    candidate_canon = _canonical(candidate_ids)
    topk = similarities.topk(min(k, similarities.size(1)), dim=1).indices
    ap_values = []
    for q_idx, row in enumerate(topk.tolist()):
        query_id = query_ids[q_idx]
        hits = 0
        precision_sum = 0.0
        for rank, candidate_idx in enumerate(row, start=1):
            if candidate_canon[candidate_idx] == query_id:
                hits += 1
                precision_sum += hits / rank
        if hits:
            ap_values.append(precision_sum / hits)
    return float(sum(ap_values) / len(ap_values)) if ap_values else 0.0
