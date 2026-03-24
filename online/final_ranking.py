from __future__ import annotations

import math
from typing import Iterable


def _softmax(values: list[float]) -> list[float]:
    """Numerically stable softmax."""
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def rank_candidates(
    candidates: Iterable[tuple[str, float]],
    occasion_scores: dict[str, float],
    alpha: float = 0.6,
    beta: float = 0.4,
) -> list[tuple[str, float, float, float]]:
    pairs = [(str(row_id), float(rel)) for row_id, rel in candidates]
    if not pairs:
        return []

    ids = [row_id for row_id, _ in pairs]
    rel_raw = [rel for _, rel in pairs]

    # Softmax on relevance so it's on the same scale as occasion scores
    rel_norm = _softmax(rel_raw)

    ranked: list[tuple[str, float, float, float]] = []
    for row_id, rel in zip(ids, rel_norm):
        occ_score = occasion_scores.get(row_id, 0.0)
        final_score = alpha * rel + beta * occ_score
        ranked.append((row_id, rel, occ_score, final_score))

    ranked.sort(key=lambda item: item[3], reverse=True)
    return ranked
