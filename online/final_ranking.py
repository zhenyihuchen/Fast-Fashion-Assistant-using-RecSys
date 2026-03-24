from __future__ import annotations

from typing import Iterable


def rank_candidates(
    candidates: Iterable[tuple[str, float]],
    occasion_scores: dict[str, float],
    alpha: float = 0.6,
    beta: float = 0.4,
) -> list[tuple[str, float, float, float]]:
    pairs = [(str(row_id), float(rel)) for row_id, rel in candidates]
    if not pairs:
        return []

    ranked: list[tuple[str, float, float, float]] = []
    for row_id, rel in pairs:
        occ_score = occasion_scores.get(row_id, 0.0)
        final_score = alpha * rel + beta * occ_score
        ranked.append((row_id, rel, occ_score, final_score))

    ranked.sort(key=lambda item: item[3], reverse=True)
    return ranked
