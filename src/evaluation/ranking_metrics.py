# src/evaluation/ranking_metrics.py

import math


def precision_at_k(
    recommended_items: list[str], relevant_items: set[str], k: int
) -> float:
    recommended_at_k = recommended_items[:k]

    if not recommended_at_k:
        return 0.0

    hits = sum(1 for item in recommended_at_k if item in relevant_items)

    return hits / len(recommended_at_k)


def recall_at_k(
    recommended_items: list[str], relevant_items: set[str], k: int
) -> float:
    if not relevant_items:
        return 0.0

    recommended_at_k = recommended_items[:k]

    hits = sum(1 for item in recommended_at_k if item in relevant_items)

    return hits / len(relevant_items)


def hit_rate_at_k(
    recommended_items: list[str], relevant_items: set[str], k: int
) -> float:
    recommended_at_k = recommended_items[:k]

    for item in recommended_at_k:
        if item in relevant_items:
            return 1.0

    return 0.0


def ndcg_at_k(recommended_items: list[str], relevant_items: set[str], k: int) -> float:
    recommended_at_k = recommended_items[:k]

    dcg = 0.0

    for index, item in enumerate(recommended_at_k):
        if item in relevant_items:
            rank = index + 1
            dcg += 1 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)

    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg
