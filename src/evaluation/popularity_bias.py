# src/evaluation/popularity_bias.py

# Popularity Bias Metrics
#
# Measures whether the recommender disproportionately recommends
# highly popular assets. Popularity is defined as the number of
# users who interacted with an asset in the training data.
#
# Lower recommendation popularity generally indicates more
# novel and exploratory recommendations.


import pandas as pd


def compute_item_popularity(train_matrix: pd.DataFrame) -> dict[str, float]:
    item_popularity = {}

    for symbol in train_matrix.columns:
        popularity = (train_matrix[symbol] > 0).sum()
        item_popularity[symbol] = float(popularity)

    return item_popularity


def average_recommendation_popularity(
    recommended_items: list[str],
    item_popularity: dict[str, float],
) -> float:
    if not recommended_items:
        return 0.0

    popularity_scores = [
        item_popularity.get(symbol, 0.0) for symbol in recommended_items
    ]

    return sum(popularity_scores) / len(popularity_scores)
