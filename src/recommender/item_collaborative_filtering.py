# src/recommender/item_collaborative_filtering.py

# Item-Based Collaborative Filtering
#
# Learn asset-to-asset similarity from the behavior of all users.
# If many users interact with both RELIANCE and TCS, they become similar.
# similar because users behave similarly around those assets
#
# For a target user:
#   - Find assets already interacted with.
#   - Find similar assets.
#   - Rank unseen assets using:
#
#       interaction_strength × similarity_score
#
# Recommendations are driven by collective user behavior rather than
# asset fundamentals or metadata.

# output will say something like: Users who interacted with user_1's interacted assets also commonly interacted with ITC.

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import PROCESSED_DATA_DIR, ASSETS_MASTER_PATH
from src.utils.logger import logger

USER_ITEM_MATRIX_PATH = PROCESSED_DATA_DIR / "user_item_matrix.csv"


class ItemCollaborativeFilteringRecommender:
    def __init__(self, matrix_path=USER_ITEM_MATRIX_PATH):
        self.user_item_matrix = pd.read_csv(
            matrix_path,
            index_col="user_id",
        )

        self.assets_df = pd.read_csv(ASSETS_MASTER_PATH)

        self.item_user_matrix = self.user_item_matrix.T

        # Each asset is represented as a vector of user interactions.
        # Cosine similarity measures how similarly users interact with pairs of assets.
        # Example: if many users interact with both RELIANCE and TCS, their similarity will be high.
        self.item_similarity_matrix = cosine_similarity(self.item_user_matrix)

        self.symbols = self.item_user_matrix.index.tolist()

        self.item_similarity_df = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.symbols,
            columns=self.symbols,
        )

    def recommend_for_user(self, user_id: str, top_k: int = 5):
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found")

        user_interactions = self.user_item_matrix.loc[user_id]

        interacted_assets = user_interactions[user_interactions > 0].index.tolist()

        candidate_scores = {}

        for asset in interacted_assets:
            asset_score = user_interactions[asset]

            similar_items = self.item_similarity_df[asset].sort_values(ascending=False)

            for candidate_asset, similarity_score in similar_items.items():
                if candidate_asset in interacted_assets:
                    continue

                candidate_scores[candidate_asset] = (
                    candidate_scores.get(candidate_asset, 0)
                    + asset_score * similarity_score
                )

        ranked_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        recommendations = []

        for symbol, score in ranked_candidates[:top_k]:
            asset_info = self.assets_df[self.assets_df["symbol"] == symbol].iloc[0]

            recommendations.append(
                {
                    "symbol": symbol,
                    "company_name": asset_info["company_name"],
                    "sector": asset_info["sector"],
                    "industry": asset_info["industry"],
                    "collaborative_score": round(float(score), 4),
                }
            )

        return recommendations


if __name__ == "__main__":
    recommender = ItemCollaborativeFilteringRecommender()

    results = recommender.recommend_for_user(
        user_id="user_1",
        top_k=5,
    )

    for item in results:
        logger.info(item)
