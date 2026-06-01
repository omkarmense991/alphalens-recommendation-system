# src/recommender/user_collaborative_filtering.py

# User-Based Collaborative Filtering
#
# Learn user-user similarity from historical interaction patterns.
# If two users interact with similar assets, they become similar users.
#
# For a target user:
#   - Find the most similar users.
#   - Look at assets they interacted with.
#   - Recommend unseen assets.
#
# recommendation_score += user_similarity × interaction_strength
#
# Recommendations are driven by collective user behavior,
# not by asset fundamentals or metadata.

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import PROCESSED_DATA_DIR, ASSETS_MASTER_PATH
from src.utils.logger import logger

USER_ITEM_MATRIX_PATH = PROCESSED_DATA_DIR / "user_item_matrix.csv"


class UserCollaborativeFilteringRecommender:
    def __init__(self):
        self.user_item_matrix = pd.read_csv(
            USER_ITEM_MATRIX_PATH,
            index_col="user_id",
        )

        self.assets_df = pd.read_csv(ASSETS_MASTER_PATH)

        # Measures user-user similarity based on interaction patterns.
        # If two users interact with similar assets, they get a higher similarity score.
        self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)

        self.user_ids = self.user_item_matrix.index.tolist()

        self.user_similarity_df = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.user_ids,
            columns=self.user_ids,
        )

    def recommend_for_user(
        self, user_id: str, top_k: int = 5, top_similar_users: int = 5
    ):
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found")

        target_user_interactions = self.user_item_matrix.loc[user_id]

        interacted_assets = target_user_interactions[
            target_user_interactions > 0
        ].index.tolist()

        similar_users = (
            self.user_similarity_df[user_id]
            .sort_values(ascending=False)
            .drop(labels=[user_id])
            .head(top_similar_users)
        )

        candidate_scores = {}

        for similar_user_id, similarity_score in similar_users.items():
            similar_user_interactions = self.user_item_matrix.loc[similar_user_id]

            for symbol, interaction_score in similar_user_interactions.items():
                if symbol in interacted_assets:
                    continue

                if interaction_score <= 0:
                    continue

                candidate_scores[symbol] = (
                    candidate_scores.get(symbol, 0)
                    + similarity_score * interaction_score
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
                    "user_cf_score": round(float(score), 4),
                }
            )

        return recommendations


if __name__ == "__main__":
    recommender = UserCollaborativeFilteringRecommender()

    results = recommender.recommend_for_user(
        user_id="user_1",
        top_k=5,
        top_similar_users=5,
    )

    for item in results:
        logger.info(item)
