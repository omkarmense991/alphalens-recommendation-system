# # src/candidates/candidate_generator.py

import pandas as pd
from src.config.settings import PROCESSED_DATA_DIR

from src.evaluation.popularity_bias import compute_item_popularity
from src.recommender.item_collaborative_filtering import (
    ItemCollaborativeFilteringRecommender,
)
from src.recommender.user_collaborative_filtering import (
    UserCollaborativeFilteringRecommender,
)


class CandidateGenerator:
    def __init__(self, matrix_path=None):
        if matrix_path is None:
            self.item_cf = ItemCollaborativeFilteringRecommender()
            self.user_cf = UserCollaborativeFilteringRecommender()
        else:
            self.item_cf = ItemCollaborativeFilteringRecommender(
                matrix_path=matrix_path
            )
            self.user_cf = UserCollaborativeFilteringRecommender(
                matrix_path=matrix_path
            )

        matrix_file = matrix_path or (PROCESSED_DATA_DIR / "user_item_matrix.csv")
        self.user_item_matrix = pd.read_csv(matrix_file, index_col="user_id")
        self.item_popularity = compute_item_popularity(self.user_item_matrix)

    def generate_candidates(self, user_id: str, candidate_pool_size: int = 20):
        item_candidates = self.item_cf.recommend_for_user(
            user_id=user_id,
            top_k=candidate_pool_size,
        )

        user_candidates = self.user_cf.recommend_for_user(
            user_id=user_id,
            top_k=candidate_pool_size,
        )

        candidates = {}

        for item in item_candidates:
            symbol = item["symbol"]

            candidates.setdefault(
                symbol,
                {
                    "symbol": item["symbol"],
                    "company_name": item["company_name"],
                    "sector": item["sector"],
                    "industry": item["industry"],
                },
            )

            candidates[symbol]["item_cf_raw_score"] = item["item_cf_raw_score"]

        for item in user_candidates:
            symbol = item["symbol"]

            candidates.setdefault(
                symbol,
                {
                    "symbol": item["symbol"],
                    "company_name": item["company_name"],
                    "sector": item["sector"],
                    "industry": item["industry"],
                },
            )

            candidates[symbol]["user_cf_raw_score"] = item["user_cf_raw_score"]

        for item in candidates.values():
            item.setdefault("item_cf_raw_score", 0.0)
            item.setdefault("user_cf_raw_score", 0.0)
            item["popularity_raw_score"] = self.item_popularity.get(item["symbol"], 0.0)

        return list(candidates.values())
