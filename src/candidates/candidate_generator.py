# # src/candidates/candidate_generator.py
"""
Candidate Generation Layer

Combines multiple retrieval sources to create a unified candidate
pool for downstream ranking.

Current retrieval sources:

- Item-Based Collaborative Filtering
- User-Based Collaborative Filtering
- Embedding Retrieval (Matrix Factorization Embeddings)
- Matrix Factorization Recommendations

For each candidate asset, the generator attaches retrieval
signals that can later be used by ranking models:

- item_cf_raw_score
- user_cf_raw_score
- embedding_raw_score
- mf_raw_score
- popularity_raw_score

The candidate pool is formed by taking the union of assets
retrieved from all retrieval sources and enriching them with
their respective retrieval scores.

Output:

A unified candidate pool containing assets retrieved from
multiple recommendation strategies.

Architecture:

User
 ↓
Candidate Generator
 ├── Item CF
 ├── User CF
 ├── Embedding Retrieval
 ├── Matrix Factorization
 └── Popularity Signal
 ↓
Candidate Pool
 ↓
Ranking Layer
 ↓
Final Recommendations

This mirrors the retrieval stage used in modern recommender
systems such as YouTube, Netflix, Spotify, Amazon, and many
large-scale recommendation platforms, where multiple retrieval
strategies generate candidates before a ranking model selects
the final recommendations.
"""

import pandas as pd
from src.config.settings import PROCESSED_DATA_DIR

from src.evaluation.popularity_bias import compute_item_popularity
from src.recommender.item_collaborative_filtering import (
    ItemCollaborativeFilteringRecommender,
)
from src.recommender.user_collaborative_filtering import (
    UserCollaborativeFilteringRecommender,
)

from src.retrieval.embedding_retriever import EmbeddingRetriever

from src.recommender.matrix_factorization import MatrixFactorizationRecommender


class CandidateGenerator:
    def __init__(self, matrix_path=None):
        if matrix_path is None:
            self.item_cf = ItemCollaborativeFilteringRecommender()
            self.user_cf = UserCollaborativeFilteringRecommender()
            self.embedding_retriever = EmbeddingRetriever()
            self.mf_recommender = MatrixFactorizationRecommender()
        else:
            self.item_cf = ItemCollaborativeFilteringRecommender(
                matrix_path=matrix_path
            )
            self.user_cf = UserCollaborativeFilteringRecommender(
                matrix_path=matrix_path
            )
            self.embedding_retriever = EmbeddingRetriever(matrix_path=matrix_path)
            self.mf_recommender = MatrixFactorizationRecommender(
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

        embedding_candidates = self.embedding_retriever.retrieve_for_user(
            user_id=user_id,
            top_k=candidate_pool_size,
        )

        mf_candidates = self.mf_recommender.recommend_for_user(
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

        for item in embedding_candidates:
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

            candidates[symbol]["embedding_raw_score"] = item["retrieval_score"]

        for item in mf_candidates:
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

            candidates[symbol]["mf_raw_score"] = item["mf_score"]

        for item in candidates.values():
            item.setdefault("item_cf_raw_score", 0.0)
            item.setdefault("user_cf_raw_score", 0.0)
            item.setdefault("embedding_raw_score", 0.0)
            item.setdefault("mf_raw_score", 0.0)
            item["popularity_raw_score"] = self.item_popularity.get(item["symbol"], 0.0)

        return list(candidates.values())
