# src/recommender/ranking_recommender.py

"""
Production Ranking Recommender

Coordinates the complete recommendation serving pipeline:

1. Candidate Generation
2. Feature Aggregation
3. Ranking

Candidate sources:

- Item-Based Collaborative Filtering
- User-Based Collaborative Filtering
- Embedding Retrieval
- Matrix Factorization
- Popularity Signals

Workflow:

User
 ↓
Candidate Generator
    ├─ Item CF Candidates
    ├─ User CF Candidates
    ├─ Embedding Candidates
    ├─ Matrix Factorization Candidates
    └─ Popularity Features
 ↓
Feature Aggregation
 ↓
Weighted Ranker
 ↓
Top-K Recommendations

Ranking Score:

ranking_score =
    item_cf_weight      × item_cf_score
  + user_cf_weight      × user_cf_score
  + embedding_weight    × embedding_score
  + mf_weight           × mf_score
  + popularity_weight   × popularity_score

This module represents the final recommendation layer used
for personalized recommendation serving.

The architecture follows a production-style
Retrieval → Ranking paradigm, where multiple candidate
sources are combined and ranked using weighted feature fusion.
"""

from src.candidates.candidate_generator import CandidateGenerator
from src.ranking.learning_to_rank import WeightedRanker
from src.utils.logger import logger


class RankingRecommender:
    def __init__(
        self,
        matrix_path=None,
        item_cf_weight: float = 0.25,
        user_cf_weight: float = 0.25,
        embedding_weight: float = 0.20,
        mf_weight: float = 0.20,
        popularity_weight: float = 0.10,
    ):
        self.candidate_generator = CandidateGenerator(matrix_path=matrix_path)

        self.ranker = WeightedRanker(
            item_cf_weight=item_cf_weight,
            user_cf_weight=user_cf_weight,
            embedding_weight=embedding_weight,
            mf_weight=mf_weight,
            popularity_weight=popularity_weight,
        )

    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = 5,
        candidate_pool_size: int = 20,
    ):
        candidates = self.candidate_generator.generate_candidates(
            user_id=user_id,
            candidate_pool_size=candidate_pool_size,
        )

        return self.ranker.rank(
            candidates=candidates,
            top_k=top_k,
        )


if __name__ == "__main__":
    recommender = RankingRecommender(
        item_cf_weight=0.25,
        user_cf_weight=0.25,
        embedding_weight=0.20,
        mf_weight=0.20,
        popularity_weight=0.10,
    )

    results = recommender.recommend_for_user(
        user_id="user_1",
        top_k=5,
        candidate_pool_size=20,
    )

    for item in results:
        logger.info(item)
