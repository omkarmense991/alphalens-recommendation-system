# src/recommender/ranking_recommender.py
from src.candidates.candidate_generator import CandidateGenerator
from src.ranking.learning_to_rank import WeightedRanker
from src.utils.logger import logger


class RankingRecommender:
    def __init__(
        self,
        matrix_path=None,
        item_cf_weight: float = 0.45,
        user_cf_weight: float = 0.45,
        popularity_weight: float = 0.10,
    ):
        self.candidate_generator = CandidateGenerator(matrix_path=matrix_path)

        self.ranker = WeightedRanker(
            item_cf_weight=item_cf_weight,
            user_cf_weight=user_cf_weight,
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
        item_cf_weight=0.45,
        user_cf_weight=0.45,
        popularity_weight=0.10,
    )

    results = recommender.recommend_for_user(
        user_id="user_1",
        top_k=5,
        candidate_pool_size=20,
    )

    for item in results:
        logger.info(item)
