# src/recommender/hybrid_collaborative_recommender.py

from src.recommender.item_collaborative_filtering import (
    ItemCollaborativeFilteringRecommender,
)
from src.recommender.user_collaborative_filtering import (
    UserCollaborativeFilteringRecommender,
)
from src.utils.logger import logger


class HybridCollaborativeRecommender:
    def __init__(self):
        self.item_cf = ItemCollaborativeFilteringRecommender()
        self.user_cf = UserCollaborativeFilteringRecommender()

    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = 5,
        item_weight: float = 0.5,
        user_weight: float = 0.5,
    ):
        item_recommendations = self.item_cf.recommend_for_user(
            user_id=user_id,
            top_k=20,
        )

        user_recommendations = self.user_cf.recommend_for_user(
            user_id=user_id,
            top_k=20,
        )

        item_scores = {
            item["symbol"]: item["collaborative_score"] for item in item_recommendations
        }

        user_scores = {
            item["symbol"]: item["user_cf_score"] for item in user_recommendations
        }

        normalized_item_scores = self._normalize_scores(item_scores)
        normalized_user_scores = self._normalize_scores(user_scores)

        all_symbols = set(normalized_item_scores.keys()) | set(
            normalized_user_scores.keys()
        )

        recommendations = []

        for symbol in all_symbols:
            item_score = normalized_item_scores.get(symbol, 0)
            user_score = normalized_user_scores.get(symbol, 0)

            final_score = item_weight * item_score + user_weight * user_score

            asset_info = self._get_asset_info(
                symbol=symbol,
                item_recommendations=item_recommendations,
                user_recommendations=user_recommendations,
            )

            recommendations.append(
                {
                    "symbol": symbol,
                    "company_name": asset_info["company_name"],
                    "sector": asset_info["sector"],
                    "industry": asset_info["industry"],
                    "item_cf_score": round(float(item_score), 4),
                    "user_cf_score": round(float(user_score), 4),
                    "hybrid_score": round(float(final_score), 4),
                }
            )

        recommendations = sorted(
            recommendations,
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        return recommendations[:top_k]

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}

        max_score = max(scores.values())

        if max_score == 0:
            return {symbol: 0 for symbol in scores}

        return {symbol: score / max_score for symbol, score in scores.items()}

    @staticmethod
    def _get_asset_info(
        symbol: str,
        item_recommendations: list[dict],
        user_recommendations: list[dict],
    ):
        for recommendation in item_recommendations + user_recommendations:
            if recommendation["symbol"] == symbol:
                return recommendation

        raise ValueError(f"Asset info not found for {symbol}")


if __name__ == "__main__":
    recommender = HybridCollaborativeRecommender()

    results = recommender.recommend_for_user(
        user_id="user_1",
        top_k=5,
    )

    for item in results:
        logger.info(item)
