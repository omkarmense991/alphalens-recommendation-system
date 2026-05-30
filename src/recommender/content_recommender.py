# src/recommender/content_recommender.py
from src.retrieval.similarity_search import AssetSimilaritySearch
from src.explainability.content_explainer import ContentRecommendationExplainer
from src.utils.logger import logger


class ContentBasedRecommender:
    def __init__(self):
        self.search_engine = AssetSimilaritySearch()
        self.explainer = ContentRecommendationExplainer()

    def find_similar_assets(self, symbol: str, top_k: int = 5, min_score: float = 0.25):
        recommendations = self.search_engine.find_similar_assets(
            symbol=symbol, top_k=top_k, min_score=min_score
        )

        for recommendation in recommendations:
            recommendation["explanations"] = self.explainer.explain(
                source_symbol=symbol, recommended_symbol=recommendation["symbol"]
            )

        return recommendations


if __name__ == "__main__":
    recommender = ContentBasedRecommender()

    results = recommender.find_similar_assets(symbol="RELIANCE.NS", top_k=5)

    for item in results:
        logger.info(f"\n{item}")
