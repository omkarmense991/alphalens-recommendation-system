# src/recommender/content_recommender.py

from src.retrieval.similarity_search import AssetSimilaritySearch
from src.utils.logger import logger


class ContentBasedRecommender:
    def __init__(self):
        self.search_engine = AssetSimilaritySearch()

    def recommend_similar_assets(self, symbol: str, top_k: int = 5):
        return self.search_engine.find_similar_assets(symbol=symbol, top_k=top_k)


if __name__ == "__main__":
    recommender = ContentBasedRecommender()

    results = recommender.recommend_similar_assets(symbol="RELIANCE.NS", top_k=5)

    for item in results:
        logger.info(item)
