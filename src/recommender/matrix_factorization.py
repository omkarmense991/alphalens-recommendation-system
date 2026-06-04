# src/recommender/matrix_factorization.py

import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import PROCESSED_DATA_DIR, ASSETS_MASTER_PATH
from src.utils.logger import logger

USER_ITEM_MATRIX_PATH = PROCESSED_DATA_DIR / "user_item_matrix.csv"


class MatrixFactorizationRecommender:
    def __init__(
        self,
        matrix_path=USER_ITEM_MATRIX_PATH,
        n_factors: int = 10,
        random_state: int = 42,
    ):
        self.user_item_matrix = pd.read_csv(
            matrix_path,
            index_col="user_id",
        )

        self.assets_df = pd.read_csv(ASSETS_MASTER_PATH)

        self.user_ids = self.user_item_matrix.index.tolist()
        self.symbols = self.user_item_matrix.columns.tolist()

        self.n_factors = min(
            n_factors,
            min(self.user_item_matrix.shape) - 1,
        )

        self.model = TruncatedSVD(
            n_components=self.n_factors,
            random_state=random_state,
        )

        self.user_embeddings = self.model.fit_transform(self.user_item_matrix)

        self.asset_embeddings = self.model.components_.T

        self.predicted_scores = self.user_embeddings @ self.asset_embeddings.T

        self.predicted_scores_df = pd.DataFrame(
            self.predicted_scores,
            index=self.user_ids,
            columns=self.symbols,
        )

    def recommend_for_user(self, user_id: str, top_k: int = 5):
        if user_id not in self.predicted_scores_df.index:
            raise ValueError(f"User {user_id} not found")

        user_scores = self.predicted_scores_df.loc[user_id]

        user_interactions = self.user_item_matrix.loc[user_id]
        interacted_assets = user_interactions[user_interactions > 0].index.tolist()

        candidate_scores = user_scores.drop(labels=interacted_assets)

        ranked_candidates = candidate_scores.sort_values(ascending=False)

        recommendations = []

        for symbol, score in ranked_candidates.head(top_k).items():
            asset_info = self.assets_df[self.assets_df["symbol"] == symbol].iloc[0]

            recommendations.append(
                {
                    "symbol": symbol,
                    "company_name": asset_info["company_name"],
                    "sector": asset_info["sector"],
                    "industry": asset_info["industry"],
                    "mf_score": round(float(score), 4),
                }
            )

        return recommendations

    # Assets consumed by similar users get similar embeddings.
    # These are not based on: Sector,Industry or PE Ratio They are based on: User interaction patterns
    def find_similar_assets_by_embedding(self, symbol: str, top_k: int = 5):
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not found")

        asset_index = self.symbols.index(symbol)
        target_embedding = self.asset_embeddings[asset_index].reshape(1, -1)

        similarities = cosine_similarity(
            target_embedding,
            self.asset_embeddings,
        )[0]

        similarity_scores = list(enumerate(similarities))

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True,
        )

        results = []

        for index, score in similarity_scores:
            candidate_symbol = self.symbols[index]

            if candidate_symbol == symbol:
                continue

            asset_info = self.assets_df[
                self.assets_df["symbol"] == candidate_symbol
            ].iloc[0]

            results.append(
                {
                    "symbol": candidate_symbol,
                    "company_name": asset_info["company_name"],
                    "sector": asset_info["sector"],
                    "industry": asset_info["industry"],
                    "embedding_similarity": round(float(score), 4),
                }
            )

            if len(results) == top_k:
                break

        return results


if __name__ == "__main__":
    recommender = MatrixFactorizationRecommender(
        n_factors=10,
    )

    logger.info("Matrix factorization recommendations:")
    results = recommender.recommend_for_user(
        user_id="user_1",
        top_k=5,
    )

    for item in results:
        logger.info(item)

    logger.info("Similar assets by latent embedding:")
    similar_assets = recommender.find_similar_assets_by_embedding(
        symbol="RELIANCE.NS",
        top_k=5,
    )

    for item in similar_assets:
        logger.info(item)