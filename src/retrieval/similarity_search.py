# src/retrieval/similarity_search.py

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import ASSET_FEATURES_PATH, ASSETS_MASTER_PATH


class AssetSimilaritySearch:
    def __init__(self):
        self.features_df = pd.read_csv(ASSET_FEATURES_PATH)
        self.assets_df = pd.read_csv(ASSETS_MASTER_PATH)

        self.symbols = self.features_df["symbol"].tolist()
        self.feature_matrix = self.features_df.drop(columns=["symbol"]).values

        self.similarity_matrix = cosine_similarity(self.feature_matrix)

    def find_similar_assets(self, symbol: str, top_k: int = 5, min_score: float = 0.25):
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not found")

        asset_index = self.symbols.index(symbol)

        similarity_scores = list(enumerate(self.similarity_matrix[asset_index]))

        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommendations = []

        for index, score in similarity_scores:
            candidate_symbol = self.symbols[index]

            if candidate_symbol == symbol:
                continue

            if score < min_score:
                continue

            asset_info = self.assets_df[
                self.assets_df["symbol"] == candidate_symbol
            ].iloc[0]

            recommendations.append(
                {
                    "symbol": candidate_symbol,
                    "company_name": asset_info["company_name"],
                    "sector": asset_info["sector"],
                    "industry": asset_info["industry"],
                    "similarity_score": round(float(score), 4),
                }
            )

            if len(recommendations) == top_k:
                break

        return recommendations
