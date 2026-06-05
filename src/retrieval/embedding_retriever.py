# src/retrieval/embedding_retriever.py

"""
Embedding-Based Retrieval

This module implements the retrieval stage of a recommender system using
latent embeddings learned from Matrix Factorization.

The MatrixFactorizationRecommender learns:

    - User embeddings
    - Asset embeddings

These embeddings capture hidden behavioral patterns from historical
user-asset interactions.

Retrieval is performed using nearest-neighbor search in the embedding space:

    Asset Retrieval:
        Find assets whose embeddings are closest to a target asset.

    User Retrieval:
        Find assets whose embeddings are closest to a target user's embedding.

This mimics the retrieval stage used in large-scale recommender systems:

    User/Item Embeddings
              ↓
      Vector Similarity Search
              ↓
       Candidate Generation
              ↓
          Ranking Layer
              ↓
      Final Recommendations

Current implementation:
    - Matrix Factorization (TruncatedSVD) embeddings
    - Scikit-learn NearestNeighbors
    - Cosine similarity

Future improvements:
    - FAISS-based Approximate Nearest Neighbor (ANN) search
    - Two-Tower Retrieval Networks
    - Real-time embedding updates
    - Retrieval evaluation metrics (Recall@K, Candidate Coverage)

The goal of this module is candidate generation, not final ranking.
A ranking model should be applied on the retrieved candidates before
serving recommendations to users.
"""

import pandas as pd

from sklearn.neighbors import NearestNeighbors

from src.config.settings import PROCESSED_DATA_DIR, ASSETS_MASTER_PATH
from src.recommender.matrix_factorization import MatrixFactorizationRecommender
from src.utils.logger import logger


USER_ITEM_MATRIX_PATH = PROCESSED_DATA_DIR / "user_item_matrix.csv"


class EmbeddingRetriever:
    def __init__(
        self,
        matrix_path=USER_ITEM_MATRIX_PATH,
        n_factors: int = 20,
        n_neighbors: int = 20,
    ):
        self.mf_recommender = MatrixFactorizationRecommender(
            matrix_path=matrix_path,
            n_factors=n_factors,
        )

        self.assets_df = pd.read_csv(ASSETS_MASTER_PATH)

        self.symbols = self.mf_recommender.symbols
        self.asset_embeddings = self.mf_recommender.asset_embeddings
        self.user_embeddings = self.mf_recommender.user_embeddings
        self.user_ids = self.mf_recommender.user_ids

        self.neighbor_model = NearestNeighbors(
            n_neighbors=min(n_neighbors, len(self.symbols)),
            metric="cosine",
            algorithm="brute",
        )

        self.neighbor_model.fit(self.asset_embeddings)

    def retrieve_similar_assets(
        self,
        symbol: str,
        top_k: int = 10,
    ):
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not found")

        asset_index = self.symbols.index(symbol)

        distances, indices = self.neighbor_model.kneighbors(
            self.asset_embeddings[asset_index].reshape(1, -1),
            n_neighbors=min(top_k + 1, len(self.symbols)),
        )

        results = []

        for distance, index in zip(distances[0], indices[0]):
            candidate_symbol = self.symbols[index]

            if candidate_symbol == symbol:
                continue

            asset_info = self.assets_df[
                self.assets_df["symbol"] == candidate_symbol
            ].iloc[0]

            similarity = 1 - distance

            results.append(
                {
                    "symbol": candidate_symbol,
                    "company_name": asset_info["company_name"],
                    "sector": asset_info["sector"],
                    "industry": asset_info["industry"],
                    "embedding_similarity": round(float(similarity), 4),
                }
            )

            if len(results) == top_k:
                break

        return results

    def retrieve_for_user(
        self,
        user_id: str,
        top_k: int = 10,
    ):
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not found")

        user_index = self.user_ids.index(user_id)
        user_embedding = self.user_embeddings[user_index].reshape(1, -1)

        distances, indices = self.neighbor_model.kneighbors(
            user_embedding,
            n_neighbors=min(top_k + 20, len(self.symbols)),
        )

        user_interactions = self.mf_recommender.user_item_matrix.loc[user_id]
        interacted_assets = set(
            user_interactions[user_interactions > 0].index.tolist()
        )

        results = []

        for distance, index in zip(distances[0], indices[0]):
            candidate_symbol = self.symbols[index]

            if candidate_symbol in interacted_assets:
                continue

            asset_info = self.assets_df[
                self.assets_df["symbol"] == candidate_symbol
            ].iloc[0]

            similarity = 1 - distance

            results.append(
                {
                    "symbol": candidate_symbol,
                    "company_name": asset_info["company_name"],
                    "sector": asset_info["sector"],
                    "industry": asset_info["industry"],
                    "retrieval_score": round(float(similarity), 4),
                }
            )

            if len(results) == top_k:
                break

        return results


if __name__ == "__main__":
    retriever = EmbeddingRetriever(
        n_factors=20,
        n_neighbors=30,
    )

    logger.info("Similar assets from embedding retriever:")
    similar_assets = retriever.retrieve_similar_assets(
        symbol="RELIANCE.NS",
        top_k=5,
    )

    for item in similar_assets:
        logger.info(item)

    logger.info("User candidates from embedding retriever:")
    user_candidates = retriever.retrieve_for_user(
        user_id="user_1",
        top_k=5,
    )

    for item in user_candidates:
        logger.info(item)