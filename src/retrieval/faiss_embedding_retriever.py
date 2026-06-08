# src/recommender/faiss_embedding_retriever.py
"""
FAISS-based embedding retriever for AlphaLens.

This module implements an embedding retrieval layer using:

1. Matrix Factorization
   - Learns user embeddings and asset embeddings from the user-item matrix.

2. L2 Normalization
   - Converts embeddings to unit-length vectors.
   - This makes FAISS inner-product search behave like cosine similarity search.

3. FAISS Index
   - Stores asset embeddings in a vector index.
   - Allows fast nearest-neighbor search.

Supported retrieval modes:

1. Asset-to-Asset Retrieval
   - Given an asset symbol, retrieve assets with similar latent embeddings.

2. User-to-Asset Retrieval
   - Given a user ID, retrieve assets closest to the user's embedding.
   - Already-interacted assets are excluded.

This represents the retrieval stage of a production-style recommender system:

    User / Asset Embedding
            ↓
    FAISS Vector Search
            ↓
    Candidate Assets
            ↓
    Ranking Model
"""

import numpy as np
import pandas as pd
import faiss

from src.config.settings import PROCESSED_DATA_DIR, ASSETS_MASTER_PATH
from src.recommender.matrix_factorization import MatrixFactorizationRecommender
from src.utils.logger import logger

USER_ITEM_MATRIX_PATH = PROCESSED_DATA_DIR / "user_item_matrix.csv"


class FaissEmbeddingRetriever:
    def __init__(
        self,
        matrix_path=USER_ITEM_MATRIX_PATH,
        n_factors: int = 20,
    ):
        # Matrix Factorization learns both:
        # 1. user embeddings
        # 2. asset embeddings
        #
        # These embeddings live in the same latent space.
        # That is why a user embedding can be compared with asset embeddings.
        self.mf_recommender = MatrixFactorizationRecommender(
            matrix_path=matrix_path,
            n_factors=n_factors,
        )

        # Asset metadata is used only for enriching the final output.
        self.assets_df = pd.read_csv(ASSETS_MASTER_PATH)

        self.symbols = self.mf_recommender.symbols
        self.user_ids = self.mf_recommender.user_ids

        # FAISS expects float32 vectors.
        #
        # We also L2-normalize embeddings so that:
        #
        #     inner product == cosine similarity
        #
        # This is why we can use faiss.IndexFlatIP for cosine-style retrieval.
        self.asset_embeddings = self._normalize(
            self.mf_recommender.asset_embeddings.astype("float32")
        )

        self.user_embeddings = self._normalize(
            self.mf_recommender.user_embeddings.astype("float32")
        )

        # Each embedding has shape:
        #
        #     (embedding_dim,)
        #
        # If n_factors = 20, then every asset/user embedding has 20 values.
        # FAISS needs to know this dimension before creating the index.
        embedding_dim = self.asset_embeddings.shape[1]

        # Create an empty FAISS vector index.
        #
        # IndexFlatIP means:
        # - Flat: exact search over all vectors
        # - IP: inner product similarity
        #
        # Because embeddings are L2-normalized, inner product gives
        # cosine similarity.
        self.index = faiss.IndexFlatIP(embedding_dim)

        # Add all asset embeddings to the FAISS index.
        #
        # The index now acts like a vector database:
        # given a query embedding, it can return nearest asset embeddings.
        self.index.add(self.asset_embeddings)

    def retrieve_similar_assets(self, symbol: str, top_k: int = 10):
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not found")

        asset_index = self.symbols.index(symbol)
        query_embedding = self.asset_embeddings[asset_index].reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k + 1, len(self.symbols)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
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
                    "faiss_similarity": round(float(score), 4),
                }
            )

            if len(results) == top_k:
                break

        return results

    def retrieve_for_user(self, user_id: str, top_k: int = 10):
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not found")

        user_index = self.user_ids.index(user_id)
        query_embedding = self.user_embeddings[user_index].reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k + 20, len(self.symbols)),
        )

        user_interactions = self.mf_recommender.user_item_matrix.loc[user_id]
        interacted_assets = set(user_interactions[user_interactions > 0].index.tolist())

        results = []

        for score, index in zip(scores[0], indices[0]):
            candidate_symbol = self.symbols[index]

            if candidate_symbol in interacted_assets:
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
                    "faiss_retrieval_score": round(float(score), 4),
                }
            )

            if len(results) == top_k:
                break

        return results

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms


if __name__ == "__main__":
    retriever = FaissEmbeddingRetriever(n_factors=20)

    logger.info("FAISS similar assets:")
    similar_assets = retriever.retrieve_similar_assets(
        symbol="RELIANCE.NS",
        top_k=5,
    )

    for item in similar_assets:
        logger.info(item)

    logger.info("FAISS user recommendations:")
    user_recommendations = retriever.retrieve_for_user(
        user_id="user_1",
        top_k=5,
    )

    for item in user_recommendations:
        logger.info(item)
