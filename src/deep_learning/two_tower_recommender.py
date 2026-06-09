# src/deep_learning/two_tower_recommender.py

import pandas as pd
import torch

from src.config.settings import ASSETS_MASTER_PATH, PROCESSED_DATA_DIR
from src.deep_learning.two_tower_model import TwoTowerModel
from src.utils.logger import logger

USER_ITEM_MATRIX_PATH = PROCESSED_DATA_DIR / "user_item_matrix.csv"
TWO_TOWER_MODEL_PATH = PROCESSED_DATA_DIR / "two_tower_model.pt"


class TwoTowerRecommender:
    def __init__(
        self,
        model_path=TWO_TOWER_MODEL_PATH,
        matrix_path=USER_ITEM_MATRIX_PATH,
    ):
        checkpoint = torch.load(
            model_path,
            map_location=torch.device("cpu"),
        )

        self.user_ids = checkpoint["user_ids"]
        self.symbols = checkpoint["symbols"]
        self.embedding_dim = checkpoint["embedding_dim"]

        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}

        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.symbols)}

        self.model = TwoTowerModel(
            num_users=len(self.user_ids),
            num_items=len(self.symbols),
            embedding_dim=self.embedding_dim,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.user_item_matrix = pd.read_csv(
            matrix_path,
            index_col="user_id",
        )

        self.assets_df = pd.read_csv(ASSETS_MASTER_PATH)

    def recommend_for_user(self, user_id: str, top_k: int = 5):
        if user_id not in self.user_to_idx:
            raise ValueError(f"User {user_id} not found")

        user_idx = self.user_to_idx[user_id]

        item_indices = torch.arange(len(self.symbols), dtype=torch.long)
        user_indices = torch.full_like(item_indices, fill_value=user_idx)

        with torch.no_grad():
            logits = self.model(user_indices, item_indices)
            scores = torch.sigmoid(logits)

        user_interactions = self.user_item_matrix.loc[user_id]
        interacted_assets = set(user_interactions[user_interactions > 0].index.tolist())

        candidate_scores = []

        for symbol, score in zip(self.symbols, scores.tolist()):
            if symbol in interacted_assets:
                continue

            candidate_scores.append((symbol, score))

        ranked_candidates = sorted(
            candidate_scores,
            key=lambda x: x[1],
            reverse=True,
        )

        recommendations = []

        for symbol, score in ranked_candidates[:top_k]:
            asset_info = self.assets_df[self.assets_df["symbol"] == symbol].iloc[0]

            recommendations.append(
                {
                    "symbol": symbol,
                    "company_name": asset_info["company_name"],
                    "sector": asset_info["sector"],
                    "industry": asset_info["industry"],
                    "two_tower_score": round(float(score), 4),
                }
            )

        return recommendations


if __name__ == "__main__":
    recommender = TwoTowerRecommender()

    results = recommender.recommend_for_user(
        user_id="user_1",
        top_k=5,
    )

    for item in results:
        logger.info(item)
