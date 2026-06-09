# src/deep_learning/two_tower_dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config.settings import PROCESSED_DATA_DIR

USER_ITEM_MATRIX_PATH = PROCESSED_DATA_DIR / "user_item_matrix.csv"


class TwoTowerInteractionDataset(Dataset):
    def __init__(self, matrix_path=USER_ITEM_MATRIX_PATH):
        self.user_item_matrix = pd.read_csv(matrix_path, index_col="user_id")

        self.user_ids = self.user_item_matrix.index.tolist()
        self.symbols = self.user_item_matrix.columns.tolist()

        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}

        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.symbols)}

        self.samples = []

        for user_id in self.user_ids:
            for symbol in self.symbols:
                interaction_score = self.user_item_matrix.loc[user_id, symbol]

                label = 1.0 if interaction_score > 0 else 0.0

                self.samples.append(
                    (
                        self.user_to_idx[user_id],
                        self.symbol_to_idx[symbol],
                        label,
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        user_idx, item_idx, label = self.samples[index]

        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "item_idx": torch.tensor(item_idx, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }
