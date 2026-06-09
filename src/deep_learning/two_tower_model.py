# src/deep_learning/two_tower_model.py
import torch
from torch import nn


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_idx, item_idx):
        user_vector = self.user_embedding(user_idx)
        item_vector = self.item_embedding(item_idx)

        score = (user_vector * item_vector).sum(dim=1)

        return score
