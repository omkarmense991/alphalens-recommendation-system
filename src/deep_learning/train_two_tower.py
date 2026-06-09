# src/deep_learning/train_two_tower.py
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config.settings import PROCESSED_DATA_DIR
from src.deep_learning.two_tower_dataset import TwoTowerInteractionDataset
from src.deep_learning.two_tower_model import TwoTowerModel
from src.utils.logger import logger

TWO_TOWER_MODEL_PATH = PROCESSED_DATA_DIR / "two_tower_model.pt"


def train_two_tower(
    epochs: int = 10,
    batch_size: int = 256,
    embedding_dim: int = 32,
    learning_rate: float = 0.001,
):
    dataset = TwoTowerInteractionDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = TwoTowerModel(
        num_users=len(dataset.user_ids),
        num_items=len(dataset.symbols),
        embedding_dim=embedding_dim,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in dataloader:
            user_idx = batch["user_idx"]
            item_idx = batch["item_idx"]
            label = batch["label"]

            optimizer.zero_grad()

            logits = model(user_idx, item_idx)

            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        logger.info(f"Epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_ids": dataset.user_ids,
            "symbols": dataset.symbols,
            "embedding_dim": embedding_dim,
        },
        TWO_TOWER_MODEL_PATH,
    )

    logger.info(f"Two-tower model saved to {TWO_TOWER_MODEL_PATH}")


if __name__ == "__main__":
    train_two_tower()
