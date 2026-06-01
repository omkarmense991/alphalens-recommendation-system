import pandas as pd

from src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.logger import logger

USER_ITEM_MATRIX_PATH = PROCESSED_DATA_DIR / "user_item_matrix.csv"


def build_interaction_matrix():
    events_path = RAW_DATA_DIR / "user_events.csv"

    events_df = pd.read_csv(events_path)

    interaction_df = events_df.groupby(["user_id", "symbol"], as_index=False).agg(
        interaction_score=("event_weight", "sum")
    )

    interaction_df["interaction_score"] = interaction_df["interaction_score"].clip(
        lower=0
    )

    user_item_matrix = interaction_df.pivot_table(
        index="user_id",
        columns="symbol",
        values="interaction_score",
        fill_value=0,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    user_item_matrix.to_csv(USER_ITEM_MATRIX_PATH)

    logger.info(f"User-item matrix saved to {USER_ITEM_MATRIX_PATH}")
    logger.info(f"Matrix shape: {user_item_matrix.shape}")


if __name__ == "__main__":
    build_interaction_matrix()
