# src/evaluation/split_user_events.py
import pandas as pd

from src.config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_USER_EVENTS_PATH,
    TEST_USER_EVENTS_PATH,
)
from src.utils.logger import logger


def split_user_events(test_ratio: float = 0.2):
    events_path = RAW_DATA_DIR / "user_events.csv"

    events_df = pd.read_csv(events_path)
    events_df["event_time"] = pd.to_datetime(events_df["event_time"])

    train_rows = []
    test_rows = []

    for user_id, user_events in events_df.groupby("user_id"):
        user_events = user_events.sort_values("event_time")

        split_index = int(len(user_events) * (1 - test_ratio))

        train_rows.append(user_events.iloc[:split_index])
        test_rows.append(user_events.iloc[split_index:])

    train_df = pd.concat(train_rows)
    test_df = pd.concat(test_rows)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_USER_EVENTS_PATH, index=False)
    test_df.to_csv(TEST_USER_EVENTS_PATH, index=False)

    logger.info(f"Train events saved to {TRAIN_USER_EVENTS_PATH}")
    logger.info(f"Test events saved to {TEST_USER_EVENTS_PATH}")
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")


if __name__ == "__main__":
    split_user_events()
