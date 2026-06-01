# src/features/build_train_test_matrices.py

# src/features/build_train_test_matrices.py

import pandas as pd

from src.config.settings import (
    PROCESSED_DATA_DIR,
    TRAIN_USER_EVENTS_PATH,
    TEST_USER_EVENTS_PATH,
    TRAIN_USER_ITEM_MATRIX_PATH,
    TEST_USER_ITEM_MATRIX_PATH,
)
from src.features.build_interaction_matrix import create_interaction_matrix
from src.utils.logger import logger


def build_train_test_matrices():
    train_events_df = pd.read_csv(TRAIN_USER_EVENTS_PATH)
    test_events_df = pd.read_csv(TEST_USER_EVENTS_PATH)

    train_matrix = create_interaction_matrix(train_events_df)
    test_matrix = create_interaction_matrix(test_events_df)

    all_users = sorted(set(train_matrix.index) | set(test_matrix.index))
    all_symbols = sorted(set(train_matrix.columns) | set(test_matrix.columns))

    train_matrix = train_matrix.reindex(
        index=all_users,
        columns=all_symbols,
        fill_value=0,
    )

    test_matrix = test_matrix.reindex(
        index=all_users,
        columns=all_symbols,
        fill_value=0,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_matrix.to_csv(TRAIN_USER_ITEM_MATRIX_PATH)
    test_matrix.to_csv(TEST_USER_ITEM_MATRIX_PATH)

    logger.info(f"Train matrix saved to {TRAIN_USER_ITEM_MATRIX_PATH}")
    logger.info(f"Test matrix saved to {TEST_USER_ITEM_MATRIX_PATH}")
    logger.info(f"Train matrix shape: {train_matrix.shape}")
    logger.info(f"Test matrix shape: {test_matrix.shape}")


if __name__ == "__main__":
    build_train_test_matrices()