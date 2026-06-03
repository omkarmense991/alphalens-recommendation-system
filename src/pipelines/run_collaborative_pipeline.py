# src/pipelines/run_collaborative_pipeline.py

"""
Collaborative Filtering Pipeline

Purpose:
Generate user interaction datasets and build
interaction matrices used for recommendation
and offline evaluation.

Steps:
1. Generate user events.
2. Build user-item interaction matrix.
3. Create temporal train/test split.
4. Build train/test interaction matrices.

Outputs:
- user_events.csv
- user_item_matrix.csv
- train_user_events.csv
- test_user_events.csv
- train_user_item_matrix.csv
- test_user_item_matrix.csv

Usage:

user_item_matrix.csv
    -> Item-based collaborative filtering
    -> User-based collaborative filtering
    -> Hybrid recommender

train_user_item_matrix.csv
test_user_item_matrix.csv
    -> Offline evaluation
    -> Model comparison
    -> Ranking metrics
"""

from src.data.simulate_user_events import simulate_user_events
from src.features.build_interaction_matrix import build_interaction_matrix
from src.evaluation.split_user_events import split_user_events
from src.features.build_train_test_matrices import build_train_test_matrices
from src.utils.logger import logger


def run_collaborative_pipeline():
    logger.info("Starting collaborative pipeline")

    # Generate user behavior events
    # such as views, searches, buys and watchlist additions.
    simulate_user_events()

    # Aggregate events into interaction scores
    # and build the user-item matrix.
    build_interaction_matrix()

    # Perform temporal train/test split
    # so evaluation mimics future prediction.
    split_user_events()

    # Convert train and test events into interaction matrices.
    #
    # Train matrix:
    #   Used to fit collaborative filtering models during evaluation.
    #
    # Test matrix:
    #   Used to evaluate whether recommendations can recover
    #   future user interactions.
    build_train_test_matrices()

    logger.info("Collaborative pipeline completed")


if __name__ == "__main__":
    run_collaborative_pipeline()
