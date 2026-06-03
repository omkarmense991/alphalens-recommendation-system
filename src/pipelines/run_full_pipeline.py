# src/pipelines/run_full_pipeline.py

"""
AlphaLens End-to-End Pipeline

Purpose:
Execute the complete recommendation system workflow.

Stages:
1. Build asset catalog and features.
2. Generate collaborative filtering datasets.
3. Train/evaluate recommenders.

Typical usage:
- Initial project setup
- Full refresh
- Rebuilding datasets
- Benchmarking recommender performance

Output:
Complete recommendation artifacts required by the system.
"""

from src.pipelines.run_feature_pipeline import run_feature_pipeline
from src.pipelines.run_collaborative_pipeline import run_collaborative_pipeline
from src.pipelines.run_evaluation_pipeline import run_evaluation_pipeline
from src.utils.logger import logger


def run_full_pipeline():
    logger.info("Starting AlphaLens full pipeline")

    # Build content-based recommendation artifacts.
    run_feature_pipeline()

    # Build collaborative filtering datasets.
    run_collaborative_pipeline()

    # Evaluate recommendation quality.
    run_evaluation_pipeline()

    logger.info("AlphaLens full pipeline completed")


if __name__ == "__main__":
    run_full_pipeline()
