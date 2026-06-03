# src/pipelines/run_feature_pipeline.

"""
Feature Pipeline

Purpose:
Build the asset catalog and asset feature representations used by
the content-based recommendation system.

Steps:
1. Create the asset universe.
2. Fetch market/fundamental data.
3. Engineer recommendation features.

Output:
- assets_master.csv
- asset_features.csv

Used by:
- Content-based recommender
- Similarity search
- Future embedding generation
"""

from src.data.create_asset_universe import create_nifty50_universe
from src.data.build_assets_master import build_assets_master
from src.features.build_asset_features import build_asset_features
from src.utils.logger import logger


def run_feature_pipeline():
    logger.info("Starting feature pipeline")

    # Build the universe of assets available for recommendation.
    create_nifty50_universe()

    # Fetch market data and fundamental metrics
    # such as PE ratio, volatility, beta, volume, etc.
    build_assets_master()

    # Transform raw asset information into numerical
    # feature vectors suitable for similarity search.
    build_asset_features()

    logger.info("Feature pipeline completed")


if __name__ == "__main__":
    run_feature_pipeline()
