# src/pipelines/run_evaluation_pipeline.py

"""
Evaluation Pipeline

Purpose:
Evaluate recommendation models using offline metrics.

Metrics:
- Precision@K
- Recall@K
- Hit Rate@K
- NDCG@K
- Catalog Coverage
- Sector Diversity
- Popularity Bias

Output:
- evaluation_results.csv
- evaluation_summary.csv

Used by:
- Model comparison
- Recommender selection
- Performance monitoring
"""

from src.evaluation.evaluate_recommenders import evaluate_recommenders
from src.utils.logger import logger


def run_evaluation_pipeline():
    logger.info("Starting evaluation pipeline")

    # Evaluate all recommendation models
    # using train/test interaction data.
    evaluate_recommenders(k=5)

    logger.info("Evaluation pipeline completed")


if __name__ == "__main__":
    run_evaluation_pipeline()
