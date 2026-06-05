# src/evaluation/mlflow_tracker.py


import mlflow
import pandas as pd

from src.config.settings import (
    EVALUATION_RESULTS_PATH,
    EVALUATION_SUMMARY_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)


def log_evaluation_to_mlflow(summary_df: pd.DataFrame):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="recommender_evaluation"):
        for _, row in summary_df.iterrows():
            model_name = row["model"]

            for metric_name in [
                "precision_at_k",
                "recall_at_k",
                "hit_rate_at_k",
                "ndcg_at_k",
                "sector_diversity_at_k",
                "catalog_coverage",
                "avg_recommendation_popularity",
            ]:
                mlflow.log_metric(
                    key=f"{model_name}_{metric_name}",
                    value=float(row[metric_name]),
                )

        best_model = summary_df.sort_values(
            "ndcg_at_k",
            ascending=False,
        ).iloc[0]["model"]

        mlflow.set_tag("best_model_by_ndcg", best_model)

        mlflow.log_artifact(str(EVALUATION_RESULTS_PATH))
        mlflow.log_artifact(str(EVALUATION_SUMMARY_PATH))