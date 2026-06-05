# src/evaluation/evaluate_recommenders.py

"""
load test_user_item_matrix.csv
train recommenders using train_user_item_matrix.csv
generate recommendations for each user
compare recommendations with future test interactions
calculate precision@k, recall@k, hit-rate@k, NDCG@k
"""

import pandas as pd

from src.config.settings import (
    TRAIN_USER_ITEM_MATRIX_PATH,
    TEST_USER_ITEM_MATRIX_PATH,
    EVALUATION_RESULTS_PATH,
    EVALUATION_SUMMARY_PATH,
    ASSETS_MASTER_PATH,
)
from src.evaluation.ranking_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    ndcg_at_k,
)
from src.recommender.item_collaborative_filtering import (
    ItemCollaborativeFilteringRecommender,
)
from src.recommender.user_collaborative_filtering import (
    UserCollaborativeFilteringRecommender,
)
from src.recommender.hybrid_collaborative_recommender import (
    HybridCollaborativeRecommender,
)

from src.recommender.matrix_factorization import MatrixFactorizationRecommender

from src.utils.logger import logger

from src.evaluation.recommendation_quality import (
    catalog_coverage,
    sector_diversity,
)

from src.evaluation.popularity_bias import (
    compute_item_popularity,
    average_recommendation_popularity,
)

from src.recommender.ranking_recommender import RankingRecommender

from src.evaluation.mlflow_tracker import log_evaluation_to_mlflow


def get_recommended_symbols(recommendations: list[dict]) -> list[str]:
    return [item["symbol"] for item in recommendations]


def get_relevant_items(test_user_row: pd.Series) -> set[str]:
    return set(test_user_row[test_user_row > 0].index.tolist())


def evaluate_model(
    model_name: str,
    recommender,
    test_matrix: pd.DataFrame,
    assets_df: pd.DataFrame,
    item_popularity: dict[str, float],
    k: int = 5,
):
    user_metrics = []

    all_recommended_items = []

    for user_id in test_matrix.index:
        relevant_items = get_relevant_items(test_matrix.loc[user_id])

        if not relevant_items:
            continue

        try:
            recommendations = recommender.recommend_for_user(
                user_id=user_id,
                top_k=k,
            )
        except ValueError:
            continue

        recommended_items = get_recommended_symbols(recommendations)

        all_recommended_items.extend(recommended_items)

        user_metrics.append(
            {
                "user_id": user_id,
                "model": model_name,
                "precision_at_k": precision_at_k(
                    recommended_items,
                    relevant_items,
                    k,
                ),
                "recall_at_k": recall_at_k(
                    recommended_items,
                    relevant_items,
                    k,
                ),
                "hit_rate_at_k": hit_rate_at_k(
                    recommended_items,
                    relevant_items,
                    k,
                ),
                "ndcg_at_k": ndcg_at_k(
                    recommended_items,
                    relevant_items,
                    k,
                ),
                "sector_diversity_at_k": sector_diversity(
                    recommended_items,
                    assets_df,
                ),
                "avg_recommendation_popularity": average_recommendation_popularity(
                    recommended_items, item_popularity
                ),
            }
        )

    results_df = pd.DataFrame(user_metrics)

    results_df["catalog_coverage"] = catalog_coverage(
        all_recommended_items,
        set(assets_df["symbol"].tolist()),
    )

    return results_df


def evaluate_recommenders(k: int = 5):
    test_matrix = pd.read_csv(
        TEST_USER_ITEM_MATRIX_PATH,
        index_col="user_id",
    )

    train_matrix = pd.read_csv(TRAIN_USER_ITEM_MATRIX_PATH, index_col="user_id")

    assets_df = pd.read_csv(ASSETS_MASTER_PATH)

    item_popularity = compute_item_popularity(train_matrix)

    recommenders = {
        "item_cf": ItemCollaborativeFilteringRecommender(
            matrix_path=TRAIN_USER_ITEM_MATRIX_PATH,
        ),
        "user_cf": UserCollaborativeFilteringRecommender(
            matrix_path=TRAIN_USER_ITEM_MATRIX_PATH,
        ),
        "hybrid_cf": HybridCollaborativeRecommender(
            matrix_path=TRAIN_USER_ITEM_MATRIX_PATH,
        ),
        "ranking": RankingRecommender(
            matrix_path=TRAIN_USER_ITEM_MATRIX_PATH,
            item_cf_weight=0.25,
            user_cf_weight=0.25,
            embedding_weight=0.20,
            mf_weight=0.20,
            popularity_weight=0.10,
        ),
        "matrix_factorization": MatrixFactorizationRecommender(
            matrix_path=TRAIN_USER_ITEM_MATRIX_PATH,
            n_factors=20,  # Best latent dimension discovered via offline evaluation: 20
        ),
    }

    evaluation_results = []

    for model_name, recommender in recommenders.items():
        logger.info(f"Evaluating {model_name}")

        model_results = evaluate_model(
            model_name=model_name,
            recommender=recommender,
            test_matrix=test_matrix,
            item_popularity=item_popularity,
            assets_df=assets_df,
            k=k,
        )

        evaluation_results.append(model_results)

    results_df = pd.concat(evaluation_results)

    summary_df = (
        results_df.groupby("model", as_index=False)
        .agg(
            precision_at_k=("precision_at_k", "mean"),
            recall_at_k=("recall_at_k", "mean"),
            hit_rate_at_k=("hit_rate_at_k", "mean"),
            ndcg_at_k=("ndcg_at_k", "mean"),
            sector_diversity_at_k=("sector_diversity_at_k", "mean"),
            catalog_coverage=("catalog_coverage", "mean"),
            avg_recommendation_popularity=("avg_recommendation_popularity", "mean"),
        )
        .sort_values("ndcg_at_k", ascending=False)
    )

    results_df.to_csv(EVALUATION_RESULTS_PATH, index=False)
    summary_df.to_csv(EVALUATION_SUMMARY_PATH, index=False)

    logger.info("\nEvaluation Summary:")
    logger.info(f"\n{summary_df}")

    logger.info(f"Detailed evaluation results saved to {EVALUATION_RESULTS_PATH}")
    logger.info(f"Evaluation summary saved to {EVALUATION_SUMMARY_PATH}")

    log_evaluation_to_mlflow(summary_df)
    logger.info("Evaluation metrics logged to MLflow")

    return summary_df


if __name__ == "__main__":
    evaluate_recommenders(k=5)
