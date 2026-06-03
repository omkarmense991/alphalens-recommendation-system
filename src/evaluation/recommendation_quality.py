# src/evaluation/recommendation_quality.py
import pandas as pd


def catalog_coverage(
    all_recommended_items: list[str],
    all_catalog_items: set[str],
) -> float:
    if not all_catalog_items:
        return 0.0

    unique_recommended_items = set(all_recommended_items)

    return len(unique_recommended_items) / len(all_catalog_items)


def sector_diversity(
    recommended_items: list[str],
    assets_df: pd.DataFrame,
) -> float:
    if not recommended_items:
        return 0.0

    recommended_assets = assets_df[assets_df["symbol"].isin(recommended_items)]

    if recommended_assets.empty:
        return 0.0

    unique_sectors = recommended_assets["sector"].nunique()

    return unique_sectors / len(recommended_items)
