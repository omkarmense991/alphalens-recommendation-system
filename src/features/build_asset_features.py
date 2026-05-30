# src/features/build_asset_features.py

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.config.settings import ASSETS_MASTER_PATH, ASSET_FEATURES_PATH
from src.utils.logger import logger

CATEGORICAL_FEATURES = [
    "sector",
    "industry",
]

NUMERIC_FEATURES = [
    "market_cap",
    "pe_ratio",
    "pb_ratio",
    "dividend_yield",
    "beta",
    "volatility",
    "returns_1y",
    "avg_volume",
]


def build_asset_features():
    df = pd.read_csv(ASSETS_MASTER_PATH)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )

    feature_matrix = preprocessor.fit_transform(df)

    categorical_size = (
        preprocessor.named_transformers_["categorical"]
        .named_steps["encoder"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
        .shape[0]
    )

    categorical_weight = 3.0
    numeric_weight = 1.0

    feature_matrix[:, :categorical_size] *= categorical_weight
    feature_matrix[:, categorical_size:] *= numeric_weight

    feature_df = pd.DataFrame(feature_matrix)

    feature_df.insert(0, "symbol", df["symbol"])

    feature_df.to_csv(ASSET_FEATURES_PATH, index=False)

    logger.info(f"Asset features saved to {ASSET_FEATURES_PATH}")


if __name__ == "__main__":
    build_asset_features()
