import pandas as pd

from src.config.settings import RAW_DATA_DIR
from src.utils.logger import logger

SOURCE_UNIVERSE_PATH = RAW_DATA_DIR / "nifty_universe.csv"
ASSET_UNIVERSE_PATH = RAW_DATA_DIR / "asset_universe.csv"

REQUIRED_COLUMNS = [
    "symbol",
    "company_name",
    "sector",
    "industry",
]


def create_asset_universe():
    universe_df = pd.read_csv(SOURCE_UNIVERSE_PATH)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in universe_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in universe file: {missing_columns}"
        )

    universe_df = universe_df[REQUIRED_COLUMNS].drop_duplicates(subset=["symbol"])

    universe_df["symbol"] = universe_df["symbol"].astype(str)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    universe_df.to_csv(ASSET_UNIVERSE_PATH, index=False)

    logger.info(f"Asset universe saved to {ASSET_UNIVERSE_PATH}")
    logger.info(f"Asset universe shape: {universe_df.shape}")


if __name__ == "__main__":
    create_asset_universe()
