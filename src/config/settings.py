from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ASSETS_MASTER_PATH = PROCESSED_DATA_DIR / "assets_master.csv"
ASSET_FEATURES_PATH = PROCESSED_DATA_DIR / "asset_features.csv"
ASSET_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "asset_embeddings.csv"

TRAIN_USER_EVENTS_PATH = PROCESSED_DATA_DIR / "train_user_events.csv"
TEST_USER_EVENTS_PATH = PROCESSED_DATA_DIR / "test_user_events.csv"

TRAIN_USER_ITEM_MATRIX_PATH = PROCESSED_DATA_DIR / "train_user_item_matrix.csv"
TEST_USER_ITEM_MATRIX_PATH = PROCESSED_DATA_DIR / "test_user_item_matrix.csv"

EVALUATION_RESULTS_PATH = PROCESSED_DATA_DIR / "evaluation_results.csv"
EVALUATION_SUMMARY_PATH = PROCESSED_DATA_DIR / "evaluation_summary.csv"


DATABASE_URL = "sqlite:///./alphalens.db"
