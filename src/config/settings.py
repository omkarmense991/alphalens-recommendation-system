from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ASSETS_MASTER_PATH = PROCESSED_DATA_DIR / "assets_master.csv"
ASSET_FEATURES_PATH = PROCESSED_DATA_DIR / "asset_features.csv"
ASSET_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "asset_embeddings.csv"
