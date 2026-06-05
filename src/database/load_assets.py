# src/database/load_assets.py

import pandas as pd

from sqlalchemy.orm import Session

from src.config.settings import ASSETS_MASTER_PATH
from src.database.models import Asset
from src.database.session import SessionLocal
from src.utils.logger import logger


def load_assets():

    assets_df = pd.read_csv(ASSETS_MASTER_PATH)

    db: Session = SessionLocal()

    try:

        db.query(Asset).delete()

        for _, row in assets_df.iterrows():

            asset = Asset(
                symbol=row["symbol"],
                company_name=row["company_name"],
                sector=row["sector"],
                industry=row["industry"],
                market_cap=row.get("market_cap"),
                pe_ratio=row.get("pe_ratio"),
                pb_ratio=row.get("pb_ratio"),
                dividend_yield=row.get("dividend_yield"),
                beta=row.get("beta"),
                volatility=row.get("volatility"),
                returns_1y=row.get("returns_1y"),
                avg_volume=row.get("avg_volume"),
                current_price=row.get("current_price"),
            )

            db.add(asset)

        db.commit()

        logger.info(f"Loaded {len(assets_df)} assets into database")

    finally:
        db.close()


if __name__ == "__main__":
    load_assets()
