# src/database/check_assets.py

from src.database.models import Asset
from src.database.session import SessionLocal


db = SessionLocal()

assets = db.query(Asset).limit(5).all()

for asset in assets:
    print(
        asset.symbol,
        asset.company_name,
        asset.sector,
    )

db.close()