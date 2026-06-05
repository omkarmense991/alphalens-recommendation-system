# src/database/recommendation_logger.py

import json

from sqlalchemy.orm import Session

from src.database.models import RecommendationLog
from src.database.session import SessionLocal
from src.utils.logger import logger


def log_recommendations(
    user_id: str,
    method: str,
    recommendations: list[dict],
):
    db: Session = SessionLocal()

    try:
        log_entry = RecommendationLog(
            user_id=user_id,
            method=method,
            recommendations_json=json.dumps(recommendations),
        )

        db.add(log_entry)
        db.commit()

        logger.info(
            f"Logged {len(recommendations)} recommendations for user={user_id}, method={method}"
        )

    finally:
        db.close()
