# src/database/check_recommendation_logs.py
import json

from src.database.models import RecommendationLog
from src.database.session import SessionLocal

db = SessionLocal()

logs = db.query(RecommendationLog).order_by(RecommendationLog.id.desc()).limit(5).all()

for log in logs:
    print("ID:", log.id)
    print("User:", log.user_id)
    print("Method:", log.method)
    print("Created:", log.created_at)
    print("Recommendations:", json.loads(log.recommendations_json)[:2])
    print("-" * 50)

db.close()
