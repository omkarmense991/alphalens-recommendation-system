# src/database/user_event_repository.py

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.database.models import UserEvent
from src.database.session import SessionLocal


def save_user_event(
    user_id: str,
    symbol: str,
    event_type: str,
    event_weight: float,
    event_time=None,
):
    db: Session = SessionLocal()

    try:
        event = UserEvent(
            user_id=user_id,
            symbol=symbol,
            event_type=event_type,
            event_weight=event_weight,
            event_time=event_time or datetime.now(timezone.utc),
        )

        db.add(event)
        db.commit()

        return event

    finally:
        db.close()