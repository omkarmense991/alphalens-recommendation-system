# src/data/simulate_user_events.py

import random
from datetime import datetime, timedelta

import pandas as pd

from src.config.settings import RAW_DATA_DIR, ASSETS_MASTER_PATH
from src.utils.logger import logger

EVENT_WEIGHTS = {
    "view": 1,
    "search": 2,
    "watchlist_add": 4,
    "recommendation_click": 5,
    "buy": 6,
    "sell": -2,
}


def simulate_user_events(
    num_users: int = 50,
    events_per_user: int = 20,
):
    assets_df = pd.read_csv(ASSETS_MASTER_PATH)

    symbols = assets_df["symbol"].tolist()
    sectors = assets_df.set_index("symbol")["sector"].to_dict()

    users = [f"user_{i}" for i in range(1, num_users + 1)]

    events = []

    for user_id in users:
        preferred_sector = random.choice(list(set(sectors.values())))

        preferred_symbols = [
            symbol for symbol in symbols if sectors[symbol] == preferred_sector
        ]

        for _ in range(events_per_user):
            if random.random() < 0.7 and preferred_symbols:
                symbol = random.choice(preferred_symbols)
            else:
                symbol = random.choice(symbols)

            event_type = random.choices(
                population=list(EVENT_WEIGHTS.keys()),
                weights=[35, 20, 15, 10, 15, 5],
                k=1,
            )[0]

            event_time = datetime.now() - timedelta(
                days=random.randint(0, 90),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )

            events.append(
                {
                    "user_id": user_id,
                    "symbol": symbol,
                    "event_type": event_type,
                    "event_weight": EVENT_WEIGHTS[event_type],
                    "event_time": event_time.isoformat(),
                }
            )

    events_df = pd.DataFrame(events)

    output_path = RAW_DATA_DIR / "user_events.csv"
    events_df.to_csv(output_path, index=False)

    logger.info(f"User events saved to {output_path}")


if __name__ == "__main__":
    simulate_user_events()
