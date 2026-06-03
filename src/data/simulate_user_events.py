# ==========================================================
# User Behavior Simulation Strategy
# ==========================================================
#
# This simulator intentionally generates non-random user
# preferences to create realistic interaction patterns for
# collaborative filtering experiments.
#
# Each simulated user is assigned:
#
#   1. Preferred Sector
#      Example:
#      Energy, Financial Services, IT
#
#   2. Preferred Industry within that Sector
#      Example:
#      Oil & Gas within Energy
#
# Event generation follows:
#
#   60% -> Industry-level preference
#   25% -> Sector-level preference
#   15% -> Exploration of any asset
#
# This mimics real investor behavior:
#
#   Users typically have strong interest in a specific
#   industry, broader interest in a sector, and occasional
#   exploration outside their usual interests.
#
# Example:
#
#   Preferred Sector   = Energy
#   Preferred Industry = Oil & Gas
#
#   Most interactions:
#       ONGC, IOC, BPCL, HINDPETRO
#
#   Some interactions:
#       GAIL
#
#   Few interactions:
#       Any stock from other sectors
#
# This produces meaningful user-user and item-item
# interaction patterns, making collaborative filtering
# evaluation more realistic than purely random event
# generation.
#
# ==========================================================

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
    num_users: int = 100,
    events_per_user: int = 40,
):
    assets_df = pd.read_csv(ASSETS_MASTER_PATH)

    symbols = assets_df["symbol"].tolist()
    sectors = assets_df.set_index("symbol")["sector"].to_dict()
    industries = assets_df.set_index("symbol")["industry"].to_dict()

    users = [f"user_{i}" for i in range(1, num_users + 1)]

    events = []

    for user_id in users:
        preferred_sector = random.choice(list(set(sectors.values())))

        sector_symbols = [
            symbol for symbol in symbols if sectors[symbol] == preferred_sector
        ]

        preferred_industry = random.choice(
            list(set(industries[symbol] for symbol in sector_symbols))
        )

        industry_symbols = [
            symbol
            for symbol in sector_symbols
            if industries[symbol] == preferred_industry
        ]

        for _ in range(events_per_user):
            selection_bucket = random.random()

            if selection_bucket < 0.60 and industry_symbols:
                symbol = random.choice(industry_symbols)

            elif selection_bucket < 0.85 and sector_symbols:
                symbol = random.choice(sector_symbols)

            else:
                symbol = random.choice(symbols)

            event_type = random.choices(
                population=list(EVENT_WEIGHTS.keys()),
                weights=[35, 20, 18, 12, 12, 3],
                k=1,
            )[0]

            event_time = datetime.now() - timedelta(
                days=random.randint(0, 120),
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
                    "preferred_sector": preferred_sector,
                    "preferred_industry": preferred_industry,
                }
            )

    events_df = pd.DataFrame(events)

    output_path = RAW_DATA_DIR / "user_events.csv"
    events_df.to_csv(output_path, index=False)

    logger.info(f"User events saved to {output_path}")
    logger.info(f"Events shape: {events_df.shape}")


if __name__ == "__main__":
    simulate_user_events()
