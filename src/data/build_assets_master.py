# src/data/build_assets_master.py

import pandas as pd
import yfinance as yf

from src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, ASSETS_MASTER_PATH
from src.utils.logger import logger


def fetch_asset_features(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    info = ticker.info

    hist = ticker.history(period="1y")

    volatility = None
    returns_1y = None

    if not hist.empty:
        daily_returns = hist["Close"].pct_change().dropna()
        volatility = daily_returns.std() * (252**0.5)
        returns_1y = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1

    return {
        "symbol": symbol,
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "pb_ratio": info.get("priceToBook"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "volatility": volatility,
        "returns_1y": returns_1y,
        "avg_volume": info.get("averageVolume"),
        "current_price": info.get("currentPrice"),
    }


def build_assets_master():
    input_path = RAW_DATA_DIR / "asset_universe.csv"
    universe_df = pd.read_csv(input_path)

    feature_rows = []

    for symbol in universe_df["symbol"]:
        logger.info(f"Fetching data for {symbol}")
        features = fetch_asset_features(symbol)
        feature_rows.append(features)

    market_df = pd.DataFrame(feature_rows)

    assets_master = universe_df.merge(market_df, on="symbol", how="left")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    assets_master.to_csv(ASSETS_MASTER_PATH, index=False)

    logger.info(f"Assets master saved to {ASSETS_MASTER_PATH}")


if __name__ == "__main__":
    build_assets_master()
