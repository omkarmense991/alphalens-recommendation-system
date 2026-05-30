# src/explainability/content_explainer.py

import pandas as pd

from src.config.settings import ASSETS_MASTER_PATH


class ContentRecommendationExplainer:
    def __init__(self):
        self.assets_df = pd.read_csv(ASSETS_MASTER_PATH)

    def explain(self, source_symbol: str, recommended_symbol: str) -> list[str]:
        source = self._get_asset(source_symbol)
        recommended = self._get_asset(recommended_symbol)

        explanations = []

        if source["sector"] == recommended["sector"]:
            explanations.append(f"Both assets belong to the {source['sector']} sector.")

        if source["industry"] == recommended["industry"]:
            explanations.append(
                f"Both assets are from the {source['industry']} industry."
            )

        if self._is_similar(
            source["market_cap"], recommended["market_cap"], tolerance=0.50
        ):
            explanations.append("They have a similar market capitalization profile.")

        if self._is_similar(
            source["pe_ratio"], recommended["pe_ratio"], tolerance=0.35
        ):
            explanations.append(
                "They have a similar valuation profile based on PE ratio."
            )

        if self._is_similar(
            source["volatility"], recommended["volatility"], tolerance=0.30
        ):
            explanations.append(
                "They have a similar risk profile based on historical volatility."
            )

        if not explanations:
            explanations.append(
                "This asset has the closest overall feature similarity based on available data."
            )

        return explanations

    def _get_asset(self, symbol: str):
        rows = self.assets_df[self.assets_df["symbol"] == symbol]

        if rows.empty:
            raise ValueError(f"Symbol {symbol} not found in assets master")

        return rows.iloc[0]

    @staticmethod
    def _is_similar(value_1, value_2, tolerance: float) -> bool:
        if pd.isna(value_1) or pd.isna(value_2):
            return False

        if value_1 == 0:
            return False

        relative_difference = abs(value_1 - value_2) / abs(value_1)

        return relative_difference <= tolerance
