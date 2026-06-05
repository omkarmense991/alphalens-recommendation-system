# src/ranking/learning_to_rank.py


"""
Ranking Layer

Ranks candidate assets using multiple recommendation signals.

Current signals:
- Item Collaborative Filtering score
- User Collaborative Filtering score
- Embedding Retrieval score
- Popularity score

The ranker normalizes each signal and computes a weighted
ranking score.

Purpose:
Transform a broad candidate pool into a final personalized
Top-K recommendation list.

Architecture:

Candidate Pool
    ↓
Signal Normalization
    ↓
Weighted Scoring
    ↓
Sorted Top-K Recommendations

This represents a simplified Learning-to-Rank architecture.
In a production system, the weighted scoring function could later
be replaced with a trained ranking model such as Logistic Regression,
XGBoost, LightGBM, or a neural ranking model.
"""


class WeightedRanker:
    def __init__(
        self,
        item_cf_weight: float = 0.35,
        user_cf_weight: float = 0.35,
        embedding_weight: float = 0.20,
        popularity_weight: float = 0.10,
    ):
        self.item_cf_weight = item_cf_weight
        self.user_cf_weight = user_cf_weight
        self.embedding_weight = embedding_weight
        self.popularity_weight = popularity_weight

    def rank(self, candidates: list[dict], top_k: int = 5):
        item_scores = {
            candidate["symbol"]: candidate.get("item_cf_raw_score", 0.0)
            for candidate in candidates
        }

        user_scores = {
            candidate["symbol"]: candidate.get("user_cf_raw_score", 0.0)
            for candidate in candidates
        }

        popularity_scores = {
            candidate["symbol"]: candidate.get("popularity_raw_score", 0.0)
            for candidate in candidates
        }

        embedding_scores = {
            candidate["symbol"]: candidate.get("embedding_raw_score", 0.0)
            for candidate in candidates
        }

        normalized_item_scores = self._normalize_scores(item_scores)
        normalized_user_scores = self._normalize_scores(user_scores)
        normalized_popularity_scores = self._normalize_scores(popularity_scores)
        normalized_embedding_scores = self._normalize_scores(embedding_scores)

        ranked_candidates = []

        for candidate in candidates:
            symbol = candidate["symbol"]

            item_cf_score = normalized_item_scores.get(symbol, 0.0)
            user_cf_score = normalized_user_scores.get(symbol, 0.0)
            popularity_score = normalized_popularity_scores.get(symbol, 0.0)
            embedding_score = normalized_embedding_scores.get(symbol, 0.0)

            ranking_score = (
                self.item_cf_weight * item_cf_score
                + self.user_cf_weight * user_cf_score
                + self.embedding_weight * embedding_score
                + self.popularity_weight * popularity_score
            )
            ranked_candidates.append(
                {
                    "symbol": candidate["symbol"],
                    "company_name": candidate["company_name"],
                    "sector": candidate["sector"],
                    "industry": candidate["industry"],
                    "item_cf_score": round(float(item_cf_score), 4),
                    "user_cf_score": round(float(user_cf_score), 4),
                    "popularity_score": round(float(popularity_score), 4),
                    "embedding_score": round(float(embedding_score), 4),
                    "ranking_score": round(float(ranking_score), 4),
                }
            )

        ranked_candidates = sorted(
            ranked_candidates,
            key=lambda x: x["ranking_score"],
            reverse=True,
        )

        return ranked_candidates[:top_k]

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}

        max_score = max(scores.values())

        if max_score == 0:
            return {symbol: 0.0 for symbol in scores}

        return {symbol: score / max_score for symbol, score in scores.items()}
