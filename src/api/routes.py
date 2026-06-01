# src/api/routes.py

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    HealthResponse,
    SimilarAssetsResponse,
    UserRecommendationsResponse,
)

from src.recommender.content_recommender import ContentBasedRecommender
from src.recommender.item_collaborative_filtering import (
    ItemCollaborativeFilteringRecommender,
)
from src.recommender.user_collaborative_filtering import (
    UserCollaborativeFilteringRecommender,
)
from src.recommender.hybrid_collaborative_recommender import (
    HybridCollaborativeRecommender,
)

router = APIRouter()

content_recommender = ContentBasedRecommender()
item_cf_recommender = ItemCollaborativeFilteringRecommender()
user_cf_recommender = UserCollaborativeFilteringRecommender()
hybrid_cf_recommender = HybridCollaborativeRecommender()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}


@router.get("/similar-assets/{symbol}", response_model=SimilarAssetsResponse)
def get_similar_assets(
    symbol: str,
    top_k: int = Query(default=5, ge=1, le=20),
    min_score: float = Query(default=0.25, ge=-1.0, le=1.0),
):
    try:
        recommendations = content_recommender.find_similar_assets(
            symbol=symbol,
            top_k=top_k,
            min_score=min_score,
        )

        return {
            "source_symbol": symbol,
            "count": len(recommendations),
            "recommendations": recommendations,
        }

    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/recommendations/{user_id}")
def get_user_recommendations(
    user_id: str,
    method: str = Query(default="item", pattern="^(item|user|hybrid)$"),
    top_k: int = Query(default=5, ge=1, le=20),
):
    try:
        if method == "item":
            recommendations = item_cf_recommender.recommend_for_user(
                user_id=user_id,
                top_k=top_k,
            )
        elif method == "user":
            recommendations = user_cf_recommender.recommend_for_user(
                user_id=user_id,
                top_k=top_k,
            )
        else:
            recommendations = hybrid_cf_recommender.recommend_for_user(
                user_id=user_id,
                top_k=top_k,
            )

        return {
            "user_id": user_id,
            "method": method,
            "count": len(recommendations),
            "recommendations": recommendations,
        }

    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
