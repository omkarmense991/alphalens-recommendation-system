# src/api/routes.py
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import HealthResponse, SimilarAssetsResponse
from src.recommender.content_recommender import ContentBasedRecommender


router = APIRouter()

recommender = ContentBasedRecommender()


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
        recommendations = recommender.find_similar_assets(
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