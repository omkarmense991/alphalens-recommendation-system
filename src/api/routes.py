# src/api/routes.py

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    HealthResponse,
    SimilarAssetsResponse,
    UserEventRequest,
    UserEventResponse,
)

from src.database.user_event_repository import save_user_event

from src.recommender.ranking_recommender import RankingRecommender

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

from src.recommender.matrix_factorization import MatrixFactorizationRecommender

from src.retrieval.embedding_retriever import EmbeddingRetriever

from src.database.recommendation_logger import log_recommendations

router = APIRouter()

content_recommender = ContentBasedRecommender()
item_cf_recommender = ItemCollaborativeFilteringRecommender()
user_cf_recommender = UserCollaborativeFilteringRecommender()
hybrid_cf_recommender = HybridCollaborativeRecommender()
mf_recommender = MatrixFactorizationRecommender()
embedding_retriever = EmbeddingRetriever(
    n_factors=20,
    n_neighbors=30,
)


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}


@router.get("/similar-assets/{symbol}", response_model=SimilarAssetsResponse)
def get_similar_assets(
    symbol: str,
    top_k: int = Query(default=5, ge=1, le=20),
    min_score: float = Query(default=0.25, ge=-1.0, le=1.0),
    same_sector_only: bool = Query(default=False),
):
    try:
        recommendations = content_recommender.find_similar_assets(
            symbol=symbol,
            top_k=top_k,
            min_score=min_score,
            same_sector_only=same_sector_only,
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
    method: str = Query(default="item", pattern="^(item|user|hybrid|ranking|mf)$"),
    top_k: int = Query(default=5, ge=1, le=20),
    item_weight: float = Query(default=0.25, ge=0.0, le=1.0),
    user_weight: float = Query(default=0.25, ge=0.0, le=1.0),
    embedding_weight: float = Query(default=0.20, ge=0.0, le=1.0),
    mf_weight: float = Query(default=0.20, ge=0.0, le=1.0),
    popularity_weight: float = Query(default=0.10, ge=0.0, le=1.0),
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
        elif method == "hybrid":
            recommendations = hybrid_cf_recommender.recommend_for_user(
                user_id=user_id,
                top_k=top_k,
            )
        elif method == "mf":
            recommendations = mf_recommender.recommend_for_user(
                user_id=user_id,
                top_k=top_k,
            )
        else:
            total_weight = (
                item_weight
                + user_weight
                + embedding_weight
                + mf_weight
                + popularity_weight
            )

            if abs(total_weight - 1.0) > 1e-6:
                raise HTTPException(
                    status_code=400,
                    detail="Ranking weights must sum to 1.0",
                )

            dynamic_ranking_recommender = RankingRecommender(
                item_cf_weight=item_weight,
                user_cf_weight=user_weight,
                embedding_weight=embedding_weight,
                mf_weight=mf_weight,
                popularity_weight=popularity_weight,
            )

            recommendations = dynamic_ranking_recommender.recommend_for_user(
                user_id=user_id,
                top_k=top_k,
                candidate_pool_size=20,
            )

        log_recommendations(
            user_id=user_id,
            method=method,
            recommendations=recommendations,
        )
        return {
            "user_id": user_id,
            "method": method,
            "count": len(recommendations),
            "recommendations": recommendations,
        }

    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/embedding/similar-assets/{symbol}")
def get_embedding_similar_assets(
    symbol: str,
    top_k: int = Query(default=5, ge=1, le=20),
):
    try:
        results = embedding_retriever.retrieve_similar_assets(
            symbol=symbol,
            top_k=top_k,
        )

        return {
            "source_symbol": symbol,
            "method": "embedding_retrieval",
            "count": len(results),
            "recommendations": results,
        }

    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/embedding/recommendations/{user_id}")
def get_embedding_user_recommendations(
    user_id: str,
    top_k: int = Query(default=5, ge=1, le=20),
):
    try:
        results = embedding_retriever.retrieve_for_user(
            user_id=user_id,
            top_k=top_k,
        )

        return {
            "user_id": user_id,
            "method": "embedding_retrieval",
            "count": len(results),
            "recommendations": results,
        }

    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.post("/user-events", response_model=UserEventResponse)
def create_user_event(event: UserEventRequest):
    allowed_events = {
        "view",
        "search",
        "watchlist_add",
        "recommendation_click",
        "buy",
        "sell",
    }

    if event.event_type not in allowed_events:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type. Allowed values: {sorted(allowed_events)}",
        )

    save_user_event(
        user_id=event.user_id,
        symbol=event.symbol,
        event_type=event.event_type,
        event_weight=event.event_weight,
        event_time=event.event_time,
    )

    return {
        "status": "saved",
        "user_id": event.user_id,
        "symbol": event.symbol,
        "event_type": event.event_type,
    }
