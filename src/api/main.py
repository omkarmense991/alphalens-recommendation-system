# src/api/main.py

from fastapi import FastAPI, HTTPException, Query

from src.recommender.content_recommender import ContentBasedRecommender

app = FastAPI(
    title="AlphaLens Recommendation System",
    description="Production-grade personalized investment recommendation system.",
    version="0.1.0",
)

recommender = ContentBasedRecommender()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/similar-assets/{symbol}")
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
