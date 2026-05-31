# src/api/main.py

from fastapi import FastAPI, HTTPException, Query

from src.recommender.content_recommender import ContentBasedRecommender

from src.api.routes import router

app = FastAPI(
    title="AlphaLens Recommendation System",
    description="Production-grade personalized investment recommendation system.",
    version="0.1.0",
)

app.include_router(router)