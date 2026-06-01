from pydantic import BaseModel


class RecommendationItem(BaseModel):
    symbol: str
    company_name: str
    sector: str
    industry: str
    similarity_score: float
    explanations: list[str]


class SimilarAssetsResponse(BaseModel):
    source_symbol: str
    count: int
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str


class CollaborativeRecommendationItem(BaseModel):
    symbol: str
    company_name: str
    sector: str
    industry: str
    collaborative_score: float


class UserRecommendationsResponse(BaseModel):
    user_id: str
    count: int
    recommendations: list[CollaborativeRecommendationItem]
