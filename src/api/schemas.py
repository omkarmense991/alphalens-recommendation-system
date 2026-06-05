from pydantic import BaseModel
from datetime import datetime
from pydantic import BaseModel


class UserEventRequest(BaseModel):
    user_id: str
    symbol: str
    event_type: str
    event_weight: float
    event_time: datetime | None = None


class UserEventResponse(BaseModel):
    status: str
    user_id: str
    symbol: str
    event_type: str


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
    item_cf_raw_score: float


class UserRecommendationsResponse(BaseModel):
    user_id: str
    count: int
    recommendations: list[CollaborativeRecommendationItem]



