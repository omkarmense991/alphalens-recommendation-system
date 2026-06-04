from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_similar_assets_endpoint():
    response = client.get(
        "/similar-assets/RELIANCE.NS",
        params={
            "top_k": 5,
            "same_sector_only": True,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["source_symbol"] == "RELIANCE.NS"
    assert data["count"] > 0
    assert "recommendations" in data

    first_item = data["recommendations"][0]

    assert "symbol" in first_item
    assert "similarity_score" in first_item
    assert "explanations" in first_item


def test_ranking_recommendations_endpoint():
    response = client.get(
        "/recommendations/user_1",
        params={
            "method": "ranking",
            "top_k": 5,
            "item_weight": 0.45,
            "user_weight": 0.45,
            "popularity_weight": 0.10,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == "user_1"
    assert data["method"] == "ranking"
    assert data["count"] > 0

    first_item = data["recommendations"][0]

    assert "item_cf_score" in first_item
    assert "user_cf_score" in first_item
    assert "popularity_score" in first_item
    assert "ranking_score" in first_item


def test_ranking_weights_must_sum_to_one():
    response = client.get(
        "/recommendations/user_1",
        params={
            "method": "ranking",
            "item_weight": 0.5,
            "user_weight": 0.5,
            "popularity_weight": 0.5,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Ranking weights must sum to 1.0"
