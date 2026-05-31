from src.recommender.content_recommender import ContentBasedRecommender


def test_find_similar_assets_returns_recommendations():
    recommender = ContentBasedRecommender()

    results = recommender.find_similar_assets(
        symbol="RELIANCE.NS",
        top_k=5,
        min_score=0.25,
    )

    assert isinstance(results, list)
    assert len(results) > 0

    first_result = results[0]

    assert "symbol" in first_result
    assert "similarity_score" in first_result
    assert "explanations" in first_result
    assert isinstance(first_result["explanations"], list)


def test_find_similar_assets_excludes_source_symbol():
    recommender = ContentBasedRecommender()

    results = recommender.find_similar_assets(
        symbol="RELIANCE.NS",
        top_k=5,
        min_score=0.25,
    )

    symbols = [item["symbol"] for item in results]

    assert "RELIANCE.NS" not in symbols
