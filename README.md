# AlphaLens Recommendation System

Production-grade personalized investment recommendation system featuring retrieval, ranking, embeddings, and ML-powered asset recommendations.

AlphaLens is not a stock price prediction system. It is a personalized asset discovery platform that recommends relevant stocks based on asset similarity, user behavior, collaborative filtering, and hybrid recommendation strategies.

## Current Features

* Content-based asset similarity
* Weighted asset feature vectors
* Similar asset retrieval using cosine similarity
* Same-sector recommendation filtering
* Explainable recommendations
* Simulated user interaction events
* User-item interaction matrix
* Item-based collaborative filtering
* User-based collaborative filtering
* Hybrid collaborative recommender
* FastAPI recommendation endpoints
* Offline evaluation pipeline
* Precision@K, Recall@K, Hit Rate@K, NDCG@K
* Catalog coverage, sector diversity, and popularity bias metrics
* Reproducible end-to-end ML pipeline

## Recommendation Methods

### Content-Based Recommendation

Represents each asset using metadata and market features such as:

* sector
* industry
* market capitalization
* PE ratio
* PB ratio
* dividend yield
* volatility
* returns
* trading volume

Similar assets are retrieved using cosine similarity over weighted feature vectors.

### Collaborative Filtering

User behavior is simulated using events such as:

* view
* search
* watchlist add
* recommendation click
* buy
* sell

These events are converted into a user-item interaction matrix.

Implemented collaborative filtering methods:

* item-based collaborative filtering
* user-based collaborative filtering
* hybrid collaborative filtering

### Hybrid Recommendation

The hybrid recommender combines normalized item-CF and user-CF scores to create a balanced recommendation ranking.

## Project Pipeline

Run the full pipeline:

```bash
python -m src.pipelines.run_full_pipeline
```

This executes:

```text
asset universe creation
→ market data extraction
→ asset feature engineering
→ user event simulation
→ interaction matrix creation
→ train/test split
→ recommender evaluation
→ metric artifact generation
```

Individual pipelines:

```bash
python -m src.pipelines.run_feature_pipeline
python -m src.pipelines.run_collaborative_pipeline
python -m src.pipelines.run_evaluation_pipeline
```

## API Endpoints

Start API:

```bash
uvicorn src.api.main:app --reload
```

Health check:

```http
GET /health
```

Similar assets:

```http
GET /similar-assets/{symbol}
```

Example:

```bash
curl "http://127.0.0.1:8000/similar-assets/RELIANCE.NS?top_k=5&same_sector_only=true"
```

Personalized recommendations:

```http
GET /recommendations/{user_id}
```

Examples:

```bash
curl "http://127.0.0.1:8000/recommendations/user_1?method=item&top_k=5"
curl "http://127.0.0.1:8000/recommendations/user_1?method=user&top_k=5"
curl "http://127.0.0.1:8000/recommendations/user_1?method=hybrid&top_k=5"
```

## Evaluation

AlphaLens uses a temporal offline evaluation strategy.

Past user interactions are used for training, and future interactions are used as test data.

Evaluation metrics:

* Precision@K
* Recall@K
* Hit Rate@K
* NDCG@K
* Sector Diversity@K
* Catalog Coverage
* Average Recommendation Popularity

Generated artifacts:

```text
data/processed/evaluation_results.csv
data/processed/evaluation_summary.csv
```

## Current Architecture

```text
Raw Asset Universe
        ↓
Market Data Fetching
        ↓
Asset Master Dataset
        ↓
Feature Engineering
        ↓
Content-Based Retrieval
        ↓
Similar Asset API
```

```text
User Events
        ↓
Interaction Matrix
        ↓
Item CF / User CF
        ↓
Hybrid CF
        ↓
Recommendation API
        ↓
Offline Evaluation
```

## Tech Stack

* Python
* Pandas
* Scikit-learn
* FastAPI
* Pydantic
* yfinance
* pytest

## Future Roadmap

* Replace hardcoded asset universe with NSE/NIFTY data source
* Add PostgreSQL persistence
* Add recommendation logging
* Add watchlist APIs
* Add candidate generation and ranking layer
* Add content + collaborative hybrid ranking
* Add MLflow experiment tracking
* Add model/recommender registry
* Add matrix factorization
* Add neural collaborative filtering
* Add two-tower retrieval model
* Add Docker deployment
* Add monitoring dashboard

## Project Highlights

AlphaLens is a production-style recommendation system for personalized investment discovery.

The project demonstrates:

* recommendation system fundamentals
* retrieval and ranking architecture
* collaborative filtering
* hybrid recommendation
* feature engineering
* offline evaluation
* recommendation quality monitoring
* API serving
* reproducible ML pipelines

The goal is not to predict stock prices, but to recommend relevant assets based on user preferences, behavior patterns, and asset similarity.
