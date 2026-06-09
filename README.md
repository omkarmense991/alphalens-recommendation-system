# AlphaLens Recommendation System

Production-grade investment recommendation system featuring retrieval, ranking, embeddings, vector search, neural retrieval, experiment tracking, and API serving.

AlphaLens is not a stock price prediction system.

Its objective is to help users discover relevant investment opportunities using asset similarity, collaborative filtering, latent embeddings, vector retrieval, and learning-to-rank techniques.

---

# Overview

AlphaLens was built to simulate how modern recommendation systems are designed and deployed in production environments.

The project implements multiple retrieval strategies, compares them using offline evaluation metrics, tracks experiments with MLflow, persists recommendation data in PostgreSQL, and exposes recommendations through FastAPI.

The system demonstrates the evolution of recommender systems from traditional similarity-based methods to latent factor models and neural retrieval architectures.

---

# System Architecture

```text
                    Asset Universe
                           |
                           v
                  Market Data Pipeline
                           |
                           v
                 Asset Feature Engineering
                           |
                           v
                 Asset Feature Store
                           |
        +------------------+------------------+
        |                                     |
        v                                     v

 Content-Based Retrieval           Similar Asset Search
```

```text
                 User Interaction Events
     (view, search, watchlist, click, buy, sell)
                           |
                           v
                  User-Item Matrix
                           |
        +---------+--------+---------+---------+
        |          |                  |        |
        v          v                  v        v

     Item CF    User CF    Matrix Factorization
                                       |
                                       v
                              Latent Embeddings
                                       |
                      +----------------+----------------+
                      |                                 |
                      v                                 v

            Embedding Retrieval              FAISS Retrieval
                      |                                 |
                      +---------------+-----------------+
                                      |
                                      v

                           Candidate Generation
                                      |
                                      v

                            Learning-to-Rank
                                      |
                                      v

                          Final Recommendations
```

---

# Key Features

### Content-Based Recommendation

* Asset similarity search
* Weighted feature vectors
* Explainable recommendations
* Same-sector filtering
* Cosine similarity retrieval

### Collaborative Filtering

* Item-based collaborative filtering
* User-based collaborative filtering
* Hybrid collaborative filtering

### Matrix Factorization

* Latent factor recommendation model
* User embeddings
* Asset embeddings
* Personalized recommendations

### Embedding Retrieval

* Embedding-based candidate generation
* User-to-item retrieval
* Asset-to-asset retrieval

### FAISS Vector Search

* Approximate nearest neighbor retrieval
* Vector similarity search
* Scalable embedding retrieval

### Neural Retrieval

* Two-Tower neural recommendation model
* User embedding tower
* Asset embedding tower
* Dot-product retrieval

### Learning-to-Rank

Combines multiple recommendation signals:

* Item CF
* User CF
* Matrix Factorization
* Embedding Retrieval
* Popularity Signals

Produces final ranked recommendations.

### Experiment Tracking

* MLflow experiment tracking
* Offline evaluation tracking
* Model comparison
* Metric logging
* Artifact storage

### Persistence Layer

* PostgreSQL database
* Asset storage
* User event storage
* Recommendation logging

### API Serving

* FastAPI
* REST endpoints
* Recommendation serving
* User event ingestion

### Containerization

* Docker
* Docker Compose
* PostgreSQL container
* API container

---

# Recommendation Methods

## 1. Content-Based Recommendation

Assets are represented using:

* Sector
* Industry
* Market Capitalization
* PE Ratio
* PB Ratio
* Dividend Yield
* Volatility
* Historical Returns
* Trading Volume

Recommendations are generated using cosine similarity over engineered feature vectors.

---

## 2. Item-Based Collaborative Filtering

Learns asset-to-asset similarity from user behavior.

If users frequently interact with the same assets, those assets become similar.

Example:

```text
Users who interacted with RELIANCE
also interacted with ONGC
```

---

## 3. User-Based Collaborative Filtering

Learns user-to-user similarity.

If two users behave similarly, assets preferred by one user can be recommended to the other.

---

## 4. Hybrid Collaborative Filtering

Combines:

```text
Item CF Score
+
User CF Score
```

to create a balanced recommendation strategy.

---

## 5. Matrix Factorization

Factorizes the user-item interaction matrix into:

```text
User Latent Factors
Item Latent Factors
```

This captures hidden behavioral patterns and generates personalized recommendations.

---

## 6. Embedding Retrieval

Uses latent asset and user embeddings generated by matrix factorization.

Supports:

* Similar asset retrieval
* User recommendation retrieval

---

## 7. FAISS Retrieval

Uses Facebook AI Similarity Search (FAISS) for efficient nearest-neighbor retrieval over embedding vectors.

Supports scalable vector search.

---

## 8. Two-Tower Neural Retrieval

A neural recommendation architecture consisting of:

```text
User Tower
Asset Tower
```

Both towers learn embeddings in a shared latent space.

Recommendations are generated using embedding similarity.

---

## 9. Learning-to-Rank

Final ranking model combines:

```text
Item CF
User CF
Embedding Retrieval
Matrix Factorization
Popularity Signals
```

into a unified recommendation score.

---

# Data Pipeline

Run complete pipeline:

```bash
python -m src.pipelines.run_full_pipeline
```

Pipeline stages:

```text
Asset Universe
→ Market Data Collection
→ Feature Engineering
→ User Event Simulation
→ User-Item Matrix Construction
→ Train/Test Split
→ Recommender Training
→ Offline Evaluation
→ MLflow Tracking
```

Individual pipelines:

```bash
python -m src.pipelines.run_feature_pipeline
python -m src.pipelines.run_collaborative_pipeline
python -m src.pipelines.run_evaluation_pipeline
```

---

# API Endpoints

Start API:

```bash
uvicorn src.api.main:app --reload
```

---

## Health Check

```http
GET /health
```

Example:

```bash
curl "http://127.0.0.1:8000/health"
```

---

## Similar Assets

```http
GET /similar-assets/{symbol}
```

Example:

```bash
curl "http://127.0.0.1:8000/similar-assets/RELIANCE.NS"
```

---

## Personalized Recommendations

```http
GET /recommendations/{user_id}
```

Methods:

```text
item
user
hybrid
ranking
mf
two_tower
```

Examples:

```bash
curl "http://127.0.0.1:8000/recommendations/user_1?method=item"
curl "http://127.0.0.1:8000/recommendations/user_1?method=user"
curl "http://127.0.0.1:8000/recommendations/user_1?method=hybrid"
curl "http://127.0.0.1:8000/recommendations/user_1?method=ranking"
curl "http://127.0.0.1:8000/recommendations/user_1?method=mf"
curl "http://127.0.0.1:8000/recommendations/user_1?method=two_tower"
```

---

## Embedding Retrieval

```http
GET /embedding/similar-assets/{symbol}
GET /embedding/recommendations/{user_id}
```

---

## FAISS Retrieval

```http
GET /faiss/similar-assets/{symbol}
GET /faiss/recommendations/{user_id}
```

---

## User Event Ingestion

```http
POST /user-events
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/user-events" \
-H "Content-Type: application/json" \
-d '{
  "user_id": "user_1",
  "symbol": "RELIANCE.NS",
  "event_type": "watchlist_add",
  "event_weight": 4
}'
```

---

# Offline Evaluation

AlphaLens uses temporal offline evaluation.

```text
Past interactions
→ Training

Future interactions
→ Evaluation
```

Metrics:

* Precision@K
* Recall@K
* Hit Rate@K
* NDCG@K
* Catalog Coverage
* Sector Diversity
* Popularity Bias

Artifacts:

```text
evaluation_results.csv
evaluation_summary.csv
```

---

# Evaluation Results

Best observed evaluation results:

| Model                | NDCG@5 |
| -------------------- | ------ |
| Matrix Factorization | 0.0836 |
| Learning-to-Rank     | 0.0727 |
| Hybrid CF            | 0.0659 |
| Item CF              | 0.0610 |
| User CF              | 0.0567 |
| Two-Tower Retrieval  | 0.0425 |

Matrix Factorization achieved the strongest ranking quality on the current dataset.

The Two-Tower model serves as an experimental neural retrieval baseline and can be improved using richer user and asset features.

---

# MLflow Experiment Tracking

Experiment Name:

```text
AlphaLens Recommendation Evaluation
```

Tracks:

* Evaluation metrics
* Model comparisons
* Offline experiments
* Artifacts

Launch UI:

```bash
mlflow ui
```

---

# Database Layer

Database:

```text
PostgreSQL
```

Stores:

* Assets
* User Events
* Recommendation Logs

Initialize database:

```bash
python -m src.database.init_db
```

Load assets:

```bash
python -m src.database.load_assets
```

---

# Docker Deployment

Build and run:

```bash
docker compose up --build -d
```

Check services:

```bash
docker compose ps
```

Stop:

```bash
docker compose down
```

---

# Repository Structure

```text
src/
├── api/
├── candidates/
├── database/
├── deep_learning/
├── evaluation/
├── feature_engineering/
├── pipelines/
├── ranking/
├── recommender/
├── retrieval/
├── simulation/
└── utils/
```

---

# Project Highlights

This project demonstrates:

* Content-Based Recommendation
* Collaborative Filtering
* Matrix Factorization
* Embedding Retrieval
* FAISS Vector Search
* Neural Retrieval (Two-Tower)
* Learning-to-Rank
* Offline Evaluation
* MLflow Experiment Tracking
* PostgreSQL Persistence
* FastAPI Serving
* Docker Deployment

The architecture mirrors production recommendation systems used by modern technology companies, adapted to the stock recommendation domain.

---

# Future Work

* Real user interaction ingestion
* Online A/B testing
* Feature-rich Two-Tower architecture
* Automated retraining pipelines
* Recommendation monitoring dashboards
* Model registry and deployment workflows

---

# Project Goal

The goal of AlphaLens is not to predict stock prices.

The objective is to build a production-style recommendation platform capable of helping users discover relevant investment opportunities through retrieval, ranking, embeddings, and recommendation system engineering.
