# Movie Recommendation System
**Cognevance Technologies — Level 1 (Easy)**

A beginner-friendly collaborative-filtering movie recommender built with Python, pandas, scikit-learn, and Streamlit. It recommends movies using user rating patterns and includes rating/popularity visualizations.

## Features
- Loads MovieLens `ratings.csv` and `movies.csv`
- Validates and cleans ratings
- Popular-movie chart and genre summary
- User-based collaborative filtering with cosine similarity
- Fallback recommendations for unknown users
- Streamlit UI for exploring users and recommendations

## Requirements
- Python 3.10+
- MovieLens small dataset from GroupLens: https://grouplens.org/datasets/movielens/latest/
- Extract the archive and place `ratings.csv` and `movies.csv` in `data/ml-latest-small/`

The dataset is not redistributed in this repository. Download it from the official GroupLens source.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Workflow
1. Load and clean movie and rating data.
2. Build a user–movie rating matrix.
3. Compute similarity between users with cosine similarity.
4. Recommend highly rated movies from similar users that the selected user has not rated.
5. Display popular movies and genre counts.

## Project structure
```text
.
├── app.py
├── recommender.py
├── requirements.txt
├── data/
│   └── ml-latest-small/   # place downloaded CSV files here
└── reports/
    └── project_report.md
```

## Limitations and next steps
This is a learning/demo recommender, not a production recommendation engine. User-based collaborative filtering can be sparse for users with few ratings. Next steps: add item-based filtering, offline evaluation (precision@k/recall@k), and model persistence.

## Author
Rupesh — B.Tech CSE | AI/ML learning projects
