from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data" / "ml-latest-small"

def load_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    ratings_path, movies_path = data_dir / "ratings.csv", data_dir / "movies.csv"
    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            "Dataset files missing. Download MovieLens small from "
            "https://grouplens.org/datasets/movielens/latest/ and place ratings.csv "
            "and movies.csv in data/ml-latest-small/."
        )
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings = ratings[ratings["rating"].between(0.5, 5.0)]
    movies = movies.dropna(subset=["movieId", "title"]).drop_duplicates("movieId")
    return ratings, movies

def popular_movies(ratings, movies, min_ratings=20, limit=10):
    stats = ratings.groupby("movieId").agg(mean_rating=("rating", "mean"),
                                          rating_count=("rating", "count")).reset_index()
    stats = stats[stats["rating_count"] >= min_ratings]
    return stats.merge(movies, on="movieId", how="left").sort_values(
        ["mean_rating", "rating_count"], ascending=False
    ).head(limit)

def recommend_for_user(user_id, ratings, movies, limit=10, neighbors=15):
    if ratings.empty:
        return pd.DataFrame(columns=["title", "score", "rating_count"])
    matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    if user_id not in matrix.index:
        return popular_movies(ratings, movies, min_ratings=1, limit=limit).rename(
            columns={"mean_rating": "score"}
        )[["title", "score", "rating_count"]]

    # Mean-center ratings per user to reduce individual rating-scale bias.
    means = matrix.mean(axis=1)
    centered = matrix.sub(means, axis=0).fillna(0)
    similarities = cosine_similarity(centered)
    sim = pd.Series(similarities[matrix.index.get_loc(user_id)], index=matrix.index)
    sim = sim.drop(index=user_id).sort_values(ascending=False).head(neighbors)
    user_rated = set(matrix.loc[user_id].dropna().index)
    candidate_ids = [mid for mid in matrix.columns if mid not in user_rated]
    scores = {}
    for movie_id in candidate_ids:
        vals = matrix.loc[sim.index, movie_id]
        valid = vals.notna()
        if not valid.any():
            continue
        weights = sim[valid]
        denom = weights.abs().sum()
        if denom == 0:
            continue
        scores[movie_id] = float((vals[valid] * weights).sum() / denom)
    if not scores:
        return popular_movies(ratings, movies, min_ratings=1, limit=limit).rename(
            columns={"mean_rating": "score"}
        )[["title", "score", "rating_count"]]
    result = pd.DataFrame({"movieId": list(scores), "score": list(scores.values())})
    counts = ratings.groupby("movieId").size().rename("rating_count")
    result = result.merge(movies, on="movieId", how="left").join(counts, on="movieId")
    return result.sort_values("score", ascending=False).head(limit)[
        ["title", "score", "rating_count"]
    ].reset_index(drop=True)
