# Project Report: Movie Recommendation System

## Objective
Recommend movies to users based on rating patterns from similar users.

## Dataset
Use the MovieLens small dataset from GroupLens. The application expects `ratings.csv` and `movies.csv` under `data/ml-latest-small/`. The data is intentionally not bundled; download it from the official provider.

## Method
1. Remove rows with missing user, movie, or rating values.
2. Restrict ratings to the dataset's expected 0.5–5.0 range.
3. Create a user–movie matrix and mean-center ratings per user.
4. Compute cosine similarity between users.
5. Score unseen movies using ratings from the most similar users.
6. Fall back to popular movies for a new user or sparse results.

## Outputs
- Table of recommended movie titles and scores.
- Most-rated movie chart.
- Genre distribution chart.

## Evaluation and limitations
This starter implementation is functionally demonstrable but does not claim a measured accuracy score. A next iteration should use a held-out test set and report ranking metrics such as Precision@K and Recall@K. Sparse user histories and popularity bias can affect results.
