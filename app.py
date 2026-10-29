import streamlit as st
import matplotlib.pyplot as plt
from recommender import load_data, popular_movies, recommend_for_user

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")
st.caption("Collaborative filtering demo using MovieLens ratings.")

try:
    ratings, movies = load_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

ratings_tab, explore_tab = st.tabs(["Personalized recommendations", "Explore data"])
with ratings_tab:
    users = sorted(ratings["userId"].unique().tolist())
    user_id = st.selectbox("Choose a MovieLens user ID", users)
    n = st.slider("Number of recommendations", 5, 20, 10)
    if st.button("Recommend movies", type="primary"):
        recs = recommend_for_user(int(user_id), ratings, movies, limit=n)
        st.subheader(f"Recommended for user {user_id}")
        st.dataframe(recs, use_container_width=True, hide_index=True)
with explore_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Most-rated movies")
        pop = ratings.groupby("movieId").size().sort_values(ascending=False).head(10)
        plot = pop.rename("ratings").reset_index().merge(movies, on="movieId")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(plot["title"].str.slice(0, 35)[::-1], plot["ratings"][::-1])
        ax.set_xlabel("Number of ratings")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Movie genres")
        genre_counts = movies.assign(genres=movies["genres"].str.split("|")).explode("genres")
        genre_counts = genre_counts[genre_counts["genres"] != "(no genres listed)"]["genres"].value_counts().head(12)
        fig, ax = plt.subplots(figsize=(9, 5))
        genre_counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Number of movies")
        st.pyplot(fig)
        plt.close(fig)

st.divider()
st.caption("Educational project. Recommendations depend on the selected dataset and user history.")
