import os
import pickle
import requests
import gdown
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
TMDB_API_KEY = "6f135fe03126ac6a83ff54eafc691c22"
PKL_PATH = "movie_data.pkl"
PKL_URL = "https://drive.google.com/uc?id=1FaykR5kIP9WCbSE5VZGxZOseR2ABWcaW"

# =========================================================
# LOAD PKL (DOWNLOAD ONCE)
# =========================================================
@st.cache_resource
def load_data():
    if not os.path.exists(PKL_PATH):
        gdown.download(PKL_URL, PKL_PATH, quiet=False)

    with open(PKL_PATH, "rb") as file:
        movies, cosine_sim = pickle.load(file)

    return movies, cosine_sim


movies, cosine_sim = load_data()

# =========================================================
# FETCH MOVIE POSTER
# =========================================================
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return None

    except Exception:
        return None


# =========================================================
# RECOMMENDATION LOGIC (SKIP MOVIES WITHOUT POSTERS)
# =========================================================
def recommend(movie_title, n=5):
    idx = movies[movies["title"] == movie_title].index[0]
    distances = cosine_sim[idx]

    sorted_movies = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )

    recommendations = []

    for i in sorted_movies[1:]:  # skip the selected movie
        movie_id = movies.iloc[i[0]].movie_id
        title = movies.iloc[i[0]].title

        poster = fetch_poster(movie_id)

        # ✅ ONLY KEEP MOVIES WITH POSTERS
        if poster:
            recommendations.append((title, poster))

        if len(recommendations) == n:
            break

    return recommendations


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie you like:",
    movies["title"].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("Top 5 Recommended Movies (With Posters)")

    if not recommendations:
        st.warning("No posters available for similar movies.")
    else:
        cols = st.columns(len(recommendations))
        for col, (title, poster_url) in zip(cols, recommendations):
            with col:
                st.image(poster_url, use_container_width=True)
                st.caption(title)
