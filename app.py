import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import urllib.parse

# =========================================================
# 🔑 PUT YOUR TMDB API KEY HERE
# =========================================================
TMDB_API_KEY = "6f135fe03126ac6a83ff54eafc691c22"

# ---------------- LOAD MOVIES & SIMILARITY ----------------
import os

PKL_URL = "https://drive.google.com/uc?id=1FaykR5kIP9WcbSE5VZGxZOseR2ABWcaw"
PKL_FILE = "movie_data.pkl"

@st.cache_resource
def load_pkl_from_url():
    if not os.path.exists(PKL_FILE):
        st.info("Downloading model file... please wait ⏳")
        response = requests.get(PKL_URL)
        response.raise_for_status()
        with open(PKL_FILE, "wb") as f:
            f.write(response.content)

    with open(PKL_FILE, "rb") as file:
        return pickle.load(file)

movies, cosine_sim = load_pkl_from_url()

movies = movies.dropna(subset=["title"]).reset_index(drop=True)

# ---------------- FETCH POSTER ----------------
@st.cache_data(show_spinner=False)
def fetch_poster(title):
    try:
        query = urllib.parse.quote(title)
        url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={TMDB_API_KEY}&query={query}"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            for movie in data["results"]:
                poster_path = movie.get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"

        return None

    except Exception:
        return None

# ---------------- RECOMMEND ONLY MOVIES WITH POSTERS ----------------
def get_recommendations_with_posters(title, top_n=5):
    idx = movies[movies["title"] == title].index[0]

    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []

    # Take more candidates so filtering is possible
    for i, score in scores[1:50]:
        movie_title = movies.iloc[i]["title"]
        poster = fetch_poster(movie_title)

        if poster:
            results.append((movie_title, poster))

        if len(results) == top_n:
            break

    return results

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie you like:",
    movies["title"].values
)

if st.button("Recommend"):
    recommendations = get_recommendations_with_posters(selected_movie, top_n=5)

    if not recommendations:
        st.warning("No recommended movies with posters found.")
    else:
        st.subheader("Top 5 Recommended Movies (With Posters)")

        cols = st.columns(5)

        for col, (title, poster_url) in zip(cols, recommendations):
            with col:
                st.image(poster_url, use_container_width=True)

                st.caption(title)
