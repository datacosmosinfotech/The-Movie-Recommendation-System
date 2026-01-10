import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import urllib.parse
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

# ---------------- TMDB API KEY ----------------
TMDB_API_KEY = "6f135fe03126ac6a83ff54eafc691c22"

# ---------------- LOAD PKL ----------------
import os
import gdown
import pickle

PKL_PATH = "movie_data.pkl"

# Download only if file does not exist
if not os.path.exists(PKL_PATH):
    url = "https://drive.google.com/uc?id=1FaykR5kIP9WCbSE5VZGxZOseR2ABWcaW"
    gdown.download(url, PKL_PATH, quiet=False)

# Now safely load the file
with open(PKL_PATH, "rb") as file:
    movies, cosine_sim = pickle.load(file)

# ---------------- FETCH POSTER ----------------
@st.cache_data(show_spinner=False)
def fetch_poster(title):
    try:
        query = urllib.parse.quote(title)
        url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={TMDB_API_KEY}&query={query}"
        )
        response = requests.get(url, timeout=5)
        data = response.json()

        if data["results"]:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass

    return None

# ---------------- RECOMMEND FUNCTION ----------------
def get_recommendations_with_posters(title, top_n=5):
    idx = movies[movies["title"] == title].index[0]

    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []

    for i, score in scores[1:50]:  # take more for filtering
        movie_title = movies.iloc[i]["title"]
        poster = fetch_poster(movie_title)

        if poster:
            results.append((movie_title, poster))

        if len(results) == top_n:
            break

    return results

# ---------------- STREAMLIT UI ----------------
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


