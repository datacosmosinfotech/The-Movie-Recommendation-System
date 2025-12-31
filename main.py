#%%
import pandas as pd
import numpy as np
import ast
#%%
credits=pd.read_csv(r"C:\Users\Dell\Downloads\tmdb_5000_credits.csv\tmdb_5000_credits.csv")
#%%
credits.head(5)
#%%
movies=pd.read_csv(r"C:\Users\Dell\Downloads\tmdb_5000_movies.csv\tmdb_5000_movies.csv")
#%%
movies.head(5)
#%%
print(credits.head())
pd.set_option('display.max_columns', None)
print(credits.head())
print(movies.head())
movies=movies.merge(credits,left_on='title',right_on='title')
#%%
def convert (obj):
    L=[]
    for i in ast.literal_eval(str(obj)):
        L.append(i['name'])

        return L
#%%
import ast

def convert(text):
    if isinstance(text, str):
        return [i['name'] for i in ast.literal_eval(text)]
    return []
#%%
movies['genres']=movies['genres'].apply(convert)
movies['genres']
#%%
movies['keywords'] = movies['keywords'].apply(convert)
movies['keywords'].head(5)
#%%
movies = movies.merge(credits, on='id')
#%%
def ensure_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return x.split()   # convert string → list of words
    return []

cols = ['genres', 'keywords', 'cast', 'crew']
for col in cols:
    movies[col] = movies[col].apply(ensure_list)

#%%
movies['tags'] = (
    movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)

movies['tags'] = movies['tags'].apply(lambda x: ' '.join(x))
#%%
movies['cast'][0]
#%%
import ast

def extract_top_cast(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)

    names = []
    for i in x:
        if isinstance(i, dict) and 'name' in i:
            names.append(i['name'])
    return names[:3]

movies['cast'] = movies['cast'].apply(extract_top_cast)
#%%
def extract_director(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)

    return [
        i['name']
        for i in x
        if isinstance(i, dict) and i.get('job') == 'Director'
    ]

movies['crew'] = movies['crew'].apply(extract_director)
#%%
movies['tags']=movies['genres']+movies['keywords']+movies['cast']+movies['crew']
#%%
movies['tags']
#%%
movies=movies[['id','title','overview','tags']]
#%%
movies['tags']=movies['tags'].apply(lambda x: " ".join(x))
#%%
movies['tags']=movies['tags'].apply(lambda x:x.lower())
#%%
movies['tags']
#%%
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])
#%%
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
#%%
def get_recommendations(title, cosine_sim, movies):
    # get index of the movie
    idx = movies[movies['title'] == title].index[0]

    # similarity scores with all movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # top 10 similar movies (exclude itself)
    sim_scores = sim_scores[1:11]

    # movie indices
    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices]
#%%
recommendations = get_recommendations("The Dark Knight Rises", cosine_sim, movies)
print(recommendations)
#%%
import pickle
with open('movie_data.pkl','wb') as file:
    pickle.dump((movies,cosine_sim),file)


