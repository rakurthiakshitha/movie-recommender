from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

app = Flask(__name__)

print("Loading data and building recommender...")
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Build recommender
movies['genres'] = movies['genres'].str.replace('|', ' ')
rated_movie_ids = ratings['movieId'].unique()
movies_filtered = movies[movies['movieId'].isin(rated_movie_ids)].reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_filtered = tfidf.fit_transform(movies_filtered['genres'])
content_sim_filtered = cosine_similarity(tfidf_matrix_filtered)

user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
movie_sim_collab = cosine_similarity(user_movie_matrix.T)
movie_sim_df = pd.DataFrame(movie_sim_collab, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Get all movie titles for dropdown
movie_list = sorted(movies_filtered['title'].tolist())
print("Ready!")

def fetch_poster(movie_title):
    """Fetch movie poster from TMDB API"""
    try:
        # Clean title: remove year
        title = movie_title.split(' (')[0]
        api_key = "8265bd1679663a7ea12ac168da84d2e8" # Public demo key
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}"
        data = requests.get(url).json()
        if data['results']:
            poster_path = data['results'][0]['poster_path']
            if poster_path:
                return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        pass
    return "https://via.placeholder.com/500x750.png?text=No+Poster"

def hybrid_recommend(title, top_n=10):
    try:
        movie_row = movies_filtered[movies_filtered['title'] == title]
        movie_id = movie_row['movieId'].values[0]
        idx = movie_row.index[0]
    except:
        return []
    
    content_scores = content_sim_filtered[idx]
    collab_scores = movie_sim_df[movie_id].values
    hybrid_scores = 0.6 * collab_scores + 0.4 * content_scores
    similar_indices = hybrid_scores.argsort()[::-1][1:top_n+1]
    
    recs = []
    for i in similar_indices:
        movie_title = movies_filtered.iloc[i]['title']
        poster = fetch_poster(movie_title)
        recs.append({"title": movie_title, "poster": poster})
    return recs

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    movie_name = ""
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recommendations = hybrid_recommend(movie_name)
    return render_template('index.html', recommendations=recommendations, movie_name=movie_name, movie_list=movie_list)

if __name__ == '__main__':
    app.run(debug=True)
    app = app