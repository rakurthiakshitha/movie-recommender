import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load both datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

print("Building smart recommender... This takes 20 seconds")

# --- Part 1: Content-Based ---
movies['genres'] = movies['genres'].str.replace('|', ' ')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
content_sim = cosine_similarity(tfidf_matrix)

# --- Part 2: Collaborative Filtering ---
rated_movie_ids = ratings['movieId'].unique()
movies_filtered = movies[movies['movieId'].isin(rated_movie_ids)].reset_index(drop=True)

# Rebuild content similarity for filtered movies only
tfidf_matrix_filtered = tfidf.fit_transform(movies_filtered['genres'])
content_sim_filtered = cosine_similarity(tfidf_matrix_filtered)

# Create user-movie rating matrix
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
movie_sim_collab = cosine_similarity(user_movie_matrix.T)

movie_sim_df = pd.DataFrame(movie_sim_collab, 
                            index=user_movie_matrix.columns, 
                            columns=user_movie_matrix.columns)

# --- Hybrid Recommender Function - V2 BLENDED ---
def hybrid_recommend(title, top_n=10):
    try:
        movie_row = movies_filtered[movies_filtered['title'] == title]
        movie_id = movie_row['movieId'].values[0]
        idx = movie_row.index[0]
    except:
        return ["Movie not found. Try exact title like 'Toy Story (1995)'"]
    
    # 1. Get content similarity scores
    content_scores = content_sim_filtered[idx]
    
    # 2. Get collaborative scores 
    collab_scores = movie_sim_df[movie_id].values
    
    # 3. BLEND: 60% collaborative + 40% content = Netflix formula
    hybrid_scores = 0.6 * collab_scores + 0.4 * content_scores
    
    # 4. Get top N, exclude the movie itself
    similar_indices = hybrid_scores.argsort()[::-1][1:top_n+1]
    recommended_titles = movies_filtered.iloc[similar_indices]['title'].tolist()
    
    return recommended_titles

# --- Run the program ---
movie_name = input("\nEnter a movie name: ")
print(f"\nSmart Recommendations for '{movie_name}':")
recs = hybrid_recommend(movie_name)
for i, movie in enumerate(recs, 1):
    print(f"{i}. {movie}")