from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv(r'C:\Users\SANTHOSH\OneDrive\ドキュメント\vs_code_programs\movie_recommandation_part_2\final2.csv')  # Use the uploaded file path

# Prepare data
new_df = df.drop(["title", "overview"], axis=1)
X = new_df.drop(['movie_id'], axis=1)
Y = new_df['movie_id']

# Train KNN model
model = KNeighborsClassifier(n_neighbors=20, p=2)
model.fit(X, Y)

@app.route('/healthcheck')
def check():
    return "Server is running"

@app.route('/recommend_movies', methods=["POST"])
def get_recommendations():
    movie_name = request.get_data(as_text=True)
    
    # Check if movie exists
    if movie_name not in df['title'].values:
        return jsonify({"error": "Movie not found"}), 404
    
    df_new = df[df['title'] == movie_name]
    
    # Perform KNN for recommendations
    results = model.kneighbors(df_new.drop(['title', 'overview', 'movie_id'], axis=1), return_distance=False)[0]
    
    movies = df[df.index.isin(results)]['title'].tolist()
    genres = df[df.index.isin(results)]['overview'].tolist()

    recommended_movies = {movie: genre for movie, genre in zip(movies, genres)}
    
    return jsonify(recommended_movies)

@app.route('/director_related', methods=['POST'])
def recommend_movies_by_director():
    movie_name = request.get_data(as_text=True)
    
    # Check if the movie exists
    if movie_name not in df['title'].values:
        return jsonify({"error": "Movie not found"}), 404
    
    # Get the director of the input movie
    director = df[df['title'] == movie_name]['Director_name'].values[0]
    
    # Filter movies directed by the same director
    director_movies = df[df['Director_name'] == director]
    
    if director_movies.shape[0] <= 1:
        return jsonify({"error": "No other movies found by this director"}), 404
    
    # Prepare feature matrix for the movies by the same director
    director_X = director_movies.drop(['title', 'overview', 'movie_id'], axis=1)
    
    # Get the row for the input movie to find its neighbors
    df_new = df[df['title'] == movie_name].drop(['title', 'overview', 'movie_id'], axis=1)
    
    # Perform KNN within the filtered set (movies by the same director)
    knn = KNeighborsClassifier(n_neighbors=5, p=2)
    knn.fit(director_X, director_movies['movie_id'])
    
    results = knn.kneighbors(df_new, return_distance=False)[0]
    
    # Get the recommended movies
    recommended_titles = director_movies.iloc[results]['title'].tolist()
    recommended_overviews = director_movies.iloc[results]['overview'].tolist()

    recommended_movies = {title: overview for title, overview in zip(recommended_titles, recommended_overviews)}

    return jsonify(recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
