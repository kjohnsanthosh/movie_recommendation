from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# df = pd.read_csv(r'C:\Users\SANTHOSH\OneDrive\ドキュメント\vs_code_programs\movie_recommandation_part_2\final2.csv')
df = pd.read_csv('movie_recommandation_part_2/final3.csv')

df1 = pd.read_csv('movie_recommandation_part_2/final3.csv')
df1['Director_name'] = df1['Director_name'] * 200

df2 = pd.read_csv('movie_recommandation_part_2/final3.csv')
df2['Actor_names'] = df2['Actor_names'] * 200

df3 = pd.read_csv('movie_recommandation_part_2/final3.csv')
df3['production'] = df3['production'] * 100



new_df = df.drop(["title", "overview"], axis = 1)


X = new_df.drop(['movie_id'], axis=  1)

X1 = df1.drop(["title", "overview", 'movie_id'], axis=1)

X2 = df2.drop(["title", "overview", 'movie_id'], axis=1)

X3 = df3.drop(["title", "overview", 'movie_id'], axis=1)

Y = new_df['movie_id']


model = KNeighborsClassifier(n_neighbors=20, p= 2)
model1 = KNeighborsClassifier(n_neighbors=10, p=2)
model2 = KNeighborsClassifier(n_neighbors=10, p=2)
model3 = KNeighborsClassifier(n_neighbors=10, p=2)


model.fit(X, Y)
model1.fit(X1, Y)
model2.fit(X2, Y)
model3.fit(X3, Y)


@app.route('/healthcheck')
def check():
    return "Server is running"

@app.route('/recommend_movies', methods= ["POST"])
def get_recommendations():

    movie_name = request.get_data(as_text= True)

    if movie_name not in df['title'].values:
        return jsonify({"error": "Movie not found"}), 404
    
    df_new = df[df['title'] == movie_name]


    results = model.kneighbors(df_new.drop(['title', 'overview', 'movie_id'], axis=1), return_distance=False)[0]
    
    movies = df[df.index.isin(results)]['title'].tolist()
    genres = df[df.index.isin(results)]['overview'].tolist()

    recommeded_movies = {}
    
    for movie, genre in zip(movies, genres):
        if movie != movie_name: 
            recommeded_movies[movie] = genre

    if not recommeded_movies:
        return jsonify({"error": "No recommendations found."}), 404
    
    return jsonify(recommeded_movies)


@app.route('/director_related', methods=['POST'])
def recommend_movies():

    movie_name = request.get_data(as_text= True)  

    df_new1 = df1[df1['title'] == movie_name]

    print(df_new1)

    results = model1.kneighbors(df_new1.drop(['title', 'overview', 'movie_id'], axis=1), return_distance=False)[0]


    movies = df[df.index.isin(results)]['title'].tolist()
    genres = df[df.index.isin(results)]['overview'].tolist()
 
    recommeded_movies = {}
    
    for movie, genre in zip(movies, genres):
        recommeded_movies[movie] = genre

    
    return jsonify(recommeded_movies)

@app.route('/actor_related', methods=['POST'])
def actor_movies():

    movie_name = request.get_data(as_text= True)

    df_new1 = df2[df2['title'] == movie_name]

    print(df_new1)

    results = model2.kneighbors(df_new1.drop(['title', 'overview', 'movie_id'], axis=1), return_distance=False)[0]


    movies = df[df.index.isin(results)]['title'].tolist()
    genres = df[df.index.isin(results)]['overview'].tolist()
 
    recommeded_movies = {}
    
    for movie, genre in zip(movies, genres):
        recommeded_movies[movie] = genre

    
    return jsonify(recommeded_movies)

@app.route('/production_related', methods=['POST'])
def production_movies():

    movie_name = request.get_data(as_text= True)
 
    df_new1 = df3[df3['title'] == movie_name]

    print(df_new1)

    results = model3.kneighbors(df_new1.drop(['title', 'overview', 'movie_id'], axis=1), return_distance=False)[0]

    movies = df[df.index.isin(results)]['title'].tolist()
    genres = df[df.index.isin(results)]['overview'].tolist()
 
    recommeded_movies = {}
    
    for movie, genre in zip(movies, genres):
        recommeded_movies[movie] = genre

    
    return jsonify(recommeded_movies)



if __name__ == '__main__':
    app.run(debug= True)