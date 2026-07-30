from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, tfidf_matrix, movies_df, user_vector, top_k):
        self.tfidf_matrix = tfidf_matrix
        self.movies_df = movies_df
        self.user_vector = user_vector
        self.top_k = top_k

    def calculate_simcos(self):
        return cosine_similarity(self.user_vector, self.tfidf_matrix).flatten()

    def show_recommendations(self, watched_movie_ids):
        sim_scores = self.calculate_simcos()

        movie_scores = []
        for idx, score in enumerate(sim_scores):
            movie_id = self.movies_df.iloc[idx]['movieId']
            if movie_id not in watched_movie_ids:
                movie_scores.append((idx, score))
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
        top_k_movies = movie_scores[:self.top_k]

        print(f"\nTop {self.top_k} Recommandations :")
        for idx, score in top_k_movies:
            movie_title = self.movies_df.iloc[idx]['title']
            print(f"- {movie_title} (Similarité: {score:.4f})")