import numpy as np


class UserProfileBuilder:
    def __init__(self, user_id, ratings_df, movies_df, tfidf_matrix):
        self.user_id = user_id
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.tfidf_matrix = tfidf_matrix

    def show_last_watched(self, n=15):
        user_history = self.ratings_df[self.ratings_df['userId'] == self.user_id]
        top_rated = user_history.sort_values(by='rating', ascending=False).head(n)

        for _, row in top_rated.iterrows():
            movie_title = self.movies_df[self.movies_df['movieId'] == row['movieId']]['title'].values[0]
            print(f"- {movie_title} (Note: {row['rating']})")

    def build_profile_vector(self):
        liked_movies = self.ratings_df[(self.ratings_df['userId'] == self.user_id) &
                                       (self.ratings_df['rating'] >= 4.0)]

        # Recup indices des films dans dataFrame
        liked_indices = self.movies_df[self.movies_df['movieId'].isin(liked_movies['movieId'])].index

        if len(liked_indices) == 0:
            return None

        liked_vectors = self.tfidf_matrix[liked_indices].toarray()

        # Recup notes pour s'en servir de poids
        weights = liked_movies.set_index('movieId').loc[self.movies_df.iloc[liked_indices]['movieId']]['rating'].values
        profile_vector = np.average(liked_vectors, axis=0, weights=weights)

        return profile_vector.reshape(1, -1)