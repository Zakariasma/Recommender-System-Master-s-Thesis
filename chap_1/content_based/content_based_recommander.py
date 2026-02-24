import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from chap_1.content_based.utils.init_tf_idf import InitMovieTfIdf


class ContentBasedRecommender:
    def __init__(
            self,
            user_history_path: str = "data/user_history/user_movie_ids.csv",
            movies_path: str = "data/user_history/user_movie_ids.csv"
    ):
        self.user_history_path = user_history_path
        self.tfidf = InitMovieTfIdf(movies_path)
        self.vectors = self.tfidf.get_vectors()
        self.movies_df = self.tfidf.movies_df.reset_index(drop=True)
        self.movies_tmdb = self.movies_df['tmdb_id'].astype(str).str.strip().str.replace('.0', '')
        self.load_user_history()

    def load_user_history(self):
        self.user_history = pd.read_csv(self.user_history_path, dtype={'tmdbId': str, 'imdbId': str})

    def _get_user_tmdb_ids(self, user_id: int):
        return self.user_history[self.user_history['userId'] == user_id]['tmdbId'].dropna().str.strip().tolist()

    def get_user_profile(self, user_id: int):
        tmdb_ids = self._get_user_tmdb_ids(user_id)

        history_vectors = []
        for tmdb_id in tmdb_ids:
            matches = self.movies_df[self.movies_tmdb == tmdb_id]
            if not matches.empty:
                history_vectors.append(self.vectors[matches.index[0]])

        if len(history_vectors) == 0:
            raise ValueError(f"Aucun film match pour user {user_id}")

        return np.mean(history_vectors, axis=0)

    def compute_similarities(self, user_profile: np.ndarray):
        scores = cosine_similarity(self.vectors, user_profile.reshape(1, -1)).flatten()
        return scores

    def get_seen_indices(self, user_id: int):
        tmdb_ids = self._get_user_tmdb_ids(user_id)
        seen_indices = []
        for tmdb_id in tmdb_ids:
            matches = self.movies_df[self.movies_tmdb == tmdb_id]
            if not matches.empty:
                seen_indices.append(matches.index[0])
        return seen_indices

    def get_top_recommendations(self, scores: np.ndarray, seen_indices: list, top_k: int = 10):
        unseen_scores = [(i, scores[i]) for i in range(len(scores)) if i not in seen_indices]
        unseen_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in unseen_scores[:top_k]]

        recs = self.movies_df.iloc[top_indices][['title', 'tmdb_id', 'imdb_id']].reset_index(drop=True)
        recs['score'] = [s for _, s in unseen_scores[:top_k]]
        return recs

    def print_user_history(self, user_id: int, top_k: int = 10):
        tmdb_ids = self._get_user_tmdb_ids(user_id)

        seen_rows = []
        for tmdb_id in tmdb_ids:
            matches = self.movies_df[self.movies_tmdb == tmdb_id]
            if not matches.empty:
                seen_rows.append(matches.index[0])

        history = self.movies_df.iloc[seen_rows][['title', 'tmdb_id', 'imdb_id']].reset_index(drop=True)
        print(f"\nHistorique user {user_id} ({len(history)} films matchés / {len(tmdb_ids)})")
        print(history.head(top_k).to_string())

    def recommend(self, user_id: int, top_k: int = 10):
        user_profile = self.get_user_profile(user_id)
        scores = self.compute_similarities(user_profile)
        seen_indices = self.get_seen_indices(user_id)
        self.print_user_history(user_id, top_k)
        return self.get_top_recommendations(scores, seen_indices, top_k)


if __name__ == "__main__":
    rec = ContentBasedRecommender()
    recommendations = rec.recommend(10, top_k=50)
    print('\n', recommendations.round({'score': 4}))
