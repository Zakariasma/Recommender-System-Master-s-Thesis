import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class CosineRecommender:
    def __init__(self, user_history_path: str, movies_path: str, k: int = 50, min_ratings: int = 20):
        self.k = k
        self.movies_df = self._load_movies(movies_path)
        history = self._load_history(user_history_path, min_ratings)

        self.users = sorted(history["userId"].unique().tolist())
        self.movies = sorted(history["tmdbId"].unique().tolist())
        self.user_idx = {user: i for i, user in enumerate(self.users)}
        self.movie_idx = {movie: j for j, movie in enumerate(self.movies)}

        self.matrix = self._build_matrix(history)

        print(f"Dataset : {self.matrix.shape[0]} users, {self.matrix.shape[1]} films, {self.matrix.nnz} ratings")

    def _load_movies(self, movies_path: str) -> pd.DataFrame:
        movies = pd.read_csv(movies_path)
        movies["tmdb_id"] = movies["tmdb_id"].astype(str).str.strip().str.replace(".0", "", regex=False)
        return movies

    def _load_history(self, user_history_path: str, min_ratings: int) -> pd.DataFrame:
        history = pd.read_csv(user_history_path, dtype={"tmdbId": str, "imdbId": str})
        return self._clean(history, min_ratings)

    def _clean(self, history: pd.DataFrame, min_ratings: int) -> pd.DataFrame:
        known_movies = set(self.movies_df["tmdb_id"])
        history["tmdbId"] = history["tmdbId"].str.strip().str.replace(".0", "", regex=False)
        history = history[
            history["tmdbId"].notna()
            & (history["tmdbId"] != "nan")
            & (history["tmdbId"] != "")
            & history["tmdbId"].isin(known_movies)
            ]
        enough_users = history["userId"].map(history["userId"].value_counts()) >= min_ratings
        enough_movies = history["tmdbId"].map(history["tmdbId"].value_counts()) >= min_ratings
        return history[enough_users & enough_movies]

    def _build_matrix(self, history: pd.DataFrame) -> csr_matrix:
        # Binaire : 1 = vu, peu importe la note
        row = history["userId"].map(self.user_idx).values
        col = history["tmdbId"].map(self.movie_idx).values
        data = np.ones(len(history))
        return csr_matrix((data, (row, col)), shape=(len(self.users), len(self.movies)))

    def _get_neighbors(self, user_idx: int) -> dict:
        user_ratings = self.matrix.getrow(user_idx).toarray().flatten()

        similarities = {}
        for other_idx in range(self.matrix.shape[0]):
            if other_idx == user_idx:
                continue
            other_ratings = self.matrix.getrow(other_idx).toarray().flatten()
            norm = np.linalg.norm(user_ratings) * np.linalg.norm(other_ratings)
            if norm == 0:
                continue
            similarities[other_idx] = float(np.dot(user_ratings, other_ratings) / norm)

        return dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:self.k])

    def _predict(self, movie_idx: int, neighbors: dict):
        num, den = 0.0, 0.0
        for neighbor_idx, similarity in neighbors.items():
            neighbor_rating = self.matrix[neighbor_idx, movie_idx]
            if neighbor_rating == 0:
                continue
            num += similarity * neighbor_rating  # sim * note (binaire = 1)
            den += abs(similarity)
        if den == 0:
            return None
        k = 1 / den
        return round(k * num, 4)

    def recommend(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        user_idx = self.user_idx[user_id]
        neighbors = self._get_neighbors(user_idx)
        seen = set(self.matrix.getrow(user_idx).indices)

        predictions = sorted(
            [(j, self._predict(j, neighbors)) for j in range(self.matrix.shape[1]) if j not in seen],
            key=lambda x: x[1] if x[1] is not None else -999,
            reverse=True
        )[:top_k]

        rows = []
        for movie_j, score in predictions:
            if score is None:
                continue
            tmdb = self.movies[movie_j]
            m = self.movies_df[self.movies_df["tmdb_id"] == tmdb]
            rows.append({
                "title": m["title"].values[0] if not m.empty else "?",
                "tmdb_id": tmdb,
                "imdb_id": m["imdb_id"].values[0] if not m.empty else "?",
                "score": score,
            })
        return pd.DataFrame(rows).reset_index(drop=True)

    def print_user_history(self, user_id: int):
        user_row = self.matrix.getrow(self.user_idx[user_id])
        seen_movie_indices = user_row.indices

        rows = []
        for j in seen_movie_indices:
            tmdb = self.movies[j]
            m = self.movies_df[self.movies_df["tmdb_id"] == tmdb]
            rows.append({"title": m["title"].values[0] if not m.empty else "?", "tmdb_id": tmdb})

        df = pd.DataFrame(rows)
        print(f"\nHistorique user {user_id} — {len(df)} films")
        print(df.to_string())
