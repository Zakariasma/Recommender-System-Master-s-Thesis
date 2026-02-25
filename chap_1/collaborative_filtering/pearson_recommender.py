import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix


class PearsonRecommender:
    def __init__(self, user_history_path: str, movies_path: str, k: int = 50, min_ratings: int = 20):
        self.k = k
        self.movies_df = self._load_movies(movies_path)
        self.history = self._load_history(user_history_path, min_ratings)

        self.users = sorted(self.history["userId"].unique().tolist())
        self.movies = sorted(self.history["tmdbId"].unique().tolist())
        self.user_idx = {user: i for i, user in enumerate(self.users)}
        self.movie_idx = {movie: j for j, movie in enumerate(self.movies)}

        self.matrix = self._build_matrix(self.history)
        self.means = self._compute_means()

        print(f"Dataset : {self.matrix.shape[0]} users, {self.matrix.shape[1]} films, {self.matrix.nnz} ratings")

    # Load les datas
    def _load_movies(self, movies_path: str) -> pd.DataFrame:
        movies = pd.read_csv(movies_path)
        movies["tmdb_id"] = movies["tmdb_id"].astype(str).str.strip().str.replace(".0", "", regex=False)
        return movies

    def _load_history(self, user_history_path: str, min_ratings: int) -> pd.DataFrame:
        history = pd.read_csv(user_history_path, dtype={"tmdbId": str, "imdbId": str})
        history = self._clean(history, min_ratings)
        return history

    # Réduit le dataframe
    # Garde film qu'on connait
    # Remove utilisateurs/films qui n'ont pas assez de données
    # Remove données pas exploitable
    def _clean(self, history: pd.DataFrame, min_ratings: int) -> pd.DataFrame:
        known_movies = set(self.movies_df["tmdb_id"])

        history["tmdbId"] = history["tmdbId"].str.strip().str.replace(".0", "", regex=False)
        history = history[
            history["tmdbId"].notna()
            & (history["tmdbId"] != "nan")
            & (history["tmdbId"] != "")
            & history["tmdbId"].isin(known_movies)
            ]

        enough_user_ratings = history["userId"].map(history["userId"].value_counts()) >= min_ratings
        enough_movie_ratings = history["tmdbId"].map(history["tmdbId"].value_counts()) >= min_ratings
        return history[enough_user_ratings & enough_movie_ratings]

    # Build de la matrice
    # CSR garde pas les 0 -> save mémoire
    def _build_matrix(self, history: pd.DataFrame) -> csr_matrix:
        row = history["userId"].map(self.user_idx).values
        col = history["tmdbId"].map(self.movie_idx).values
        data = history["rating"].values
        # Forme de la matrice -> user en ligne, film en collone, rating comme data
        return csr_matrix((data, (row, col)), shape=(len(self.users), len(self.movies)))

    # Calcul des moyennes par utilisateur
    def _compute_means(self) -> np.ndarray:
        means = []
        for user_position in range(self.matrix.shape[0]):
            user_ratings = self.matrix.getrow(user_position).data  # notes non-nulles du user
            means.append(float(np.mean(user_ratings)) if len(user_ratings) > 0 else 0.0)
        return np.array(means)

    # Garde que les K voisins pour eviter temps trop long
    def _get_neighbors(self, user_idx: int) -> dict:
        user_ratings = self.matrix.getrow(user_idx).toarray().flatten() # Recup note du user cible
        rated_mask = user_ratings > 0

        similarities = {}
        for other_idx in range(self.matrix.shape[0]):
            if other_idx == user_idx:
                continue
            other_ratings = self.matrix.getrow(other_idx).toarray().flatten()
            co_rated_mask = rated_mask & (other_ratings > 0) # On récupère les notes communes entre le user (vecteur de bool)
            if co_rated_mask.sum() < 2:
                continue
            # Pearson sur les notes communes
            corr = pearsonr(user_ratings[co_rated_mask], other_ratings[co_rated_mask])[0]
            if not np.isnan(corr):
                similarities[other_idx] = corr

        return dict(sorted(similarities.items(), key=lambda x: abs(x[1]), reverse=True)[:self.k])

    def _predict(self, user_idx: int, movie_idx: int, neighbors: dict):
        num, den = 0.0, 0.0
        for neighbor_idx, similarity in neighbors.items():
            neighbor_rating = self.matrix[neighbor_idx, movie_idx]
            if neighbor_rating == 0:
                continue
            num += similarity * (neighbor_rating - self.means[neighbor_idx])
            den += abs(similarity)
        if den == 0:
            return None
        k = 1 / den
        score = self.means[user_idx] + k * num
        return round(float(np.clip(score, 0, 5)), 4) # Doit clip pour borner le resultat entre 0 et 5

    def recommend(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        user_idx = self.user_idx[user_id]
        neighbors = self._get_neighbors(user_idx)
        seen = set(self.matrix.getrow(user_idx).indices)

        predictions = sorted(
            [(j, self._predict(user_idx, j, neighbors)) for j in range(self.matrix.shape[1]) if j not in seen],
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
        seen_ratings = user_row.data

        rows = []
        for j, rating in zip(seen_movie_indices, seen_ratings):
            tmdb = self.movies[j]
            m = self.movies_df[self.movies_df["tmdb_id"] == tmdb]
            rows.append({
                "title": m["title"].values[0] if not m.empty else "?",
                "tmdb_id": tmdb,
                "rating": rating,
            })

        df = pd.DataFrame(rows)
        print(f"\nHistorique user {user_id} — {len(df)} films")
        print(df.to_string())
