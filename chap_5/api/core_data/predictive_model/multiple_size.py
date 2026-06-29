import os
from collections import defaultdict

import pandas as pd
from sqlalchemy import create_engine

from chap_5.api.core_data.predictive_model.db.db import MODEL_DIR, TransitionDB
from chap_5.api.core_data.predictive_model.skipping import SkippingModel
from chap_5.api.core_data.predictive_model.similarity import SimilarityModel

DB_PATH = os.path.join(MODEL_DIR, "transitions.db")


class MultipleSizeModel:

    def __init__(
        self,
        db_path_movies: str,
        fraction: float = None,
        k_sizes: list   = None,
        k: int          = 3,
        max_skip: int   = 5,
        batch_size: int = 10_000,
    ):
        self.db             = TransitionDB(DB_PATH)
        self.db_path_movies = db_path_movies
        self.fraction       = fraction
        self.k_sizes        = k_sizes or [1, 2, 3]
        self.k              = k
        self.max_skip       = max_skip
        self.batch_size     = batch_size

    def fit(self):
        movie_genres = self._load_genre_mapping()

        for k in self.k_sizes:
            print(f"\n=== Skipping k={k} ===")
            SkippingModel(k, self.db, self.max_skip, self.batch_size, self.fraction).fit()

        for k in self.k_sizes:
            print(f"\n=== Similarity k={k} ===")
            SimilarityModel(k, self.db, self.batch_size, movie_genres).fit()

    def _load_genre_mapping(self) -> dict:
        engine = create_engine(self.db_path_movies)
        with engine.connect() as conn:
            mg = pd.read_sql("SELECT movie_id, genre_id FROM movie_genre", conn)
        engine.dispose()

        mapping = defaultdict(set)
        for row in mg.itertuples(index=False):
            mapping[row.movie_id].add(row.genre_id)
        return mapping