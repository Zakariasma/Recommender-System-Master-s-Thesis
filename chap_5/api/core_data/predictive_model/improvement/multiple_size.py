import os

from chap_5.api.core_data.predictive_model.db.db import MODEL_DIR, TransitionDB
from chap_5.api.core_data.predictive_model.improvement.skipping import SkippingModel

DB_PATH = os.path.join(MODEL_DIR, "transitions.db")


class MultipleSizeModel:

    def __init__(
        self,
        db_path_movies: str,
        fraction: float = None,
        k_sizes: list = None,
        k: int = 3,
        max_skip: int = 5,
        batch_size: int = 10_000,
    ):
        self.db = TransitionDB(DB_PATH)
        self.db_path_movies = db_path_movies
        self.fraction = fraction
        self.k_sizes = k_sizes or [1, 2, 3]
        self.k = k
        self.max_skip = max_skip
        self.batch_size = batch_size

    def fit(self):
        for k in self.k_sizes:
            print(f"\n=== Skipping k={k} ===")
            SkippingModel(
                k=k,
                db=self.db,
                max_skip=self.max_skip,
                batch_size=self.batch_size,
                fraction=self.fraction
            ).fit()