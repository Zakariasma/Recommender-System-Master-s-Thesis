from sqlalchemy import create_engine, text
import pandas as pd

from chap_5.api.mdp.config import DATABASE_URL


class RewardModel:
    """Charge et cache les scores des films (table movies) pour la policy iteration."""

    def __init__(self, database_url: str = DATABASE_URL):
        self.engine = create_engine(database_url)
        self._cache = self._load_scores()

    def _load_scores(self) -> dict:
        df = pd.read_sql(text("SELECT id, score FROM movies"), self.engine)
        return dict(zip(df['id'], df['score']))

    def get(self, item_id: int, default: float = 0.0) -> float:
        return self._cache.get(item_id, default)
