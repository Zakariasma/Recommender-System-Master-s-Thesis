from sqlalchemy import create_engine, text
import pandas as pd


class RewardModel:
    """
    Charge les scores des films depuis la table `movies` et les met en cache
    pour éviter des requêtes SQL répétées lors de la policy iteration.
    """

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self._cache = self._load_scores()

    def _load_scores(self) -> dict:
        query = text("SELECT id, score FROM movies")
        df = pd.read_sql(query, self.engine)
        return dict(zip(df['id'], df['score']))

    def get(self, item_id: int, default: float = 0.0) -> float:
        return self._cache.get(item_id, default)