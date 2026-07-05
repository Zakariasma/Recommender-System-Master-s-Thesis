from chap_5.api.mdp.predictive_model.state_key import decode
from chap_5.api.mdp.parralelized.sqlite_stores import SQLiteTransitionStoreReadOnly

K_WEIGHTS = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "models")

TRANSITIONS_DB_PATH = os.path.join(DATA_DIR, "transitions.db")
TRANSITIONS_TABLE = "transitions_skipping"


class PredictiveModelSQLite:
    """Utilisée uniquement pour get_observed_states (cache) et accès générique.
    Le calcul batché des successeurs par chunk se fait directement dans para.py."""

    def __init__(self, db_path: str = TRANSITIONS_DB_PATH, table: str = TRANSITIONS_TABLE,
                 k_weights: dict = None):
        self.store = SQLiteTransitionStoreReadOnly(db_path, table)
        self.k_weights = k_weights or K_WEIGHTS

    def get_observed_states(self, k: int = 1) -> list:
        return [decode(s_str) for s_str in self.store.get_states(k)]