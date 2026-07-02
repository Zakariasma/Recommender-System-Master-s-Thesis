import os
from collections import defaultdict

from chap_5.api.core_data.predictive_model.db.db import TransitionDB
from chap_5.api.core_data.predictive_model.improvement.multiple_size import MultipleSizeModel, DB_PATH


class PredictiveModel:

    def __init__(
        self,
        db_path_movies: str,
        fraction: float,
        k_sizes: list = None,
        k_weights: dict = None,
        k: int = 3,
        max_skip: int = 5,
        batch_size: int = 10_000,
    ):
        self.db = TransitionDB(DB_PATH)
        self.db_path_movies = db_path_movies
        self.fraction = fraction
        self.k_sizes = k_sizes or [1, 2, 3]
        self.k_weights = k_weights or {1: 1/3, 2: 1/3, 3: 1/3}
        self.k = k
        self.max_skip = max_skip
        self.batch_size = batch_size

    def init(self):
        if self.db.tables_empty():
            print("Tables vides, entraînement en cours...")
            MultipleSizeModel(
                db_path_movies=self.db_path_movies,
                fraction=self.fraction,
                k_sizes=self.k_sizes,
                k=self.k,
                max_skip=self.max_skip,
                batch_size=self.batch_size,
            ).fit()
        else:
            print("Modèle déjà entraîné.")

    def _fmt(self, items) -> str:
        return ','.join(str(x) for x in items)

    def get_trmc(self, s: tuple, s_: tuple) -> float:
        if isinstance(s, int):
            s = (s,)
        if isinstance(s_, int):
            s_ = (s_,)

        s  = tuple(int(x) for x in s)
        s_ = tuple(int(x) for x in s_)

        total = 0.0
        for k, weight in self.k_weights.items():
            s_k  = self._fmt(s[-k:])      # on prend les k derniers éléments
            s_k_ = self._fmt(s_[-k:])     # idem pour l'état futur
            total += weight * self.db.get_prob(s_k, s_k_, k)

        return total

    def get_successors(self, s: tuple) -> dict:
        """
        Retourne {item_r: proba} — juste le dernier item du successeur comme clé.
        ex: {5869: 0.15, 111: 0.08, ...}  ← pas de tuples imbriqués
        """
        if isinstance(s, int):
            s = (s,)
        s = tuple(int(x) for x in s)

        merged = defaultdict(float)

        for k, weight in self.k_weights.items():
            s_k = self._fmt(s[-k:])
            for s_str, prob in self.db.get_successors(s_k, k).items():
                # On prend uniquement le DERNIER item du successeur
                last_item = int(s_str.split(',')[-1])
                merged[last_item] += weight * prob

        total = sum(merged.values())
        return {item: p / total for item, p in merged.items()}

    def get_observed_states(self, k: int = None) -> list:
        """
        Retourne la liste des états observés (sous forme de tuples d'int),
        utilisés par le MDPSolver pour la policy iteration.
        """
        k = k or self.k
        raw_states = self.db.get_states(k)

        states = []
        for s_str in raw_states:
            items = tuple(int(x) for x in s_str.split(',') if x)
            states.append(items)

        return states

if __name__ == "__main__":
    db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/master')
    model = PredictiveModel(db_url, fraction=1/10000)
    model.init()

    s = (10000)
    s_ = (8491)
    print(model.get_trmc(s, s_))