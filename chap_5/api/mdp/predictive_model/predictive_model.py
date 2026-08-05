from collections import defaultdict
from psycopg2.extras import execute_values

from chap_5.api.mdp.config import DATABASE_URL, K, FRACTION, MAX_SKIP, BATCH_SIZE
from chap_5.api.mdp.predictive_model.helper.encoder import as_tuple, encode, decode
from chap_5.api.mdp.predictive_model.helper.sql_query import TransitionStore
from chap_5.api.mdp.predictive_model.improvement.multiple_size import MultipleSizeModel

K_WEIGHTS = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}


class PredictiveModel:
    def __init__(self, k_sizes: list = None, k_weights: dict = None):
        self.store = TransitionStore(DATABASE_URL)
        self.k_sizes = k_sizes or [1, 2, 3]
        self.k_weights = k_weights or K_WEIGHTS

    def improve_model(self):
        if not self.store.is_empty():
            print("Modèle déjà amélioré.")
            return
        print("Tables vides, amélioration en cours...")
        MultipleSizeModel(self.store, self.k_sizes, MAX_SKIP, BATCH_SIZE, FRACTION).fit()

    def get_trmc(self, s: tuple, s_: tuple) -> float:
        s, s_ = as_tuple(s), as_tuple(s_)
        return sum(
            weight * self.store.get_prob(encode(s[-k:]), encode(s_[-k:]), k)
            for k, weight in self.k_weights.items()
        )

    def get_successors(self, s: tuple) -> dict:
        s = as_tuple(s)
        merged = defaultdict(float)
        for k, weight in self.k_weights.items():
            for s_bytes, prob in self.store.get_successors(encode(s[-k:]), k).items():
                merged[decode(s_bytes)[-1]] += weight * prob

        total = sum(merged.values())
        return {item: p / total for item, p in merged.items()} if total else {}

    def _get_strict_successors_batch(self, states_list: list) -> dict:
        """
        Récupère UNIQUEMENT les transitions strictes (k = self.k) pour une liste d'états.
        Pas de K_WEIGHTS, pas de suffixes.
        """
        if not states_list:
            return {}

        # 1. On encode les états (tuples) en bytes (BYTEA) pour PostgreSQL
        unique_keys = list(set(encode(state_tuple) for state_tuple in states_list))

        raw_transitions = defaultdict(dict)
        raw_conn = self.sqlite_store.engine.raw_connection()  # ou ton engine PG

        try:
            with raw_conn.cursor() as cur:
                cur.execute("SET work_mem = '1GB'")

                # 2. Requête directe avec ANY sur la taille k exacte
                # On filtre avec k = self.k (ex: 3)
                cur.execute("""
                            SELECT s, s_, prob
                            FROM transitions
                            WHERE s = ANY (%s)
                              AND k = %s
                            """, (unique_keys, self.k))

                # 3. Fusion des résultats
                for row in cur:
                    s_bytes = bytes(row[0])
                    s_next_bytes = bytes(row[1])
                    prob = row[2]

                    # On décode juste le dernier item de l'état destination
                    item = decode(s_next_bytes)[-1]
                    raw_transitions[s_bytes][item] = prob

        finally:
            raw_conn.close()

        return dict(raw_transitions)

    def get_observed_states(self, k: int = K) -> list:
        return [decode(s_bytes) for s_bytes in self.store.get_states(k)]


if __name__ == "__main__":
    model = PredictiveModel()
    model.improve_model()
    print(model.get_trmc((10000,), (8491,)))