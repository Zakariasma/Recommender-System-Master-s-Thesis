from collections import defaultdict

from chap_5.api.mdp.config import DATABASE_URL, K, FRACTION, MAX_SKIP, BATCH_SIZE
from chap_5.api.mdp.predictive_model.state_key import as_tuple, encode, decode
from chap_5.api.mdp.predictive_model.transition_store import TransitionStore
from chap_5.api.mdp.predictive_model.improvement.multiple_size import MultipleSizeModel

K_WEIGHTS = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}


class PredictiveModel:
    def __init__(self, k_sizes: list = None, k_weights: dict = None):
        self.store = TransitionStore(DATABASE_URL)
        self.k_sizes = k_sizes or [1, 2, 3]
        self.k_weights = k_weights or K_WEIGHTS

    def improve_model(self):
        if not self.store.is_empty():
            print("Modèle déjà améliorer.")
            return
        print("Tables vides, amelioration en cours...")
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
            for s_str, prob in self.store.get_successors(encode(s[-k:]), k).items():
                merged[decode(s_str)[-1]] += weight * prob

        total = sum(merged.values())
        return {item: p / total for item, p in merged.items()} if total else {}

    def get_observed_states(self, k: int = K) -> list:
        return [decode(s_str) for s_str in self.store.get_states(k)]


if __name__ == "__main__":
    model = PredictiveModel()
    model.improve_model()
    print(model.get_trmc((10000,), (8491,)))