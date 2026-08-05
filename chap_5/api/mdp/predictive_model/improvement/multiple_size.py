from chap_5.api.mdp.predictive_model.helper.sql_query import TransitionStore
from chap_5.api.mdp.predictive_model.improvement.skipping import SkippingModel


class MultipleSizeModel:
    def __init__(self, store: TransitionStore, k_sizes: list, max_skip: int, batch_size: int, fraction: float = 1.0):
        self.store = store
        self.k_sizes = k_sizes
        self.max_skip = max_skip
        self.batch_size = batch_size
        self.fraction = fraction

    def fit(self):
        for k in self.k_sizes:
            print(f"\n=== Skipping k={k} ===")
            SkippingModel(k, self.store, self.max_skip, self.batch_size, self.fraction).fit()
