from chap_5.api.mdp_rework.predictive_model.improvement.skipping.data.sql import fetch_distinct_states, \
    create_skipping_database, retrieve_distinct_states

EPSILON = 1e-6


class PredictiveModel:
    def __init__(self, k: int):
        self.k = k
        self.skipping_engine = create_skipping_database()
        pass

    def _build_info_index(self):
        observed_states = retrieve_distinct_states(self.skipping_engine, self.k)
        print(len(observed_states))



engine = create_skipping_database()
model = PredictiveModel(k=3)
model.build_info_index()


