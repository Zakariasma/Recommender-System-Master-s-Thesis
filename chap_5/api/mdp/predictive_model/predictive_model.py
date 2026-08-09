from chap_5.api.mdp.predictive_model.improvement.multi_size import MultiSize
from chap_5.api.mdp.predictive_model.improvement.setup_info_dict.setup_info_dict import SetupInfoDict
from chap_5.api.mdp.predictive_model.improvement.skipping.data.sql import fetch_distinct_states, \
    create_skipping_database, retrieve_distinct_states

class PredictiveModel:
    def __init__(self, k: int):
        self.k = k
        self.skipping_engine = create_skipping_database()
        self.setup_full_infos = SetupInfoDict(self.k)
        self.multi_size = MultiSize(k=self.k)
        pass

    def _build_info_index(self):
        self.multi_size.improve_model()
        self.setup_full_infos.build_info_index()









