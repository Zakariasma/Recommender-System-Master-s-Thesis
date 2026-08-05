from chap_5.api.mdp_rework.predictive_model.improvement.skipping.skipping import SkippingModel
from chap_5.api.mdp_rework.predictive_model.improvement.skipping.data.sql import (
    create_skipping_database, setup_skiping_transition_table, create_transition_dict
)
from chap_5.api.mdp_rework.predictive_model.improvement.similarity.similarity import Similarity
from chap_5.api.mdp_rework.predictive_model.improvement.similarity.data.sql import (
    create_similarity_database, setup_similarity_dict_table, create_similarity_dict
)


class MultiSize:
    def __init__(self, k: int):
        self.k = k

    def run_skipping(self):
        engine = create_skipping_database()
        setup_skiping_transition_table(engine)

        for current_k in range(self.k, 0, -1):
            model = SkippingModel(k=current_k)
            model.skipping()

        create_transition_dict(engine, self.k)

    def process_similarity(self):
        engine = create_similarity_database()
        setup_similarity_dict_table(engine)

        for current_k in range(self.k, 0, -1):
            sim = Similarity(k=current_k)
            sim.similarity()

        create_similarity_dict(engine, self.k)


if __name__ == "__main__":
    multi = MultiSize(k=3)

    multi.run_skipping()
    multi.process_similarity()