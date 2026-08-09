from chap_5.api.mdp.config import K, GAMMA_RL, THRESHOLD, LIST_SIZE
from chap_5.api.mdp.mdp.data.sql import create_database, init_kv_store_db, flush_value_states
from chap_5.api.mdp.predictive_model.improvement.setup_info_dict.data.sql import create_full_info_database
from chap_5.api.mdp.predictive_model.improvement.skipping.data.sql import (
    retrieve_distinct_states, create_skipping_database
)
from chap_5.api.mdp.mdp.policy_evaluation import PolicyEvaluation
from chap_5.api.mdp.mdp.policy_improvement import PolicyImprovement
from chap_5.api.mdp.predictive_model.improvement.setup_info_dict.data.pg_sql import fetch_movie_scores, \
    get_pg_engine


class MDPSolver:
    def __init__(self):
        self.k = K
        self.gamma = GAMMA_RL
        self.threshold = THRESHOLD
        self.list_size = LIST_SIZE
        self.skipping_engine = create_skipping_database()
        self.full_info_engine = create_full_info_database()
        self.kv_engine = create_database()
        self.pg_engine = get_pg_engine()
        self.movie_scores = fetch_movie_scores(self.pg_engine)
        self.v_states = {}

    def solve_mdp(self, states: list = None):
        if states is None:
            states = retrieve_distinct_states(self.skipping_engine, self.k)
        init_kv_store_db(self.kv_engine)
        evaluator = PolicyEvaluation(
            full_info_engine=self.full_info_engine,
            kv_engine=self.kv_engine,
            v_states=self.v_states,
            movie_scores=self.movie_scores,
            gamma=self.gamma,
            threshold=self.threshold
        )
        improver = PolicyImprovement(
            full_info_engine=self.full_info_engine,
            kv_engine=self.kv_engine,
            v_states=self.v_states,
            movie_scores=self.movie_scores,
            list_size=self.list_size
        )
        iteration = 1
        while True:
            evaluator.evaluate(states)
            stable = improver.improve(states)
            if stable:
                break
            iteration += 1
        flush_rows = [{"s": s, "value": v} for s, v in self.v_states.items()]
        if flush_rows:
            flush_value_states(self.kv_engine, flush_rows)


if __name__ == "__main__":
    solver = MDPSolver()
    solver.solve_mdp()
