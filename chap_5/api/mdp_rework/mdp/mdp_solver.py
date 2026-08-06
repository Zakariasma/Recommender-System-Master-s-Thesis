from chap_5.api.mdp_rework.config import K, GAMMA_RL, THRESHOLD, LIST_SIZE
from chap_5.api.mdp_rework.mdp.data.sql import create_database, init_kv_store_db
from chap_5.api.mdp_rework.predictive_model.improvement.setup_info_dict.data.sql import create_full_info_database
from chap_5.api.mdp_rework.predictive_model.improvement.skipping.data.sql import (
    retrieve_distinct_states, create_skipping_database
)
from chap_5.api.mdp_rework.mdp.policy_evaluation import PolicyEvaluation




class MDPSolver:
    def __init__(self):
        self.k = K
        self.gamma = GAMMA_RL
        self.threshold = THRESHOLD
        self.list_size = LIST_SIZE

        self.skipping_engine = create_skipping_database()
        self.full_info_engine = create_full_info_database()
        self.kv_engine = create_database()

    def solve_mdp(self, states: list = None):
        if states is None:
            states = retrieve_distinct_states(self.skipping_engine, self.k)

        init_kv_store_db(self.kv_engine)

        evaluator = PolicyEvaluation(
            full_info_engine=self.full_info_engine,
            kv_engine=self.kv_engine,
            gamma=self.gamma,
            threshold=self.threshold
        )

        improver = PolicyImprovement(
            full_info_engine=self.full_info_engine,
            kv_engine=self.kv_engine,
            list_size=self.list_size
        )

        # Boucle principale de Policy Iteration
        iteration = 1
        while True:
            print(f"\n=== Itération {iteration} ===")

            print("Évaluation de la politique...")
            evaluator.evaluate(states)

            print("Amélioration de la politique...")
            stable = improver.improve(states)

            if stable:
                print("\nPolitique stable atteinte ! MDP résolu.")
                break

            iteration += 1


if __name__ == "__main__":
    solver = MDPSolver()
    solver.solve_mdp()