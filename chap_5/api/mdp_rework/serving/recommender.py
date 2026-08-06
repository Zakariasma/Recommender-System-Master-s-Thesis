from chap_5.api.mdp_rework.config import LIST_SIZE, BOLTZMANN_TEMPERATURE
from chap_5.api.mdp_rework.mdp.data.sql import create_database, get_by_batch_policy, get_by_batch_states
from chap_5.api.mdp_rework.predictive_model.improvement.setup_info_dict.data.sql import create_full_info_database, retrieve_full_info_for_batch
from chap_5.api.mdp_rework.serving.boltzmann import boltzmann, sample


class Recommender:

    def __init__(self, temperature: float = BOLTZMANN_TEMPERATURE):
        self.kv_engine = create_database()
        self.full_info_engine = create_full_info_database()
        self.temperature = temperature

    def recommend(self, s: tuple, n: int = LIST_SIZE) -> list:
        policies = get_by_batch_policy(self.kv_engine, {s})
        policy_items = policies.get(s, [])

        full_info_batch = retrieve_full_info_for_batch(self.full_info_engine, {s})
        info = full_info_batch.get(s)

        if not info and not policy_items:
            return []

        pool = {}

        if info:
            for i_k in range(3):
                s_primes = info[i_k * 5]
                tr_predicts = info[i_k * 5 + 1]

                if len(s_primes) > 0:
                    last_items = s_primes[:, -1]

                    for i, r in enumerate(last_items.tolist()):
                        if r not in pool:
                            pool[r] = float(tr_predicts[i])

        recommended_items = []

        if policy_items:
            top_item = policy_items[0]
            recommended_items.append(top_item)

            if top_item in pool:
                del pool[top_item]

        if len(recommended_items) < n and pool:
            next_states = {r: s[1:] + (r,) for r in pool.keys()}

            v_dict = get_by_batch_states(self.kv_engine, set(next_states.values()))

            v_pool = {}

            for r, s_next in next_states.items():
                v_next = v_dict.get(s_next)

                if v_next is not None:
                    v_pool[r] = v_next

            if v_pool:
                distribution = boltzmann(v_pool, self.temperature)
            else:
                distribution = boltzmann(pool, self.temperature)

            rest = sample(distribution, n - len(recommended_items))
            recommended_items.extend(rest)

        return recommended_items


if __name__ == "__main__":
    recommender = Recommender()

    state_test = (11, 1536, 8396)

    print(f"Recommandations pour l'état {state_test} :")
    print(recommender.recommend(state_test))