from collections import defaultdict

from chap_5.api.mdp_rework.predictive_model.improvement.setup_info_dict.data.pg_sql import fetch_movie_scores, \
    get_pg_engine
from chap_5.api.mdp_rework.predictive_model.improvement.skipping.data.sql import fetch_distinct_states, \
    create_skipping_database, retrieve_distinct_states, retrieve_skipping_full_info_for_batch
from chap_5.api.mdp_rework.predictive_model.improvement.similarity.data.sql import create_similarity_database, \
    retrieve_similarity_full_info_for_batch
from chap_5.api.mdp_rework.shared.endode_to_blob import decode, unpack_transitions

EPSILON = 1e-6


class SetupInfoDict:
    def __init__(self, k: int):
        self.k = k
        self.skipping_engine = create_skipping_database()
        self.similarity_engine = create_similarity_database()
        self.pg_engine = get_pg_engine()

    def build_info_index(self):
        observed_states = retrieve_distinct_states(self.skipping_engine, self.k)
        r = fetch_movie_scores(self.pg_engine)

        if not observed_states:
            return

        # On se limite aux 100 premiers états pour le test
        for i, state in enumerate(observed_states[:100]):
            print(f"\n=== État {i + 1} : {decode(state)} ===")

            skipping_successors, similarity_successors = self._get_successors(state)
            skip_by_k = self.transform_successor_to_dict(state, skipping_successors)
            sim_by_k = self.transform_successor_to_dict(state, similarity_successors)

            unified_probs = self.get_tr(skip_by_k, sim_by_k)

            # Affichage de Skip, Sim et TR Unifié sur la même ligne pour chaque k
            for k_val in range(self.k, 0, -1):
                sum_skip = sum(skip_by_k.get(k_val, {}).values())
                sum_sim = sum(sim_by_k.get(k_val, {}).values())
                sum_unified = sum(unified_probs.get(k_val, {}).values())
                print(
                    f"  k={k_val} | Sum Skip: {sum_skip:.4f} | Sum Sim: {sum_sim:.4f} | Sum TR Unifié: {sum_unified:.4f}")

            # Calcul et affichage de la somme globale de tr_predict
            tr_predict = self._build_tr_predict_function(state)
            sum_tr_predict = sum(tr_predict.values())
            print(f"  --> SOMME TR_PREDICT: {sum_tr_predict:.4f}")

    def _get_successors(self, state):
        skipping_successors = retrieve_skipping_full_info_for_batch(self.skipping_engine, {state})
        similarity_successors = retrieve_similarity_full_info_for_batch(self.similarity_engine, {state})
        return skipping_successors, similarity_successors

    def transform_successor_to_dict(self, state, successors):
        res = defaultdict(dict)
        if state in successors:
            for succ_blob, proba_blob in successors[state]:
                transitions = unpack_transitions(succ_blob, proba_blob)
                for s_prime, proba in transitions:
                    k_val = len(decode(s_prime))
                    res[k_val][s_prime] = proba
        return res

    def get_tr(self, skip_by_k, sim_by_k):
        unified_probs = defaultdict(dict)
        for k_val in range(self.k, 0, -1):
            skip_k = skip_by_k.get(k_val, {})
            sim_k = sim_by_k.get(k_val, {})

            sum_skip = sum(skip_k.values())
            sum_sim = sum(sim_k.values())

            # On détermine les poids en fonction de la disponibilité des modèles
            if sum_skip > 0 and sum_sim > 0:
                w_skip = 0.5
                w_sim = 0.5
            elif sum_skip > 0:
                w_skip = 1.0
                w_sim = 0.0
            elif sum_sim > 0:
                w_skip = 0.0
                w_sim = 1.0
            else:
                continue

            all_s_primes = set(skip_k.keys()) | set(sim_k.keys())

            for s_prime in all_s_primes:
                p_skip = skip_k.get(s_prime, 0.0)
                p_sim = sim_k.get(s_prime, 0.0)

                unified_probs[k_val][s_prime] = (w_skip * p_skip) + (w_sim * p_sim)

        return unified_probs

    def _build_tr_function(self, state):
        skipping_successors, similarity_successors = self._get_successors(state)
        skip_by_k = self.transform_successor_to_dict(state, skipping_successors)
        sim_by_k = self.transform_successor_to_dict(state, similarity_successors)
        unified_probs = self.get_tr(skip_by_k, sim_by_k)
        return unified_probs

    def _build_tr_predict_function(self, state):
        unified_probs = self._build_tr_function(state)

        valid_k_count = 0
        for k_val in range(self.k, 0, -1):
            if unified_probs.get(k_val):
                valid_k_count += 1

        if valid_k_count == 0:
            return {}

        tr_predict = defaultdict(float)
        weight = 1.0 / valid_k_count

        for k_val in range(self.k, 0, -1):
            probs = unified_probs.get(k_val)
            if probs:
                for s_prime, proba in probs.items():
                    tr_predict[s_prime] += weight * proba

        return dict(tr_predict)


if __name__ == "__main__":
    setup = SetupInfoDict(k=3)
    setup.build_info_index()