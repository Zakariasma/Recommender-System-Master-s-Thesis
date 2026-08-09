from chap_5.api.mdp.predictive_model.improvement.setup_info_dict.data.sql import create_full_info_database, \
    init_full_info_db, flush, create_index
from chap_5.api.mdp.predictive_model.improvement.setup_info_dict.utils.get_popularity_dict import get_item_counts
from chap_5.api.mdp.predictive_model.improvement.skipping.data.sql import create_skipping_database, \
    retrieve_distinct_states, retrieve_skipping_full_info_for_batch
from chap_5.api.mdp.predictive_model.improvement.similarity.data.sql import create_similarity_database, \
    retrieve_similarity_full_info_for_batch
from chap_5.api.mdp.shared.debug_log import log_progress, reset_progress
from chap_5.api.mdp.predictive_model.improvement.setup_info_dict.alpha_beta import AlphaBeta


class SetupInfoDict:
    def __init__(self, k: int):
        self.k = k
        self.skipping_engine = create_skipping_database()
        self.similarity_engine = create_similarity_database()

        item_counts, total_items = get_item_counts()
        self.alpha_beta_calc = AlphaBeta(item_counts, total_items)

        self.full_info_engine = create_full_info_database()
        init_full_info_db(self.full_info_engine)

    def _preprocess_successors(self, successors: dict) -> dict:
        result = {}
        for state, k_dict in successors.items():
            by_k = {}
            for k_val, (s_primes, probs) in k_dict.items():
                by_k[k_val] = {}
                for s_prime, proba in zip(s_primes, probs):
                    by_k[k_val][tuple(s_prime)] = proba
            result[state] = by_k
        return result

    def build_info_index(self):
        observed_states = retrieve_distinct_states(self.skipping_engine, self.k)
        total_states = len(observed_states)

        batch_rows = []
        processed = 0
        reset_progress()

        for i in range(0, total_states, 10000):
            batch_states = observed_states[i:i + 10000]
            batch_set = set(batch_states)

            raw_skipping = retrieve_skipping_full_info_for_batch(self.skipping_engine, batch_set)
            skipping_pre = self._preprocess_successors(raw_skipping)

            raw_similarity = retrieve_similarity_full_info_for_batch(self.similarity_engine, batch_set)
            similarity_pre = self._preprocess_successors(raw_similarity)

            for state_tuple in batch_states:
                tr_predict = self._build_tr_predict_function(state_tuple, skipping_pre, similarity_pre)
                if not tr_predict:
                    processed += 1
                    continue

                batch_rows.append(self._create_row(state_tuple, tr_predict))
                processed += 1

            if batch_rows:
                flush(self.full_info_engine, batch_rows)
                batch_rows.clear()

            log_progress(processed, total_states, 'SetupInfo')

        if batch_rows:
            flush(self.full_info_engine, batch_rows)

        create_index(self.full_info_engine)

    def _create_row(self, state_tuple: tuple, tr_predict: dict) -> dict:
        sum_alpha_p = 0.0
        alpha_cache = {}
        for s_prime_tuple, proba in tr_predict.items():
            r = s_prime_tuple[-1]
            alpha = self.alpha_beta_calc.compute_alpha(r)
            alpha_cache[r] = alpha
            sum_alpha_p += alpha * proba

        row = {"s": state_tuple}
        for k_val in [1, 2, 3]:
            row[f"k{k_val}_s_"] = []
            row[f"k{k_val}_tr"] = []
            row[f"k{k_val}_p_reco"] = []
            row[f"k{k_val}_p_not_reco"] = []

        for s_prime_tuple, p_s_r in tr_predict.items():
            k_val = len(s_prime_tuple)
            if k_val not in [1, 2, 3]:
                continue

            r = s_prime_tuple[-1]
            alpha = alpha_cache[r]
            beta = self.alpha_beta_calc.compute_beta(alpha, p_s_r, sum_alpha_p)

            proba_reco = alpha * p_s_r
            proba_not_reco = beta * p_s_r

            row[f"k{k_val}_s_"].append(list(s_prime_tuple))
            row[f"k{k_val}_tr"].append(p_s_r)
            row[f"k{k_val}_p_reco"].append(proba_reco)
            row[f"k{k_val}_p_not_reco"].append(proba_not_reco)

        return row

    def get_tr(self, skip_by_k: dict, sim_by_k: dict) -> dict:
        unified_probs = {}
        for k_val in range(self.k, 0, -1):
            skip_k = skip_by_k.get(k_val, {})
            sim_k = sim_by_k.get(k_val, {})

            sum_skip = sum(skip_k.values())
            sum_sim = sum(sim_k.values())

            if sum_skip > 0 and sum_sim > 0:
                w_skip, w_sim = 0.5, 0.5
            elif sum_skip > 0:
                w_skip, w_sim = 1.0, 0.0
            elif sum_sim > 0:
                w_skip, w_sim = 0.0, 1.0
            else:
                continue

            all_s_primes = skip_k.keys() | sim_k.keys()
            unified_probs[k_val] = {}

            for s_prime_tuple in all_s_primes:
                p_skip = skip_k.get(s_prime_tuple, 0.0)
                p_sim = sim_k.get(s_prime_tuple, 0.0)
                unified_probs[k_val][s_prime_tuple] = (w_skip * p_skip) + (w_sim * p_sim)

        return unified_probs

    def _build_tr_function(self, state_tuple: tuple, skipping_pre: dict, similarity_pre: dict) -> dict:
        skip_by_k = skipping_pre.get(state_tuple, {})
        sim_by_k = similarity_pre.get(state_tuple, {})
        return self.get_tr(skip_by_k, sim_by_k)

    def _build_tr_predict_function(self, state_tuple: tuple, skipping_pre: dict, similarity_pre: dict) -> dict:
        unified_probs = self._build_tr_function(state_tuple, skipping_pre, similarity_pre)

        valid_k_count = len(unified_probs)
        if valid_k_count == 0:
            return {}

        weight = 1.0 / valid_k_count
        tr_predict = {}

        for probs in unified_probs.values():
            for s_prime, proba in probs.items():
                tr_predict[s_prime] = tr_predict.get(s_prime, 0.0) + weight * proba

        return tr_predict


if __name__ == "__main__":
    setup = SetupInfoDict(k=3)
    setup.build_info_index()