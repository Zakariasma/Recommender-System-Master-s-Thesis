import time
from collections import defaultdict
import orjson

from chap_5.api.mdp_rework.predictive_model.improvement.setup_info_dict.data.pg_sql import fetch_movie_scores, \
    get_pg_engine
from chap_5.api.mdp_rework.predictive_model.improvement.setup_info_dict.data.sql import create_full_info_database, \
    init_full_info_db, flush, create_index
from chap_5.api.mdp_rework.predictive_model.improvement.setup_info_dict.utils.get_popularity_dict import get_item_counts
from chap_5.api.mdp_rework.predictive_model.improvement.skipping.data.sql import create_skipping_database, \
    retrieve_distinct_states, retrieve_skipping_full_info_for_batch
from chap_5.api.mdp_rework.predictive_model.improvement.similarity.data.sql import create_similarity_database, \
    retrieve_similarity_full_info_for_batch
from chap_5.api.mdp_rework.shared.debug_log import log_progress, reset_progress
from chap_5.api.mdp_rework.predictive_model.improvement.setup_info_dict.alpha_beta import AlphaBeta


class SetupInfoDict:
    def __init__(self, k: int):
        self.k = k
        self.skipping_engine = create_skipping_database()
        self.similarity_engine = create_similarity_database()
        self.pg_engine = get_pg_engine()
        self.movie_scores = fetch_movie_scores(self.pg_engine)

        item_counts, total_items = get_item_counts()
        self.alpha_beta_calc = AlphaBeta(item_counts, total_items)

        self.full_info_engine = create_full_info_database()
        init_full_info_db(self.full_info_engine)

    def build_info_index(self):
        t0 = time.perf_counter()
        observed_states = retrieve_distinct_states(self.skipping_engine, self.k)
        print(f"[TIME] load_states={time.perf_counter() - t0:.2f}s n={len(observed_states)}", flush=True)

        total_states = len(observed_states)

        batch_rows = []
        processed = 0
        reset_progress()

        for i in range(0, total_states, 10000):
            batch_states = observed_states[i:i + 10000]
            batch_set = set(batch_states)

            t0 = time.perf_counter()
            skipping_successors = retrieve_skipping_full_info_for_batch(self.skipping_engine, batch_set)
            t_skip = time.perf_counter() - t0

            t0 = time.perf_counter()
            similarity_successors = retrieve_similarity_full_info_for_batch(self.similarity_engine, batch_set)
            t_sim = time.perf_counter() - t0

            t0 = time.perf_counter()
            for state_tuple in batch_states:
                tr_predict = self._build_tr_predict_function(state_tuple, skipping_successors, similarity_successors)
                if not tr_predict:
                    processed += 1
                    continue

                row = self._create_row(state_tuple, tr_predict)
                batch_rows.append(row)

                processed += 1
            t_compute = time.perf_counter() - t0

            t0 = time.perf_counter()
            if batch_rows:
                flush(self.full_info_engine, batch_rows)
                batch_rows.clear()
            t_flush = time.perf_counter() - t0

            log_progress(processed, total_states, 'SetupInfo')
            print(
                f"[TIME] skip={t_skip:.2f}s sim={t_sim:.2f}s "
                f"compute={t_compute:.2f}s flush={t_flush:.2f}s",
                flush=True,
            )

        if batch_rows:
            flush(self.full_info_engine, batch_rows)

        t_idx = time.perf_counter()
        create_index(self.full_info_engine)
        print(f"[TIME] create_index={time.perf_counter() - t_idx:.2f}s", flush=True)

    def _create_row(self, state_tuple: tuple, tr_predict: dict) -> dict:
        sum_alpha_p = 0.0
        alpha_cache = {}
        for s_prime_tuple, proba in tr_predict.items():
            r = s_prime_tuple[-1]
            alpha = self.alpha_beta_calc.compute_alpha(r)
            alpha_cache[r] = alpha
            sum_alpha_p += alpha * proba

        s_prime_list = []
        tr_predict_list = []
        reward_list = []
        proba_reco_list = []
        proba_not_reco_list = []

        for s_prime_tuple, p_s_r in tr_predict.items():
            r = s_prime_tuple[-1]

            alpha = alpha_cache[r]
            beta = self.alpha_beta_calc.compute_beta(alpha, p_s_r, sum_alpha_p)

            proba_reco = alpha * p_s_r
            proba_not_reco = beta * p_s_r

            immediate_reward = self.movie_scores.get(r, 0.0)
            weighted_reward = immediate_reward * p_s_r

            s_prime_list.append(list(s_prime_tuple))
            tr_predict_list.append(p_s_r)
            reward_list.append(weighted_reward)
            proba_reco_list.append(proba_reco)
            proba_not_reco_list.append(proba_not_reco)

        return {
            "s": orjson.dumps(list(state_tuple)).decode('utf-8'),
            "s_": orjson.dumps(s_prime_list).decode('utf-8'),
            "tr_predict": orjson.dumps(tr_predict_list).decode('utf-8'),
            "reward": orjson.dumps(reward_list).decode('utf-8'),
            "proba_reco": orjson.dumps(proba_reco_list).decode('utf-8'),
            "proba_not_reco": orjson.dumps(proba_not_reco_list).decode('utf-8')
        }

    def transform_successor_to_dict(self, state_tuple: tuple, successors: dict) -> dict:
        res = defaultdict(dict)
        if state_tuple in successors:
            for s_prime_tuple, proba in successors[state_tuple]:
                k_val = len(s_prime_tuple)
                res[k_val][s_prime_tuple] = proba
        return res

    def get_tr(self, skip_by_k: dict, sim_by_k: dict) -> dict:
        unified_probs = defaultdict(dict)
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

            all_s_primes = set(skip_k.keys()) | set(sim_k.keys())

            for s_prime_tuple in all_s_primes:
                p_skip = skip_k.get(s_prime_tuple, 0.0)
                p_sim = sim_k.get(s_prime_tuple, 0.0)
                unified_probs[k_val][s_prime_tuple] = (w_skip * p_skip) + (w_sim * p_sim)

        return unified_probs

    def _build_tr_function(self, state_tuple: tuple, skipping_successors: dict, similarity_successors: dict) -> dict:
        skip_by_k = self.transform_successor_to_dict(state_tuple, skipping_successors)
        sim_by_k = self.transform_successor_to_dict(state_tuple, similarity_successors)
        return self.get_tr(skip_by_k, sim_by_k)

    def _build_tr_predict_function(self, state_tuple: tuple, skipping_successors: dict,
                                   similarity_successors: dict) -> dict:
        unified_probs = self._build_tr_function(state_tuple, skipping_successors, similarity_successors)

        valid_k_count = sum(1 for k_val in range(self.k, 0, -1) if unified_probs.get(k_val))
        if valid_k_count == 0:
            return {}

        tr_predict = defaultdict(float)
        weight = 1.0 / valid_k_count

        for k_val in range(self.k, 0, -1):
            probs = unified_probs.get(k_val)
            if probs:
                for s_prime_tuple, proba in probs.items():
                    tr_predict[s_prime_tuple] += weight * proba

        return dict(tr_predict)


if __name__ == "__main__":
    setup = SetupInfoDict(k=3)
    setup.build_info_index()