import numpy as np
from chap_5.api.mdp.mdp.data.sql import (
    get_by_batch_policy, flush_policy
)
from chap_5.api.mdp.predictive_model.improvement.setup_info_dict.data.sql import retrieve_full_info_for_batch


class PolicyImprovement:
    def __init__(self, full_info_engine, kv_engine, v_states: dict, movie_scores: dict, list_size: int,
                 batch_size: int = 10000):
        self.full_info_engine = full_info_engine
        self.kv_engine = kv_engine
        self.v_states = v_states
        self.list_size = list_size
        self.batch_size = batch_size

        max_movie_id = max(movie_scores.keys()) if movie_scores else 0
        self.scores_lookup = np.zeros(max_movie_id + 1, dtype=np.float64)
        for movie_id, score in movie_scores.items():
            self.scores_lookup[movie_id] = score

    def improve(self, states: list) -> bool:
        stable = True
        total_states = len(states)

        for i in range(0, total_states, self.batch_size):
            batch_states = states[i:i + self.batch_size]
            if not self._improve_batch(batch_states):
                stable = False

        return stable

    def _improve_batch(self, batch_states: list) -> bool:
        full_info_batch, old_policies = self._prepare_batch(batch_states)

        flush_rows = []
        batch_stable = True

        for s in batch_states:
            if s not in full_info_batch:
                continue

            info = full_info_batch[s]
            new_policy = self._compute_top_k(info)
            old_policy = old_policies.get(s, [])

            if set(new_policy) != set(old_policy):
                batch_stable = False
            flush_rows.append({"s": s, "value": new_policy})

        if flush_rows:
            flush_policy(self.kv_engine, flush_rows)

        return batch_stable

    def _prepare_batch(self, batch_states: list) -> tuple:
        batch_set = set(batch_states)
        full_info_batch = retrieve_full_info_for_batch(self.full_info_engine, batch_set)
        policies = get_by_batch_policy(self.kv_engine, batch_set)
        return full_info_batch, policies

    def _compute_top_k(self, info: tuple) -> list:
        last_items_list = []
        p_reco_list = []
        p_not_reco_list = []
        v_primes_list = []

        for i_k in range(3):
            s_primes = info[i_k * 4]
            if len(s_primes) > 0:
                last_items_list.append(s_primes[:, -1])
                p_reco_list.append(info[i_k * 4 + 2])
                p_not_reco_list.append(info[i_k * 4 + 3])

                s_primes_list = s_primes.tolist()
                v_primes_list.append(np.array([
                    self.v_states.get(tuple(row), self.scores_lookup[row[-1]])
                    for row in s_primes_list
                ]))

        if not last_items_list:
            return []

        last_items = np.concatenate(last_items_list)
        p_reco = np.concatenate(p_reco_list)
        p_not_reco = np.concatenate(p_not_reco_list)
        v_primes = np.concatenate(v_primes_list)

        delta = (p_reco - p_not_reco) * v_primes

        sorted_indices = np.argsort(delta)[::-1]
        top_k_indices = sorted_indices[:self.list_size]

        return last_items[top_k_indices].tolist()
