from chap_5.api.mdp_rework.mdp.data.sql import (
    get_by_batch_policy, get_by_batch_states, flush_value_states
)
from chap_5.api.mdp_rework.predictive_model.improvement.setup_info_dict.data.sql import retrieve_full_info_for_batch


class PolicyEvaluation:
    def __init__(self, full_info_engine, kv_engine, gamma: float, threshold: float, batch_size: int = 10000):
        self.full_info_engine = full_info_engine
        self.kv_engine = kv_engine
        self.gamma = gamma
        self.threshold = threshold
        self.batch_size = batch_size

    def evaluate(self, states: list) -> dict:
        total_states = len(states)

        while True:
            delta = 0.0

            for i in range(0, total_states, self.batch_size):
                batch_states = states[i:i + self.batch_size]
                batch_delta = self._evaluate_batch(batch_states)
                delta = max(delta, batch_delta)

            if delta < self.threshold:
                break

    def _evaluate_batch(self, batch_states: list) -> float:
        full_info_batch, policies = self._prepare_batch(batch_states)

        all_s_primes = self._get_all_s_primes(full_info_batch)
        all_states_to_fetch = all_s_primes.union(set(batch_states))
        v_states = get_by_batch_states(self.kv_engine, all_states_to_fetch)

        flush_rows = []
        max_delta = 0.0

        for s in batch_states:
            if s not in full_info_batch:
                continue

            info = full_info_batch[s]
            policy_items = set(policies.get(s, []))

            v_old = v_states.get(s, 0.0)
            v_new = self._compute_v_new(info, policy_items, v_states)

            max_delta = max(max_delta, abs(v_new - v_old))
            flush_rows.append({"s": s, "value": v_new})

        if flush_rows:
            flush_value_states(self.kv_engine, flush_rows)

        return max_delta

    def _prepare_batch(self, batch_states: list) -> tuple:
        batch_set = set(batch_states)
        full_info_batch = retrieve_full_info_for_batch(self.full_info_engine, batch_set)
        policies = get_by_batch_policy(self.kv_engine, batch_set)
        return full_info_batch, policies

    def _get_all_s_primes(self, full_info_batch: dict) -> set:
        all_s_primes = set()
        for info in full_info_batch.values():
            for s_prime_list in info["s_"]:
                all_s_primes.add(tuple(s_prime_list))
        return all_s_primes

    def _compute_v_new(self, info: dict, policy_items: set, v_states: dict) -> float:
        v_new = 0.0
        n = len(info["s_"])

        sum_proba = 0.0
        for j in range(n):
            last_item = tuple(info["s_"][j])[-1]
            proba_mdp = info["proba_reco"][j] if last_item in policy_items else info["proba_not_reco"][j]
            sum_proba += proba_mdp

        max_allowed_sum = 0.99
        if sum_proba > max_allowed_sum:
            scale = max_allowed_sum / sum_proba
        else:
            scale = 1.0

        for j in range(n):
            s_prime = tuple(info["s_"][j])
            tr_pred = info["tr_predict"][j]
            reward = info["reward"][j]

            if tr_pred == 0:
                continue

            immediate_reward = reward / tr_pred
            last_item = s_prime[-1]

            proba_mdp = info["proba_reco"][j] if last_item in policy_items else info["proba_not_reco"][j]
            if proba_mdp == 0:
                continue

            proba_mdp *= scale

            v_prime = v_states.get(s_prime, 0.0)
            v_new += proba_mdp * (immediate_reward + self.gamma * v_prime)

        return v_new