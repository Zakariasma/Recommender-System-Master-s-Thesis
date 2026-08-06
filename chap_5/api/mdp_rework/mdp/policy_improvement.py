from chap_5.api.mdp_rework.mdp.data.sql import (
    get_by_batch_policy, get_by_batch_states, flush_policy
)
from chap_5.api.mdp_rework.predictive_model.improvement.setup_info_dict.data.sql import retrieve_full_info_for_batch


class PolicyImprovement:
    def __init__(self, full_info_engine, kv_engine, list_size: int, batch_size: int = 10000):
        self.full_info_engine = full_info_engine
        self.kv_engine = kv_engine
        self.list_size = list_size
        self.batch_size = batch_size

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
        all_s_primes = self._get_all_s_primes(full_info_batch)
        v_states = get_by_batch_states(self.kv_engine, all_s_primes)
        flush_rows = []
        batch_stable = True

        for s in batch_states:
            if s not in full_info_batch:
                continue

            info = full_info_batch[s]
            new_policy = self._compute_top_k(info, v_states)
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

    def _get_all_s_primes(self, full_info_batch: dict) -> set:
        all_s_primes = set()
        for info in full_info_batch.values():
            for s_prime_list in info["s_"]:
                all_s_primes.add(tuple(s_prime_list))
        return all_s_primes

    def _compute_top_k(self, info: dict, v_states: dict) -> list:
        gains = []
        n = len(info["s_"])

        for j in range(n):
            s_prime = tuple(info["s_"][j])
            p_reco = info["proba_reco"][j]
            p_not_reco = info["proba_not_reco"][j]
            delta = (p_reco - p_not_reco) * v_states.get(s_prime, 0.0)
            r = s_prime[-1]
            gains.append((r, delta))

        top_k = sorted(gains, key=lambda x: -x[1])[:self.list_size]
        return [item for item, _ in top_k]