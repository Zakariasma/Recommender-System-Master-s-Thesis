import time
import numpy as np

from chap_5.api.mdp_rework.mdp.data.sql import get_by_batch_policy
from chap_5.api.mdp_rework.predictive_model.improvement.setup_info_dict.data.sql import retrieve_full_info_for_batch
from chap_5.api.mdp_rework.shared.debug_log import reset_progress


class PolicyEvaluation:
    def __init__(self, full_info_engine, kv_engine, v_states: dict, gamma: float, threshold: float,
                 batch_size: int = 10000):
        self.full_info_engine = full_info_engine
        self.kv_engine = kv_engine
        self.v_states = v_states  # Dictionnaire en mémoire partagée
        self.gamma = gamma
        self.threshold = threshold
        self.batch_size = batch_size

    def evaluate(self, states: list) -> dict:
        total_states = len(states)
        sweep = 0

        while True:
            delta = 0.0
            processed = 0
            reset_progress()
            t_sweep_start = time.perf_counter()

            for i in range(0, total_states, self.batch_size):
                batch_states = states[i:i + self.batch_size]
                batch_delta = self._evaluate_batch(batch_states)
                delta = max(delta, batch_delta)

                processed += len(batch_states)
                elapsed = time.perf_counter() - t_sweep_start
                speed = processed / elapsed if elapsed > 0 else 0
                print(f"  Eval {processed:,}/{total_states:,} | {speed:,.0f} etats/s", flush=True)

            print(f"\n  [Eval] sweep {sweep} terminé — delta={delta:.6f} ({time.perf_counter() - t_sweep_start:.1f}s)",
                  flush=True)
            sweep += 1

            if delta < self.threshold:
                break

    def _evaluate_batch(self, batch_states: list) -> float:
        t0 = time.perf_counter()
        full_info_batch, policies = self._prepare_batch(batch_states)
        t_prep = time.perf_counter() - t0

        max_delta = 0.0

        t0 = time.perf_counter()
        for s in batch_states:
            if s not in full_info_batch:
                continue

            info = full_info_batch[s]
            policy_items = set(policies.get(s, []))

            v_old = self.v_states.get(s, 0.0)
            v_new = self._compute_v_new(info, policy_items)

            max_delta = max(max_delta, abs(v_new - v_old))
            self.v_states[s] = v_new
        t_compute = time.perf_counter() - t0

        print(f"    [TIME] prep={t_prep:.2f}s compute={t_compute:.2f}s", flush=True)

        return max_delta

    def _prepare_batch(self, batch_states: list) -> tuple:
        batch_set = set(batch_states)
        full_info_batch = retrieve_full_info_for_batch(self.full_info_engine, batch_set)
        policies = get_by_batch_policy(self.kv_engine, batch_set)
        return full_info_batch, policies

    def _compute_v_new(self, info: tuple, policy_items: set) -> float:
        policy_arr = np.array(list(policy_items), dtype='>u2') if policy_items else np.array([], dtype='>u2')

        proba_mdp_list = []
        sum_proba = 0.0

        for i_k in range(3):
            s_primes = info[i_k * 5]
            p_reco = info[i_k * 5 + 3]
            p_not_reco = info[i_k * 5 + 4]

            if len(s_primes) == 0:
                proba_mdp_list.append(np.array([]))
                continue

            last_items = s_primes[:, -1]
            if len(policy_arr) > 0:
                is_reco = np.isin(last_items, policy_arr)
            else:
                is_reco = np.zeros(len(last_items), dtype=bool)

            proba_mdp = np.where(is_reco, p_reco, p_not_reco)
            proba_mdp_list.append(proba_mdp)
            sum_proba += np.sum(proba_mdp)

        max_allowed_sum = 0.99
        scale = max_allowed_sum / sum_proba if sum_proba > max_allowed_sum else 1.0

        v_new = 0.0

        for i_k in range(3):
            s_primes = info[i_k * 5]
            if len(s_primes) == 0:
                continue

            tr_predicts = info[i_k * 5 + 1]
            rewards = info[i_k * 5 + 2]
            proba_mdp = proba_mdp_list[i_k] * scale

            immediate_rewards = np.zeros_like(rewards)
            mask = tr_predicts != 0
            immediate_rewards[mask] = rewards[mask] / tr_predicts[mask]

            s_primes_list = s_primes.tolist()
            v_primes = np.array([self.v_states.get(tuple(s), 0.0) for s in s_primes_list])

            raw_contributions = proba_mdp * (immediate_rewards + self.gamma * v_primes)
            v_new += float(np.sum(raw_contributions))

        return v_new