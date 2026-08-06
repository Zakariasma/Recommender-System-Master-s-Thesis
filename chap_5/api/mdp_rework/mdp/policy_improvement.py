import time
import numpy as np
from chap_5.api.mdp_rework.mdp.data.sql import (
    get_by_batch_policy, flush_policy
)
from chap_5.api.mdp_rework.predictive_model.improvement.setup_info_dict.data.sql import retrieve_full_info_for_batch


class PolicyImprovement:
    def __init__(self, full_info_engine, kv_engine, v_states: dict, list_size: int, batch_size: int = 10000):
        self.full_info_engine = full_info_engine
        self.kv_engine = kv_engine
        self.v_states = v_states  # Dictionnaire en mémoire partagée
        self.list_size = list_size
        self.batch_size = batch_size

    def improve(self, states: list) -> bool:
        stable = True
        total_states = len(states)
        t_improve_start = time.perf_counter()
        processed = 0

        for i in range(0, total_states, self.batch_size):
            batch_states = states[i:i + self.batch_size]
            if not self._improve_batch(batch_states):
                stable = False

            processed += len(batch_states)
            elapsed = time.perf_counter() - t_improve_start
            speed = processed / elapsed if elapsed > 0 else 0
            print(f"  Improve {processed:,}/{total_states:,} | {speed:,.0f} etats/s", flush=True)

        print(f"\n  [Improve] terminé — {time.perf_counter() - t_improve_start:.1f}s. Politique stable : {stable}", flush=True)
        return stable

    def _improve_batch(self, batch_states: list) -> bool:
        t0 = time.perf_counter()
        full_info_batch, old_policies = self._prepare_batch(batch_states)
        t_prep = time.perf_counter() - t0

        flush_rows = []
        batch_stable = True

        t0 = time.perf_counter()
        for s in batch_states:
            if s not in full_info_batch:
                continue

            info = full_info_batch[s]
            new_policy = self._compute_top_k(info)
            old_policy = old_policies.get(s, [])

            if set(new_policy) != set(old_policy):
                batch_stable = False
            flush_rows.append({"s": s, "value": new_policy})
        t_compute = time.perf_counter() - t0

        t0 = time.perf_counter()
        if flush_rows:
            flush_policy(self.kv_engine, flush_rows)
        t_flush = time.perf_counter() - t0

        print(f"    [TIME] prep={t_prep:.2f}s compute={t_compute:.2f}s flush={t_flush:.2f}s", flush=True)

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
            s_primes = info[i_k * 5]
            if len(s_primes) > 0:
                last_items_list.append(s_primes[:, -1])
                p_reco_list.append(info[i_k * 5 + 3])
                p_not_reco_list.append(info[i_k * 5 + 4])

                # On calcule V(s') pour chaque k séparément, puis on ajoute à la liste
                v_primes_list.append(np.array([self.v_states.get(tuple(row), 0.0) for row in s_primes.tolist()]))

        if not last_items_list:
            return []

        # On concatène uniquement des tableaux 1D, ça marche parfaitement !
        last_items = np.concatenate(last_items_list)
        p_reco = np.concatenate(p_reco_list)
        p_not_reco = np.concatenate(p_not_reco_list)
        v_primes = np.concatenate(v_primes_list)

        # Calcul vectorisé du Gain Additionnel
        delta = (p_reco - p_not_reco) * v_primes

        # Tri par ordre décroissant et top kappa
        sorted_indices = np.argsort(delta)[::-1]
        top_k_indices = sorted_indices[:self.list_size]

        return last_items[top_k_indices].tolist()