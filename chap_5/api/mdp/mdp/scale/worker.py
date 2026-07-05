import gc
import numpy as np
from collections import defaultdict

from chap_5.api.mdp.config import DATABASE_URL, GAMMA_BOOST
from chap_5.api.mdp.mdp.scale.postgres_batch_store import PostgresBatchStore
from chap_5.api.mdp.predictive_model.state_key import encode, decode
from chap_5.api.mdp.mdp.alpha_beta import AlphaBeta
from chap_5.api.mdp.mdp.popularity import PopularityModel

EPSILON = 1e-6
K_WEIGHTS = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}

# Globaux pour les workers (hérités via Copy-On-Write)
_global_states = None
_global_state_to_idx = None
_V_raw = None
_ctx = {}


def init_worker(states, state_to_idx, V_raw, pi_raw, list_size, counter):
    """Initialise le contexte du worker."""
    gc.disable()
    global _global_states, _global_state_to_idx, _V_raw
    _global_states = states
    _global_state_to_idx = state_to_idx
    _V_raw = V_raw

    _ctx["list_size"] = list_size
    _ctx["counter"] = counter
    _ctx["store"] = PostgresBatchStore(DATABASE_URL)

    # On instancie directement AlphaBeta ici !
    _ctx["ab"] = AlphaBeta(PopularityModel(), GAMMA_BOOST)

    _ctx["V_np"] = np.frombuffer(V_raw, dtype=np.float64)
    _ctx["pi_np"] = np.frombuffer(pi_raw, dtype=np.int32).reshape(len(states), list_size)


def transition_chunk_worker(task):
    """Construit une portion de la matrice de transition P."""
    idx, start, end = task
    states = _global_states[start:end]
    store = _ctx["store"]
    ab = _ctx["ab"]
    pi_np = _ctx["pi_np"]
    list_size = _ctx["list_size"]
    counter = _ctx["counter"]

    cols, data, indptr = [], [], [0]

    prefixes_by_k = defaultdict(set)
    for s in states:
        for k in K_WEIGHTS:
            prefixes_by_k[k].add(encode(s[-k:]))

    raw_by_k = {}
    for k, prefixes in prefixes_by_k.items():
        raw_by_k[k] = store.batch_get_successors(list(prefixes), k)

    for i, s in enumerate(states):
        sidx = start + i
        R_list = [int(pi_np[sidx, j]) for j in range(list_size) if pi_np[sidx, j] >= 0]

        merged = defaultdict(float)
        for k, weight in K_WEIGHTS.items():
            prefix = encode(s[-k:])
            for s_str, prob in raw_by_k[k].get(prefix, {}).items():
                merged[decode(s_str)[-1]] += weight * prob
        total = sum(merged.values())
        successors = {item: p / total for item, p in merged.items()} if total else {}

        n_succ = len(successors)
        if n_succ <= 1:
            dist = {}
            for r in R_list:
                dist[r] = ab.alpha(r) * successors.get(r, 0.0)
            for r_prime, q in successors.items():
                if r_prime not in R_list:
                    dist[r_prime] = (1.0 - EPSILON) * q
        else:
            alpha_map = {r: ab.alpha(r) for r in successors}
            weighted_sum = sum(alpha_map[r] * q for r, q in successors.items())

            dist = {}
            for r in R_list:
                dist[r] = alpha_map.get(r, ab.alpha(r)) * successors.get(r, 0.0)

            for r_prime, q in successors.items():
                if r_prime not in R_list:
                    beta_r = ab.beta_fast(alpha_map[r_prime], q, n_succ, weighted_sum)
                    dist[r_prime] = beta_r * q

        total_dist = sum(dist.values())
        if total_dist > 0 and abs(total_dist - 1.0) > 1e-9:
            dist = {k: v / total_dist for k, v in dist.items()}

        count = 0
        for r, p in dist.items():
            s_next = s[1:] + (r,)
            if s_next in _global_state_to_idx:
                cols.append(_global_state_to_idx[s_next])
                data.append(p)
                count += 1
        indptr.append(indptr[-1] + count)

        with counter.get_lock():
            counter.value += 1

    return idx, np.array(cols, dtype=np.int32), np.array(data, dtype=np.float64), np.array(indptr, dtype=np.int32)


def improve_chunk_worker(task):
    """Améliore la politique pour une portion d'états."""
    idx, start, end = task
    states = _global_states[start:end]
    store = _ctx["store"]
    ab = _ctx["ab"]
    V = _ctx["V_np"]
    list_size = _ctx["list_size"]
    counter = _ctx["counter"]

    new_policies, new_scores = [], []

    prefixes_by_k = defaultdict(set)
    for s in states:
        for k in K_WEIGHTS:
            prefixes_by_k[k].add(encode(s[-k:]))

    raw_by_k = {}
    for k, prefixes in prefixes_by_k.items():
        raw_by_k[k] = store.batch_get_successors(list(prefixes), k)

    for i, s in enumerate(states):
        merged = defaultdict(float)
        for k, weight in K_WEIGHTS.items():
            prefix = encode(s[-k:])
            for s_str, prob in raw_by_k[k].get(prefix, {}).items():
                merged[decode(s_str)[-1]] += weight * prob
        total = sum(merged.values())
        successors = {item: p / total for item, p in merged.items()} if total else {}

        n_succ = len(successors)
        if n_succ <= 1:
            scores = {}
            for r, q_r in successors.items():
                s_next = s[1:] + (r,)
                v_next = V[_global_state_to_idx[s_next]] if s_next in _global_state_to_idx else 0.0
                alpha_r = ab.alpha(r)
                scores[r] = (alpha_r * q_r - (1.0 - EPSILON) * q_r) * v_next
        else:
            alpha_map = {r: ab.alpha(r) for r in successors}
            weighted_sum = sum(alpha_map[r] * q for r, q in successors.items())

            scores = {}
            for r, q_r in successors.items():
                s_next = s[1:] + (r,)
                v_next = V[_global_state_to_idx[s_next]] if s_next in _global_state_to_idx else 0.0

                alpha_r = alpha_map[r]
                beta_r = ab.beta_fast(alpha_r, q_r, n_succ, weighted_sum)

                delta_prob = (alpha_r * q_r) - (beta_r * q_r)
                scores[r] = delta_prob * v_next

        top = sorted(scores.items(), key=lambda x: -x[1])[:list_size]
        new_policies.append([k for k, _ in top])
        new_scores.append([v for _, v in top])

        with counter.get_lock():
            counter.value += 1

    return idx, new_policies, new_scores