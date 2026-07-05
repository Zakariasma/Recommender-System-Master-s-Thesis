from collections import defaultdict
from chap_5.api.mdp.predictive_model.state_key import encode, decode
from chap_5.api.mdp.mdp.alpha_beta import AlphaBeta
from chap_5.api.mdp.mdp.popularity import PopularityModel
from chap_5.api.mdp.config import GAMMA_BOOST

K_WEIGHTS = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}

def batch_get_successors(transition_store, states: list, k_weights: dict = None) -> dict:
    k_weights = k_weights or K_WEIGHTS
    prefixes_by_k = defaultdict(set)
    for s in states:
        for k in k_weights:
            prefixes_by_k[k].add(encode(s[-k:]))

    raw_by_k = {}
    for k, prefixes in prefixes_by_k.items():
        raw_by_k[k] = transition_store.batch_get_successors(list(prefixes), k)

    successors_map = {}
    for s in states:
        merged = defaultdict(float)
        for k, weight in k_weights.items():
            prefix = encode(s[-k:])
            for s_str, prob in raw_by_k[k].get(prefix, {}).items():
                merged[decode(s_str)[-1]] += weight * prob
        total = sum(merged.values())
        successors_map[s] = {item: p / total for item, p in merged.items()} if total else {}
    return successors_map

def dist_list(ab: AlphaBeta, s: tuple, R: list, successors: dict) -> dict:
    """Retourne {r: prob} au lieu de {s_next: prob} pour économiser la RAM."""
    if not R:
        result = {r_prime: q for r_prime, q in successors.items()}
    else:
        result = {}
        for r in R:
            result[r] = ab.alpha(r) * successors.get(r, 0.0)
        for r_prime, q in successors.items():
            if r_prime not in R:
                result[r_prime] = ab.beta(r_prime, successors) * q

    total = sum(result.values())
    if total > 0 and abs(total - 1.0) > 1e-9:
        return {k: v / total for k, v in result.items()}
    return result

def dist_single(ab: AlphaBeta, s: tuple, r: int, successors: dict) -> dict:
    """Retourne {r: prob} au lieu de {s_next: prob}."""
    alpha_r = ab.alpha(r)
    result = {r: alpha_r * successors.get(r, 0.0)}
    for r_prime, q in successors.items():
        if r_prime != r:
            beta_r_prime = ab.beta(r_prime, successors)
            result[r_prime] = beta_r_prime * q

    total = sum(result.values())
    if total > 0 and abs(total - 1.0) > 1e-9:
        return {k: v / total for k, v in result.items()}
    return result

def make_alpha_beta(gamma: float = GAMMA_BOOST) -> AlphaBeta:
    return AlphaBeta(PopularityModel(), gamma)


def batch_get_successors_flat(transition_store, states: list, k_weights: dict = None):
    """Version optimisée qui renvoie 3 listes plates au lieu d'un dict de dicts."""
    k_weights = k_weights or K_WEIGHTS
    prefixes_by_k = defaultdict(set)
    for s in states:
        for k in k_weights:
            prefixes_by_k[k].add(encode(s[-k:]))

    raw_by_k = {}
    for k, prefixes in prefixes_by_k.items():
        raw_by_k[k] = transition_store.batch_get_successors(list(prefixes), k)

    state_indices = []
    rs = []
    probs = []

    for i, s in enumerate(states):
        merged = defaultdict(float)
        for k, weight in k_weights.items():
            prefix = encode(s[-k:])
            for s_str, prob in raw_by_k[k].get(prefix, {}).items():
                merged[decode(s_str)[-1]] += weight * prob
        total = sum(merged.values())
        if total > 0:
            for item, p in merged.items():
                state_indices.append(i)
                rs.append(item)
                probs.append(p / total)

    return state_indices, rs, probs