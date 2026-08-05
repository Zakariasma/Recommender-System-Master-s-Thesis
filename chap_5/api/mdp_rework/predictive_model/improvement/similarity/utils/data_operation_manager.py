import struct
from collections import defaultdict

from chap_5.api.mdp_rework.config import BITS_MOVIE_REPRESENTATION


# {Romance} -> {<165,122,966>,...}
def generate_state_per_substate(state_genres):
    grouped_subsets = defaultdict(list)
    for state_bytes, genre_ids_str in state_genres.items():
        if not genre_ids_str:
            continue

        genres = genre_ids_str.split(',')
        c = len(genres)
        if c <= 1:
            continue

        for i in range(c):
            subset_key = ",".join(genres[:i] + genres[i + 1:])
            grouped_subsets[subset_key].append(state_bytes)
    return grouped_subsets


# {<165,122,966>} -> {[Romance, Action], ...}
def generate_subsets_per_state(state_genres: dict) -> dict:
    state_to_subsets = {}

    for state_bytes, genre_ids_str in state_genres.items():
        if not genre_ids_str:
            continue

        genres = genre_ids_str.split(',')
        if len(genres) <= 1:
            continue

        subsets = [
            ",".join(genres[:i] + genres[i + 1:])
            for i in range(len(genres))
        ]
        state_to_subsets[state_bytes] = subsets

    return state_to_subsets

def merge_candidates(subsets_per_state: dict, subset_to_states: dict, k: int) -> dict:
    # taille en octets d'un état (par ex pour: k=3, 15 bits/film = 45 bits -> 6 octets)
    state_byte_size = (k * BITS_MOVIE_REPRESENTATION + 7) // 8
    merged = {}

    for state, subsets in subsets_per_state.items():
        candidates = set()

        for subset in subsets:
            states_blob = subset_to_states.get(subset)
            if states_blob:
                # On découpe le gros BLOB en tranches de state_byte_size pour isoler chaque état
                for i in range(0, len(states_blob), state_byte_size):
                    candidate = states_blob[i:i + state_byte_size]
                    candidates.add(candidate)
        merged[state] = list(candidates)

    return merged


def build_inverted_index(cand_list: list, successors_for_candidates: dict, k: int) -> tuple:
    state_byte_size = (k * BITS_MOVIE_REPRESENTATION + 7) // 8
    all_successors = set()
    inv_index = defaultdict(list)

    for cand_bytes, sim_score in cand_list:
        if cand_bytes not in successors_for_candidates:
            continue

        succ_blob, proba_blob = successors_for_candidates[cand_bytes]

        num_succ = len(succ_blob) // state_byte_size
        if num_succ > 0:
            probs = struct.unpack(f'{num_succ}f', proba_blob)
            for i in range(num_succ):
                s_prime = succ_blob[i * state_byte_size: (i + 1) * state_byte_size]
                proba = probs[i]

                all_successors.add(s_prime)
                inv_index[s_prime].append((sim_score, proba))

    return all_successors, inv_index