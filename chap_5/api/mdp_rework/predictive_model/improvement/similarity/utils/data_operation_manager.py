import struct
from collections import defaultdict

from chap_5.api.mdp_rework.config import BITS_MOVIE_REPRESENTATION
from chap_5.api.mdp_rework.shared.endode_to_blob import unpack_transitions


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
    state_byte_size = (k * BITS_MOVIE_REPRESENTATION + 7) // 8
    merged = {}

    for state, subsets in subsets_per_state.items():
        candidates = set()

        for subset in subsets:
            states_blob = subset_to_states.get(subset)
            if states_blob:
                for i in range(0, len(states_blob), state_byte_size):
                    candidate = states_blob[i:i + state_byte_size]
                    candidates.add(candidate)
        merged[state] = list(candidates)

    return merged


def build_inverted_index(cand_list: list, successors_for_candidates: dict) -> tuple:
    all_successors = set()
    inv_index = defaultdict(list)

    for cand_bytes, sim_score in cand_list:
        if cand_bytes not in successors_for_candidates:
            continue

        for s_prime, proba in successors_for_candidates[cand_bytes]:
            all_successors.add(s_prime)
            inv_index[s_prime].append((sim_score, proba))

    return all_successors, inv_index