from collections import defaultdict


def generate_state_per_substate(state_genres):
    grouped_subsets = defaultdict(list)
    for state_tuple, genre_ids_str in state_genres.items():
        if not genre_ids_str:
            continue

        genres = genre_ids_str.split(',')
        c = len(genres)
        if c <= 1:
            continue

        for i in range(c):
            subset_key = ",".join(genres[:i] + genres[i + 1:])
            grouped_subsets[subset_key].append(state_tuple)
    return grouped_subsets


def generate_subsets_per_state(state_genres: dict) -> dict:
    state_to_subsets = {}

    for state_tuple, genre_ids_str in state_genres.items():
        if not genre_ids_str:
            continue

        genres = genre_ids_str.split(',')
        if len(genres) <= 1:
            continue

        subsets = [
            ",".join(genres[:i] + genres[i + 1:])
            for i in range(len(genres))
        ]
        state_to_subsets[state_tuple] = subsets

    return state_to_subsets


def merge_candidates(subsets_per_state: dict, subset_to_states: dict, k: int) -> dict:
    merged = {}

    for state, subsets in subsets_per_state.items():
        candidates = set()

        for subset in subsets:
            states_list = subset_to_states.get(subset)
            if states_list:
                # states_list est maintenant une liste de tuples d'entiers
                for candidate in states_list:
                    candidates.add(candidate)  # candidate est un tuple, donc hashable

        merged[state] = list(candidates)

    return merged


def build_inverted_index(cand_list: list, successors_for_candidates: dict) -> tuple:
    all_successors = set()
    inv_index = defaultdict(list)

    for cand_tuple, sim_score in cand_list:
        if cand_tuple not in successors_for_candidates:
            continue

        # successors_for_candidates[cand_tuple] est une liste de (s_prime_list, proba)
        for s_prime_list, proba in successors_for_candidates[cand_tuple]:
            # On convertit s_prime_list en tuple pour pouvoir l'ajouter à un set
            s_prime_tuple = tuple(s_prime_list)
            all_successors.add(s_prime_tuple)
            inv_index[s_prime_tuple].append((sim_score, proba))

    return all_successors, inv_index