from collections import defaultdict

from chap_5.api.mdp_rework.config import BATCH_SIZE
from chap_5.api.mdp_rework.predictive_model.improvement.similarity.build_similarity_index import BuildSimilarityIndex
from chap_5.api.mdp_rework.predictive_model.improvement.similarity.data.sql import (
    get_states_to_process, create_similarity_database, get_state_by_subset,
    insert_similarity_transition_batch, count_states_to_process
)
from chap_5.api.mdp_rework.predictive_model.improvement.similarity.utils.data_operation_manager import \
    generate_subsets_per_state, merge_candidates, build_inverted_index
from chap_5.api.mdp_rework.predictive_model.improvement.skipping.data.sql import \
    create_skipping_database, retrieve_skipping_full_info_for_batch, retrieve_transition_for_batch
from chap_5.api.mdp_rework.shared.debug_log import reset_progress, log_progress


class Similarity:
    def __init__(self, k: int):
        self.k = k

    def similarity(self):
        build_index = BuildSimilarityIndex(self.k)
        build_index.build_index()
        self.sim_engine = create_similarity_database()
        self.skip_engine = create_skipping_database()

        total_states = count_states_to_process(self.sim_engine, self.k)
        total_processed = 0
        reset_progress()

        all_rows_to_insert = []

        while True:
            state_to_process = get_states_to_process(self.sim_engine, self.k)
            if not state_to_process:
                break

            candidats = self._get_candidates(state_to_process)
            filtered_candidates = self._score_and_filter_candidates(candidats)
            sucessors_for_candidates = self._get_successor_for_candidates(filtered_candidates)
            dict_scores = self._calcul_transition_scores(filtered_candidates, sucessors_for_candidates)
            row_score_normalized = self._normalize_score_to_proba(dict_scores)

            all_rows_to_insert.extend(row_score_normalized)

            if len(all_rows_to_insert) >= BATCH_SIZE:
                insert_similarity_transition_batch(self.sim_engine, self.k, all_rows_to_insert)
                all_rows_to_insert.clear()

            total_processed += len(state_to_process)
            log_progress(total_processed, total_states, 'similarity')

        if all_rows_to_insert:
            insert_similarity_transition_batch(self.sim_engine, self.k, all_rows_to_insert)

    def _get_candidates(self, states: dict):
        subsate_per_state = generate_subsets_per_state(states)
        subset_set = set()
        for subsets in subsate_per_state.values():
            subset_set.update(subsets)
        subset_to_states = get_state_by_subset(self.sim_engine, self.k, subset_set)
        final_candidates = merge_candidates(subsate_per_state, subset_to_states, self.k)
        return final_candidates

    def _score_and_filter_candidates(self, candidates: dict) -> dict:
        retained = {}
        for state_tuple, cand_list in candidates.items():
            self_movies = state_tuple
            kept = []

            for cand_tuple in cand_list:
                max_m = min(len(self_movies), len(cand_tuple))

                sim_score = 0
                for m in range(max_m):
                    if self_movies[m] == cand_tuple[m]:
                        sim_score += (m + 2)

                if sim_score > 0:
                    kept.append((cand_tuple, sim_score))

            if kept:
                retained[state_tuple] = kept

        return retained

    def _get_successor_for_candidates(self, filtered_candidates: dict) -> dict:
        candidates_set = set()
        for cand_list in filtered_candidates.values():
            for cand_tuple, _ in cand_list:
                candidates_set.add(cand_tuple)
        return retrieve_transition_for_batch(self.skip_engine, candidates_set, self.k)

    def _calcul_transition_scores(self, filtered_candidates: dict, sucessors_for_candidates: dict) -> dict:
        dict_scores = {}

        for state_base, cand_list in filtered_candidates.items():
            all_successors, inv_index = build_inverted_index(cand_list, sucessors_for_candidates)
            fstate_scores = {}
            for s_prime in all_successors:
                score_s_to_sprime = 0.0
                for sim_score, proba in inv_index[s_prime]:
                    score_s_to_sprime += sim_score * proba
                if score_s_to_sprime > 0:
                    fstate_scores[s_prime] = score_s_to_sprime
            dict_scores[state_base] = fstate_scores
        return dict_scores

    def _normalize_score_to_proba(self, dict_scores: dict) -> list:
        rows_to_insert = []
        for state_base, fstate_scores in dict_scores.items():
            global_score = sum(fstate_scores.values())
            if global_score > 0:
                for fstate, score in fstate_scores.items():
                    proba = score / global_score
                    # fstate est un tuple, on le met en liste pour le JSON
                    rows_to_insert.append((state_base, list(fstate), proba))
        return rows_to_insert


if __name__ == "__main__":
    sim = Similarity(k=1)
    sim.similarity()