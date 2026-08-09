from chap_5.api.mdp.config import BATCH_SIZE
from chap_5.api.mdp.predictive_model.improvement.similarity.data.sql import (
    create_similarity_database, init_similarity_db,
    insert_state_genres_batch, insert_reverse_index_batch
)
from chap_5.api.mdp.predictive_model.improvement.similarity.utils.data_operation_manager import \
    generate_state_per_substate
from chap_5.api.mdp.predictive_model.improvement.skipping.data.sql import (
    create_skipping_database, fetch_distinct_states
)
from chap_5.api.mdp.predictive_model.improvement.similarity.data.pg_sql import (
    get_pg_engine, fetch_movie_genres
)


class BuildSimilarityIndex:
    def __init__(self, k: int):
        self.k = k
        self.skip_engine = create_skipping_database()
        self.sim_engine = create_similarity_database()
        self.pg_engine = get_pg_engine()
        init_similarity_db(self.sim_engine, k)

    def _build_state_genre_index(self, movie_genres: dict) -> dict:
        rows_to_insert = []
        state_genres = {}

        for state_tuple in fetch_distinct_states(self.skip_engine, self.k):
            genre_ids = set()
            for mid in state_tuple:
                genre_ids.update(movie_genres.get(mid, set()))

            genre_ids_str = ",".join(str(g) for g in sorted(genre_ids))
            state_genres[state_tuple] = genre_ids_str
            rows_to_insert.append((state_tuple, genre_ids_str))

            if len(rows_to_insert) >= BATCH_SIZE:
                insert_state_genres_batch(self.sim_engine, self.k, rows_to_insert)
                rows_to_insert.clear()

        if rows_to_insert:
            insert_state_genres_batch(self.sim_engine, self.k, rows_to_insert)

        return state_genres

    def _build_reverse_index(self, state_genres: dict):
        grouped_subsets = generate_state_per_substate(state_genres)
        rows_to_insert = []
        for subset_key, states_list in grouped_subsets.items():
            rows_to_insert.append({"genre_subset": subset_key, "states": [list(s) for s in states_list]})

            if len(rows_to_insert) >= BATCH_SIZE / 10:
                insert_reverse_index_batch(self.sim_engine, self.k, rows_to_insert)
                rows_to_insert.clear()

        if rows_to_insert:
            insert_reverse_index_batch(self.sim_engine, self.k, rows_to_insert)

    def build_index(self):
        movie_genres = fetch_movie_genres(self.pg_engine)
        state_genres = self._build_state_genre_index(movie_genres)
        self._build_reverse_index(state_genres)

