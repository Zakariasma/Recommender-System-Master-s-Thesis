import time
from collections import defaultdict
from sqlalchemy import create_engine, text

from setup_sqlite import SQLiteStore
from chap_5.api.mdp.config import DATABASE_URL
from chap_5.api.mdp.predictive_model.helper.encoder import decode, encode

BATCH_SIZE = 50_000
LOG_EVERY_SECONDS = 5


class Similarity:
    def __init__(self, k: int, sqlite_store: SQLiteStore, pg_url: str = DATABASE_URL):
        self.k = k
        self.sqlite_store = sqlite_store
        self.engine = create_engine(pg_url)
        self.BATCH_SIZE = 50_000

    def run(self):
        print(f"=== Démarrage de la similarité pour k={self.k} ===")
        self.sqlite_store.connect()
        self.sqlite_store.setup_tables()

        # 1. Construction des index
        self._build_indexes()

        # 2. Calcul des transitions
        self._compute_transitions()

        # 3. Nettoyage des tables d'index
        print("\nNettoyage des tables d'index...")
        self.sqlite_store.clean_index_tables()
        self.sqlite_store.conn.commit()
        self.sqlite_store.close()
        print("=== Terminé ===")

    def _build_indexes(self):
        movie_genres = self._fetch_movie_genres()
        state_genres = self._build_persistence(movie_genres)
        self._build_reverse_index(state_genres)
        print("\nCréation des index SQLite...")
        self.sqlite_store.create_indexes()

    def _fetch_movie_genres(self) -> dict:
        """Récupère les genres depuis PostgreSQL."""
        print("Récupération des genres de films depuis PG...")
        movie_genres = {}
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT movie_id, genre_id FROM movie_genre")).fetchall()
        for movie_id, genre_id in rows:
            if movie_id not in movie_genres:
                movie_genres[movie_id] = set()
            movie_genres[movie_id].add(int(genre_id))
        print(f"  -> {len(movie_genres):,} films chargés.")
        return movie_genres

    def _build_persistence(self, movie_genres: dict) -> dict:
        """Stream les états de taille k depuis PG, calcule les genres, les insère dans SQLite."""
        print(f"\nConstruction de la table state_genres pour k={self.k}...")
        raw_conn = self.engine.raw_connection()

        total_processed = 0
        start = time.perf_counter()
        last_log = start
        rows_to_insert = []
        state_genres_in_ram = {}

        try:
            with raw_conn.cursor(name="stream_states_cursor") as pg_cur:
                pg_cur.itersize = 10_000
                # FILTRAGE PAR TAILLE K
                pg_cur.execute("SELECT DISTINCT s FROM transitions WHERE k = %s", (self.k,))
                state_id = 0

                for (state_bytes,) in pg_cur:
                    state_id += 1
                    total_processed += 1
                    genre_ids = set()

                    movie_ids = decode(state_bytes)
                    for mid in movie_ids:
                        genre_ids.update(movie_genres.get(mid, set()))

                    # On garde en RAM pour l'étape suivante (évite de relire la BDD)
                    state_genres_in_ram[state_id] = frozenset(genre_ids)
                    genre_ids_str = ",".join(str(g) for g in sorted(genre_ids))
                    rows_to_insert.append((state_id, state_bytes, genre_ids_str))

                    if len(rows_to_insert) >= BATCH_SIZE:
                        self.sqlite_store.insert_state_genres_batch(rows_to_insert)
                        rows_to_insert.clear()

                    now = time.perf_counter()
                    if now - last_log >= LOG_EVERY_SECONDS:
                        speed = total_processed / (now - start)
                        print(f"\r  {total_processed:,} états traités | {speed:,.0f} états/s", end="", flush=True)
                        last_log = now

            if rows_to_insert:
                self.sqlite_store.insert_state_genres_batch(rows_to_insert)

            print(f"\r  state_genres terminé en {time.perf_counter() - start:.1f}s.       ")
            return state_genres_in_ram

        finally:
            raw_conn.close()

    def _build_reverse_index(self, state_genres: dict):
        """Génère les sous-ensembles de genres et les insère dans SQLite."""
        print("\nConstruction de la table reverse_index...")

        total_subsets = 0
        total_states = len(state_genres)
        processed = 0
        start = time.perf_counter()
        last_log = start
        rows_to_insert = []

        for state_id, genre_ids in state_genres.items():
            processed += 1
            c = len(genre_ids)

            if c <= 1:
                continue

            # Calcul pur : on trie une fois, on génère les clés par exclusion d'index
            sorted_str_genres = sorted(str(g) for g in genre_ids)
            for i in range(c):
                subset_key = ",".join(sorted_str_genres[:i] + sorted_str_genres[i + 1:])
                rows_to_insert.append((subset_key, state_id))
                total_subsets += 1

            if len(rows_to_insert) >= BATCH_SIZE:
                self.sqlite_store.insert_reverse_index_batch(rows_to_insert)
                rows_to_insert.clear()

            now = time.perf_counter()
            if now - last_log >= LOG_EVERY_SECONDS:
                speed = total_subsets / (now - start)
                print(f"\r  {total_subsets:,} sous-ensembles générés | {speed:,.0f} subsets/s", end="", flush=True)
                last_log = now

        if rows_to_insert:
            self.sqlite_store.insert_reverse_index_batch(rows_to_insert)

        print(
            f"\r  reverse_index terminé en {time.perf_counter() - start:.1f}s. Total : {total_subsets:,} sous-ensembles.      ")

    def _compute_transitions(self):
        print("\nChargement des états depuis SQLite...")
        rows = self.sqlite_store.get_all_states()

        states = []
        states_dict = {}
        movies_dict = {}

        for state_id, state_bytes, genre_ids_str in rows:
            categories = frozenset(int(x) for x in genre_ids_str.split(",")) if genre_ids_str else frozenset()
            movies = decode(state_bytes)
            states.append((state_id, state_bytes, movies, categories))
            states_dict[state_id] = state_bytes
            movies_dict[state_id] = movies

        total = len(states)
        global_start = time.perf_counter()

        for batch_start in range(0, total, self.BATCH_SIZE):
            chunk = states[batch_start:batch_start + self.BATCH_SIZE]

            # --- ÉTAPE 1 ---
            t1 = time.perf_counter()
            candidates_dict = self._get_candidates_batch(chunk)
            t2 = time.perf_counter()

            # --- ÉTAPE 2 ---
            chunk_data = []
            all_cand_movies = set()
            for state_id, state_bytes, movies, _ in chunk:
                candidates = candidates_dict.get(state_id, set())
                retained = self._keep_sim_candidats(movies, candidates, movies_dict)
                if retained:
                    chunk_data.append((state_id, state_bytes, retained))
                    for c_id in retained.keys():
                        all_cand_movies.add(movies_dict[c_id])
            t3 = time.perf_counter()

            # --- ÉTAPE 3 ---
            all_successors = self._get_strict_successors_batch(list(all_cand_movies)) if all_cand_movies else {}
            t4 = time.perf_counter()

            # --- ÉTAPE 4 ---
            sim_rows = []
            for state_id, state_bytes, retained in chunk_data:
                rows = self._calcul_new_transition(state_bytes, retained, movies_dict, all_successors)
                sim_rows.extend(rows)
            t5 = time.perf_counter()

            # --- ÉTAPE 5 ---
            if sim_rows:
                self.sqlite_store.insert_sim_transition_batch(sim_rows)
            t6 = time.perf_counter()

            speed = (batch_start + len(chunk)) / (t6 - global_start) if t6 != global_start else 0
            print(
                f"Lot {batch_start} | [1]{t2 - t1:.2f}s [2]{t3 - t2:.2f}s [3]{t4 - t3:.2f}s [4]{t5 - t4:.2f}s [5]{t6 - t5:.2f}s | {speed:,.0f} états/s")

        self.sqlite_store.conn.commit()
        print("\nCalcul des transitions terminé.")

    def _get_strict_successors_batch(self, states_list: list) -> dict:
        """
        Récupère UNIQUEMENT les transitions strictes (k = self.k) pour une liste d'états.
        Pas de K_WEIGHTS, pas de suffixes.
        """
        if not states_list:
            return {}

        # On encode les tuples en bytes
        unique_keys = list(set(encode(state_tuple) for state_tuple in states_list))

        raw_transitions = defaultdict(dict)
        raw_conn = self.engine.raw_connection()

        try:
            with raw_conn.cursor() as cur:
                cur.execute("SET work_mem = '1GB'")

                # Requête stricte : on filtre avec k = self.k !
                cur.execute("""
                            SELECT s, s_, prob
                            FROM transitions
                            WHERE s = ANY (%s)
                              AND k = %s
                            """, (unique_keys, self.k))

                for row in cur:
                    s_bytes = bytes(row[0])
                    s_next_bytes = bytes(row[1])
                    prob = row[2]

                    item = decode(s_next_bytes)[-1]
                    raw_transitions[s_bytes][item] = prob

        finally:
            raw_conn.close()

        return dict(raw_transitions)

    def _get_candidates_batch(self, chunk):
        all_subsets = set()
        state_subsets = {}
        for state_id, state_bytes, _, categories in chunk:
            if not categories or len(categories) <= 1:
                continue
            keys = []
            for g in categories:
                subset = categories - {g}
                subset_key = ",".join(str(x) for x in sorted(subset))
                keys.append(subset_key)
                all_subsets.add(subset_key)
            state_subsets[state_id] = keys

        if not all_subsets:
            return {}

        candidates_rows = self.sqlite_store.get_reverse_candidates(list(all_subsets))
        reverse_map = defaultdict(set)
        for subset_key, cand_id in candidates_rows:
            reverse_map[subset_key].add(cand_id)

        candidates_dict = {}
        for state_id, keys in state_subsets.items():
            cands = set()
            for k in keys:
                cands.update(reverse_map.get(k, set()))
            cands.discard(state_id)
            candidates_dict[state_id] = cands
        return candidates_dict

    def _keep_sim_candidats(self, self_movies, candidates, movies_dict):
        retained = {}
        for cand_id in candidates:
            cand_movies = movies_dict.get(cand_id)
            if not cand_movies:
                continue
            sim_score = 0
            max_m = min(len(self_movies), len(cand_movies))
            for m in range(max_m):
                if self_movies[m] == cand_movies[m]:
                    sim_score += (m + 2)
            if sim_score > 0:
                retained[cand_id] = sim_score
        return retained

    def _calcul_new_transition(self, state_bytes, retained_candidates, movies_dict, all_successors):
        simcount = defaultdict(float)
        for cand_id, sim_score in retained_candidates.items():
            cand_movies = movies_dict.get(cand_id)
            if not cand_movies:
                continue
            cand_bytes = encode(cand_movies)
            cand_transitions = all_successors.get(cand_bytes, {})
            for next_item, prob in cand_transitions.items():
                simcount[next_item] += sim_score * prob

        total_score = sum(simcount.values())
        rows = []
        if total_score > 0:
            for next_item, score in simcount.items():
                s_next_blob = encode((next_item,))
                rows.append((state_bytes, s_next_blob, score / total_score))
        return rows


if __name__ == "__main__":
    store = SQLiteStore("sim_index.sqlite")
    sim = Similarity(k=3, sqlite_store=store)
    sim.run()