import time
from sqlalchemy import create_engine, text

from chap_5.api.mdp.config import DATABASE_URL
from chap_5.api.mdp.predictive_model.helper.encoder import decode
from setup_sqlite import SQLiteStore

BATCH_SIZE = 50_000
LOG_EVERY_SECONDS = 5


class IndexSimBuilder:
    def __init__(self, pg_url: str, sqlite_store: SQLiteStore, k: int = 3):
        self.pg_engine = create_engine(pg_url)
        self.sqlite_store = sqlite_store
        self.k = k  # On stocke la taille de l'état

    def run(self):
        print(f"=== Initialisation de la base SQLite pour k={self.k} ===")
        self.sqlite_store.connect()
        self.sqlite_store.setup_tables()

        movie_genres = self._fetch_movie_genres()
        state_genres_in_ram = self._build_persistence(movie_genres)
        self._build_reverse_index(state_genres_in_ram)

        print("\nCréation des index SQLite...")
        self.sqlite_store.create_indexes()
        self.sqlite_store.close()
        print("=== Construction terminée ===")

    def _fetch_movie_genres(self) -> dict:
        """Récupère les genres depuis PostgreSQL."""
        print("Récupération des genres de films depuis PG...")
        movie_genres = {}
        with self.pg_engine.connect() as conn:
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
        raw_conn = self.pg_engine.raw_connection()

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

        print(f"\r  reverse_index terminé en {time.perf_counter() - start:.1f}s. Total : {total_subsets:,} sous-ensembles.      ")


if __name__ == "__main__":
    # On instancie le store SQLite et le builder
    sqlite_store = SQLiteStore("sim_index.sqlite")
    builder = IndexSimBuilder(DATABASE_URL, sqlite_store, k=3)
    builder.run()