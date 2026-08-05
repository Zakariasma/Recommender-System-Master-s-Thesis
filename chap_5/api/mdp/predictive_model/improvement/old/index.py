import os
import random
import sqlite3
import time
from collections import defaultdict
from sqlalchemy import create_engine

from chap_5.api.mdp.config import DATABASE_URL

K = 3
SAMPLE_SIZE = 1000
SQLITE_PATH = "similarity_index.sqlite"

MOVIE_TO_GENRES = {}


def load_movie_genres():
    print("Chargement des genres depuis la base de données...")
    engine = create_engine(DATABASE_URL)
    raw_conn = engine.raw_connection()

    mapping = defaultdict(int)
    try:
        with raw_conn.cursor() as cur:
            cur.execute("SELECT movie_id, genre_id FROM movie_genre")
            for movie_id, genre_id in cur:
                mapping[int(movie_id)] |= (1 << int(genre_id))
    finally:
        raw_conn.close()

    print(f"  -> {len(mapping):,} films chargés avec leurs genres.")
    return dict(mapping)


def get_mask_for_state(state_str):
    mask = 0
    for mid in state_str.split(","):
        try:
            mask |= MOVIE_TO_GENRES.get(int(mid), 0)
        except ValueError:
            continue
    return mask


def submasks_minus_one(mask):
    m = mask
    while m:
        bit = m & -m
        yield mask ^ bit
        m ^= bit


def init_sqlite(path):
    if os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -200000")  # ~200 MB cache SQLite

    conn.execute("""
        CREATE TABLE states (
            state_id INTEGER PRIMARY KEY,
            state_str TEXT NOT NULL,
            genre_mask TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE state_subsets (
            subset_mask TEXT NOT NULL,
            state_id INTEGER NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    conn.commit()
    return conn


def flush_batches(conn, state_rows, subset_rows):
    if state_rows:
        conn.executemany(
            "INSERT INTO states(state_id, state_str, genre_mask) VALUES (?, ?, ?)",
            state_rows,
        )

    if subset_rows:
        conn.executemany(
            "INSERT INTO state_subsets(subset_mask, state_id) VALUES (?, ?)",
            subset_rows,
        )

    conn.commit()


class SimilarityIndexBuilder:
    def __init__(self, k: int, sqlite_path: str):
        self.k = k
        self.sqlite_path = sqlite_path

    def fit(self):
        print("=== Construction de l'index de similarité sur disque ===")
        print(f"k            : {self.k}")
        print(f"sqlite file  : {self.sqlite_path}")

        global MOVIE_TO_GENRES
        MOVIE_TO_GENRES = load_movie_genres()

        self._build_index()

    def _build_index(self):
        print("\nExtraction des états et écriture de l'index sur disque...")
        engine = create_engine(DATABASE_URL)
        raw_conn = engine.raw_connection()
        sqlite_conn = init_sqlite(self.sqlite_path)

        total_states = 0
        total_subset_rows = 0
        sample_states = []

        state_rows = []
        subset_rows = []

        flush_every_states = 10_000
        start = time.perf_counter()

        try:
            with raw_conn.cursor(name="stream_states_cursor") as cur:
                cur.itersize = 10_000
                cur.execute("SELECT DISTINCT s FROM transitions")

                for (state_str,) in cur:
                    total_states += 1
                    state_id = total_states

                    if len(sample_states) < SAMPLE_SIZE:
                        sample_states.append(state_str)
                    else:
                        idx = random.randint(0, total_states - 1)
                        if idx < SAMPLE_SIZE:
                            sample_states[idx] = state_str

                    mask = get_mask_for_state(state_str)
                    mask_str = str(mask)

                    state_rows.append((state_id, state_str, mask_str))

                    if mask.bit_count() <= 1:
                        subset_rows.append((mask_str, state_id))
                        total_subset_rows += 1
                    else:
                        for sub in submasks_minus_one(mask):
                            subset_rows.append((str(sub), state_id))
                            total_subset_rows += 1

                    if total_states % flush_every_states == 0:
                        flush_batches(sqlite_conn, state_rows, subset_rows)
                        state_rows.clear()
                        subset_rows.clear()

                        elapsed = time.perf_counter() - start
                        print(
                            f"\r  États: {total_states:,} | "
                            f"Lignes subsets: {total_subset_rows:,} | "
                            f"Vitesse: {total_states / elapsed:,.0f} états/s",
                            end="",
                            flush=True,
                        )

            flush_batches(sqlite_conn, state_rows, subset_rows)
            print()

            elapsed = time.perf_counter() - start
            print(f"\nTraitement terminé : {total_states:,} états en {elapsed:.1f}s.")
            print(f"Lignes écrites dans state_subsets : {total_subset_rows:,}")

            print("\nCréation des index SQLite...")
            idx_start = time.perf_counter()

            sqlite_conn.execute(
                "CREATE INDEX idx_state_subsets_subset_mask ON state_subsets(subset_mask)"
            )
            sqlite_conn.execute(
                "CREATE INDEX idx_state_subsets_state_id ON state_subsets(state_id)"
            )
            sqlite_conn.execute(
                "CREATE INDEX idx_states_mask ON states(genre_mask)"
            )

            sqlite_conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                ("total_states", str(total_states)),
            )
            sqlite_conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                ("total_subset_rows", str(total_subset_rows)),
            )
            sqlite_conn.commit()

            print(f"Index SQLite créés en {time.perf_counter() - idx_start:.1f}s.")

            size_mb = os.path.getsize(self.sqlite_path) / (1024 * 1024)
            print(f"Taille du fichier SQLite : {size_mb:,.1f} MB")

        finally:
            sqlite_conn.close()
            raw_conn.close()


if __name__ == "__main__":
    model = SimilarityIndexBuilder(k=K, sqlite_path=SQLITE_PATH)
    model.fit()