import time
import sqlite3
import struct
from collections import defaultdict

from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values

from chap_5.api.mdp.config import DATABASE_URL
from chap_5.api.mdp.predictive_model.helper.encoder import encode, decode

PG_ENGINE = create_engine(DATABASE_URL)
PG_TRANSITIONS_PATH = "pg_transitions.sqlite"
SQLITE_OPTIMIZED_PATH = "optimized_transitions.sqlite"
BATCH_SIZE = 10_000


def setup_sqlite_table():
    """Fonction 1 : Crée la nouvelle table SQLite propre."""
    print("=== Création de la base SQLite optimisée ===")
    conn = sqlite3.connect(SQLITE_OPTIMIZED_PATH)
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("PRAGMA synchronous = OFF;")

    conn.execute("DROP TABLE IF EXISTS opt_transitions;")
    # On stocke par état source et par taille k
    conn.execute("""
                 CREATE TABLE opt_transitions
                 (
                     s          BLOB    NOT NULL,
                     k          INTEGER NOT NULL,
                     successors BLOB    NOT NULL,
                     probs      BLOB    NOT NULL,
                     PRIMARY KEY (s, k)
                 )
                 """)
    conn.commit()
    conn.close()
    print("Table opt_transitions créée.")


def fetch_unique_states_sqlite(k: int) -> list:
    """Fonction 2 : Récupère tous les états uniques d'une taille k depuis pg_transitions.sqlite."""
    print(f"\nRécupération des états uniques pour k={k} depuis SQLite...")
    conn = sqlite3.connect(PG_TRANSITIONS_PATH)
    # Optimisations de lecture pour SQLite
    conn.execute("PRAGMA cache_size = -1000000;")  # 1 Go de cache
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA mmap_size = 268435456;")  # 256 MB de mapping mémoire

    cur = conn.cursor()
    # Si l'index n'existe pas, la requête sera lente, alors on s'assure qu'il est là
    cur.execute("CREATE INDEX IF NOT EXISTS idx_transitions_s_k ON transitions(s, k);")

    cur.execute("SELECT DISTINCT s FROM transitions WHERE k = ?", (k,))
    states = [row[0] for row in cur.fetchall()]
    conn.close()
    print(f"  -> {len(states):,} états trouvés pour k={k}.")
    return states


def fetch_strict_successors_batch_pg(states_list: list, k: int) -> dict:
    """Fonction 3 : Récupère les transitions strictes par batch depuis PG."""
    if not states_list:
        return {}

    # On déduplique les clés
    unique_keys = list(set(states_list))

    raw_transitions = defaultdict(dict)
    raw_conn = PG_ENGINE.raw_connection()

    try:
        with raw_conn.cursor() as cur:
            cur.execute("SET work_mem = '1GB'")
            cur.execute("""
                        SELECT s, s_, prob
                        FROM transitions
                        WHERE s = ANY (%s)
                          AND k = %s
                        """, (unique_keys, k))

            for row in cur:
                s_bytes = bytes(row[0])
                s_next_bytes = bytes(row[1])
                prob = row[2]

                item = decode(s_next_bytes)[-1]
                raw_transitions[s_bytes][item] = prob
    finally:
        raw_conn.close()

    return dict(raw_transitions)


def orchestrator():
    """Fonction 4 : Orchestre la copie par batch de PG vers SQLite."""
    # 1. Préparer la table SQLite de destination
    setup_sqlite_table()

    sqlite_conn = sqlite3.connect(SQLITE_OPTIMIZED_PATH)
    sqlite_conn.execute("PRAGMA journal_mode = MEMORY;")
    sqlite_conn.execute("PRAGMA synchronous = OFF;")
    sqlite_cur = sqlite_conn.cursor()

    # 2. Boucler sur les tailles k = 1, 2, 3
    for k in [1, 2, 3]:
        print(f"\n=== Traitement des transitions pour k={k} ===")

        # On récupère les états depuis SQLite
        states = fetch_unique_states_sqlite(k)
        total = len(states)

        start_time = time.perf_counter()
        last_log = start_time
        processed = 0
        rows_to_insert = []

        # 3. Parcourir par batch de 10 000
        for i in range(0, total, BATCH_SIZE):
            batch = states[i:i + BATCH_SIZE]

            # Récupérer les transitions du batch depuis PG
            transitions_dict = fetch_strict_successors_batch_pg(batch, k)

            # Compresser en BLOB pour l'insertion SQLite
            for s_bytes, succ_dict in transitions_dict.items():
                items_list = list(succ_dict.keys())
                probs_list = list(succ_dict.values())

                # 'I' = Unsigned Int (4 octets), 'f' = Float (4 octets)
                successors_blob = b''.join(struct.pack('I', item) for item in items_list)
                probs_blob = b''.join(struct.pack('f', prob) for prob in probs_list)

                rows_to_insert.append((s_bytes, k, successors_blob, probs_blob))

            # Insertion en masse dans SQLite
            if rows_to_insert:
                sqlite_cur.executemany(
                    "INSERT OR REPLACE INTO opt_transitions VALUES (?, ?, ?, ?)",
                    rows_to_insert
                )
                sqlite_conn.commit()
                rows_to_insert.clear()

            processed += len(batch)
            now = time.perf_counter()
            if now - last_log >= 5:
                speed = processed / (now - start_time)
                eta = (total - processed) / speed if speed else 0
                print(f"\r  k={k} | {processed:,}/{total:,} états traités | {speed:,.0f} états/s | ETA: {eta:.0f}s",
                      end="", flush=True)
                last_log = now

        print(f"\r  k={k} terminé en {time.perf_counter() - start_time:.1f}s.      ")

    # 4. Création de l'index à la toute fin
    print("\nCréation de l'index SQLite...")
    sqlite_cur.execute("CREATE INDEX idx_opt_s_k ON opt_transitions(s, k);")
    sqlite_conn.commit()
    sqlite_conn.close()
    print("=== Base optimisée prête ! ===")


if __name__ == "__main__":
    orchestrator()