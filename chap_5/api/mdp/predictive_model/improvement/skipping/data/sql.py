import os
import sqlite3
from collections import defaultdict

from chap_5.api.mdp.shared.create_transition_sql_dict import generate_transition_dict
from chap_5.api.mdp.shared.binary_encoder import (
    encode_state, decode_state,
    decode_successors_fixed, decode_proba_list
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skipping.sqlite")


def create_skipping_database():
    return DB_PATH

def _get_conn(db_path: str) -> sqlite3.Connection:
    """Ouvre une connexion native sqlite3 et applique les PRAGMAs d'optimisation."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-2000000")
    conn.execute("PRAGMA mmap_size=268435456")
    return conn


def setup_counts_table(db_path: str):
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS counts")
        cur.execute("CREATE TABLE counts (s BLOB, s_ BLOB, count REAL)")
        conn.commit()
    finally:
        conn.close()


def setup_transition_table(db_path: str, k: int):
    table_name = f"k{k}_transition"
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"CREATE TABLE {table_name} (s BLOB NOT NULL, s_ BLOB NOT NULL, k INTEGER NOT NULL, prob REAL NOT NULL, PRIMARY KEY (s, s_, k))")
        conn.commit()
    finally:
        conn.close()


def create_index_transition_table(db_path: str, k: int):
    table_name = f"k{k}_transition"
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_s ON {table_name}(s)")
        conn.commit()
    finally:
        conn.close()


def setup_skiping_transition_table(db_path: str):
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS skipping_transition_dict")
        cur.execute("""CREATE TABLE skipping_transition_dict
                       (
                           s            BLOB,
                           k1_successor BLOB,
                           k1_proba     BLOB,
                           k2_successor BLOB,
                           k2_proba     BLOB,
                           k3_successor BLOB,
                           k3_proba     BLOB
                       )""")
        conn.commit()
    finally:
        conn.close()


def flush_counts_conn(conn: sqlite3.Connection, counts: dict):
    if not counts: return
    rows = (
        (encode_state(s), encode_state(s_), c)
        for s, transitions in counts.items()
        for s_, c in transitions.items()
    )
    cur = conn.cursor()
    cur.execute("BEGIN TRANSACTION")
    try:
        cur.executemany("INSERT INTO counts (s, s_, count) VALUES (?, ?, ?)", rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

def open_persistent_connection(db_path: str) -> sqlite3.Connection:
    return _get_conn(db_path)


def normalize_and_clean(db_path: str, k: int):
    table_name = f"k{k}_transition"
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("BEGIN TRANSACTION")
        cur.execute(f"""INSERT INTO {table_name} (s, s_, k, prob) 
                        SELECT s, s_, {k}, cnt / SUM(cnt) OVER (PARTITION BY s) 
                        FROM (SELECT s, s_, SUM(count) AS cnt FROM counts GROUP BY s, s_)""")
        cur.execute("DELETE FROM counts")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # On crée l'index maintenant que la table est pleine !
    create_index_transition_table(db_path, k)


def create_transition_dict(db_path: str, max_k: int):
    # On passe le chemin à la fonction de génération
    generate_transition_dict(db_path, max_k, source_suffix="transition", target_table="skipping_transition_dict")


def fetch_distinct_states(db_path: str, k: int):
    table_name = f"k{k}_transition"
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT s FROM {table_name}")
        while True:
            rows = cur.fetchmany(10000)
            if not rows: break
            for (state_blob,) in rows:
                yield decode_state(state_blob)
    finally:
        conn.close()


def retrieve_distinct_states(db_path: str, k: int):
    return list(fetch_distinct_states(db_path, k))


def retrieve_skipping_full_info_for_batch(db_path: str, candidates_set: set) -> dict:
    if not candidates_set: return {}
    candidates_blobs = [encode_state(c) for c in candidates_set]
    result = {}

    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        for i in range(0, len(candidates_blobs), 500):
            chunk = candidates_blobs[i:i + 500]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""SELECT s, k1_successor, k1_proba, k2_successor, k2_proba, k3_successor, k3_proba
                        FROM skipping_transition_dict
                        WHERE s IN ({placeholders})"""
            cur.execute(query, chunk)
            rows = cur.fetchall()

            for row in rows:
                s_tuple = decode_state(row[0])
                k_dict = {}
                if row[1]: k_dict[1] = (decode_successors_fixed(row[1], 1), decode_proba_list(row[2]))
                if row[3]: k_dict[2] = (decode_successors_fixed(row[3], 2), decode_proba_list(row[4]))
                if row[5]: k_dict[3] = (decode_successors_fixed(row[5], 3), decode_proba_list(row[6]))
                result[s_tuple] = k_dict
    finally:
        conn.close()
    return result


def retrieve_transition_for_batch(db_path: str, candidates_set: set, k: int) -> dict:
    if not candidates_set: return {}
    table_name = f"k{k}_transition"
    candidates_blobs = [encode_state(c) for c in candidates_set]
    result = defaultdict(list)

    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        for i in range(0, len(candidates_blobs), 500):
            chunk = candidates_blobs[i:i + 500]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"SELECT s, s_, prob FROM {table_name} WHERE s IN ({placeholders})"
            cur.execute(query, chunk)
            rows = cur.fetchall()

            for s_blob, s_prime_blob, prob in rows:
                result[decode_state(s_blob)].append((decode_state(s_prime_blob), prob))
    finally:
        conn.close()
    return result


def init_skipping_db(db_path: str, k: int):
    setup_counts_table(db_path)
    setup_transition_table(db_path, k)