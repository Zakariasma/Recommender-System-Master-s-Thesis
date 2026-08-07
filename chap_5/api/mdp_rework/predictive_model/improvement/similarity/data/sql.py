import os
import sqlite3

from chap_5.api.mdp_rework.shared.create_transition_sql_dict import generate_transition_dict
from chap_5.api.mdp_rework.shared.binary_encoder import (
    encode_state, decode_state,
    decode_successors_fixed, decode_proba_list
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "similarity.sqlite")


def create_similarity_database():
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


def setup_state_genres_table(db_path: str, k: int):
    table_name = f"k{k}_state_genres"
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(f"CREATE TABLE {table_name} (state BLOB NOT NULL UNIQUE, genre_ids TEXT NOT NULL)")
        conn.commit()
    finally:
        conn.close()


def setup_reverse_index_table(db_path: str, k: int):
    table_name = f"k{k}_reverse_index"
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(f"CREATE TABLE {table_name} (genre_subset TEXT PRIMARY KEY, states BLOB NOT NULL) WITHOUT ROWID")
        conn.commit()
    finally:
        conn.close()


def insert_state_genres_batch(db_path: str, k: int, rows):
    if not rows: return
    table_name = f"k{k}_state_genres"
    data = [(encode_state(s), g) for s, g in rows]
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("BEGIN TRANSACTION")
        cur.executemany(
            f"INSERT INTO {table_name} (state, genre_ids) VALUES (?, ?) "
            f"ON CONFLICT(state) DO UPDATE SET genre_ids = excluded.genre_ids",
            data)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_reverse_index_batch(db_path: str, k: int, rows):
    if not rows: return
    table_name = f"k{k}_reverse_index"
    data = []
    for row in rows:
        states_list = row.get("states")
        if not isinstance(states_list, bytes):
            states_list = b''.join([encode_state(s) for s in states_list])
        data.append((row["genre_subset"], states_list))
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("BEGIN TRANSACTION")
        cur.executemany(
            f"INSERT INTO {table_name} (genre_subset, states) VALUES (?, ?) "
            f"ON CONFLICT(genre_subset) DO UPDATE SET states = excluded.states",
            data)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_states_to_process(db_path: str, k: int, batch_size: int = 10000) -> dict:
    table_name = f"k{k}_state_genres"
    rows = []
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("BEGIN TRANSACTION")
        cur.execute(f"SELECT state, genre_ids FROM {table_name} LIMIT {batch_size}")
        rows = cur.fetchall()
        if rows:
            states_to_delete = [row[0] for row in rows]
            for i in range(0, len(states_to_delete), 500):
                chunk = states_to_delete[i:i + 500]
                placeholders = ",".join(["?"] * len(chunk))
                cur.execute(f"DELETE FROM {table_name} WHERE state IN ({placeholders})", chunk)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    if not rows:
        return {}
    return {decode_state(state): genre_ids for state, genre_ids in rows}


def get_state_by_subset(db_path: str, k: int, subsets: set) -> dict:
    if not subsets: return {}
    table_name = f"k{k}_reverse_index"
    subsets_list = list(subsets)
    state_byte_size = k * 2
    result = {}

    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        for i in range(0, len(subsets_list), 500):
            chunk = subsets_list[i:i + 500]
            placeholders = ",".join(["?"] * len(chunk))
            cur.execute(
                f"SELECT genre_subset, states FROM {table_name} WHERE genre_subset IN ({placeholders})",
                chunk)
            rows = cur.fetchall()
            for subset, states_blob in rows:
                states_list = []
                for j in range(0, len(states_blob), state_byte_size):
                    states_list.append(decode_state(states_blob[j:j + state_byte_size]))
                result[subset] = states_list
    finally:
        conn.close()
    return result


def setup_similarity_transition_table(db_path: str, k: int):
    table_name = f"k{k}_similarity_transition"
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"CREATE TABLE {table_name} (s BLOB NOT NULL, s_ BLOB NOT NULL, k INTEGER NOT NULL, prob REAL NOT NULL, PRIMARY KEY (s, s_, k))")
        cur.execute(f"CREATE INDEX idx_{table_name}_s ON {table_name}(s)")
        conn.commit()
    finally:
        conn.close()


def insert_similarity_transition_batch(db_path: str, k: int, rows):
    if not rows: return
    table_name = f"k{k}_similarity_transition"
    data = []
    for s, s_, p in rows:
        actual_k = len(s_) if isinstance(s_, (list, tuple)) else k
        data.append((encode_state(s), encode_state(s_), actual_k, p))
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("BEGIN TRANSACTION")
        cur.executemany(
            f"INSERT INTO {table_name} (s, s_, k, prob) VALUES (?, ?, ?, ?) "
            f"ON CONFLICT(s, s_, k) DO UPDATE SET prob = excluded.prob",
            data)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def setup_similarity_dict_table(db_path: str):
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE similarity_transition_dict
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


def count_states_to_process(db_path: str, k: int) -> int:
    table_name = f"k{k}_state_genres"
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()[0] or 0
    finally:
        conn.close()


def create_similarity_dict(db_path: str, max_k: int):
    generate_transition_dict(db_path, max_k, source_suffix="similarity_transition", target_table="similarity_transition_dict")


def retrieve_similarity_full_info_for_batch(db_path: str, candidates_set: set) -> dict:
    if not candidates_set: return {}
    candidates_blobs = [encode_state(c) for c in candidates_set]
    result = {}

    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        for i in range(0, len(candidates_blobs), 500):
            chunk = candidates_blobs[i:i + 500]
            placeholders = ",".join(["?"] * len(chunk))
            cur.execute(f"""SELECT s, k1_successor, k1_proba, k2_successor, k2_proba, k3_successor, k3_proba
                            FROM similarity_transition_dict
                            WHERE s IN ({placeholders})""", chunk)
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


def init_similarity_db(db_path: str, k: int):
    setup_state_genres_table(db_path, k)
    setup_reverse_index_table(db_path, k)
    setup_similarity_transition_table(db_path, k)
