
import os
import sqlite3
from chap_5.api.mdp_rework.shared.binary_encoder import (
    encode_state, decode_state,
    encode_successors_fixed, decode_successors_fixed,
    encode_proba_list, decode_proba_list
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_info.sqlite")


def create_full_info_database():
    return DB_PATH


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA wal_autocheckpoint = 0")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -2000000")
    conn.execute("PRAGMA mmap_size = 268435456")
    return conn


def setup_full_info_table(db_path: str):
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS state_full_info")
        cur.execute("""
                    CREATE TABLE state_full_info
                    (
                        s             BLOB,
                        k1_s_         BLOB,
                        k1_tr         BLOB,
                        k1_p_reco     BLOB,
                        k1_p_not_reco BLOB,
                        k2_s_         BLOB,
                        k2_tr         BLOB,
                        k2_p_reco     BLOB,
                        k2_p_not_reco BLOB,
                        k3_s_         BLOB,
                        k3_tr         BLOB,
                        k3_p_reco     BLOB,
                        k3_p_not_reco BLOB
                    )
                    """)
        conn.commit()
    finally:
        conn.close()


def create_index(db_path: str):
    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("CREATE INDEX idx_state_full_info_s ON state_full_info(s)")
        cur.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()


def flush(db_path: str, rows: list):
    if not rows:
        return

    encoded_rows = []
    for row in rows:
        enc = [encode_state(row["s"])]
        for k_val in [1, 2, 3]:
            prefix = f"k{k_val}_"
            enc.append(encode_successors_fixed(row.get(prefix + "s_", [])))
            enc.append(encode_proba_list(row.get(prefix + "tr", [])))
            enc.append(encode_proba_list(row.get(prefix + "p_reco", [])))
            enc.append(encode_proba_list(row.get(prefix + "p_not_reco", [])))
        encoded_rows.append(tuple(enc))

    # 13 colonnes -> 13 points d'interrogation
    placeholders = ",".join(["?"] * 13)
    query = f"INSERT INTO state_full_info VALUES ({placeholders})"

    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("BEGIN TRANSACTION")
        cur.executemany(query, encoded_rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def retrieve_full_info_for_batch(db_path: str, candidates_set: set) -> dict:
    if not candidates_set:
        return {}

    candidates_blobs = [encode_state(c) for c in candidates_set]
    result = {}

    conn = _get_conn(db_path)
    try:
        cur = conn.cursor()
        for i in range(0, len(candidates_blobs), 500):
            chunk = candidates_blobs[i:i + 500]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT s, k1_s_, k1_tr, k1_p_reco, k1_p_not_reco,
                       k2_s_, k2_tr, k2_p_reco, k2_p_not_reco,
                       k3_s_, k3_tr, k3_p_reco, k3_p_not_reco
                FROM state_full_info 
                WHERE s IN ({placeholders})
            """
            cur.execute(query, chunk)
            rows = cur.fetchall()

            for row in rows:
                s_tuple = decode_state(row[0])
                result[s_tuple] = (
                    decode_successors_fixed(row[1], 1), decode_proba_list(row[2]),
                    decode_proba_list(row[3]), decode_proba_list(row[4]),
                    decode_successors_fixed(row[5], 2), decode_proba_list(row[6]),
                    decode_proba_list(row[7]), decode_proba_list(row[8]),
                    decode_successors_fixed(row[9], 3), decode_proba_list(row[10]),
                    decode_proba_list(row[11]), decode_proba_list(row[12])
                )
    finally:
        conn.close()

    return result


def init_full_info_db(db_path: str):
    setup_full_info_table(db_path)
