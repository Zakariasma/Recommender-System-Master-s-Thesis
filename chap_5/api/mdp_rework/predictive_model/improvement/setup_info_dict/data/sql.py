import os
from sqlalchemy import create_engine, text, bindparam
from chap_5.api.mdp_rework.shared.binary_encoder import (
    encode_state, decode_state,
    encode_successors_fixed, decode_successors_fixed,
    encode_proba_list, decode_proba_list
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_info.sqlite")


def create_full_info_database():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA wal_autocheckpoint = 0"))
        conn.execute(text("PRAGMA journal_mode = WAL"))
        conn.execute(text("PRAGMA synchronous = OFF"))
        conn.execute(text("PRAGMA temp_store = MEMORY"))
        conn.execute(text("PRAGMA cache_size = -2000000"))
        conn.execute(text("PRAGMA mmap_size = 268435456"))
    return engine


def setup_full_info_table(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS state_full_info"))
        # 15 colonnes séparées par k
        conn.execute(text("""
                          CREATE TABLE state_full_info
                          (
                              s             BLOB PRIMARY KEY,
                              k1_s_         BLOB,
                              k1_tr         BLOB,
                              k1_rew        BLOB,
                              k1_p_reco     BLOB,
                              k1_p_not_reco BLOB,
                              k2_s_         BLOB,
                              k2_tr         BLOB,
                              k2_rew        BLOB,
                              k2_p_reco     BLOB,
                              k2_p_not_reco BLOB,
                              k3_s_         BLOB,
                              k3_tr         BLOB,
                              k3_rew        BLOB,
                              k3_p_reco     BLOB,
                              k3_p_not_reco BLOB
                          ) WITHOUT ROWID
                          """))


def create_index(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX idx_state_full_info_s ON state_full_info(s)"))


def flush(engine, rows: list):
    if not rows:
        return
    cols = "s, k1_s_, k1_tr, k1_rew, k1_p_reco, k1_p_not_reco, k2_s_, k2_tr, k2_rew, k2_p_reco, k2_p_not_reco, k3_s_, k3_tr, k3_rew, k3_p_reco, k3_p_not_reco"
    vals = ":s, :k1_s_, :k1_tr, :k1_rew, :k1_p_reco, :k1_p_not_reco, :k2_s_, :k2_tr, :k2_rew, :k2_p_reco, :k2_p_not_reco, :k3_s_, :k3_tr, :k3_rew, :k3_p_reco, :k3_p_not_reco"

    query = text(
        f"INSERT INTO state_full_info ({cols}) VALUES ({vals}) ON CONFLICT(s) DO UPDATE SET {', '.join([f'{c}=excluded.{c}' for c in cols.split(', ')[1:]])}")

    encoded_rows = []
    for row in rows:
        enc = {"s": encode_state(row["s"])}
        for k_val in [1, 2, 3]:
            prefix = f"k{k_val}_"
            enc[prefix + "s_"] = encode_successors_fixed(row.get(prefix + "s_", []))
            enc[prefix + "tr"] = encode_proba_list(row.get(prefix + "tr", []))
            enc[prefix + "rew"] = encode_proba_list(row.get(prefix + "rew", []))
            enc[prefix + "p_reco"] = encode_proba_list(row.get(prefix + "p_reco", []))
            enc[prefix + "p_not_reco"] = encode_proba_list(row.get(prefix + "p_not_reco", []))
        encoded_rows.append(enc)

    with engine.begin() as conn:
        conn.execute(query, encoded_rows)


def retrieve_full_info_for_batch(engine, candidates_set: set) -> dict:
    if not candidates_set:
        return {}
    candidates_blobs = [encode_state(c) for c in candidates_set]
    query = text("""
                 SELECT s,
                        k1_s_,
                        k1_tr,
                        k1_rew,
                        k1_p_reco,
                        k1_p_not_reco,
                        k2_s_,
                        k2_tr,
                        k2_rew,
                        k2_p_reco,
                        k2_p_not_reco,
                        k3_s_,
                        k3_tr,
                        k3_rew,
                        k3_p_reco,
                        k3_p_not_reco
                 FROM state_full_info
                 WHERE s IN :candidates
                 """)
    query = query.bindparams(bindparam("candidates", expanding=True))

    result = {}
    with engine.connect() as conn:
        for i in range(0, len(candidates_blobs), 500):
            chunk = candidates_blobs[i:i + 500]
            rows = conn.execute(query, {"candidates": chunk}).fetchall()
            for row in rows:
                s_tuple = decode_state(row[0])
                # On renvoie un tuple de 15 tableaux numpy
                result[s_tuple] = (
                    decode_successors_fixed(row[1], 1), decode_proba_list(row[2]), decode_proba_list(row[3]),
                    decode_proba_list(row[4]), decode_proba_list(row[5]),
                    decode_successors_fixed(row[6], 2), decode_proba_list(row[7]), decode_proba_list(row[8]),
                    decode_proba_list(row[9]), decode_proba_list(row[10]),
                    decode_successors_fixed(row[11], 3), decode_proba_list(row[12]), decode_proba_list(row[13]),
                    decode_proba_list(row[14]), decode_proba_list(row[15])
                )
    return result


def init_full_info_db(engine):
    setup_full_info_table(engine)