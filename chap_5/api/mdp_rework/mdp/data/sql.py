import os
from sqlalchemy import create_engine, text, bindparam
from chap_5.api.mdp_rework.shared.binary_encoder import encode_state, decode_state

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "kv_store.sqlite",
)

def create_database():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA journal_mode = WAL"))
        conn.execute(text("PRAGMA synchronous = NORMAL"))
        conn.execute(text("PRAGMA temp_store = MEMORY"))
        conn.execute(text("PRAGMA cache_size = -2000000"))
        conn.execute(text("PRAGMA mmap_size = 268435456"))
    return engine

def setup_policy_table(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS policy"))
        conn.execute(text("CREATE TABLE policy (s BLOB PRIMARY KEY, value BLOB NOT NULL) WITHOUT ROWID"))

def setup_state_values_table(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS value_state"))
        conn.execute(text("CREATE TABLE value_state (s BLOB PRIMARY KEY, value REAL NOT NULL) WITHOUT ROWID"))

def init_kv_store_db(engine):
    setup_policy_table(engine)
    setup_state_values_table(engine)

def flush_policy(engine, rows: list):
    if not rows: return
    data = [{"s": encode_state(r["s"]), "value": encode_state(tuple(r["value"]))} for r in rows]
    query = text("INSERT INTO policy (s, value) VALUES (:s, :value) ON CONFLICT(s) DO UPDATE SET value = excluded.value")
    with engine.begin() as conn: conn.execute(query, data)

def flush_value_states(engine, rows: list):
    if not rows: return
    data = [{"s": encode_state(r["s"]), "value": r["value"]} for r in rows]
    query = text("INSERT INTO value_state (s, value) VALUES (:s, :value) ON CONFLICT(s) DO UPDATE SET value = excluded.value")
    with engine.begin() as conn: conn.execute(query, data)

def get_by_batch_policy(engine, candidates_set: set) -> dict:
    if not candidates_set: return {}
    candidates_list = [encode_state(c) for c in candidates_set]
    query = text("SELECT s, value FROM policy WHERE s IN :candidates")
    query = query.bindparams(bindparam("candidates", expanding=True))
    result = {}
    with engine.connect() as conn:
        for i in range(0, len(candidates_list), 500):
            chunk = candidates_list[i:i+500]
            rows = conn.execute(query, {"candidates": chunk}).fetchall()
            for s, value in rows:
                result[decode_state(s)] = list(decode_state(value))
    return result

def get_by_batch_states(engine, candidates_set: set) -> dict:
    if not candidates_set: return {}
    candidates_list = [encode_state(c) for c in candidates_set]
    query = text("SELECT s, value FROM value_state WHERE s IN :candidates")
    query = query.bindparams(bindparam("candidates", expanding=True))
    result = {}
    with engine.connect() as conn:
        for i in range(0, len(candidates_list), 500):
            chunk = candidates_list[i:i+500]
            rows = conn.execute(query, {"candidates": chunk}).fetchall()
            for s, value in rows:
                result[decode_state(s)] = value
    return result