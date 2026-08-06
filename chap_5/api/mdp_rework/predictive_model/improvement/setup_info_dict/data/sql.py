import os
import orjson
from sqlalchemy import create_engine, text, bindparam

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "full_info.sqlite",
)

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
        # Plus de PRIMARY KEY ni WITHOUT ROWID pour des inserts ultra rapides
        conn.execute(text("""
            CREATE TABLE state_full_info (
                s              TEXT,
                s_             TEXT,
                tr_predict     TEXT,
                reward         TEXT,
                proba_reco     TEXT,
                proba_not_reco TEXT
            )
        """))

def create_index(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX idx_state_full_info_s ON state_full_info(s)"))

def flush(engine, rows: list):
    if not rows:
        return
    # INSERT simple sans ON CONFLICT car la table n'a plus de PK
    query = text("""
        INSERT INTO state_full_info (s, s_, tr_predict, reward, proba_reco, proba_not_reco)
        VALUES (:s, :s_, :tr_predict, :reward, :proba_reco, :proba_not_reco)
    """)
    with engine.begin() as conn:
        conn.execute(query, rows)

def retrieve_full_info_for_batch(engine, candidates_set: set) -> dict:
    if not candidates_set:
        return {}
    candidates_list = [orjson.dumps(list(c)).decode('utf-8') for c in candidates_set]
    query = text("""
        SELECT s, s_, tr_predict, reward, proba_reco, proba_not_reco
        FROM state_full_info
        WHERE s IN :candidates
    """)
    query = query.bindparams(bindparam("candidates", expanding=True))
    result = {}
    with engine.connect() as conn:
        rows = conn.execute(query, {"candidates": candidates_list}).fetchall()
    for s, s_, tr_predict, reward, proba_reco, proba_not_reco in rows:
        result[tuple(orjson.loads(s))] = {
            "s_": orjson.loads(s_),
            "tr_predict": orjson.loads(tr_predict),
            "reward": orjson.loads(reward),
            "proba_reco": orjson.loads(proba_reco),
            "proba_not_reco": orjson.loads(proba_not_reco)
        }
    return result

def retrieve_full_info(engine, s_tuple: tuple) -> dict:
    s_json = orjson.dumps(list(s_tuple)).decode('utf-8')
    query = text("""
        SELECT s_, tr_predict, reward, proba_reco, proba_not_reco
        FROM state_full_info
        WHERE s = :s
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"s": s_json}).fetchone()

    if not row:
        return None

    return {
        "s_": orjson.loads(row[0]),
        "tr_predict": orjson.loads(row[1]),
        "reward": orjson.loads(row[2]),
        "proba_reco": orjson.loads(row[3]),
        "proba_not_reco": orjson.loads(row[4])
    }

def init_full_info_db(engine):
    setup_full_info_table(engine)