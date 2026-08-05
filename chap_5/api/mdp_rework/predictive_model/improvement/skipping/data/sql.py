import os
import struct
from collections import defaultdict

from sqlalchemy import create_engine, text, bindparam

from chap_5.api.mdp_rework.shared.create_transition_sql_dict import generate_transition_dict
from chap_5.api.mdp_rework.shared.endode_to_blob import encode

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "skipping.sqlite",
)


def create_skipping_database():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA journal_mode = WAL"))
        conn.execute(text("PRAGMA synchronous = NORMAL"))
        conn.execute(text("PRAGMA temp_store = MEMORY"))
        conn.execute(text("PRAGMA cache_size = -2000000"))
        conn.execute(text("PRAGMA mmap_size = 268435456"))
    return engine


def setup_counts_table(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS counts"))
        conn.execute(text("""
                          CREATE TABLE counts
                          (
                              s     BLOB,
                              s_    BLOB,
                              count REAL,
                              PRIMARY KEY (s, s_)
                          ) WITHOUT ROWID
                          """))


def setup_transition_table(engine, k: int):
    table_name = f"k{k}_transition"
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(f"""
            CREATE TABLE {table_name} (
                s BLOB NOT NULL,
                s_ BLOB NOT NULL,
                k INTEGER NOT NULL,
                prob REAL NOT NULL,
                PRIMARY KEY (s, s_, k)
            )
        """))
        conn.execute(text(f"CREATE INDEX idx_{table_name}_s ON {table_name}(s)"))

def setup_skiping_transition_table(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS skipping_transition_dict"))
        conn.execute(text("""
                          CREATE TABLE skipping_transition_dict
                          (
                              s         BLOB PRIMARY KEY,
                              successor BLOB,
                              proba     BLOB
                          ) WITHOUT ROWID
                          """))


def flush_counts(engine, counts):
    if not counts:
        return
    rows = [
        {"s": encode(s), "s_": encode(s_), "count": c}
        for s, transitions in counts.items()
        for s_, c in transitions.items()
    ]
    query = text("""
                 INSERT INTO counts (s, s_, count)
                 VALUES (:s, :s_, :count) ON CONFLICT(s, s_)
        DO
                 UPDATE SET count = counts.count + excluded.count
                 """)
    with engine.begin() as conn:
        conn.execute(query, rows)


def normalize_and_clean(engine, k: int):
    table_name = f"k{k}_transition"
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {table_name} (s, s_, k, prob)
            SELECT s, s_, {k}, cnt / SUM(cnt) OVER (PARTITION BY s)
            FROM (
                SELECT s, s_, SUM(count) AS cnt
                FROM counts
                GROUP BY s, s_
            )
        """))
        conn.execute(text("DELETE FROM counts"))


def create_transition_dict(engine, max_k: int):
    generate_transition_dict(engine, max_k, source_suffix="transition", target_table="skipping_transition_dict")


def retrieve_successor_for_batch(engine, candidates_set: set) -> dict:
    if not candidates_set:
        return {}
    query = text("SELECT s, successor, proba FROM skipping_transition_dict WHERE s IN :candidates")
    query = query.bindparams(bindparam("candidates", expanding=True))
    result = {}
    with engine.connect() as conn:
        rows = conn.execute(query, {"candidates": list(candidates_set)}).fetchall()
    for s_bytes, succ_blob, proba_blob in rows:
        result[s_bytes] = (succ_blob, proba_blob)
    return result

def fetch_distinct_states(engine, k: int):
    table_name = f"k{k}_transition"
    query = text(f"SELECT DISTINCT s FROM {table_name}")
    with engine.connect() as conn:
        result = conn.execute(query)
        while True:
            rows = result.fetchmany(10000)
            if not rows:
                break
            for (state_bytes,) in rows:
                yield state_bytes

def retrieve_distinct_states(engine, k: int):
    table_name = f"k{k}_transition"
    query = text(f"SELECT DISTINCT s FROM {table_name}")
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result.fetchall()]


def retrieve_skipping_full_info_for_batch(engine, candidates_set: set) -> dict:
    if not candidates_set:
        return {}
    query = text("SELECT s, successor, proba FROM skipping_transition_dict WHERE s IN :candidates")
    query = query.bindparams(bindparam("candidates", expanding=True))
    result = defaultdict(list)
    with engine.connect() as conn:
        rows = conn.execute(query, {"candidates": list(candidates_set)}).fetchall()
    for s_bytes, successor, proba in rows:
        result[s_bytes].append((successor, proba))
    return result

def init_skipping_db(engine, k: int):
    setup_counts_table(engine)
    setup_transition_table(engine, k)