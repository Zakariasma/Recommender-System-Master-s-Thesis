import os
from sqlalchemy import create_engine, text, bindparam

from chap_5.api.mdp_rework.shared.create_transition_sql_dict import generate_transition_dict

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "similarity.sqlite",
)

def create_similarity_database():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA journal_mode = WAL"))
        conn.execute(text("PRAGMA synchronous = NORMAL"))
        conn.execute(text("PRAGMA temp_store = MEMORY"))
        conn.execute(text("PRAGMA cache_size = -2000000"))
        conn.execute(text("PRAGMA mmap_size = 268435456"))
    return engine

def setup_state_genres_table(engine, k: int):
    table_name = f"k{k}_state_genres"
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(f"""
            CREATE TABLE {table_name} (
                state     BLOB NOT NULL UNIQUE,
                genre_ids TEXT NOT NULL
            )
        """))

def setup_reverse_index_table(engine, k: int):
    table_name = f"k{k}_reverse_index"
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(f"""
            CREATE TABLE {table_name} (
                genre_subset TEXT PRIMARY KEY,
                states       BLOB NOT NULL
            ) WITHOUT ROWID
        """))

def insert_state_genres_batch(engine, k: int, rows):
    if not rows:
        return
    table_name = f"k{k}_state_genres"
    data = [{"state": s, "genre_ids": g} for s, g in rows]
    query = text(f"""
        INSERT INTO {table_name} (state, genre_ids)
        VALUES (:state, :genre_ids)
        ON CONFLICT(state) DO UPDATE SET genre_ids = excluded.genre_ids
    """)
    with engine.begin() as conn:
        conn.execute(query, data)

def insert_reverse_index_batch(engine, k: int, rows):
    if not rows:
        return
    table_name = f"k{k}_reverse_index"
    query = text(f"""
        INSERT INTO {table_name} (genre_subset, states)
        VALUES (:genre_subset, :states)
        ON CONFLICT(genre_subset) DO UPDATE SET states = excluded.states
    """)
    with engine.begin() as conn:
        conn.execute(query, rows)

def get_states_to_process(engine, k: int, batch_size: int = 10000) -> dict:
    table_name = f"k{k}_state_genres"
    with engine.begin() as conn:
        rows = conn.execute(text(f"SELECT state, genre_ids FROM {table_name} LIMIT {batch_size}")).fetchall()
        if not rows:
            return {}
        conn.execute(
            text(f"DELETE FROM {table_name} WHERE state IN (SELECT state FROM {table_name} LIMIT {batch_size})"))
    return {state: genre_ids for state, genre_ids in rows}

def get_state_by_subset(engine, k: int, subsets: set) -> dict:
    if not subsets:
        return {}

    table_name = f"k{k}_reverse_index"
    query = text(f"SELECT genre_subset, states FROM {table_name} WHERE genre_subset IN :subsets")
    query = query.bindparams(bindparam("subsets", expanding=True))

    result = {}
    with engine.connect() as conn:
        rows = conn.execute(query, {"subsets": list(subsets)}).fetchall()

    for subset, states_bytes in rows:
        result[subset] = states_bytes

    return result

def setup_similarity_transition_table(engine, k: int):
    table_name = f"k{k}_similarity_transition"
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

def insert_similarity_transition_batch(engine, k: int, rows):
    if not rows:
        return
    table_name = f"k{k}_similarity_transition"
    data = [{"s": s, "s_": s_, "k": k, "prob": p} for s, s_, p in rows]
    query = text(f"""
        INSERT INTO {table_name} (s, s_, k, prob)
        VALUES (:s, :s_, :k, :prob)
        ON CONFLICT(s, s_, k) DO UPDATE SET prob = excluded.prob
    """)
    with engine.begin() as conn:
        conn.execute(query, data)

def setup_similarity_dict_table(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS similarity_transition_dict"))
        conn.execute(text("""
                          CREATE TABLE similarity_transition_dict
                          (
                              s         BLOB PRIMARY KEY,
                              successor BLOB,
                              proba     BLOB
                          ) WITHOUT ROWID
                          """))

def count_states_to_process(engine, k: int) -> int:
    table_name = f"k{k}_state_genres"
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    return result or 0

def create_similarity_dict(engine, max_k: int):
    generate_transition_dict(engine, max_k, source_suffix="similarity_transition", target_table="similarity_transition_dict")

def init_similarity_db(engine, k: int):
    setup_state_genres_table(engine, k)
    setup_reverse_index_table(engine, k)
    setup_similarity_transition_table(engine, k)
    setup_similarity_dict_table(engine)