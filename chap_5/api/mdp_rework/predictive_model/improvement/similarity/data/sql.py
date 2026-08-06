import os
from collections import defaultdict

import orjson
from sqlalchemy import create_engine, event, text, bindparam

from chap_5.api.mdp_rework.shared.create_transition_sql_dict import generate_transition_dict

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "similarity.sqlite",
)


def _dumps(obj) -> str:
    return orjson.dumps(obj).decode()


def _loads(data):
    return orjson.loads(data)


def create_similarity_database():
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def set_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.execute("PRAGMA cache_size=-2000000")
        cur.execute("PRAGMA mmap_size=268435456")
        cur.close()

    return engine


def setup_state_genres_table(engine, k: int):
    table_name = f"k{k}_state_genres"
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(f"""
            CREATE TABLE {table_name} (
                state     TEXT NOT NULL UNIQUE,
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
                states       TEXT NOT NULL
            ) WITHOUT ROWID
        """))


def insert_state_genres_batch(engine, k: int, rows):
    if not rows:
        return
    table_name = f"k{k}_state_genres"
    data = [{"state": _dumps(list(s)), "genre_ids": g} for s, g in rows]
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
    data = []
    for row in rows:
        states_list = row.get("states")
        if not isinstance(states_list, str):
            states_list = _dumps([list(s) for s in states_list])
        data.append({"genre_subset": row["genre_subset"], "states": states_list})

    query = text(f"""
        INSERT INTO {table_name} (genre_subset, states)
        VALUES (:genre_subset, :states)
        ON CONFLICT(genre_subset) DO UPDATE SET states = excluded.states
    """)
    with engine.begin() as conn:
        conn.execute(query, data)


def get_states_to_process(engine, k: int, batch_size: int = 10000) -> dict:
    table_name = f"k{k}_state_genres"
    with engine.begin() as conn:
        rows = conn.execute(
            text(f"SELECT state, genre_ids FROM {table_name} LIMIT {batch_size}")
        ).fetchall()
        if not rows:
            return {}

        states_to_delete = [row[0] for row in rows]
        delete_query = text(f"DELETE FROM {table_name} WHERE state IN :states")
        delete_query = delete_query.bindparams(bindparam("states", expanding=True))
        conn.execute(delete_query, {"states": states_to_delete})

    return {tuple(_loads(state)): genre_ids for state, genre_ids in rows}


def get_state_by_subset(engine, k: int, subsets: set) -> dict:
    if not subsets:
        return {}

    table_name = f"k{k}_reverse_index"
    query = text(
        f"SELECT genre_subset, states FROM {table_name} WHERE genre_subset IN :subsets"
    )
    query = query.bindparams(bindparam("subsets", expanding=True))

    result = {}
    with engine.connect() as conn:
        rows = conn.execute(query, {"subsets": list(subsets)}).fetchall()

    for subset, states_json in rows:
        result[subset] = [tuple(s) for s in _loads(states_json)]

    return result


def setup_similarity_transition_table(engine, k: int):
    table_name = f"k{k}_similarity_transition"
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(f"""
            CREATE TABLE {table_name} (
                s TEXT NOT NULL,
                s_ TEXT NOT NULL,
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
    data = []
    for s, s_, p in rows:
        actual_k = len(s_) if isinstance(s_, (list, tuple)) else k
        data.append({
            "s": _dumps(list(s)),
            "s_": _dumps(list(s_)),
            "k": actual_k,
            "prob": p,
        })

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
            CREATE TABLE similarity_transition_dict (
                s         TEXT NOT NULL,
                successor TEXT,
                proba     TEXT
            )
        """))


def count_states_to_process(engine, k: int) -> int:
    table_name = f"k{k}_state_genres"
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    return result or 0


def create_similarity_dict(engine, max_k: int):
    generate_transition_dict(
        engine,
        max_k,
        source_suffix="similarity_transition",
        target_table="similarity_transition_dict",
    )


def retrieve_similarity_full_info_for_batch(engine, candidates_set: set) -> dict:
    if not candidates_set:
        return {}
    candidates_list = [_dumps(list(c)) for c in candidates_set]
    query = text(
        "SELECT s, successor, proba FROM similarity_transition_dict WHERE s IN :candidates"
    )
    query = query.bindparams(bindparam("candidates", expanding=True))
    result = defaultdict(list)
    with engine.connect() as conn:
        rows = conn.execute(query, {"candidates": candidates_list}).fetchall()

    for s, successor, proba in rows:
        s_tuple = tuple(_loads(s))
        succ_list = _loads(successor)
        proba_list = _loads(proba)
        for s_prime, p in zip(succ_list, proba_list):
            result[s_tuple].append((tuple(s_prime), p))

    return result


def init_similarity_db(engine, k: int):
    setup_state_genres_table(engine, k)
    setup_reverse_index_table(engine, k)
    setup_similarity_transition_table(engine, k)
    setup_similarity_dict_table(engine)