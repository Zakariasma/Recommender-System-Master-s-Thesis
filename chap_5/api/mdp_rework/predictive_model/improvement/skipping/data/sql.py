import os
from collections import defaultdict

import orjson
from sqlalchemy import create_engine, event, text, bindparam

from chap_5.api.mdp_rework.shared.create_transition_sql_dict import generate_transition_dict

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "skipping.sqlite",
)


def _dumps(obj) -> str:
    return orjson.dumps(obj).decode()


def _loads(data):
    return orjson.loads(data)


def create_skipping_database():
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def set_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA wal_autocheckpoint = 0")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.execute("PRAGMA cache_size=-2000000")
        cur.execute("PRAGMA mmap_size=268435456")
        cur.close()

    return engine


def setup_counts_table(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS counts"))
        conn.execute(text("""
            CREATE TABLE counts (
                s TEXT,
                s_ TEXT,
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
                s TEXT NOT NULL,
                s_ TEXT NOT NULL,
                k INTEGER NOT NULL,
                prob REAL NOT NULL,
                PRIMARY KEY (s, s_, k)
            )
        """))
        conn.execute(text(f"CREATE INDEX idx_{table_name}_s ON {table_name}(s)"))


def setup_skiping_transition_table(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS skipping_transition_dict"))
        # pas de PK / index pendant le bulk load
        conn.execute(text("""
            CREATE TABLE skipping_transition_dict (
                s TEXT NOT NULL,
                successor TEXT,
                proba TEXT
            )
        """))


def flush_counts(engine, counts):
    if not counts:
        return
    rows = [
        {"s": _dumps(list(s)), "s_": _dumps(list(s_)), "count": c}
        for s, transitions in counts.items()
        for s_, c in transitions.items()
    ]
    query = text("""
        INSERT INTO counts (s, s_, count)
        VALUES (:s, :s_, :count) ON CONFLICT(s, s_)
        DO UPDATE SET count = counts.count + excluded.count
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
    generate_transition_dict(
        engine, max_k,
        source_suffix="transition",
        target_table="skipping_transition_dict",
    )


def fetch_distinct_states(engine, k: int):
    table_name = f"k{k}_transition"
    query = text(f"SELECT DISTINCT s FROM {table_name}")
    with engine.connect() as conn:
        result = conn.execute(query)
        while True:
            rows = result.fetchmany(10000)
            if not rows:
                break
            for (state,) in rows:
                yield tuple(_loads(state))


def retrieve_distinct_states(engine, k: int):
    return list(fetch_distinct_states(engine, k))


def retrieve_skipping_full_info_for_batch(engine, candidates_set: set) -> dict:
    if not candidates_set:
        return {}

    candidates_list = [orjson.dumps(list(c)).decode('utf-8') for c in candidates_set]

    query = text("SELECT s, successor, proba FROM skipping_transition_dict WHERE s IN :candidates")
    query = query.bindparams(bindparam("candidates", expanding=True))

    result = {}
    with engine.connect() as conn:
        rows = conn.execute(query, {"candidates": candidates_list}).fetchall()

    for s, successor, proba in rows:
        result[tuple(orjson.loads(s))] = (orjson.loads(successor), orjson.loads(proba))

    return result


def retrieve_transition_for_batch(engine, candidates_set: set, k: int) -> dict:
    if not candidates_set:
        return {}
    table_name = f"k{k}_transition"
    candidates_list = [_dumps(list(c)) for c in candidates_set]
    query = text(f"SELECT s, s_, prob FROM {table_name} WHERE s IN :candidates")
    query = query.bindparams(bindparam("candidates", expanding=True))
    result = defaultdict(list)
    with engine.connect() as conn:
        rows = conn.execute(query, {"candidates": candidates_list}).fetchall()
    for s, s_, prob in rows:
        result[tuple(_loads(s))].append((tuple(_loads(s_)), prob))
    return result


def init_skipping_db(engine, k: int):
    setup_counts_table(engine)
    setup_transition_table(engine, k)