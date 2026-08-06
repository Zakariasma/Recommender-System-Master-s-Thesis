import os
from collections import defaultdict
from sqlalchemy import create_engine, event, text, bindparam
from chap_5.api.mdp_rework.shared.create_transition_sql_dict import generate_transition_dict
from chap_5.api.mdp_rework.shared.binary_encoder import (
    encode_state, decode_state,
    decode_successors_fixed, decode_proba_list
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skipping.sqlite")


def create_skipping_database():
    engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

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


def setup_counts_table(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS counts"))
        conn.execute(text("""CREATE TABLE counts
                             (
                                 s     BLOB,
                                 s_    BLOB,
                                 count REAL,
                                 PRIMARY KEY (s, s_)
                             ) WITHOUT ROWID"""))


def setup_transition_table(engine, k: int):
    table_name = f"k{k}_transition"
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(
            f"""CREATE TABLE {table_name} (s BLOB NOT NULL, s_ BLOB NOT NULL, k INTEGER NOT NULL, prob REAL NOT NULL, PRIMARY KEY (s, s_, k))"""))
        conn.execute(text(f"CREATE INDEX idx_{table_name}_s ON {table_name}(s)"))


def setup_skiping_transition_table(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS skipping_transition_dict"))
        # 6 colonnes : k1, k2, k3 pour successor et proba
        conn.execute(text("""CREATE TABLE skipping_transition_dict
                             (
                                 s            BLOB PRIMARY KEY,
                                 k1_successor BLOB,
                                 k1_proba     BLOB,
                                 k2_successor BLOB,
                                 k2_proba     BLOB,
                                 k3_successor BLOB,
                                 k3_proba     BLOB
                             ) WITHOUT ROWID"""))


def flush_counts(engine, counts):
    if not counts: return
    rows = [{"s": encode_state(s), "s_": encode_state(s_), "count": c} for s, transitions in counts.items() for s_, c in
            transitions.items()]
    query = text(
        "INSERT INTO counts (s, s_, count) VALUES (:s, :s_, :count) ON CONFLICT(s, s_) DO UPDATE SET count = counts.count + excluded.count")
    with engine.begin() as conn: conn.execute(query, rows)


def normalize_and_clean(engine, k: int):
    table_name = f"k{k}_transition"
    with engine.begin() as conn:
        conn.execute(text(
            f"""INSERT INTO {table_name} (s, s_, k, prob) SELECT s, s_, {k}, cnt / SUM(cnt) OVER (PARTITION BY s) FROM (SELECT s, s_, SUM(count) AS cnt FROM counts GROUP BY s, s_)"""))
        conn.execute(text("DELETE FROM counts"))


def create_transition_dict(engine, max_k: int):
    generate_transition_dict(engine, max_k, source_suffix="transition", target_table="skipping_transition_dict")


def fetch_distinct_states(engine, k: int):
    table_name = f"k{k}_transition"
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT DISTINCT s FROM {table_name}"))
        while True:
            rows = result.fetchmany(10000)
            if not rows: break
            for (state_blob,) in rows: yield decode_state(state_blob)


def retrieve_distinct_states(engine, k: int):
    return list(fetch_distinct_states(engine, k))


def retrieve_skipping_full_info_for_batch(engine, candidates_set: set) -> dict:
    if not candidates_set: return {}
    candidates_blobs = [encode_state(c) for c in candidates_set]
    query = text("""SELECT s, k1_successor, k1_proba, k2_successor, k2_proba, k3_successor, k3_proba
                    FROM skipping_transition_dict
                    WHERE s IN :candidates""")
    query = query.bindparams(bindparam("candidates", expanding=True))

    result = {}
    with engine.connect() as conn:
        for i in range(0, len(candidates_blobs), 500):
            chunk = candidates_blobs[i:i + 500]
            rows = conn.execute(query, {"candidates": chunk}).fetchall()
            for row in rows:
                s_tuple = decode_state(row[0])
                k_dict = {}
                # k=1
                if row[1]: k_dict[1] = (decode_successors_fixed(row[1], 1), decode_proba_list(row[2]))
                # k=2
                if row[3]: k_dict[2] = (decode_successors_fixed(row[3], 2), decode_proba_list(row[4]))
                # k=3
                if row[5]: k_dict[3] = (decode_successors_fixed(row[5], 3), decode_proba_list(row[6]))
                result[s_tuple] = k_dict
    return result


def retrieve_transition_for_batch(engine, candidates_set: set, k: int) -> dict:
    if not candidates_set: return {}
    table_name = f"k{k}_transition"
    candidates_blobs = [encode_state(c) for c in candidates_set]
    query = text(f"SELECT s, s_, prob FROM {table_name} WHERE s IN :candidates")
    query = query.bindparams(bindparam("candidates", expanding=True))
    result = defaultdict(list)
    with engine.connect() as conn:
        for i in range(0, len(candidates_blobs), 500):
            chunk = candidates_blobs[i:i + 500]
            rows = conn.execute(query, {"candidates": chunk}).fetchall()
            for s_blob, s_prime_blob, prob in rows:
                result[decode_state(s_blob)].append((decode_state(s_prime_blob), prob))
    return result


def init_skipping_db(engine, k: int):
    setup_counts_table(engine)
    setup_transition_table(engine, k)