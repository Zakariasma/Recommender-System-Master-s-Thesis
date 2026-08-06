import orjson
from collections import defaultdict
from sqlalchemy import text, bindparam
from chap_5.api.mdp_rework.shared.debug_log import log_progress, reset_progress


def _dumps(obj) -> str:
    return orjson.dumps(obj).decode()


def _loads(data):
    return orjson.loads(data)


def generate_substates(s_tuple: tuple) -> dict:
    substates = {}
    k = len(s_tuple)
    while k > 0:
        substates[k] = s_tuple[-k:]
        k -= 1
    return substates


def fetch_transitions_for_batch(conn, s_tuples: list, source_suffix: str) -> dict:
    sub_to_orig = defaultdict(lambda: defaultdict(list))
    for s_tuple in s_tuples:
        for k, sub_s_tuple in generate_substates(s_tuple).items():
            sub_s_json = _dumps(list(sub_s_tuple))
            sub_to_orig[k][sub_s_json].append(s_tuple)

    transitions_by_s = {s_tuple: ([], []) for s_tuple in s_tuples}

    for k in sorted(sub_to_orig.keys(), reverse=True):
        sub_map = sub_to_orig[k]
        table_name = f"k{k}_{source_suffix}"
        query = text(f"SELECT s, s_, prob FROM {table_name} WHERE s IN :candidates")
        query = query.bindparams(bindparam("candidates", expanding=True))
        rows = conn.execute(query, {"candidates": list(sub_map.keys())}).fetchall()

        for sub_s_json, s_prime_json, prob in rows:
            for orig_s in sub_map[sub_s_json]:
                succ, probas = transitions_by_s[orig_s]
                succ.append(s_prime_json)  # s_prime_json est déjà une chaîne JSON
                probas.append(prob)

    return transitions_by_s


def create_row_from_transitions(s_json: str, succ_probas: tuple) -> dict:
    successors_json, probas = succ_probas
    if not successors_json:
        return None
    return {
        "s": s_json,
        "successor": "[" + ",".join(successors_json) + "]",
        "proba": _dumps(probas),
    }


def insert_rows(conn, target_table: str, rows: list):
    if not rows:
        return
    conn.execute(text(f"""
        INSERT INTO {target_table} (s, successor, proba)
        VALUES (:s, :successor, :proba)
    """), rows)


def generate_transition_dict(engine, max_k: int, source_suffix: str, target_table: str):
    table_name = f"k{max_k}_{source_suffix}"

    with engine.connect() as read_conn:
        total_count = read_conn.execute(
            text(f"SELECT COUNT(DISTINCT s) FROM {table_name}")
        ).scalar()
        if not total_count:
            return

        result = read_conn.execute(text(f"SELECT DISTINCT s FROM {table_name}"))
        processed = 0
        reset_progress()

        while True:
            rows = result.fetchmany(10000)
            if not rows:
                break

            # on garde le JSON brut pour s, et on decode avec orjson pour la clé tuple
            s_items = [(tuple(_loads(row[0])), row[0]) for row in rows]
            s_tuples = [s for s, _ in s_items]

            batch_transitions = fetch_transitions_for_batch(
                read_conn, s_tuples, source_suffix
            )

            batch = []
            for s_tuple, s_json in s_items:
                row = create_row_from_transitions(s_json, batch_transitions[s_tuple])
                if row:
                    batch.append(row)

            if batch:
                with engine.begin() as write_conn:
                    insert_rows(write_conn, target_table, batch)

            processed += len(rows)
            log_progress(processed, total_count, 'TransitionDict')

    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_{target_table}_s_cover "
            f"ON {target_table}(s, successor, proba)"
        ))