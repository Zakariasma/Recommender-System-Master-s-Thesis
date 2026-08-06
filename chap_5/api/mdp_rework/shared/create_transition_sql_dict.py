from collections import defaultdict
from sqlalchemy import text, bindparam
from chap_5.api.mdp_rework.shared.binary_encoder import (
    decode_state, encode_state,
    encode_successors_fixed, encode_proba_list
)
from chap_5.api.mdp_rework.shared.debug_log import log_progress, reset_progress


def generate_substates(s_tuple: tuple) -> dict:
    """Génère les sous-états de taille k à 1 en décalant la fenêtre."""
    substates = {}
    k = len(s_tuple)
    while k > 0:
        substates[k] = s_tuple[-k:]
        k -= 1
    return substates


def fetch_transitions_for_k(conn, s_blobs: list, k: int, source_suffix: str) -> dict:
    """Récupère les transitions pour une taille k donnée en générant les sous-états."""
    sub_to_orig = defaultdict(list)  # <-- CORRECTION ICI
    for s_blob in s_blobs:
        s_tuple = decode_state(s_blob)
        sub_s_tuple = s_tuple[-k:] if k <= len(s_tuple) else s_tuple
        sub_s_blob = encode_state(sub_s_tuple)
        sub_to_orig[sub_s_blob].append(s_blob)

    table_name = f"k{k}_{source_suffix}"
    query = text(f"SELECT s, s_, prob FROM {table_name} WHERE s IN :candidates")
    query = query.bindparams(bindparam("candidates", expanding=True))

    trans_map = {s: ([], []) for s in s_blobs}

    candidates = list(sub_to_orig.keys())
    for i in range(0, len(candidates), 500):
        chunk = candidates[i:i + 500]
        rows = conn.execute(query, {"candidates": chunk}).fetchall()
        for sub_s_blob, s_prime_blob, prob in rows:
            for orig_s_blob in sub_to_orig[sub_s_blob]:
                if orig_s_blob in trans_map:
                    succ, probas = trans_map[orig_s_blob]
                    succ.append(decode_state(s_prime_blob))
                    probas.append(prob)

    return trans_map

def insert_rows(conn, target_table: str, rows: list):
    if not rows: return
    conn.execute(text(f"""
        INSERT INTO {target_table} (s, k1_successor, k1_proba, k2_successor, k2_proba, k3_successor, k3_proba)
        VALUES (:s, :k1_successor, :k1_proba, :k2_successor, :k2_proba, :k3_successor, :k3_proba)
    """), rows)


def generate_transition_dict(engine, max_k: int, source_suffix: str, target_table: str):
    table_name = f"k{max_k}_{source_suffix}"
    with engine.connect() as read_conn:
        total_count = read_conn.execute(text(f"SELECT COUNT(DISTINCT s) FROM {table_name}")).scalar()
        if not total_count: return

        result = read_conn.execute(text(f"SELECT DISTINCT s FROM {table_name}"))
        processed = 0
        reset_progress()

        while True:
            rows = result.fetchmany(10000)
            if not rows: break

            s_blobs = [row[0] for row in rows]

            # Récupérer les transitions pour k=1, k=2, k=3 en utilisant les sous-états
            trans_k1 = fetch_transitions_for_k(read_conn, s_blobs, 1, source_suffix)
            trans_k2 = fetch_transitions_for_k(read_conn, s_blobs, 2, source_suffix)
            trans_k3 = fetch_transitions_for_k(read_conn, s_blobs, 3, source_suffix)

            batch = []
            for s_blob in s_blobs:
                s1, p1 = trans_k1.get(s_blob, ([], []))
                s2, p2 = trans_k2.get(s_blob, ([], []))
                s3, p3 = trans_k3.get(s_blob, ([], []))

                # S'il n'y a aucune transition du tout, on skip
                if not s1 and not s2 and not s3: continue

                batch.append({
                    "s": s_blob,
                    "k1_successor": encode_successors_fixed(s1),
                    "k1_proba": encode_proba_list(p1),
                    "k2_successor": encode_successors_fixed(s2),
                    "k2_proba": encode_proba_list(p2),
                    "k3_successor": encode_successors_fixed(s3),
                    "k3_proba": encode_proba_list(p3)
                })

            if batch:
                with engine.begin() as write_conn:
                    insert_rows(write_conn, target_table, batch)

            processed += len(rows)
            log_progress(processed, total_count, 'TransitionDict')

    with engine.begin() as conn:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{target_table}_s ON {target_table}(s)"))