import struct
from sqlalchemy import text, bindparam
from chap_5.api.mdp_rework.shared.endode_to_blob import decode, encode


def generate_substates(s_bytes: bytes) -> dict:
    """Génère les sous-états de taille k à 1 en décalant la fenêtre."""
    substates = {}
    items_tuple = decode(s_bytes)
    k = len(items_tuple)

    while k > 0:
        substates[k] = encode(items_tuple[-k:])
        k -= 1
    return substates


# On ajoute source_suffix en paramètre
def fetch_transitions_for_batch(engine, s_bytes_list: list, source_suffix: str) -> dict:
    """
    Récupère les transitions en batch pour tout un groupe d'états s.
    Retourne un dict: { s_bytes_original: { k: [(s_, prob), ...] } }
    """
    sub_to_orig = {}
    for s_bytes in s_bytes_list:
        substates = generate_substates(s_bytes)
        for k, sub_s in substates.items():
            if k not in sub_to_orig:
                sub_to_orig[k] = {}
            sub_to_orig[k][sub_s] = s_bytes

    transitions_by_s = {}

    with engine.connect() as conn:
        for k, sub_map in sub_to_orig.items():
            # CORRECTION ICI : utilisation de source_suffix
            table_name = f"k{k}_{source_suffix}"
            query = text(f"SELECT s, s_, prob FROM {table_name} WHERE s IN :candidates")
            query = query.bindparams(bindparam("candidates", expanding=True))

            rows = conn.execute(query, {"candidates": list(sub_map.keys())}).fetchall()

            for sub_s, s_, prob in rows:
                orig_s = sub_map[sub_s]
                if orig_s not in transitions_by_s:
                    transitions_by_s[orig_s] = {}
                if k not in transitions_by_s[orig_s]:
                    transitions_by_s[orig_s][k] = []
                transitions_by_s[orig_s][k].append((s_, prob))

    return transitions_by_s


def create_row_from_transitions(s_bytes: bytes, transitions_by_k: dict) -> dict:
    if not transitions_by_k:
        return None

    packed_succ = struct.pack('i', len(transitions_by_k))
    packed_probs = struct.pack('i', len(transitions_by_k))

    for k in sorted(transitions_by_k.keys(), reverse=True):
        trans = transitions_by_k[k]

        packed_succ += struct.pack('i', k)
        packed_succ += struct.pack('i', len(trans))

        packed_probs += struct.pack('i', k)
        packed_probs += struct.pack('i', len(trans))

        for s_, prob in trans:
            packed_succ += s_
            packed_probs += struct.pack('f', prob)

    return {"s": s_bytes, "successor": packed_succ, "proba": packed_probs}


def insert_rows(engine, target_table: str, rows: list):
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {target_table} (s, successor, proba)
            VALUES (:s, :successor, :proba) ON CONFLICT(s) DO
            UPDATE SET
                successor = excluded.successor,
                proba = excluded.proba
        """), rows)


def generate_transition_dict(engine, max_k: int, source_suffix: str, target_table: str):
    table_name = f"k{max_k}_{source_suffix}"
    with engine.connect() as conn:
        try:
            conn.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1")).fetchall()
        except Exception:
            return

        result = conn.execute(text(f"SELECT DISTINCT s FROM {table_name}"))
        batch = []

        while True:
            rows = result.fetchmany(10000)
            if not rows:
                break

            s_bytes_list = [s_bytes for (s_bytes,) in rows]

            # On passe source_suffix à la fonction de batch
            batch_transitions = fetch_transitions_for_batch(engine, s_bytes_list, source_suffix)

            for s_bytes in s_bytes_list:
                transitions_by_k = batch_transitions.get(s_bytes, {})
                row = create_row_from_transitions(s_bytes, transitions_by_k)
                if row:
                    batch.append(row)

            insert_rows(engine, target_table, batch)
            batch.clear()

        insert_rows(engine, target_table, batch)