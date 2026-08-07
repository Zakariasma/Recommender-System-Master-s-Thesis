import sqlite3
from collections import defaultdict

from chap_5.api.mdp_rework.shared.binary_encoder import encode_proba_list
from chap_5.api.mdp_rework.shared.debug_log import log_progress, reset_progress


def _get_conn(db_path: str) -> sqlite3.Connection:
    """Ouvre une connexion native sqlite3 et applique les PRAGMAs d'optimisation."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-2000000")
    conn.execute("PRAGMA mmap_size=268435456")
    return conn


def insert_rows(conn, target_table: str, rows: list):
    if not rows: return
    data = [
        (r["s"], r["k1_successor"], r["k1_proba"],
         r["k2_successor"], r["k2_proba"],
         r["k3_successor"], r["k3_proba"])
        for r in rows
    ]
    conn.executemany(f"""
        INSERT INTO {target_table} (s, k1_successor, k1_proba, k2_successor, k2_proba, k3_successor, k3_proba)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, data)


def generate_transition_dict(db_path: str, max_k: int, source_suffix: str, target_table: str):
    table_name = f"k{max_k}_{source_suffix}"

    read_conn = _get_conn(db_path)
    write_conn = _get_conn(db_path)
    try:
        stream_cur = read_conn.cursor()
        query_cur = read_conn.cursor()

        # Table temp de candidats, créée une seule fois et réutilisée à chaque batch
        read_conn.execute("CREATE TEMP TABLE candidates (k INTEGER, blob BLOB)")
        read_conn.execute("CREATE INDEX idx_candidates_k_blob ON candidates(k, blob)")

        stream_cur.execute(f"SELECT COUNT(DISTINCT s) FROM {table_name}")
        total_count = stream_cur.fetchone()[0]
        if not total_count: return

        stream_cur.execute(f"SELECT DISTINCT s FROM {table_name}")
        processed = 0
        reset_progress()

        while True:
            rows = stream_cur.fetchmany(10000)
            if not rows: break

            s_blobs = [row[0] for row in rows]

            sub_maps = {1: defaultdict(list), 2: defaultdict(list), 3: defaultdict(list)}
            for s_blob in s_blobs:
                sub_maps[1][s_blob[4:]].append(s_blob)
                sub_maps[2][s_blob[2:]].append(s_blob)
                sub_maps[3][s_blob].append(s_blob)

            batch_trans = {s: {1: ([], []), 2: ([], []), 3: ([], [])} for s in s_blobs}

            # --- Population de la table candidats en une fois ---
            read_conn.execute("DELETE FROM candidates")
            candidate_rows = [
                (k_val, blob)
                for k_val, sub_map in sub_maps.items()
                for blob in sub_map.keys()
            ]
            read_conn.executemany("INSERT INTO candidates (k, blob) VALUES (?, ?)", candidate_rows)

            fetched_by_k = {}
            for k_val in [1, 2, 3]:
                query_cur.execute(f"""
                    SELECT t.s, t.s_, t.prob
                    FROM k{k_val}_{source_suffix} t
                    JOIN candidates c ON c.blob = t.s AND c.k = ?
                """, (k_val,))
                fetched_by_k[k_val] = query_cur.fetchall()

            # --- Collecte : on garde les blobs bruts, aucun decode ---
            for k_val in [1, 2, 3]:
                sub_map = sub_maps[k_val]
                for sub_s_blob, s_prime_blob, prob in fetched_by_k[k_val]:
                    for orig_s_blob in sub_map[sub_s_blob]:
                        succ, probas = batch_trans[orig_s_blob][k_val]
                        succ.append(s_prime_blob)
                        probas.append(prob)

            # --- Encode : join des blobs (identique à encode_successors_fixed) ---
            batch = []
            for s_blob in s_blobs:
                s1, p1 = batch_trans[s_blob][1]
                s2, p2 = batch_trans[s_blob][2]
                s3, p3 = batch_trans[s_blob][3]
                if not s1 and not s2 and not s3: continue
                batch.append({
                    "s": s_blob,
                    "k1_successor": b"".join(s1), "k1_proba": encode_proba_list(p1),
                    "k2_successor": b"".join(s2), "k2_proba": encode_proba_list(p2),
                    "k3_successor": b"".join(s3), "k3_proba": encode_proba_list(p3)
                })

            if batch:
                try:
                    write_conn.execute("BEGIN TRANSACTION")
                    insert_rows(write_conn, target_table, batch)
                    write_conn.commit()
                except Exception:
                    write_conn.rollback()
                    raise

            processed += len(rows)
            log_progress(processed, total_count, 'TransitionDict')

        write_conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{target_table}_s ON {target_table}(s)")
        write_conn.execute("ANALYZE")
        write_conn.commit()
    finally:
        read_conn.close()
        write_conn.close()