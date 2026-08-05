import struct
from sqlalchemy import text

def generate_transition_dict(engine, max_k: int, source_suffix: str, target_table: str):
    for current_k in range(max_k, 0, -1):
        table_name = f"k{current_k}_{source_suffix}"
        with engine.connect() as conn:
            try:
                conn.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1")).fetchall()
            except Exception:
                continue

        query = text(f"SELECT s, s_, prob FROM {table_name} ORDER BY s")

        data = []
        current_s = None
        current_transitions = []

        with engine.connect() as conn:
            result = conn.execute(query)

            while True:
                rows = result.fetchmany(10000)
                if not rows:
                    break

                for s, s_, prob in rows:
                    if s != current_s:
                        if current_s is not None:
                            packed_succ = b"".join(blob for blob, _ in current_transitions)
                            packed_probs = struct.pack(f'{len(current_transitions)}f',
                                                       *[p for _, p in current_transitions])
                            data.append({"s": current_s, "successor": packed_succ, "proba": packed_probs})

                            if len(data) >= 10000:
                                with engine.begin() as conn_insert:
                                    conn_insert.execute(text(f"""
                                        INSERT INTO {target_table} (s, successor, proba)
                                        VALUES (:s, :successor, :proba) ON CONFLICT(s) DO
                                        UPDATE SET
                                            successor = excluded.successor,
                                            proba = excluded.proba
                                        """), data)
                                data.clear()
                        current_s = s
                        current_transitions = []
                    current_transitions.append((s_, prob))

            if current_s is not None:
                packed_succ = b"".join(blob for blob, _ in current_transitions)
                packed_probs = struct.pack(f'{len(current_transitions)}f', *[p for _, p in current_transitions])
                data.append({"s": current_s, "successor": packed_succ, "proba": packed_probs})

            if data:
                with engine.begin() as conn:
                    conn.execute(text(f"""
                        INSERT INTO {target_table} (s, successor, proba)
                        VALUES (:s, :successor, :proba) ON CONFLICT(s) DO
                        UPDATE SET
                            successor = excluded.successor,
                            proba = excluded.proba
                        """), data)