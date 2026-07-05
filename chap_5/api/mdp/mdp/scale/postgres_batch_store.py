from collections import defaultdict
from sqlalchemy import create_engine, text


class PostgresBatchStore:
    """Remplace SQLiteTransitionStoreReadOnly pour lire par gros paquets."""

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def batch_get_successors(self, prefixes: list, k: int) -> dict:
        if not prefixes:
            return {}

        grouped = defaultdict(dict)
        # On découpe par 900 pour éviter que l'IN clause de Postgres ne soit trop grande
        with self.engine.connect() as conn:
            for i in range(0, len(prefixes), 900):
                batch = prefixes[i:i + 900]
                in_clause = ",".join(f"'{p}'" for p in batch)
                query = text(f"SELECT s, s_, prob FROM transitions WHERE k = {k} AND s IN ({in_clause})")
                rows = conn.execute(query).fetchall()
                for s, s_, prob in rows:
                    grouped[s][s_] = prob
        return grouped