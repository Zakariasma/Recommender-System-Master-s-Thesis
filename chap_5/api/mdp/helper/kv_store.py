import json
from sqlalchemy import create_engine, text


class KeyValueStore:
    """Une seule table Postgres (namespace, key, value) partagée par tous les usages."""

    def __init__(self, database_url: str, namespace: str):
        self.engine = create_engine(database_url)
        self.namespace = namespace
        self._init_table()

    def _init_table(self):
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    PRIMARY KEY (namespace, key))
            """))

    def get(self, key: str, default=None):
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT value FROM kv_store WHERE namespace = :ns AND key = :k"),
                {"ns": self.namespace, "k": key},
            ).fetchone()
        return row[0] if row else default

    def set_many(self, items: dict):
        if not items:
            return
        rows = [{"ns": self.namespace, "k": k, "v": json.dumps(v)} for k, v in items.items()]
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO kv_store (namespace, key, value)
                VALUES (:ns, :k, CAST(:v AS JSONB))
                ON CONFLICT (namespace, key) DO UPDATE SET value = excluded.value
            """), rows)

    def get_many(self, keys: list) -> dict:
        if not keys:
            return {}
        result = {}
        # On découpe par 900 pour éviter que l'IN clause de Postgres ne soit trop grande
        with self.engine.connect() as conn:
            for i in range(0, len(keys), 900):
                batch_keys = keys[i:i + 900]
                placeholders = ",".join([f":k{j}" for j in range(len(batch_keys))])
                params = {f"k{j}": k for j, k in enumerate(batch_keys)}
                params["ns"] = self.namespace

                query = text(f"SELECT key, value FROM kv_store WHERE namespace = :ns AND key IN ({placeholders})")
                rows = conn.execute(query, params).fetchall()
                for k, v in rows:
                    result[k] = json.loads(v)
        return result