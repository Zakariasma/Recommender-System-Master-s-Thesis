import os
import sqlite3
import json
from collections import defaultdict


class SQLiteKVStore:
    def __init__(self, db_path: str, namespace: str):
        self.db_path = db_path
        self.namespace = namespace
        self.conn = sqlite3.connect(db_path, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA busy_timeout = 30000")
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS kv_store (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (namespace, key))
        """)
        self.conn.commit()

    def get(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM kv_store WHERE namespace = ? AND key = ?", (self.namespace, key)).fetchone()
        return json.loads(row[0]) if row else default

    def get_many(self, keys: list) -> dict:
        if not keys: return {}
        result = {}
        for i in range(0, len(keys), 900):
            batch_keys = keys[i:i + 900]
            placeholders = ",".join("?" * len(batch_keys))
            rows = self.conn.execute(f"SELECT key, value FROM kv_store WHERE namespace = ? AND key IN ({placeholders})", [self.namespace] + batch_keys).fetchall()
            for k, v in rows:
                result[k] = json.loads(v)
        return result

    def set_many(self, batch: dict):
        if not batch: return
        data = [(self.namespace, k, json.dumps(v)) for k, v in batch.items()]
        self.conn.executemany("INSERT INTO kv_store (namespace, key, value) VALUES (?, ?, ?) ON CONFLICT(namespace, key) DO UPDATE SET value = excluded.value", data)
        self.conn.commit()

    def close(self): self.conn.close()


class SQLiteTransitionStoreReadOnly:
    def __init__(self, db_path: str, table: str = "transitions_skipping"):
        uri = f"file:{os.path.abspath(db_path)}?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        self.table = table
        self.conn.execute("PRAGMA query_only = TRUE")
        # --- OPTIMISATIONS LECTURE (sans écriture) ---
        # Pas de journal_mode = WAL ici car la base est en lecture stricte (ro)
        self.conn.execute("PRAGMA mmap_size = 268435456")  # 256MB Memory Mapped I/O
        self.conn.execute("PRAGMA cache_size = -64000")    # 64MB de cache au lieu de 2MB
        self.conn.execute("PRAGMA temp_store = MEMORY")

    def batch_get_successors(self, prefixes: list, k: int) -> dict:
        if not prefixes: return {}
        grouped = defaultdict(dict)
        for i in range(0, len(prefixes), 900):
            batch = prefixes[i:i + 900]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn.execute(f"SELECT s, s_, prob FROM {self.table} WHERE k = ? AND s IN ({placeholders})", [k] + batch).fetchall()
            for s, s_, prob in rows:
                grouped[s][s_] = prob
        return grouped

    def get_states(self, k: int) -> list:
        rows = self.conn.execute(f"SELECT DISTINCT s FROM {self.table} WHERE k = ?", (k,)).fetchall()
        return [row[0] for row in rows]

    def close(self): self.conn.close()