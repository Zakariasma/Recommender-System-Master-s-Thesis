import sqlite3
import os

MODEL_DIR = "../data/model"


class TransitionDB:
    _instance = None

    def __new__(cls, db_path: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.db_path = db_path
            cls._instance._init_tables()
        return cls._instance

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS counts (
                    s     TEXT NOT NULL,
                    s_    TEXT NOT NULL,
                    count REAL NOT NULL,
                    PRIMARY KEY (s, s_))
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transitions_skipping (
                    s    TEXT NOT NULL,
                    s_   TEXT NOT NULL,
                    k    INTEGER NOT NULL,
                    prob REAL NOT NULL,
                    PRIMARY KEY (s, s_, k))
            """)
            # La table similarity n'est plus utilisée, on peut la garder ou la supprimer.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transitions_similarity (
                    s    TEXT NOT NULL,
                    s_   TEXT NOT NULL,
                    k    INTEGER NOT NULL,
                    prob REAL NOT NULL,
                    PRIMARY KEY (s, s_, k))
            """)
            conn.commit()

    def tables_empty(self) -> bool:
        """Vérifie si la table de skipping est vide (modèle non entraîné)."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM transitions_skipping").fetchone()
            return row[0] == 0

    def flush_counts(self, rows: list):
        with self._connect() as conn:
            conn.executemany("""
                INSERT INTO counts (s, s_, count) VALUES (?, ?, ?)
                ON CONFLICT(s, s_) DO UPDATE SET count = count + excluded.count
            """, rows)
            conn.commit()

    def flush_skipping(self, rows: list):
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO transitions_skipping (s, s_, k, prob) VALUES (?, ?, ?, ?)", rows
            )
            conn.commit()

    def flush_similarity(self, rows: list):
        # Gardée pour compatibilité, mais plus utilisée
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO transitions_similarity (s, s_, k, prob) VALUES (?, ?, ?, ?)", rows
            )
            conn.commit()

    def normalize_and_clean(self, k: int):
        with self._connect() as conn:
            conn.execute(f"""
                INSERT INTO transitions_skipping (s, s_, k, prob)
                SELECT s, s_, {k}, count / SUM(count) OVER (PARTITION BY s)
                FROM counts
            """)
            conn.execute("DELETE FROM counts")
            conn.commit()

    def load_skipping_transitions(self, k: int) -> dict:
        tr = {}
        with self._connect() as conn:
            for s, s_, prob in conn.execute(
                "SELECT s, s_, prob FROM transitions_skipping WHERE k = ?", (k,)
            ):
                if s not in tr:
                    tr[s] = {}
                tr[s][s_] = prob
        return tr

    def get_prob(self, s: str, s_: str, k: int) -> float:
        """Lit la probabilité dans la table transitions_skipping."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT prob FROM transitions_skipping WHERE s = ? AND s_ = ? AND k = ?",
                (s, s_, k)
            ).fetchone()
        return row[0] if row else 0.0