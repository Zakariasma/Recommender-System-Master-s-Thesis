from sqlalchemy import create_engine, text


class TransitionStore:
    """Stocke et lit tr_predict(s, s') par taille de fenêtre k."""

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self._init_tables()

    def _init_tables(self):
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS counts (
                    s TEXT NOT NULL,
                    s_ TEXT NOT NULL,
                    count REAL NOT NULL,
                    PRIMARY KEY (s, s_)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS transitions (
                    s TEXT NOT NULL,
                    s_ TEXT NOT NULL,
                    k INTEGER NOT NULL,
                    prob REAL NOT NULL,
                    PRIMARY KEY (s, s_, k)
                )
            """))
            # Index dédié aux requêtes filtrant par k (get_states, get_successors, batch_get_successors)
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_transitions_k_s
                ON transitions (k, s) INCLUDE (s_, prob)
            """))

    def is_empty(self) -> bool:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT COUNT(*) FROM transitions")).fetchone()
        return row[0] == 0

    def flush_counts(self, rows: list):
        if not rows:
            return
        data = [{"s": s, "s_": s_, "c": c} for s, s_, c in rows]
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO counts (s, s_, count) VALUES (:s, :s_, :c)
                ON CONFLICT (s, s_) DO UPDATE SET count = counts.count + excluded.count
            """), data)

    def normalize_and_clean(self, k: int):
        with self.engine.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO transitions (s, s_, k, prob)
                SELECT s, s_, {k}, count / SUM(count) OVER (PARTITION BY s)
                FROM counts
            """))
            conn.execute(text("DELETE FROM counts"))

    def get_prob(self, s: str, s_: str, k: int) -> float:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT prob FROM transitions WHERE s = :s AND s_ = :s_ AND k = :k"),
                {"s": s, "s_": s_, "k": k},
            ).fetchone()
        return row[0] if row else 0.0

    def get_successors(self, s: str, k: int) -> dict:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT s_, prob FROM transitions WHERE s = :s AND k = :k"),
                {"s": s, "k": k},
            ).fetchall()
        return {s_: prob for s_, prob in rows}

    def get_states(self, k: int) -> list:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT s FROM transitions WHERE k = :k"), {"k": k}
            ).fetchall()
        return [row[0] for row in rows]