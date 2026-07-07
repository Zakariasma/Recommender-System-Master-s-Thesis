from sqlalchemy import create_engine, text
from chap_5.api.mdp.config import DATABASE_URL


def create_mdp_tables(engine=None):
    if engine is None:
        engine = create_engine(DATABASE_URL)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS counts (
                    s TEXT NOT NULL,
                    s_ TEXT NOT NULL,
                    count REAL NOT NULL,
                    PRIMARY KEY (s, s_)
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS transitions (
                    s TEXT NOT NULL,
                    s_ TEXT NOT NULL,
                    k INTEGER NOT NULL,
                    prob REAL NOT NULL,
                    PRIMARY KEY (s, s_, k)
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_transitions_k_s
                ON transitions (k, s) INCLUDE (s_, prob)
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS kv_store (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
        )