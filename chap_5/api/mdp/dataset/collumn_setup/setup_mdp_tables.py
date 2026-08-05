from sqlalchemy import create_engine, text
from chap_5.api.mdp.config import DATABASE_URL


def create_mdp_tables(engine=None):
    if engine is None:
        engine = create_engine(DATABASE_URL)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE UNLOGGED TABLE IF NOT EXISTS counts (
                    s BYTEA,
                    s_ BYTEA,
                    count REAL
                );
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS transitions (
                    s BYTEA NOT NULL,
                    s_ BYTEA NOT NULL,
                    k INTEGER NOT NULL,
                    prob REAL NOT NULL,
                    PRIMARY KEY (s, s_, k)
                );
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_transitions_k_s
                    ON transitions (k, s) INCLUDE (s_, prob);
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
                );
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS kv_state_genre (
                    state_id BIGSERIAL PRIMARY KEY,
                    state BYTEA NOT NULL UNIQUE,
                    genre_ids TEXT NOT NULL
                );
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_kv_state_genre_state
                    ON kv_state_genre (state);
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS reverse_kv_genre_state (
                    genre_subset TEXT NOT NULL,
                    state_id BIGINT NOT NULL,
                    PRIMARY KEY (genre_subset, state_id)
                );
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_reverse_kv_subset
                    ON reverse_kv_genre_state (genre_subset);
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS sim_transition (
                    s BYTEA NOT NULL,
                    s_ BYTEA NOT NULL,
                    prob REAL NOT NULL,
                    PRIMARY KEY (s, s_)
                );
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_sim_transition_s
                    ON sim_transition (s) INCLUDE (s_, prob);
                """
            )
        )