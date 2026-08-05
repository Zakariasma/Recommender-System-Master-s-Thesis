import os
from sqlalchemy import create_engine, text

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "predictive.sqlite",
)

def create_skipping_database():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA journal_mode = WAL"))
        conn.execute(text("PRAGMA synchronous = NORMAL"))
        conn.execute(text("PRAGMA temp_store = MEMORY"))
        conn.execute(text("PRAGMA cache_size = -2000000"))
        conn.execute(text("PRAGMA mmap_size = 268435456"))
    return engine

def setup_state_full_info_table(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS state_full_info"))
        conn.execute(text("""
            CREATE TABLE state_full_info (
                s BLOB NOT NULL,
                s_ BLOB NOT NULL,
                proba_transition REAL NOT NULL,
                reward REAL NOT NULL,
                proba_reco REAL NOT NULL,
                proba_not_reco REAL NOT NULL,
                PRIMARY KEY (s, s_)
            ) WITHOUT ROWID
        """))
        conn.execute(text("CREATE INDEX idx_state_full_info_s ON state_full_info(s)"))