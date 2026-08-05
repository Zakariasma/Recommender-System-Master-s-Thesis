import sqlite3


class SQLiteStore:
    def __init__(self, db_path: str = "sim_index.sqlite"):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode = MEMORY;")
        self.conn.execute("PRAGMA synchronous = OFF;")

    def setup_tables(self):
        """Crée toutes les tables nécessaires."""
        cur = self.conn.cursor()
        cur.executescript("""
                          DROP TABLE IF EXISTS state_genres;
                          CREATE TABLE state_genres
                          (
                              state_id  INTEGER PRIMARY KEY,
                              state     BLOB NOT NULL UNIQUE,
                              genre_ids TEXT NOT NULL
                          );

                          DROP TABLE IF EXISTS reverse_index;
                          CREATE TABLE reverse_index
                          (
                              genre_subset TEXT    NOT NULL,
                              state_id     INTEGER NOT NULL
                          );

                          DROP TABLE IF EXISTS sim_transition;
                          CREATE TABLE sim_transition
                          (
                              s    BLOB NOT NULL,
                              s_   BLOB NOT NULL,
                              prob REAL NOT NULL,
                              PRIMARY KEY (s, s_)
                          );
                          """)
        self.conn.commit()

    def insert_state_genres_batch(self, rows: list):
        cur = self.conn.cursor()
        cur.executemany("INSERT INTO state_genres VALUES (?, ?, ?)", rows)
        self.conn.commit()

    def insert_reverse_index_batch(self, rows: list):
        cur = self.conn.cursor()
        cur.executemany("INSERT INTO reverse_index VALUES (?, ?)", rows)
        self.conn.commit()

    def insert_sim_transition_batch(self, rows: list):
        """Insère les nouvelles transitions calculées par la similarité."""
        cur = self.conn.cursor()
        cur.executemany("INSERT OR REPLACE INTO sim_transition VALUES (?, ?, ?)", rows)
        self.conn.commit()

    def create_indexes(self):
        """Crée les index à la fin de l'insertion (beaucoup plus rapide)."""
        cur = self.conn.cursor()
        cur.execute("CREATE INDEX idx_state_genres ON state_genres(genre_ids);")
        cur.execute("CREATE INDEX idx_reverse ON reverse_index(genre_subset);")
        self.conn.commit()

    def get_all_states(self):
        """Récupère tous les états pour le calcul de similarité."""
        cur = self.conn.cursor()
        cur.execute("SELECT state_id, state, genre_ids FROM state_genres")
        return cur.fetchall()

    def get_reverse_candidates(self, subset_keys: list):
        """Récupère les candidats pour une liste de sous-clés (découpé en chunks sûrs pour SQLite)."""
        cur = self.conn.cursor()
        results = []
        SQLITE_CHUNK_SIZE = 500  # Limite sûre pour SQLite

        for i in range(0, len(subset_keys), SQLITE_CHUNK_SIZE):
            chunk = subset_keys[i:i + SQLITE_CHUNK_SIZE]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT genre_subset, state_id FROM reverse_index WHERE genre_subset IN ({placeholders})"
            cur.execute(query, chunk)
            results.extend(cur.fetchall())

        return results

    def close(self):
        if self.conn:
            self.conn.close()

    def clean_index_tables(self):
        """Supprime les tables d'index pour libérer de l'espace."""
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS state_genres;")
        cur.execute("DROP TABLE IF EXISTS reverse_index;")
        self.conn.commit()