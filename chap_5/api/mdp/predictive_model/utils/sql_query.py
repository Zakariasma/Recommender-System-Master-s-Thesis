import io
from collections import defaultdict
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values


class TransitionStore:

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)

    def is_empty(self) -> bool:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT COUNT(*) FROM transitions")).fetchone()
        return row[0] == 0

    def flush_counts(self, rows_iterator):
        if not rows_iterator:
            return

        merged = {}
        for s_bytes, s_next_bytes, c in rows_iterator:
            key = (s_bytes, s_next_bytes)
            merged[key] = merged.get(key, 0.0) + c

        buffer = io.StringIO("".join(
            f"\\\\x{s.hex()}\t\\\\x{s_.hex()}\t{c}\n"
            for (s, s_), c in merged.items()
        ))
        buffer.seek(0)

        raw_conn = self.engine.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                cur.execute("SET synchronous_commit = off;")
                cur.copy_from(buffer, 'counts', sep='\t')
            raw_conn.commit()
        except Exception:
            raw_conn.rollback()
            raise
        finally:
            raw_conn.close()

    def normalize_and_clean(self, k: int):
        with self.engine.begin() as conn:
            conn.execute(text("SET work_mem = '1GB'"))
            conn.execute(text("SET max_parallel_workers_per_gather = 4"))

            conn.execute(text(f"""
                INSERT INTO transitions (s, s_, k, prob)
                SELECT s, s_, {k}, cnt / SUM(cnt) OVER (PARTITION BY s)
                FROM (
                    SELECT s, s_, SUM(count) AS cnt
                    FROM counts
                    GROUP BY s, s_
                ) t
            """))
            conn.execute(text("TRUNCATE TABLE counts"))
            conn.execute(text("RESET work_mem"))

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

    # =========================================================
    # NOUVELLES MÉTHODES POUR LA SIMILARITÉ
    # =========================================================

    def get_all_distinct_states(self) -> list:
        """Récupère tous les états uniques (BYTEA) depuis la table transitions."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT DISTINCT s FROM transitions")).fetchall()
        return [row[0] for row in rows]

    def get_movie_genres(self) -> dict:
        """Récupère tous les movie_genre et les met dans un dico {movie_id: [genre_ids]}."""
        movie_genres = defaultdict(list)
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT movie_id, genre_id FROM movie_genre")).fetchall()
        for movie_id, genre_id in rows:
            movie_genres[int(movie_id)].append(int(genre_id))
        return dict(movie_genres)

    def flush_state_genres(self, rows: list):
        """Insère les mappings état -> genres en masse dans kv_state_genre."""
        if not rows:
            return
        raw_conn = self.engine.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                cur.execute("SET synchronous_commit = off;")
                query = """
                    INSERT INTO kv_state_genre (state_id, state, genre_ids) VALUES %s
                    ON CONFLICT (state) DO NOTHING
                """
                execute_values(cur, query, rows, page_size=10000)
            raw_conn.commit()
        except Exception:
            raw_conn.rollback()
            raise
        finally:
            raw_conn.close()

    def get_state_genres(self) -> list:
        """Récupère les state_id et genre_ids depuis kv_state_genre."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT state_id, genre_ids FROM kv_state_genre")).fetchall()
        return [(row[0], row[1]) for row in rows]

    def flush_reverse_index(self, csv_buffer_string: str):
        """Insère les données avec COPY (vitesse maximale absolue)."""
        if not csv_buffer_string:
            return

        import io
        buffer = io.StringIO(csv_buffer_string)
        buffer.seek(0)

        raw_conn = self.engine.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                cur.execute("SET synchronous_commit = off;")
                cur.copy_from(buffer, 'reverse_kv_genre_state', sep='\t')
            raw_conn.commit()
        except Exception:
            raw_conn.rollback()
            raise
        finally:
            raw_conn.close()

